from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scan2hwpx.contracts import EvidenceIR, EvidencePage, contract_sha256
from scan2hwpx.evaluation.strict_json import require_strict_json_bytes
from scan2hwpx.model_a.block_role_vector import (
    ModelABlockRoleVectorPageResult,
    ModelABlockRoleVectorSegmentTrace,
    build_model_a_block_role_vector_requests,
    compile_model_a_block_role_vector_responses,
)
from scan2hwpx.model_a.inference import ModelAPageAnalysis, ModelAPageImage
from scan2hwpx.model_a.table_fusion import (
    BuiltModelATableEvidencePromotion,
    BuiltModelATablePageFusion,
    assemble_model_a_table_fused_document,
    fuse_model_a_table_page_analysis,
    promote_model_a_table_evidence,
)
from scan2hwpx.model_a.table_topology import (
    BuiltModelATableTopology,
    ModelATableTopologyArtifact,
    validate_model_a_table_topology,
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_MAX_JSON_DEPTH = 96
_MAX_JSON_NODES = 2_000_000


@dataclass(frozen=True, slots=True)
class _ValidatedSemantic:
    analysis: ModelAPageAnalysis
    result_sha256: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly fuse local table topology with block-role Model A outputs."
    )
    parser.add_argument("--evidence-ir", type=Path, required=True)
    parser.add_argument("--pages-root", type=Path, required=True)
    parser.add_argument("--topology-batch-dir", type=Path, required=True)
    parser.add_argument("--block-role-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _eligibility() -> dict[str, object]:
    return {
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
    }


def _write_json(path: Path, value: object) -> bytes:
    payload = _canonical_json_bytes(value)
    path.write_bytes(payload)
    return payload


def _write_sha(path: Path, digest: str, target_name: str) -> None:
    path.write_bytes(f"{digest}  {target_name}\n".encode("ascii"))


def _write_artifact(path: Path, payload: bytes) -> str:
    digest = _sha256(payload)
    path.write_bytes(payload)
    _write_sha(path.with_suffix(".sha256"), digest, path.name)
    return digest


def _read_required(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"missing {label}")
    return path.read_bytes()


def _strict_json_dict(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_required(path, label)
    require_strict_json_bytes(
        raw,
        max_depth=_MAX_JSON_DEPTH,
        max_nodes=_MAX_JSON_NODES,
        label=label,
    )
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value, raw


def _page_output_key(page_no: int, page_id: str) -> str:
    return f"page-{page_no:04d}-{_sha256(page_id.encode('utf-8'))[:12]}"


def _child_dir(root: Path, name: str) -> Path:
    child = (root / name).resolve()
    if child.parent != root:
        raise ValueError("artifact directory escapes configured batch root")
    return child


def _page_image_path(pages_root: Path, page_id: str) -> Path:
    matches: list[Path] = []
    for suffix in _IMAGE_SUFFIXES:
        candidate = (pages_root / f"{page_id}{suffix}").resolve()
        if candidate.parent != pages_root:
            raise ValueError("page id cannot escape configured pages root")
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"page image must resolve exactly once; found {len(matches)}")
    return matches[0]


def _media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError("unsupported page image extension")


def _load_page_image(pages_root: Path, page: EvidencePage) -> ModelAPageImage:
    path = _page_image_path(pages_root, page.id)
    raw = path.read_bytes()
    return ModelAPageImage(
        page_id=page.id,
        page_no=page.page_no,
        image_source_ref=page.image_source_ref,
        media_type=_media_type(path),
        raw_bytes=raw,
        sha256=_sha256(raw),
    )


def _summary_records(
    path: Path,
    *,
    evidence_ir: EvidenceIR,
    pages: tuple[EvidencePage, ...],
    schema_version: str,
    success_outcome: str,
    success_count_key: str,
    error_count_keys: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], str]:
    summary, raw = _strict_json_dict(path, f"{schema_version} summary")
    if summary.get("schema_version") != schema_version:
        raise ValueError("batch summary schema version mismatch")
    if summary.get("evidence_ir_contract_sha256") != contract_sha256(evidence_ir):
        raise ValueError("batch summary EvidenceIR digest mismatch")
    if summary.get("page_count") != len(pages):
        raise ValueError("batch summary page count mismatch")
    if summary.get(success_count_key) != len(pages):
        raise ValueError("batch summary does not report every page successful")
    if any(summary.get(key) != 0 for key in error_count_keys):
        raise ValueError("batch summary reports blocked or failed pages")
    if (
        summary.get("research_only") is not True
        or summary.get("golden_eligible") is not False
        or summary.get("training_eligible") is not False
        or summary.get("release_eligible") is not False
    ):
        raise ValueError("batch summary eligibility boundary mismatch")
    records = summary.get("pages")
    if not isinstance(records, list):
        raise TypeError("batch summary pages must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("page_id"), str):
            raise TypeError("batch summary page record is invalid")
        page_id = record["page_id"]
        if page_id in by_id or record.get("outcome") != success_outcome:
            raise ValueError("batch summary page outcomes are invalid")
        by_id[page_id] = record
    expected_ids = {page.id for page in pages}
    if set(by_id) != expected_ids:
        raise ValueError("batch summary page coverage mismatch")
    for page in pages:
        record = by_id[page.id]
        if record.get("page_no") != page.page_no or record.get(
            "page_output_key"
        ) != _page_output_key(page.page_no, page.id):
            raise ValueError("batch summary page identity mismatch")
    return by_id, _sha256(raw)


def _validate_page_binding(
    path: Path,
    *,
    page: EvidencePage,
    page_key: str,
    page_image: ModelAPageImage,
    evidence_ir: EvidenceIR,
    topology: bool,
) -> None:
    binding, _raw = _strict_json_dict(path, "page binding")
    if (
        binding.get("page_id") != page.id
        or binding.get("page_no") != page.page_no
        or binding.get("page_output_key") != page_key
    ):
        label = "topology" if topology else "semantic"
        raise ValueError(f"{label} page binding mismatch")
    if topology and (
        binding.get("status") != "exact_digest_match"
        or binding.get("evidence_ir_contract_sha256") != contract_sha256(evidence_ir)
        or binding.get("page_image_source_ref") != page.image_source_ref
        or binding.get("actual_page_image_sha256") != page_image.sha256
        or binding.get("expected_page_image_sha256") != page_image.sha256
    ):
        raise ValueError("topology page binding mismatch")
    if not topology and binding.get("page_image_sha256") != page_image.sha256:
        raise ValueError("semantic page binding mismatch")


def _load_topology(
    evidence_ir: EvidenceIR,
    page: EvidencePage,
    page_image: ModelAPageImage,
    *,
    batch_root: Path,
    summary_record: dict[str, Any] | None,
) -> BuiltModelATableTopology:
    page_key = _page_output_key(page.page_no, page.id)
    page_dir = _child_dir(batch_root, page_key)
    raw = _read_required(page_dir / "topology.json", "topology artifact")
    require_strict_json_bytes(
        raw,
        max_depth=_MAX_JSON_DEPTH,
        max_nodes=_MAX_JSON_NODES,
        label="topology artifact",
    )
    artifact = ModelATableTopologyArtifact.model_validate_json(raw, strict=True)
    canonical = _canonical_json_bytes(artifact.model_dump(mode="json"))
    if raw != canonical:
        raise ValueError("topology artifact bytes are not canonical")
    digest = _sha256(raw)
    expected_sha_file = f"{digest}  topology.json\n".encode("ascii")
    if _read_required(page_dir / "topology.sha256", "topology SHA") != expected_sha_file:
        raise ValueError("topology SHA file mismatch")
    if summary_record is not None and summary_record.get("topology_sha256") != digest:
        raise ValueError("topology summary digest mismatch")
    _validate_page_binding(
        page_dir / "page-binding.json",
        page=page,
        page_key=page_key,
        page_image=page_image,
        evidence_ir=evidence_ir,
        topology=True,
    )
    built = validate_model_a_table_topology(
        artifact,
        evidence_ir=evidence_ir,
        page_image=page_image,
    )
    if built.artifact_bytes != raw or built.artifact_sha256 != digest:
        raise ValueError("topology replay does not match stored artifact")
    return built


def _load_semantic(
    evidence_ir: EvidenceIR,
    page: EvidencePage,
    page_image: ModelAPageImage,
    *,
    batch_root: Path,
    summary_record: dict[str, Any] | None,
) -> _ValidatedSemantic:
    page_key = _page_output_key(page.page_no, page.id)
    page_dir = _child_dir(batch_root, page_key)
    raw_result = _read_required(page_dir / "result.json", "semantic result")
    require_strict_json_bytes(
        raw_result,
        max_depth=_MAX_JSON_DEPTH,
        max_nodes=_MAX_JSON_NODES,
        label="semantic result",
    )
    stored = ModelABlockRoleVectorPageResult.model_validate_json(raw_result, strict=True)
    if raw_result != _canonical_json_bytes(stored.model_dump(mode="json")):
        raise ValueError("semantic result bytes are not canonical")
    if stored.outcome != "valid_unverified" or stored.compiled_page_analysis is None:
        raise ValueError("semantic result is not valid_unverified")
    if (
        stored.source.evidence_ir_id != evidence_ir.id
        or stored.source.evidence_ir_contract_sha256 != contract_sha256(evidence_ir)
        or stored.source.target_page_id != page.id
        or stored.source.target_page_no != page.page_no
        or stored.source.page_image_source_ref != page.image_source_ref
        or stored.source.page_image_sha256 != page_image.sha256
    ):
        raise ValueError("semantic result source binding mismatch")
    result_sha256 = _sha256(raw_result)
    if summary_record is not None and summary_record.get("result_sha256") != result_sha256:
        raise ValueError("semantic summary result digest mismatch")
    _validate_page_binding(
        page_dir / "page-binding.json",
        page=page,
        page_key=page_key,
        page_image=page_image,
        evidence_ir=evidence_ir,
        topology=False,
    )

    requests = build_model_a_block_role_vector_requests(
        evidence_ir,
        page_image=page_image,
        model=stored.model,
    )
    raw_responses: list[bytes] = []
    for request in requests:
        segment_no = request.artifact.segment_index + 1
        segment_dir = page_dir / "segments" / f"segment-{segment_no:04d}"
        if _read_required(segment_dir / "request.json", "semantic segment request") != (
            request.raw_request_bytes
        ):
            raise ValueError("semantic segment request cannot be reproduced")
        response = _read_required(
            segment_dir / "raw-assistant-content.bin",
            "semantic segment raw response",
        )
        raw_responses.append(response)
        trace_raw = _read_required(segment_dir / "trace.json", "semantic segment trace")
        require_strict_json_bytes(
            trace_raw,
            max_depth=_MAX_JSON_DEPTH,
            max_nodes=_MAX_JSON_NODES,
            label="semantic segment trace",
        )
        trace = ModelABlockRoleVectorSegmentTrace.model_validate_json(trace_raw, strict=True)
        if trace != stored.segments[request.artifact.segment_index]:
            raise ValueError("semantic segment trace mismatch")
        execution, _execution_raw = _strict_json_dict(
            segment_dir / "execution.json",
            "semantic segment execution",
        )
        if (
            execution.get("status") != "received"
            or execution.get("request_sha256") != request.raw_request_sha256
            or execution.get("assistant_content_sha256") != _sha256(response)
            or execution.get("assistant_content_size_bytes") != len(response)
        ):
            raise ValueError("semantic segment execution provenance mismatch")
    rebuilt = compile_model_a_block_role_vector_responses(
        requests,
        tuple(raw_responses),
        max_response_bytes=stored.response_byte_limit,
    )
    if rebuilt.result_bytes != raw_result or rebuilt.artifact != stored:
        raise ValueError("semantic result cannot be reproduced from stored segment artifacts")
    compiled = _read_required(
        page_dir / "compiled-page-analysis.json",
        "compiled semantic page analysis",
    )
    if rebuilt.compiled_page_analysis_bytes != compiled:
        raise ValueError("compiled semantic page analysis bytes mismatch")
    require_strict_json_bytes(
        compiled,
        max_depth=_MAX_JSON_DEPTH,
        max_nodes=_MAX_JSON_NODES,
        label="compiled semantic page analysis",
    )
    analysis = ModelAPageAnalysis.model_validate_json(compiled, strict=True)
    if analysis != rebuilt.artifact.compiled_page_analysis:
        raise ValueError("compiled semantic page analysis artifact mismatch")
    return _ValidatedSemantic(analysis=analysis, result_sha256=result_sha256)


def _issue(stage: str, exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return f"{stage}:{type(exc).__name__}:{message}"[:2_048]


def _failure_payload(
    *,
    role: str,
    issues: Sequence[str],
    page: EvidencePage | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "model-a-table-fusion-failure/1.0",
        **_eligibility(),
        "artifact_role": role,
        "status": "blocked",
        "issues": list(issues),
    }
    if page is not None:
        payload.update(
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "page_output_key": _page_output_key(page.page_no, page.id),
            }
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_path = args.evidence_ir.resolve()
    evidence_raw = _read_required(evidence_path, "base EvidenceIR")
    require_strict_json_bytes(
        evidence_raw,
        max_depth=_MAX_JSON_DEPTH,
        max_nodes=_MAX_JSON_NODES,
        label="base EvidenceIR",
    )
    evidence_ir = EvidenceIR.model_validate_json(evidence_raw, strict=True)
    pages = tuple(sorted(evidence_ir.pages, key=lambda page: (page.page_no, page.id)))
    pages_root = args.pages_root.resolve()
    topology_root = args.topology_batch_dir.resolve()
    semantic_root = args.block_role_batch_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    global_issues: list[str] = []
    topology_summary_sha256: str | None = None
    semantic_summary_sha256: str | None = None
    try:
        topology_records, topology_summary_sha256 = _summary_records(
            topology_root / "summary.json",
            evidence_ir=evidence_ir,
            pages=pages,
            schema_version="model-a-table-topology-batch/1.0",
            success_outcome="generated_unverified",
            success_count_key="generated_unverified_count",
            error_count_keys=("execution_error_count",),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        topology_records = {}
        global_issues.append(_issue("topology_batch_summary", exc))
    try:
        semantic_records, semantic_summary_sha256 = _summary_records(
            semantic_root / "summary.json",
            evidence_ir=evidence_ir,
            pages=pages,
            schema_version="model-a-block-role-vector-smoke/2.0",
            success_outcome="valid_unverified",
            success_count_key="valid_unverified_count",
            error_count_keys=("blocked_count", "execution_error_count"),
        )
        semantic_summary, _raw = _strict_json_dict(
            semantic_root / "summary.json",
            "semantic batch summary",
        )
        stability = semantic_summary.get("model_stability")
        if not isinstance(stability, dict) or stability.get("status") != "stable":
            raise ValueError("semantic batch model stability is not verified")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        semantic_records = {}
        global_issues.append(_issue("semantic_batch_summary", exc))

    page_images: dict[str, ModelAPageImage] = {}
    topologies: dict[str, BuiltModelATableTopology] = {}
    semantics: dict[str, _ValidatedSemantic] = {}
    page_input_issues: dict[str, list[str]] = {page.id: [] for page in pages}
    for page in pages:
        try:
            page_images[page.id] = _load_page_image(pages_root, page)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            page_input_issues[page.id].append(_issue("page_image", exc))
            continue
        try:
            topologies[page.id] = _load_topology(
                evidence_ir,
                page,
                page_images[page.id],
                batch_root=topology_root,
                summary_record=topology_records.get(page.id),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            page_input_issues[page.id].append(_issue("topology", exc))
        try:
            semantics[page.id] = _load_semantic(
                evidence_ir,
                page,
                page_images[page.id],
                batch_root=semantic_root,
                summary_record=semantic_records.get(page.id),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            page_input_issues[page.id].append(_issue("semantic", exc))

    promotion: BuiltModelATableEvidencePromotion | None = None
    promotion_issue: str | None = None
    topology_summary_valid = not any(
        issue.startswith("topology_batch_summary:") for issue in global_issues
    )
    if topology_summary_valid and len(topologies) == len(pages):
        try:
            promotion = promote_model_a_table_evidence(
                evidence_ir,
                topologies=tuple(topologies[page.id] for page in pages),
            )
            _write_artifact(output_dir / "promotion.json", promotion.artifact_bytes)
            promoted_bytes = _canonical_json_bytes(
                promotion.promoted_evidence_ir.model_dump(mode="json")
            )
            _write_artifact(output_dir / "promoted-evidence-ir.json", promoted_bytes)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            promotion_issue = _issue("promotion", exc)
    else:
        promotion_issue = "promotion:blocked:validated topology coverage is incomplete"
    if promotion is None:
        _write_json(
            output_dir / "promotion-failure.json",
            _failure_payload(
                role="table_evidence_promotion",
                issues=(*global_issues, promotion_issue),
            ),
        )

    semantic_summary_valid = not any(
        issue.startswith("semantic_batch_summary:") for issue in global_issues
    )
    page_fusions: dict[str, BuiltModelATablePageFusion] = {}
    page_summaries: list[dict[str, object]] = []
    for page in pages:
        page_key = _page_output_key(page.page_no, page.id)
        page_dir = output_dir / page_key
        page_dir.mkdir()
        issues = list(page_input_issues[page.id])
        if promotion is None:
            issues.append("fusion:blocked:promotion is unavailable")
        if not semantic_summary_valid:
            issues.append("fusion:blocked:semantic batch summary is invalid")
        topology = topologies.get(page.id)
        semantic = semantics.get(page.id)
        fusion: BuiltModelATablePageFusion | None = None
        if not issues and promotion is not None and topology is not None and semantic is not None:
            try:
                fusion = fuse_model_a_table_page_analysis(
                    promotion,
                    semantic_page_analysis=semantic.analysis,
                )
                page_fusions[page.id] = fusion
                _write_artifact(page_dir / "fusion.json", fusion.artifact_bytes)
                fused_analysis_bytes = _canonical_json_bytes(
                    fusion.artifact.fused_page_analysis.model_dump(mode="json")
                )
                _write_artifact(
                    page_dir / "fused-page-analysis.json",
                    fused_analysis_bytes,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                issues.append(_issue("fusion", exc))
        if fusion is None:
            _write_json(
                page_dir / "failure.json",
                _failure_payload(role="table_page_fusion", issues=issues, page=page),
            )
        _write_json(
            page_dir / "page-binding.json",
            {
                "schema_version": "model-a-table-fusion-page-binding/1.0",
                **_eligibility(),
                "status": "fused_unverified" if fusion is not None else "blocked",
                "base_evidence_ir_id": evidence_ir.id,
                "base_evidence_ir_sha256": contract_sha256(evidence_ir),
                "page_id": page.id,
                "page_no": page.page_no,
                "page_output_key": page_key,
                "page_image_sha256": (
                    None if page.id not in page_images else page_images[page.id].sha256
                ),
                "topology_artifact_sha256": (
                    None if topology is None else topology.artifact_sha256
                ),
                "semantic_result_sha256": (None if semantic is None else semantic.result_sha256),
                "promotion_artifact_sha256": (
                    None if promotion is None else promotion.artifact_sha256
                ),
                "fusion_artifact_sha256": (None if fusion is None else fusion.artifact_sha256),
            },
        )
        page_summaries.append(
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "page_output_key": page_key,
                "outcome": "fused_unverified" if fusion is not None else "blocked",
                "issues": issues,
                "table_count": 0 if topology is None else len(topology.artifact.tables),
                "semantic_node_count": (0 if semantic is None else len(semantic.analysis.nodes)),
                "fused_node_count": (
                    0 if fusion is None else len(fusion.artifact.fused_page_analysis.nodes)
                ),
                "shared_text_resolution_count": (
                    0 if fusion is None else len(fusion.artifact.shared_text_resolutions)
                ),
                "fusion_artifact_sha256": (None if fusion is None else fusion.artifact_sha256),
                "fused_page_analysis_sha256": (
                    None if fusion is None else fusion.artifact.fused_page_analysis_sha256
                ),
            }
        )

    assembly = None
    assembly_issue: str | None = None
    if promotion is not None and len(page_fusions) == len(pages):
        try:
            assembly = assemble_model_a_table_fused_document(
                promotion,
                page_fusions=tuple(page_fusions[page.id] for page in pages),
            )
            _write_artifact(output_dir / "assembly.json", assembly.artifact_bytes)
            content_bytes = _canonical_json_bytes(
                assembly.artifact.content_ir.model_dump(mode="json")
            )
            _write_artifact(output_dir / "content-ir.json", content_bytes)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            assembly_issue = _issue("assembly", exc)
    else:
        assembly_issue = "assembly:blocked:complete fused page coverage is unavailable"
    if assembly is None:
        _write_json(
            output_dir / "assembly-failure.json",
            _failure_payload(
                role="table_fused_document_assembly",
                issues=(assembly_issue,),
            ),
        )

    failed = bool(global_issues) or promotion is None or assembly is None
    summary = {
        "schema_version": "model-a-table-fusion-batch/1.0",
        **_eligibility(),
        "execution_policy": "strict_replay_local_no_network_no_retry",
        "base_evidence_ir_id": evidence_ir.id,
        "base_evidence_ir_raw_sha256": _sha256(evidence_raw),
        "base_evidence_ir_contract_sha256": contract_sha256(evidence_ir),
        "source_document_sha256": evidence_ir.source_document_sha256,
        "topology_batch_summary_sha256": topology_summary_sha256,
        "semantic_batch_summary_sha256": semantic_summary_sha256,
        "global_issues": global_issues,
        "page_count": len(pages),
        "validated_topology_count": len(topologies),
        "validated_semantic_count": len(semantics),
        "fused_unverified_count": len(page_fusions),
        "blocked_page_count": len(pages) - len(page_fusions),
        "promotion_outcome": "generated_unverified" if promotion is not None else "blocked",
        "promotion_artifact_sha256": (None if promotion is None else promotion.artifact_sha256),
        "promoted_evidence_ir_sha256": (
            None if promotion is None else contract_sha256(promotion.promoted_evidence_ir)
        ),
        "assembly_outcome": "generated_unverified" if assembly is not None else "blocked",
        "assembly_artifact_sha256": (None if assembly is None else assembly.artifact_sha256),
        "content_ir_sha256": (None if assembly is None else assembly.artifact.content_ir_sha256),
        "limitations": [
            "All topology, semantic, promoted evidence, and fused outputs remain unverified.",
            "Table fusion is deterministic evidence compilation, not learned accuracy.",
            "No paid API, network provider, retry, or hidden repair is used.",
            "Any incomplete page coverage blocks document assembly.",
        ],
        "pages": page_summaries,
    }
    summary_bytes = _write_json(output_dir / "summary.json", summary)
    _write_sha(
        output_dir / "summary.sha256",
        _sha256(summary_bytes),
        "summary.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
