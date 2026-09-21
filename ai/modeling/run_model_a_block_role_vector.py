from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scan2hwpx.contracts import EvidenceIR, contract_sha256
from scan2hwpx.model_a.block_role_vector import (
    BLOCK_CONTEXT_RADIUS,
    BLOCK_ROLE_VECTOR_COMPILER_VERSION,
    BLOCK_ROLE_VECTOR_PROMPT_VERSION,
    BLOCK_ROLE_VECTOR_RESULT_VERSION,
    MAX_TARGET_BLOCKS_PER_SEGMENT,
    build_model_a_block_role_vector_requests,
    compile_model_a_block_role_vector_responses,
)
from scan2hwpx.model_a.inference import ModelAPageImage
from scan2hwpx.model_a.ollama import (
    OllamaModelABlockRoleVectorProvider,
    resolve_ollama_model_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local block-based Model A v2 on exact EvidenceIR pages."
    )
    parser.add_argument("--evidence-ir", type=Path, required=True)
    parser.add_argument("--pages-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-id", action="append", dest="page_ids")
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=1024)
    return parser.parse_args()


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value, pretty=True))


def _media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"unsupported page image extension: {path}")


def _page_image_path(pages_root: Path, page_id: str) -> Path:
    matches: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = (pages_root / f"{page_id}{suffix}").resolve()
        if candidate.parent != pages_root:
            raise ValueError("page id cannot escape the configured pages root")
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"page image must resolve exactly once for {page_id}: {matches}")
    return matches[0]


def _page_output_key(page_no: int, page_id: str) -> str:
    digest = _sha256(page_id.encode("utf-8"))[:12]
    return f"page-{page_no:04d}-{digest}"


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _selected_pages(evidence_ir: EvidenceIR, page_ids: list[str] | None):
    ordered = tuple(sorted(evidence_ir.pages, key=lambda page: (page.page_no, page.id)))
    if page_ids is None:
        return ordered
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("--page-id values must be unique")
    page_by_id = {page.id: page for page in ordered}
    unknown = sorted(set(page_ids) - set(page_by_id))
    if unknown:
        raise ValueError("unknown --page-id values: " + ", ".join(unknown))
    requested = set(page_ids)
    return tuple(page for page in ordered if page.id in requested)


def _provider_record(args: argparse.Namespace) -> dict[str, object]:
    return {
        "provider": "ollama_local_chat",
        "base_url": args.base_url,
        "timeout_seconds": args.timeout_seconds,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": 0,
        "seed": 0,
        "think": False,
        "stream": False,
    }


def main() -> int:
    args = _parse_args()
    evidence_path = args.evidence_ir.resolve()
    evidence_raw = evidence_path.read_bytes()
    evidence_ir = EvidenceIR.model_validate_json(evidence_raw, strict=True)
    pages_root = (
        args.pages_root.resolve()
        if args.pages_root is not None
        else (evidence_path.parent / "pages").resolve()
    )
    pages = _selected_pages(evidence_ir, args.page_ids)
    if not pages:
        raise ValueError("at least one page must be selected")
    image_paths = {page.id: _page_image_path(pages_root, page.id) for page in pages}

    model = resolve_ollama_model_artifact(args.model, base_url=args.base_url)
    provider = OllamaModelABlockRoleVectorProvider(
        model=model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    page_summaries: list[dict[str, object]] = []
    elapsed_values: list[float] = []
    batch_started = time.perf_counter()
    failed = False
    for page in pages:
        page_key = _page_output_key(page.page_no, page.id)
        page_dir = output_dir / page_key
        page_dir.mkdir()
        image_path = image_paths[page.id]
        image_bytes = image_path.read_bytes()
        page_image = ModelAPageImage(
            page_id=page.id,
            page_no=page.page_no,
            image_source_ref=page.image_source_ref,
            media_type=_media_type(image_path),
            raw_bytes=image_bytes,
            sha256=_sha256(image_bytes),
        )
        _write_json(
            page_dir / "page-binding.json",
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "page_output_key": page_key,
                "page_image_path": str(image_path),
                "page_image_sha256": page_image.sha256,
            },
        )
        started = time.perf_counter()
        try:
            requests = build_model_a_block_role_vector_requests(
                evidence_ir,
                page_image=page_image,
                model=model,
            )
            segments_dir = page_dir / "segments"
            segments_dir.mkdir()
            raw_responses: list[bytes] = []
            for request in requests:
                segment_no = request.artifact.segment_index + 1
                segment_dir = segments_dir / f"segment-{segment_no:04d}"
                segment_dir.mkdir()
                (segment_dir / "request.json").write_bytes(request.raw_request_bytes)
                wire_payload = provider.wire_payload_bytes(request)
                segment_started = time.perf_counter()
                raw_response = provider.generate(request)
                segment_elapsed = time.perf_counter() - segment_started
                raw_responses.append(raw_response)
                (segment_dir / "raw-assistant-content.bin").write_bytes(raw_response)
                _write_json(
                    segment_dir / "execution.json",
                    {
                        "status": "received",
                        "elapsed_seconds": segment_elapsed,
                        "request_sha256": request.raw_request_sha256,
                        "wire_payload_sha256": _sha256(wire_payload),
                        "wire_payload_size_bytes": len(wire_payload),
                        "assistant_content_sha256": _sha256(raw_response),
                        "assistant_content_size_bytes": len(raw_response),
                        "provider": _provider_record(args),
                    },
                )
            result = compile_model_a_block_role_vector_responses(
                requests,
                tuple(raw_responses),
            )
            for request, trace in zip(requests, result.artifact.segments, strict=True):
                segment_no = request.artifact.segment_index + 1
                _write_json(
                    segments_dir / f"segment-{segment_no:04d}" / "trace.json",
                    trace.model_dump(mode="json"),
                )
            if result.compiled_page_analysis_bytes is not None:
                (page_dir / "compiled-page-analysis.json").write_bytes(
                    result.compiled_page_analysis_bytes
                )
            (page_dir / "result.json").write_bytes(result.result_bytes)

            elapsed_seconds = time.perf_counter() - started
            elapsed_values.append(elapsed_seconds)
            artifact = result.artifact
            role_plan = artifact.resolved_role_plan
            raw_decision = artifact.raw_combined_block_decision
            expanded = artifact.expanded_line_decision
            page_summaries.append(
                {
                    "page_id": page.id,
                    "page_no": page.page_no,
                    "page_output_key": page_key,
                    "elapsed_seconds": elapsed_seconds,
                    "outcome": artifact.outcome,
                    "issue_codes": [issue.code.value for issue in artifact.issues],
                    "compiled_validation_issue_codes": [
                        issue.code.value for issue in artifact.compiled_validation_issues
                    ],
                    "text_line_count": len(
                        artifact.classification_plan.ordered_text_observation_ids
                    ),
                    "classification_block_count": len(artifact.classification_plan.blocks),
                    "forced_block_count": sum(
                        block.forced_role is not None
                        for block in artifact.classification_plan.blocks
                    ),
                    "segment_count": len(artifact.segments),
                    "target_block_limit": MAX_TARGET_BLOCKS_PER_SEGMENT,
                    "context_block_radius": BLOCK_CONTEXT_RADIUS,
                    "raw_block_role_counts": dict(
                        sorted(Counter(() if raw_decision is None else raw_decision.roles).items())
                    ),
                    "resolved_block_role_counts": dict(
                        sorted(
                            Counter(
                                ()
                                if role_plan is None
                                else (
                                    resolution.resolved_role
                                    for resolution in role_plan.block_resolutions
                                )
                            ).items()
                        )
                    ),
                    "resolved_line_role_counts": dict(
                        sorted(Counter(() if expanded is None else expanded.roles).items())
                    ),
                    "deterministic_resolution_count": (
                        0
                        if role_plan is None
                        else sum(
                            resolution.role_source.value == "deterministic"
                            for resolution in role_plan.block_resolutions
                        )
                    ),
                    "deterministic_conflict_count": (
                        0
                        if role_plan is None
                        else sum(resolution.conflict for resolution in role_plan.block_resolutions)
                    ),
                    "classification_plan_sha256": artifact.classification_plan_sha256,
                    "raw_combined_block_decision_sha256": (
                        artifact.raw_combined_block_decision_sha256
                    ),
                    "resolved_role_plan_sha256": artifact.resolved_role_plan_sha256,
                    "expanded_line_decision_sha256": (artifact.expanded_line_decision_sha256),
                    "compiled_page_analysis_sha256": (artifact.compiled_page_analysis_sha256),
                    "result_sha256": result.result_sha256,
                }
            )
            if artifact.outcome != "valid_unverified":
                failed = True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failed = True
            elapsed_seconds = time.perf_counter() - started
            elapsed_values.append(elapsed_seconds)
            failure = {
                "status": "execution_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": elapsed_seconds,
            }
            _write_json(page_dir / "failure.json", failure)
            page_summaries.append(
                {
                    "page_id": page.id,
                    "page_no": page.page_no,
                    "page_output_key": page_key,
                    "elapsed_seconds": elapsed_seconds,
                    "outcome": "execution_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    try:
        final_model = resolve_ollama_model_artifact(args.model, base_url=args.base_url)
        model_stability: dict[str, object] = {
            "status": "stable" if final_model == model else "changed",
            "initial": model.model_dump(mode="json"),
            "final": final_model.model_dump(mode="json"),
        }
        if final_model != model:
            failed = True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        failed = True
        model_stability = {
            "status": "unverified",
            "initial": model.model_dump(mode="json"),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    summary = {
        "schema_version": "model-a-block-role-vector-smoke/2.0",
        "created_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "execution_policy": "sequential_pages_and_segments_single_local_gpu_no_retry",
        "evidence_ir_path": str(evidence_path),
        "evidence_ir_raw_sha256": _sha256(evidence_raw),
        "evidence_ir_contract_sha256": contract_sha256(evidence_ir),
        "model": model.model_dump(mode="json"),
        "model_stability": model_stability,
        "provider": _provider_record(args),
        "prompt_version": BLOCK_ROLE_VECTOR_PROMPT_VERSION,
        "compiler_version": BLOCK_ROLE_VECTOR_COMPILER_VERSION,
        "result_version": BLOCK_ROLE_VECTOR_RESULT_VERSION,
        "target_block_limit": MAX_TARGET_BLOCKS_PER_SEGMENT,
        "context_block_radius": BLOCK_CONTEXT_RADIUS,
        "page_count": len(pages),
        "valid_unverified_count": sum(
            item["outcome"] == "valid_unverified" for item in page_summaries
        ),
        "blocked_count": sum(item["outcome"] == "blocked" for item in page_summaries),
        "execution_error_count": sum(
            item["outcome"] == "execution_error" for item in page_summaries
        ),
        "total_elapsed_seconds": time.perf_counter() - batch_started,
        "page_latency_seconds": {
            "min": min(elapsed_values),
            "p50_nearest_rank": _nearest_rank(elapsed_values, 0.50),
            "p95_nearest_rank": _nearest_rank(elapsed_values, 0.95),
            "max": max(elapsed_values),
            "mean": sum(elapsed_values) / len(elapsed_values),
        },
        "limitations": [
            "No human-verified gold labels are used; role accuracy is not measured.",
            "All compiled nodes remain needs_review=true.",
            "Forced roles are deterministic structural constraints, not learned accuracy.",
            "Table line roles do not reconstruct cell or grid topology.",
            "The exact Ollama request body digest is recorded, but the raw response envelope is not retained.",
            "Pages and segments run sequentially on one local GPU.",
        ],
        "pages": page_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
