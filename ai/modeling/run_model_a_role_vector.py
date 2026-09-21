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
from scan2hwpx.model_a import (
    DEFAULT_MAX_ROLE_VECTOR_LINES_PER_SEGMENT,
    ROLE_VECTOR_COMPILER_VERSION,
    ROLE_VECTOR_PROMPT_VERSION,
    ModelAPageImage,
    OllamaModelARoleVectorProvider,
    execute_model_a_role_vector_chunked_page,
    resolve_ollama_model_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Model A role-vector compiler on exact EvidenceIR pages."
    )
    parser.add_argument("--evidence-ir", type=Path, required=True)
    parser.add_argument("--pages-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-id", action="append", dest="page_ids")
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument(
        "--max-lines-per-segment",
        type=int,
        default=DEFAULT_MAX_ROLE_VECTOR_LINES_PER_SEGMENT,
    )
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
    matches = tuple(
        path
        for suffix in (".png", ".jpg", ".jpeg", ".webp")
        if (path := pages_root / f"{page_id}{suffix}").is_file()
    )
    if len(matches) != 1:
        raise ValueError(f"page image must resolve exactly once for {page_id}: {matches}")
    return matches[0]


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
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    pages = _selected_pages(evidence_ir, args.page_ids)
    if not pages:
        raise ValueError("at least one page must be selected")
    model = resolve_ollama_model_artifact(args.model, base_url=args.base_url)
    provider = OllamaModelARoleVectorProvider(
        model=model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )

    page_summaries: list[dict[str, object]] = []
    elapsed_values: list[float] = []
    batch_started = time.perf_counter()
    for page in pages:
        image_path = _page_image_path(pages_root, page.id)
        image_bytes = image_path.read_bytes()
        page_image = ModelAPageImage(
            page_id=page.id,
            page_no=page.page_no,
            image_source_ref=page.image_source_ref,
            media_type=_media_type(image_path),
            raw_bytes=image_bytes,
            sha256=_sha256(image_bytes),
        )
        started = time.perf_counter()
        result = execute_model_a_role_vector_chunked_page(
            evidence_ir,
            page_image=page_image,
            model=model,
            provider=provider,
            max_lines_per_segment=args.max_lines_per_segment,
        )
        elapsed_seconds = time.perf_counter() - started
        elapsed_values.append(elapsed_seconds)

        page_dir = output_dir / page.id
        page_dir.mkdir()
        segments_dir = page_dir / "segments"
        segments_dir.mkdir()
        for index, (request, raw_response, trace) in enumerate(
            zip(
                result.requests,
                result.raw_role_response_bytes,
                result.artifact.segments,
                strict=True,
            ),
            start=1,
        ):
            segment_dir = segments_dir / f"segment-{index:04d}"
            segment_dir.mkdir()
            (segment_dir / "request.json").write_bytes(request.raw_request_bytes)
            (segment_dir / "raw-role-response.json").write_bytes(raw_response)
            (segment_dir / "trace.json").write_bytes(
                _canonical_json_bytes(trace.model_dump(mode="json"), pretty=True)
            )
        if result.compiled_page_analysis_bytes is not None:
            (page_dir / "compiled-page-analysis.json").write_bytes(
                result.compiled_page_analysis_bytes
            )
        (page_dir / "result.json").write_bytes(result.result_bytes)

        analysis = result.artifact.compiled_page_analysis
        roles = (
            Counter(node.role.value for node in analysis.nodes if node.kind == "text")
            if analysis is not None
            else Counter()
        )
        page_summaries.append(
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "elapsed_seconds": elapsed_seconds,
                "outcome": result.artifact.outcome,
                "issue_codes": [issue.code.value for issue in result.artifact.issues],
                "compiled_validation_issue_codes": [
                    issue.code.value for issue in result.artifact.compiled_validation_issues
                ],
                "observation_counts": dict(
                    sorted(Counter(item.kind.value for item in page.observations).items())
                ),
                "node_count": 0 if analysis is None else len(analysis.nodes),
                "review_required_node_count": len(result.artifact.review_required_node_ids),
                "role_counts": dict(sorted(roles.items())),
                "segment_count": len(result.artifact.segments),
                "segment_line_limit": result.artifact.segment_line_limit,
                "segment_outcomes": [segment.outcome for segment in result.artifact.segments],
                "request_sha256s": [request.raw_request_sha256 for request in result.requests],
                "raw_role_response_sha256s": [
                    segment.raw_role_response_sha256 for segment in result.artifact.segments
                ],
                "raw_combined_decision_sha256": (result.artifact.raw_combined_decision_sha256),
                "resolved_decision_sha256": result.artifact.resolved_decision_sha256,
                "deterministic_anchor_resolution_count": len(
                    result.artifact.deterministic_anchor_resolutions
                ),
                "deterministic_anchor_override_count": sum(
                    resolution.overrode_model
                    for resolution in result.artifact.deterministic_anchor_resolutions
                ),
                "compiled_page_analysis_sha256": (result.artifact.compiled_page_analysis_sha256),
                "result_sha256": result.result_sha256,
            }
        )

    summary = {
        "schema_version": "model-a-role-vector-chunked-smoke/1.1",
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
        "prompt_version": ROLE_VECTOR_PROMPT_VERSION,
        "compiler_version": ROLE_VECTOR_COMPILER_VERSION,
        "max_lines_per_segment": args.max_lines_per_segment,
        "page_count": len(pages),
        "valid_unverified_count": sum(
            item["outcome"] == "valid_unverified" for item in page_summaries
        ),
        "blocked_count": sum(item["outcome"] == "blocked" for item in page_summaries),
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
            "Role-vector compiler v1.2 blocks typed table_grid and formula observations.",
            "Deterministic resolution covers only explicit question, choice, bogi separator, and page-number anchors.",
            "The table text role is a semantic hint and does not reconstruct table topology.",
            "One-line-per-node output does not prove paragraph grouping fidelity.",
        ],
        "pages": page_summaries,
    }
    (output_dir / "summary.json").write_bytes(_canonical_json_bytes(summary, pretty=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
