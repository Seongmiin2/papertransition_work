from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from scan2hwpx.contracts import ContentIR, EvidenceIR, ObservationKind, contract_sha256
from scan2hwpx.evaluation import role_macro_f1
from scan2hwpx.model_a import (
    ModelAPageAnalysis,
    ModelARoleVectorAnchorResolution,
    ModelARoleVectorChunkedPageResult,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recorded raw-model and resolved-pipeline role decisions "
            "with a non-Golden ContentIR proxy."
        )
    )
    parser.add_argument("--evidence-ir", type=Path, required=True)
    parser.add_argument("--proxy-content-ir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-id", action="append", dest="page_ids")
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _role_owners(content_ir: ContentIR) -> dict[str, tuple[str, ...]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for node in content_ir.nodes:
        for evidence_ref in node.evidence_refs:
            owners[evidence_ref].append(node.role.value)
    return {key: tuple(values) for key, values in owners.items()}


def _batch_results(
    batch_dir: Path,
) -> tuple[
    dict[str, ModelARoleVectorChunkedPageResult],
    dict[str, ModelAPageAnalysis],
    dict[str, str],
]:
    results: dict[str, ModelARoleVectorChunkedPageResult] = {}
    analyses: dict[str, ModelAPageAnalysis] = {}
    result_raw_sha256s: dict[str, str] = {}
    for path in sorted(batch_dir.glob("*/result.json")):
        raw = path.read_bytes()
        result = ModelARoleVectorChunkedPageResult.model_validate_json(raw, strict=True)
        if result.outcome != "valid_unverified" or result.compiled_page_analysis is None:
            raise ValueError(f"batch result is not a valid compiled page: {path}")
        analysis = result.compiled_page_analysis
        if analysis.page_id in analyses:
            raise ValueError(f"duplicate compiled page analysis: {analysis.page_id}")
        if (
            analysis.id != result.source.expected_page_analysis_id
            or analysis.evidence_ir_id != result.source.evidence_ir_id
            or analysis.evidence_ir_sha256 != result.source.evidence_ir_contract_sha256
            or analysis.page_id != result.source.target_page_id
            or analysis.page_no != result.source.target_page_no
        ):
            raise ValueError(f"result source and compiled page lineage differ: {path}")
        compiled_path = path.parent / "compiled-page-analysis.json"
        compiled = ModelAPageAnalysis.model_validate_json(
            compiled_path.read_bytes(),
            strict=True,
        )
        if compiled != analysis:
            raise ValueError(f"standalone compiled page differs from result.json: {compiled_path}")
        results[analysis.page_id] = result
        analyses[analysis.page_id] = analysis
        result_raw_sha256s[analysis.page_id] = _sha256(raw)
    return results, analyses, result_raw_sha256s


def _decision_vectors(
    result: ModelARoleVectorChunkedPageResult,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if result.raw_combined_decision is None or result.resolved_decision is None:
        raise ValueError("valid batch result must contain raw and resolved decisions")
    observation_ids = tuple(
        observation_id
        for segment in result.segments
        for observation_id in segment.ordered_text_observation_ids
    )
    raw_roles = result.raw_combined_decision.roles
    resolved_roles = result.resolved_decision.roles
    if len(observation_ids) != len(raw_roles) or len(observation_ids) != len(resolved_roles):
        raise ValueError("decision roles do not cover the recorded segment observations")
    return observation_ids, raw_roles, resolved_roles


def _decision_owners(
    results: dict[str, ModelARoleVectorChunkedPageResult],
    *,
    resolved: bool,
) -> dict[str, tuple[str, ...]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for result in results.values():
        observation_ids, raw_roles, resolved_roles = _decision_vectors(result)
        roles = resolved_roles if resolved else raw_roles
        for observation_id, role in zip(observation_ids, roles, strict=True):
            owners[observation_id].append(role)
    return {key: tuple(values) for key, values in owners.items()}


def _same_role_run_diagnostics(roles: list[str]) -> dict[str, object]:
    longest_role = None
    longest_length = 0
    current_role = None
    current_length = 0
    for role in roles:
        if role == current_role:
            current_length += 1
        else:
            current_role = role
            current_length = 1
        if current_length > longest_length:
            longest_role = current_role
            longest_length = current_length
    trailing_role = roles[-1] if roles else None
    trailing_length = 0
    for role in reversed(roles):
        if role != trailing_role:
            break
        trailing_length += 1
    return {
        "longest_same_role": longest_role,
        "longest_same_role_length": longest_length,
        "trailing_same_role": trailing_role,
        "trailing_same_role_length": trailing_length,
    }


def _comparison_block(
    observation_ids: Sequence[str],
    proxy_owners: dict[str, tuple[str, ...]],
    output_owners: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    aligned_ids = tuple(
        observation_id
        for observation_id in observation_ids
        if len(proxy_owners.get(observation_id, ())) == 1
        and len(output_owners.get(observation_id, ())) == 1
    )
    reference = [proxy_owners[observation_id][0] for observation_id in aligned_ids]
    prediction = [output_owners[observation_id][0] for observation_id in aligned_ids]
    agreement_count = sum(
        expected == predicted for expected, predicted in zip(reference, prediction, strict=True)
    )
    confusion = Counter(zip(reference, prediction, strict=True))
    return {
        "aligned_count": len(aligned_ids),
        "proxy_unmapped_count": sum(
            observation_id not in proxy_owners for observation_id in observation_ids
        ),
        "output_unmapped_count": sum(
            observation_id not in output_owners for observation_id in observation_ids
        ),
        "ambiguous_proxy_count": sum(
            len(proxy_owners.get(observation_id, ())) > 1 for observation_id in observation_ids
        ),
        "ambiguous_output_count": sum(
            len(output_owners.get(observation_id, ())) > 1 for observation_id in observation_ids
        ),
        "exact_agreement_count": agreement_count,
        "exact_agreement_rate": (None if not aligned_ids else agreement_count / len(aligned_ids)),
        "role_macro_f1": (None if not aligned_ids else role_macro_f1(reference, prediction)),
        "proxy_role_counts": dict(sorted(Counter(reference).items())),
        "output_role_counts": dict(sorted(Counter(prediction).items())),
        "confusion_counts": {
            expected: dict(
                sorted(
                    (predicted, count)
                    for (candidate, predicted), count in confusion.items()
                    if candidate == expected
                )
            )
            for expected in sorted(set(reference))
        },
    }


def _recorded_anchor_block(
    resolutions: Sequence[ModelARoleVectorAnchorResolution],
    proxy_owners: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    proxy_aligned = tuple(
        resolution
        for resolution in resolutions
        if len(proxy_owners.get(resolution.observation_id, ())) == 1
    )
    raw_agreement_count = sum(
        proxy_owners[resolution.observation_id][0] == resolution.model_role
        for resolution in proxy_aligned
    )
    resolved_agreement_count = sum(
        proxy_owners[resolution.observation_id][0] == resolution.resolved_role
        for resolution in proxy_aligned
    )
    return {
        "resolution_count": len(resolutions),
        "override_count": sum(resolution.overrode_model for resolution in resolutions),
        "reason_counts": dict(
            sorted(Counter(resolution.reason.value for resolution in resolutions).items())
        ),
        "role_transition_counts": dict(
            sorted(
                Counter(
                    f"{resolution.model_role}->{resolution.resolved_role}"
                    for resolution in resolutions
                ).items()
            )
        ),
        "proxy_aligned_count": len(proxy_aligned),
        "raw_model_proxy_agreement_count": raw_agreement_count,
        "raw_model_proxy_agreement_rate": (
            None if not proxy_aligned else raw_agreement_count / len(proxy_aligned)
        ),
        "resolved_pipeline_proxy_agreement_count": resolved_agreement_count,
        "resolved_pipeline_proxy_agreement_rate": (
            None if not proxy_aligned else resolved_agreement_count / len(proxy_aligned)
        ),
    }


def main() -> int:
    args = _parse_args()
    evidence_raw = args.evidence_ir.resolve().read_bytes()
    proxy_raw = args.proxy_content_ir.resolve().read_bytes()
    evidence_ir = EvidenceIR.model_validate_json(evidence_raw, strict=True)
    proxy_content = ContentIR.model_validate_json(proxy_raw, strict=True)
    proxy_content.assert_evidence_integrity(evidence_ir)
    proxy_owners = _role_owners(proxy_content)
    results, analyses, result_raw_sha256s = _batch_results(args.batch_dir.resolve())

    ordered_pages = tuple(sorted(evidence_ir.pages, key=lambda value: (value.page_no, value.id)))
    page_by_id = {page.id: page for page in ordered_pages}
    if args.page_ids is None:
        selected_pages = ordered_pages
    else:
        if len(args.page_ids) != len(set(args.page_ids)):
            raise ValueError("--page-id values must be unique")
        unknown = sorted(set(args.page_ids) - set(page_by_id))
        if unknown:
            raise ValueError("unknown --page-id values: " + ", ".join(unknown))
        selected = set(args.page_ids)
        selected_pages = tuple(page for page in ordered_pages if page.id in selected)
    expected_pages = {page.id for page in selected_pages}
    if set(analyses) != expected_pages:
        raise ValueError(
            "batch compiled pages do not exactly cover EvidenceIR: "
            f"expected={sorted(expected_pages)}, actual={sorted(analyses)}"
        )
    evidence_contract_sha256 = contract_sha256(evidence_ir)
    for page in selected_pages:
        result = results[page.id]
        source = result.source
        if (
            source.evidence_ir_id != evidence_ir.id
            or source.evidence_ir_contract_sha256 != evidence_contract_sha256
            or source.source_document_sha256 != evidence_ir.source_document_sha256
            or source.target_page_id != page.id
            or source.target_page_no != page.page_no
            or source.page_image_source_ref != page.image_source_ref
        ):
            raise ValueError(f"batch result does not bind the selected EvidenceIR page: {page.id}")
        result_ids, _, _ = _decision_vectors(result)
        expected_ids = tuple(
            observation.id
            for observation in page.observations
            if observation.kind == ObservationKind.TEXT_LINE
        )
        if len(result_ids) != len(expected_ids) or set(result_ids) != set(expected_ids):
            raise ValueError(f"batch result does not cover page text exactly once: {page.id}")

    text_observation_ids = tuple(
        observation.id
        for page in selected_pages
        for observation in page.observations
        if observation.kind == ObservationKind.TEXT_LINE
    )
    raw_model_owners = _decision_owners(results, resolved=False)
    resolved_pipeline_owners = _decision_owners(results, resolved=True)
    compiler_keys = {
        (
            result.decision_compiler.compiler_id,
            result.decision_compiler.compiler_version,
            result.decision_compiler.compiler_sha256,
        )
        for result in results.values()
    }
    if len(compiler_keys) != 1:
        raise ValueError("comparison batch must use exactly one recorded decision compiler")
    compiler = next(iter(results.values())).decision_compiler

    per_page: list[dict[str, object]] = []
    for page in selected_pages:
        result = results[page.id]
        page_ids, raw_roles, resolved_roles = _decision_vectors(result)
        per_page.append(
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "text_observation_count": len(page_ids),
                "comparisons": {
                    "raw_model": _comparison_block(
                        page_ids,
                        proxy_owners,
                        raw_model_owners,
                    ),
                    "resolved_pipeline": _comparison_block(
                        page_ids,
                        proxy_owners,
                        resolved_pipeline_owners,
                    ),
                },
                "same_role_run_diagnostics": {
                    "raw_model": _same_role_run_diagnostics(list(raw_roles)),
                    "resolved_pipeline": _same_role_run_diagnostics(list(resolved_roles)),
                },
                "recorded_anchor_provenance": _recorded_anchor_block(
                    result.deterministic_anchor_resolutions,
                    proxy_owners,
                ),
                "result_provenance": {
                    "result_schema_version": result.schema_version,
                    "result_raw_sha256": result_raw_sha256s[page.id],
                    "compiler_version": result.decision_compiler.compiler_version,
                    "compiler_sha256": result.decision_compiler.compiler_sha256,
                    "segment_request_sha256s": [
                        segment.request_sha256 for segment in result.segments
                    ],
                    "raw_role_response_sha256s": [
                        segment.raw_role_response_sha256 for segment in result.segments
                    ],
                    "raw_combined_decision_sha256": result.raw_combined_decision_sha256,
                    "resolved_decision_sha256": result.resolved_decision_sha256,
                    "compiled_page_analysis_sha256": (result.compiled_page_analysis_sha256),
                },
            }
        )

    all_resolutions = tuple(
        resolution
        for page in selected_pages
        for resolution in results[page.id].deterministic_anchor_resolutions
    )
    report = {
        "schema_version": "model-a-unverified-role-proxy-comparison/1.2",
        "metric_status": "diagnostic_only_not_accuracy_not_release_evidence",
        "research_only": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "evidence_ir_raw_sha256": _sha256(evidence_raw),
        "evidence_ir_contract_sha256": evidence_contract_sha256,
        "proxy_content_ir_raw_sha256": _sha256(proxy_raw),
        "selected_page_ids": [page.id for page in selected_pages],
        "text_observation_count": len(text_observation_ids),
        "prediction_bases": {
            "raw_model": "raw_combined_decision_before_deterministic_anchor_resolution",
            "resolved_pipeline": (
                "resolved_decision_after_recorded_deterministic_anchor_resolution"
            ),
        },
        "compiler_provenance": compiler.model_dump(mode="json"),
        "page_result_raw_sha256s": dict(sorted(result_raw_sha256s.items())),
        "comparisons": {
            "raw_model": _comparison_block(
                text_observation_ids,
                proxy_owners,
                raw_model_owners,
            ),
            "resolved_pipeline": _comparison_block(
                text_observation_ids,
                proxy_owners,
                resolved_pipeline_owners,
            ),
        },
        "recorded_anchor_provenance": _recorded_anchor_block(
            all_resolutions,
            proxy_owners,
        ),
        "pages": per_page,
        "legacy_report_compatibility": (
            "v1.1 prediction metrics are resolved-pipeline diagnostics and are not "
            "reinterpreted as raw-model metrics"
        ),
        "limitations": [
            "The proxy ContentIR is unverified and explicitly not Golden.",
            "Projection alignment and page assignment require human review.",
            "Agreement with the proxy must not be reported as model accuracy.",
            "Raw-model and resolved-pipeline diagnostics are intentionally separate.",
            "Anchor diagnostics use the resolutions recorded by each result, not current code.",
            "A table text role does not prove reconstruction of typed table topology.",
        ],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
