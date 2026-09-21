from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from scan2hwpx.contracts import (
    ContentIR,
    EvidenceIR,
    EvidenceSourceKind,
    contract_sha256,
)
from scan2hwpx.evaluation import role_macro_f1
from scan2hwpx.evaluation.strict_json import require_strict_json_bytes
from scan2hwpx.model_a.block_role_vector import (
    ModelABlockRoleVectorDecision,
    ModelABlockRoleVectorPageResult,
    ModelABlockRoleVectorRequest,
    ModelABlockRoleVectorSegmentTrace,
)
from scan2hwpx.model_a.classification_blocks import (
    ModelAClassificationBlockResolution,
    ModelAClassificationRoleSource,
    build_model_a_classification_plan,
)
from scan2hwpx.model_a.inference import ModelAPageAnalysis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recorded Model A v2 block roles and their resolved line roles "
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
    dict[str, ModelABlockRoleVectorPageResult],
    dict[str, ModelAPageAnalysis],
    dict[str, str],
]:
    results: dict[str, ModelABlockRoleVectorPageResult] = {}
    analyses: dict[str, ModelAPageAnalysis] = {}
    result_raw_sha256s: dict[str, str] = {}
    for path in sorted(batch_dir.glob("*/result.json")):
        raw = path.read_bytes()
        require_strict_json_bytes(raw, max_depth=64, label="block role-vector page result")
        result = ModelABlockRoleVectorPageResult.model_validate_json(raw, strict=True)
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
        compiled_raw = compiled_path.read_bytes()
        require_strict_json_bytes(compiled_raw, max_depth=64, label="compiled page analysis")
        compiled = ModelAPageAnalysis.model_validate_json(
            compiled_raw,
            strict=True,
        )
        if compiled != analysis:
            raise ValueError(f"standalone compiled page differs from result.json: {compiled_path}")
        _validate_segment_artifacts(path.parent, result)
        results[analysis.page_id] = result
        analyses[analysis.page_id] = analysis
        result_raw_sha256s[analysis.page_id] = _sha256(raw)
    return results, analyses, result_raw_sha256s


def _validate_segment_artifacts(
    page_dir: Path,
    result: ModelABlockRoleVectorPageResult,
) -> None:
    segments_dir = page_dir / "segments"
    actual_segment_dirs = tuple(
        path.name for path in sorted(segments_dir.glob("segment-*")) if path.is_dir()
    )
    expected_segment_dirs = tuple(
        f"segment-{trace.segment_index + 1:04d}" for trace in result.segments
    )
    if actual_segment_dirs != expected_segment_dirs:
        raise ValueError(f"segment artifact directories do not exactly cover result: {page_dir}")
    for trace, directory_name in zip(result.segments, expected_segment_dirs, strict=True):
        segment_dir = segments_dir / directory_name
        trace_raw = (segment_dir / "trace.json").read_bytes()
        require_strict_json_bytes(
            trace_raw,
            max_depth=16,
            label="block segment trace",
        )
        standalone_trace = ModelABlockRoleVectorSegmentTrace.model_validate_json(
            trace_raw,
            strict=True,
        )
        if standalone_trace != trace:
            raise ValueError(f"standalone segment trace differs from result.json: {segment_dir}")

        request_raw = (segment_dir / "request.json").read_bytes()
        require_strict_json_bytes(
            request_raw,
            max_depth=32,
            label="block role-vector request",
        )
        request = ModelABlockRoleVectorRequest.model_validate_json(request_raw, strict=True)
        if _sha256(request_raw) != trace.request_sha256 or (
            request.model != result.model
            or request.source != result.source
            or request.classification_plan_sha256 != result.classification_plan_sha256
            or request.decision_compiler != result.decision_compiler
            or request.segment_index != trace.segment_index
            or tuple(block.block_index for block in request.target_blocks)
            != trace.target_block_indexes
            or request.response_schema_sha256 != trace.response_schema_sha256
        ):
            raise ValueError(f"segment request provenance differs from result.json: {segment_dir}")

        response_raw = (segment_dir / "raw-assistant-content.bin").read_bytes()
        if (
            _sha256(response_raw) != trace.raw_role_response_sha256
            or len(response_raw) != trace.raw_role_response_size_bytes
        ):
            raise ValueError(f"raw segment response digest differs from result.json: {segment_dir}")
        require_strict_json_bytes(
            response_raw,
            max_depth=8,
            max_nodes=len(trace.target_block_indexes) + 4,
            label="block role-vector response",
        )
        raw_decision = ModelABlockRoleVectorDecision.model_validate_json(
            response_raw,
            strict=True,
        )
        if raw_decision != trace.raw_decision:
            raise ValueError(f"raw segment response differs from recorded decision: {segment_dir}")


def _line_vectors(
    result: ModelABlockRoleVectorPageResult,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    raw = result.raw_combined_block_decision
    resolved = result.expanded_line_decision
    role_plan = result.resolved_role_plan
    if raw is None or resolved is None or role_plan is None:
        raise ValueError("valid block result must contain raw, resolved, and expanded decisions")
    plan = result.classification_plan
    if len(raw.roles) != len(plan.blocks):
        raise ValueError("raw block roles do not cover the classification plan")
    raw_line_roles = tuple(
        role
        for block, role in zip(plan.blocks, raw.roles, strict=True)
        for _ in block.ordered_text_observation_ids
    )
    observation_ids = plan.ordered_text_observation_ids
    if (
        resolved.ordered_text_observation_ids != observation_ids
        or len(raw_line_roles) != len(observation_ids)
        or len(resolved.roles) != len(observation_ids)
        or role_plan.expanded_line_roles != resolved.roles
    ):
        raise ValueError("block decisions do not expand over the canonical page lines")
    return observation_ids, raw_line_roles, resolved.roles


def _decision_owners(
    results: dict[str, ModelABlockRoleVectorPageResult],
    *,
    resolved: bool,
) -> dict[str, tuple[str, ...]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for result in results.values():
        observation_ids, raw_roles, resolved_roles = _line_vectors(result)
        roles = resolved_roles if resolved else raw_roles
        for observation_id, role in zip(observation_ids, roles, strict=True):
            owners[observation_id].append(role)
    return {key: tuple(values) for key, values in owners.items()}


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
        "exact_agreement_rate": None if not aligned_ids else agreement_count / len(aligned_ids),
        "role_macro_f1": None if not aligned_ids else role_macro_f1(reference, prediction),
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


def _raw_to_resolved_block(
    observation_ids: Sequence[str],
    proxy_owners: dict[str, tuple[str, ...]],
    raw_owners: dict[str, tuple[str, ...]],
    resolved_owners: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    decision_aligned_ids = tuple(
        observation_id
        for observation_id in observation_ids
        if len(raw_owners.get(observation_id, ())) == 1
        and len(resolved_owners.get(observation_id, ())) == 1
    )
    changed_count = sum(
        raw_owners[observation_id][0] != resolved_owners[observation_id][0]
        for observation_id in decision_aligned_ids
    )
    proxy_aligned_ids = tuple(
        observation_id
        for observation_id in decision_aligned_ids
        if len(proxy_owners.get(observation_id, ())) == 1
    )
    raw_agreement_count = sum(
        raw_owners[observation_id][0] == proxy_owners[observation_id][0]
        for observation_id in proxy_aligned_ids
    )
    resolved_agreement_count = sum(
        resolved_owners[observation_id][0] == proxy_owners[observation_id][0]
        for observation_id in proxy_aligned_ids
    )
    improved_count = sum(
        raw_owners[observation_id][0] != proxy_owners[observation_id][0]
        and resolved_owners[observation_id][0] == proxy_owners[observation_id][0]
        for observation_id in proxy_aligned_ids
    )
    regressed_count = sum(
        raw_owners[observation_id][0] == proxy_owners[observation_id][0]
        and resolved_owners[observation_id][0] != proxy_owners[observation_id][0]
        for observation_id in proxy_aligned_ids
    )
    aligned_count = len(proxy_aligned_ids)
    raw_rate = None if not aligned_count else raw_agreement_count / aligned_count
    resolved_rate = None if not aligned_count else resolved_agreement_count / aligned_count
    return {
        "decision_aligned_count": len(decision_aligned_ids),
        "changed_line_count": changed_count,
        "changed_line_rate": (
            None if not decision_aligned_ids else changed_count / len(decision_aligned_ids)
        ),
        "proxy_aligned_count": aligned_count,
        "raw_model_proxy_agreement_count": raw_agreement_count,
        "raw_model_proxy_agreement_rate": raw_rate,
        "resolved_pipeline_proxy_agreement_count": resolved_agreement_count,
        "resolved_pipeline_proxy_agreement_rate": resolved_rate,
        "proxy_agreement_rate_delta": (
            None if raw_rate is None or resolved_rate is None else resolved_rate - raw_rate
        ),
        "improved_line_count": improved_count,
        "regressed_line_count": regressed_count,
        "unchanged_correct_line_count": raw_agreement_count - regressed_count,
        "unchanged_incorrect_line_count": (aligned_count - raw_agreement_count - improved_count),
    }


def _forced_resolutions(
    result: ModelABlockRoleVectorPageResult,
) -> tuple[ModelAClassificationBlockResolution, ...]:
    role_plan = result.resolved_role_plan
    if role_plan is None:
        raise ValueError("valid block result must contain its resolved role plan")
    forced: list[ModelAClassificationBlockResolution] = []
    for block, resolution in zip(
        result.classification_plan.blocks,
        role_plan.block_resolutions,
        strict=True,
    ):
        deterministic = resolution.role_source == ModelAClassificationRoleSource.DETERMINISTIC
        if deterministic != (block.forced_role is not None):
            raise ValueError("recorded role source does not match the forced block policy")
        if deterministic:
            if (
                resolution.ordered_text_observation_ids != block.ordered_text_observation_ids
                or resolution.resolved_role != block.forced_role
            ):
                raise ValueError("forced block resolution does not bind its recorded block")
            forced.append(resolution)
    return tuple(forced)


def _forced_subset_block(
    results: Sequence[ModelABlockRoleVectorPageResult],
    proxy_owners: dict[str, tuple[str, ...]],
    raw_owners: dict[str, tuple[str, ...]],
    resolved_owners: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    forced_pairs = tuple(
        (block, resolution)
        for result in results
        for block, resolution in zip(
            result.classification_plan.blocks,
            result.resolved_role_plan.block_resolutions,
            strict=True,
        )
        if block.forced_role is not None
    )
    for result in results:
        _forced_resolutions(result)
    forced_ids = tuple(
        observation_id
        for block, _ in forced_pairs
        for observation_id in block.ordered_text_observation_ids
    )
    block_count = len(forced_pairs)
    conflict_count = sum(resolution.conflict for _, resolution in forced_pairs)
    return {
        "weighting": "text_lines_after_repeating_each_forced_block_role",
        "block_count": block_count,
        "text_line_count": len(forced_ids),
        "conflict_block_count": conflict_count,
        "conflict_block_rate": None if not block_count else conflict_count / block_count,
        "formation_counts": dict(
            sorted(Counter(block.formation.value for block, _ in forced_pairs).items())
        ),
        "role_transition_counts": dict(
            sorted(
                Counter(
                    f"{resolution.model_role}->{resolution.resolved_role}"
                    for _, resolution in forced_pairs
                ).items()
            )
        ),
        "comparisons": {
            "raw_model": _comparison_block(forced_ids, proxy_owners, raw_owners),
            "resolved_pipeline": _comparison_block(
                forced_ids,
                proxy_owners,
                resolved_owners,
            ),
        },
        "raw_to_resolved": _raw_to_resolved_block(
            forced_ids,
            proxy_owners,
            raw_owners,
            resolved_owners,
        ),
    }


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


def _validate_result_binding(
    result: ModelABlockRoleVectorPageResult,
    evidence_ir: EvidenceIR,
    page: object,
    evidence_contract_sha256: str,
) -> None:
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
    source_by_id = {item.id: item for item in evidence_ir.sources}
    page_source = source_by_id[page.image_source_ref]
    if page_source.kind != EvidenceSourceKind.PAGE_IMAGE or (
        page_source.sha256 is not None and page_source.sha256 != source.page_image_sha256
    ):
        raise ValueError(f"batch result page image provenance differs: {page.id}")
    expected_plan = build_model_a_classification_plan(
        evidence_ir,
        target_page_id=page.id,
    )
    if result.classification_plan != expected_plan:
        raise ValueError(f"classification plan cannot be reproduced from EvidenceIR: {page.id}")
    result_ids, _, _ = _line_vectors(result)
    if result_ids != expected_plan.ordered_text_observation_ids:
        raise ValueError(f"batch result does not cover page text exactly once: {page.id}")


def main() -> int:
    args = _parse_args()
    evidence_raw = args.evidence_ir.resolve().read_bytes()
    proxy_raw = args.proxy_content_ir.resolve().read_bytes()
    require_strict_json_bytes(evidence_raw, max_depth=64, label="EvidenceIR")
    require_strict_json_bytes(proxy_raw, max_depth=64, label="proxy ContentIR")
    evidence_ir = EvidenceIR.model_validate_json(evidence_raw, strict=True)
    proxy_content = ContentIR.model_validate_json(proxy_raw, strict=True)
    proxy_content.assert_evidence_integrity(evidence_ir)
    proxy_owners = _role_owners(proxy_content)
    results, analyses, result_raw_sha256s = _batch_results(args.batch_dir.resolve())

    selected_pages = _selected_pages(evidence_ir, args.page_ids)
    expected_pages = {page.id for page in selected_pages}
    if set(analyses) != expected_pages:
        raise ValueError(
            "batch compiled pages do not exactly cover selected EvidenceIR pages: "
            f"expected={sorted(expected_pages)}, actual={sorted(analyses)}"
        )
    evidence_contract_digest = contract_sha256(evidence_ir)
    for page in selected_pages:
        _validate_result_binding(
            results[page.id],
            evidence_ir,
            page,
            evidence_contract_digest,
        )

    model_keys = {
        (result.model.model_id, result.model.model_revision, result.model.artifact_sha256)
        for result in results.values()
    }
    if len(model_keys) != 1:
        raise ValueError("comparison batch must use exactly one recorded model artifact")
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
    grouping_keys = {
        (
            result.classification_plan.grouping.grouper_id,
            result.classification_plan.grouping.grouper_version,
            result.classification_plan.grouping.grouper_sha256,
        )
        for result in results.values()
    }
    if len(grouping_keys) != 1:
        raise ValueError("comparison batch must use exactly one recorded classification grouper")

    text_observation_ids = tuple(
        observation_id
        for page in selected_pages
        for observation_id in results[page.id].classification_plan.ordered_text_observation_ids
    )
    if len(text_observation_ids) != len(set(text_observation_ids)):
        raise ValueError("selected page results repeat text observation ids")
    raw_model_owners = _decision_owners(results, resolved=False)
    resolved_pipeline_owners = _decision_owners(results, resolved=True)
    model = next(iter(results.values())).model
    compiler = next(iter(results.values())).decision_compiler
    grouping = next(iter(results.values())).classification_plan.grouping

    per_page: list[dict[str, object]] = []
    for page in selected_pages:
        result = results[page.id]
        page_ids, _, _ = _line_vectors(result)
        per_page.append(
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "classification_block_count": len(result.classification_plan.blocks),
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
                "raw_to_resolved": _raw_to_resolved_block(
                    page_ids,
                    proxy_owners,
                    raw_model_owners,
                    resolved_pipeline_owners,
                ),
                "recorded_forced_subset": _forced_subset_block(
                    (result,),
                    proxy_owners,
                    raw_model_owners,
                    resolved_pipeline_owners,
                ),
                "result_provenance": {
                    "result_schema_version": result.schema_version,
                    "result_raw_sha256": result_raw_sha256s[page.id],
                    "classification_plan_sha256": result.classification_plan_sha256,
                    "segment_request_sha256s": [
                        segment.request_sha256 for segment in result.segments
                    ],
                    "raw_role_response_sha256s": [
                        segment.raw_role_response_sha256 for segment in result.segments
                    ],
                    "raw_combined_block_decision_sha256": (
                        result.raw_combined_block_decision_sha256
                    ),
                    "resolved_role_plan_sha256": result.resolved_role_plan_sha256,
                    "expanded_line_decision_sha256": result.expanded_line_decision_sha256,
                    "compiled_page_analysis_sha256": result.compiled_page_analysis_sha256,
                },
            }
        )

    selected_results = tuple(results[page.id] for page in selected_pages)
    report = {
        "schema_version": "model-a-block-role-proxy-comparison/2.0",
        "metric_status": "diagnostic_only_not_accuracy_not_release_evidence",
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "evidence_ir_raw_sha256": _sha256(evidence_raw),
        "evidence_ir_contract_sha256": evidence_contract_digest,
        "proxy_content_ir_raw_sha256": _sha256(proxy_raw),
        "proxy_content_ir_contract_sha256": contract_sha256(proxy_content),
        "selected_page_ids": [page.id for page in selected_pages],
        "classification_block_count": sum(
            len(result.classification_plan.blocks) for result in selected_results
        ),
        "text_observation_count": len(text_observation_ids),
        "comparison_unit": "evidence_text_line",
        "prediction_bases": {
            "raw_model": (
                "raw_block_roles_repeated_over_each_recorded_block_member_line_before_"
                "deterministic_resolution"
            ),
            "resolved_pipeline": (
                "recorded_expanded_line_decision_after_deterministic_forced_role_resolution"
            ),
        },
        "model_provenance": model.model_dump(mode="json"),
        "compiler_provenance": compiler.model_dump(mode="json"),
        "classification_grouping_provenance": grouping.model_dump(mode="json"),
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
        "raw_to_resolved": _raw_to_resolved_block(
            text_observation_ids,
            proxy_owners,
            raw_model_owners,
            resolved_pipeline_owners,
        ),
        "recorded_forced_subset": _forced_subset_block(
            selected_results,
            proxy_owners,
            raw_model_owners,
            resolved_pipeline_owners,
        ),
        "pages": per_page,
        "limitations": [
            "The proxy ContentIR is unverified and explicitly not human-verified Gold.",
            "Agreement with the proxy is diagnostic only and must not be reported as accuracy.",
            "Projection alignment and page assignment require human review.",
            "Raw block roles are line-weighted by repeating one role over every block member line.",
            "Forced-role diagnostics measure recorded deterministic policy effects, not learned quality.",
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
