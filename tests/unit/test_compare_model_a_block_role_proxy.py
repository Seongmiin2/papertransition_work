from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai.modeling import compare_model_a_block_role_proxy as comparison
from scan2hwpx.contracts import ContentIR, EvidenceIR, ObservationKind, contract_sha256
from scan2hwpx.model_a.block_role_vector import (
    ModelABlockRoleVectorPageResult,
    build_model_a_block_role_vector_requests,
    compile_model_a_block_role_vector_responses,
)
from scan2hwpx.model_a.classification_blocks import (
    ModelAClassificationLayoutZone,
    ModelAClassificationPlan,
    ModelAClassificationRolePlan,
)
from tests.unit.test_model_a_role_vector import _evidence, _image, _json_bytes, _model


def _write_comparison_inputs(
    root: Path,
    evidence: EvidenceIR,
    *,
    raw_roles: tuple[str, ...],
) -> tuple[Path, Path, Path, ModelABlockRoleVectorPageResult]:
    requests = build_model_a_block_role_vector_requests(
        evidence,
        page_image=_image(),
        model=_model(),
    )
    responses: list[bytes] = []
    offset = 0
    for request in requests:
        count = len(request.artifact.target_blocks)
        responses.append(_json_bytes({"roles": raw_roles[offset : offset + count]}))
        offset += count
    assert offset == len(raw_roles)
    built = compile_model_a_block_role_vector_responses(requests, tuple(responses))
    result = built.artifact
    analysis = result.compiled_page_analysis
    assert analysis is not None
    assert built.compiled_page_analysis_bytes is not None
    proxy = ContentIR(
        id="proxy-content",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        nodes=analysis.nodes,
        reading_order=analysis.reading_order,
    )

    evidence_path = root / "evidence.json"
    proxy_path = root / "proxy.json"
    batch_dir = root / "batch"
    page_dir = batch_dir / analysis.page_id
    segments_dir = page_dir / "segments"
    segments_dir.mkdir(parents=True)
    evidence_path.write_bytes(_json_bytes(evidence.model_dump(mode="json")))
    proxy_path.write_bytes(_json_bytes(proxy.model_dump(mode="json")))
    (page_dir / "result.json").write_bytes(built.result_bytes)
    (page_dir / "compiled-page-analysis.json").write_bytes(built.compiled_page_analysis_bytes)
    for request, response, trace in zip(
        requests,
        responses,
        result.segments,
        strict=True,
    ):
        segment_dir = segments_dir / f"segment-{trace.segment_index + 1:04d}"
        segment_dir.mkdir()
        (segment_dir / "request.json").write_bytes(request.raw_request_bytes)
        (segment_dir / "raw-assistant-content.bin").write_bytes(response)
        (segment_dir / "trace.json").write_bytes(_json_bytes(trace.model_dump(mode="json")))
    return evidence_path, proxy_path, batch_dir, result


def _run_comparison(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    evidence: EvidenceIR,
    *,
    raw_roles: tuple[str, ...],
) -> dict[str, object]:
    evidence_path, proxy_path, batch_dir, _ = _write_comparison_inputs(
        root,
        evidence,
        raw_roles=raw_roles,
    )
    output = root / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_model_a_block_role_proxy.py",
            "--evidence-ir",
            str(evidence_path),
            "--proxy-content-ir",
            str(proxy_path),
            "--batch-dir",
            str(batch_dir),
            "--output",
            str(output),
        ],
    )

    assert comparison.main() == 0
    return json.loads(output.read_bytes())


def test_report_expands_raw_blocks_and_separates_resolved_forced_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_comparison(
        monkeypatch,
        tmp_path,
        _evidence(),
        raw_roles=("passage", "passage", "passage"),
    )

    assert report["schema_version"] == "model-a-block-role-proxy-comparison/2.0"
    assert report["metric_status"] == "diagnostic_only_not_accuracy_not_release_evidence"
    assert report["human_review_required"] is True
    assert report["classification_block_count"] == 3
    assert report["text_observation_count"] == 3
    comparisons = report["comparisons"]
    assert comparisons["raw_model"]["exact_agreement_rate"] == pytest.approx(2 / 3)
    assert comparisons["resolved_pipeline"]["exact_agreement_rate"] == 1
    transition = report["raw_to_resolved"]
    assert transition["changed_line_count"] == 1
    assert transition["improved_line_count"] == 1
    assert transition["regressed_line_count"] == 0
    assert transition["proxy_agreement_rate_delta"] == pytest.approx(1 / 3)
    forced = report["recorded_forced_subset"]
    assert forced["block_count"] == 1
    assert forced["text_line_count"] == 1
    assert forced["conflict_block_count"] == 1
    assert forced["comparisons"]["raw_model"]["exact_agreement_rate"] == 0
    assert forced["comparisons"]["resolved_pipeline"]["exact_agreement_rate"] == 1
    assert report["model_provenance"]["artifact_sha256"]
    assert report["compiler_provenance"]["compiler_sha256"]
    assert report["classification_grouping_provenance"]["grouper_sha256"]


def test_report_uses_null_rates_when_no_text_or_forced_subset_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _evidence()
    page = base.pages[0].model_copy(
        update={
            "observations": tuple(
                observation
                for observation in base.pages[0].observations
                if observation.kind == ObservationKind.IMAGE
            )
        }
    )
    evidence = base.model_copy(update={"pages": (page,)})

    report = _run_comparison(
        monkeypatch,
        tmp_path,
        evidence,
        raw_roles=(),
    )

    assert report["text_observation_count"] == 0
    for basis in ("raw_model", "resolved_pipeline"):
        block = report["comparisons"][basis]
        assert block["aligned_count"] == 0
        assert block["exact_agreement_rate"] is None
        assert block["role_macro_f1"] is None
    transition = report["raw_to_resolved"]
    assert transition["changed_line_rate"] is None
    assert transition["raw_model_proxy_agreement_rate"] is None
    assert transition["resolved_pipeline_proxy_agreement_rate"] is None
    assert transition["proxy_agreement_rate_delta"] is None
    forced = report["recorded_forced_subset"]
    assert forced["block_count"] == 0
    assert forced["conflict_block_rate"] is None


def test_rejects_raw_segment_bytes_that_do_not_match_recorded_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path, proxy_path, batch_dir, _ = _write_comparison_inputs(
        tmp_path,
        _evidence(),
        raw_roles=("passage", "passage", "passage"),
    )
    response_path = next(batch_dir.glob("*/segments/segment-*/raw-assistant-content.bin"))
    response_path.write_bytes(_json_bytes({"roles": ["title", "title", "title"]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_model_a_block_role_proxy.py",
            "--evidence-ir",
            str(evidence_path),
            "--proxy-content-ir",
            str(proxy_path),
            "--batch-dir",
            str(batch_dir),
            "--output",
            str(tmp_path / "comparison.json"),
        ],
    )

    with pytest.raises(ValueError, match="raw segment response digest"):
        comparison.main()


def test_rejects_internally_valid_plan_not_reproducible_from_evidence() -> None:
    evidence = _evidence()
    requests = build_model_a_block_role_vector_requests(
        evidence,
        page_image=_image(),
        model=_model(),
    )
    result = compile_model_a_block_role_vector_responses(
        requests,
        (_json_bytes({"roles": ["passage", "passage", "passage"]}),),
    ).artifact
    plan_payload = result.classification_plan.model_dump(mode="python")
    first_block = dict(plan_payload["blocks"][0])
    first_block["layout_zone"] = ModelAClassificationLayoutZone.RIGHT_BODY
    plan_payload["blocks"] = (first_block, *plan_payload["blocks"][1:])
    forged_plan = ModelAClassificationPlan.model_validate(plan_payload, strict=True)
    forged_plan_digest = contract_sha256(forged_plan)

    role_plan_payload = result.resolved_role_plan.model_dump(mode="python")
    role_plan_payload["classification_plan_sha256"] = forged_plan_digest
    forged_role_plan = ModelAClassificationRolePlan.model_validate(
        role_plan_payload,
        strict=True,
    )
    result_payload = result.model_dump(mode="python")
    result_payload["classification_plan"] = forged_plan.model_dump(mode="python")
    result_payload["classification_plan_sha256"] = forged_plan_digest
    for trace in result_payload["segments"]:
        trace["classification_plan_sha256"] = forged_plan_digest
    result_payload["resolved_role_plan"] = forged_role_plan.model_dump(mode="python")
    result_payload["resolved_role_plan_sha256"] = contract_sha256(forged_role_plan)
    forged_result = ModelABlockRoleVectorPageResult.model_validate(
        result_payload,
        strict=True,
    )

    with pytest.raises(ValueError, match="classification plan cannot be reproduced"):
        comparison._validate_result_binding(
            forged_result,
            evidence,
            evidence.pages[0],
            contract_sha256(evidence),
        )
