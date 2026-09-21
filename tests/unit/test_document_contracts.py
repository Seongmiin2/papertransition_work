from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

import scan2hwpx.contracts.models as contract_models
from scan2hwpx.contracts import (
    REQUIRED_DOCUMENT_QUALITY_CHECKS_V1,
    BBox,
    CheckDecision,
    ContentIR,
    ContentPlanItem,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    HwpDocumentPlan,
    ObservationKind,
    OcrCandidate,
    PageLayoutIntent,
    QualityCheck,
    QualityOutcome,
    QualityReport,
    StyleIntent,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)


def _bbox() -> BBox:
    return BBox(pixel=(10.0, 20.0, 110.0, 60.0), normalized=(0.1, 0.2, 0.8, 0.4))


def _evidence() -> EvidenceIR:
    return EvidenceIR(
        id="evidence-1",
        source_document_sha256="a" * 64,
        sources=(
            EvidenceSource(
                id="page-image-1",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref=f"artifact://{'b' * 64}/page-image",
                producer="pdf-renderer-v1",
                sha256="b" * 64,
            ),
            EvidenceSource(
                id="ocr-1",
                kind=EvidenceSourceKind.OCR_PROVIDER,
                artifact_ref="artifact://models/korean-exam-ppocrv5",
                producer="paddle-ocr-adapter-v1",
            ),
        ),
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=1000.0,
                height=1400.0,
                image_source_ref="page-image-1",
                observations=(
                    EvidenceObservation(
                        id="observation-1",
                        kind=ObservationKind.TEXT_LINE,
                        bbox=_bbox(),
                        confidence=0.99,
                        source_refs=("ocr-1",),
                        ocr_candidates=(
                            OcrCandidate(
                                text="1. 다음 글을 읽고 답하시오.",
                                confidence=0.99,
                                quality=0.98,
                                source_ref="ocr-1",
                                selected=True,
                                provider="paddleocr",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _content() -> ContentIR:
    evidence = _evidence()
    return ContentIR(
        id="content-1",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        nodes=(
            TextContentNode(
                id="text-1",
                role=ContentRole.QUESTION,
                text="1. 다음 글을 읽고 답하시오.",
                evidence_refs=("observation-1",),
                confidence=0.99,
            ),
        ),
        reading_order=("text-1",),
    )


def _table_payload() -> dict[str, Any]:
    return {
        "id": "table-1",
        "kind": "table",
        "role": "table",
        "evidence_refs": ["observation-1"],
        "confidence": 0.99,
        "needs_review": False,
        "rows": 1,
        "columns": 1,
        "cells": [
            {
                "row": 0,
                "column": 0,
                "row_span": 1,
                "column_span": 1,
                "text": "cell",
                "evidence_refs": ["observation-1"],
            }
        ],
    }


def _plan() -> HwpDocumentPlan:
    content = _content()
    return HwpDocumentPlan(
        id="plan-1",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        capability_profile_id="exam2hwpx-authoring-v1",
        design_profile_id="exam-clean-v1",
        official_spec_refs=("hancom-hwpx-format#styles-1",),
        page_layout=PageLayoutIntent(
            width_mm=210.0,
            height_mm=297.0,
            margin_top_mm=15.0,
            margin_right_mm=15.0,
            margin_bottom_mm=15.0,
            margin_left_mm=15.0,
            columns=2,
            column_gap_mm=8.0,
        ),
        styles=(StyleIntent(id="question", semantic_role="question", bold=True),),
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="paragraph",
                content_ref="text-1",
                style_ref="question",
            ),
        ),
    )


def _quality_report() -> QualityReport:
    evidence = _evidence()
    content = _content()
    plan = _plan()
    return QualityReport(
        id="quality-1",
        document_id="document-1",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        hwp_document_plan_id=plan.id,
        hwp_document_plan_sha256=contract_sha256(plan),
        compiled_artifact_ref="artifact://documents/document-1/result.hwpx",
        compiled_artifact_sha256="d" * 64,
        compiler_version="scan2hwpx-test-v1",
        model_bundle_id="bundle-1",
        check_set_id="document-quality-v1",
        outcome=QualityOutcome.AUTO_PASS,
        checks=tuple(
            QualityCheck(
                id=check_id,
                decision=CheckDecision.PASS,
                message=f"{check_id} passed.",
            )
            for check_id in sorted(REQUIRED_DOCUMENT_QUALITY_CHECKS_V1)
        ),
    )


@pytest.mark.parametrize("model", [_evidence(), _content(), _plan(), _quality_report()])
def test_contracts_round_trip_through_json(model: Any) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


def test_evidence_rejects_duplicate_ids_and_unknown_source_refs() -> None:
    duplicate = _evidence().model_dump(mode="json")
    duplicate["sources"].append(duplicate["sources"][0])
    with pytest.raises(ValidationError, match="duplicate evidence source ids"):
        EvidenceIR.model_validate(duplicate)

    unknown = _evidence().model_dump(mode="json")
    unknown["pages"][0]["image_source_ref"] = "missing-source"
    with pytest.raises(ValidationError, match="unknown evidence source refs"):
        EvidenceIR.model_validate(unknown)


def test_evidence_rejects_multiple_selected_ocr_candidates() -> None:
    payload = _evidence().model_dump(mode="json")
    payload["pages"][0]["observations"][0]["ocr_candidates"].append(
        {
            "text": "다른 후보",
            "confidence": 0.9,
            "quality": 0.9,
            "source_ref": "ocr-1",
            "selected": True,
            "provider": "second-ocr",
        }
    )
    with pytest.raises(ValidationError, match="more than one selected OCR candidate"):
        EvidenceIR.model_validate(payload)


def test_content_rejects_incomplete_or_duplicate_reading_order() -> None:
    payload = _content().model_dump(mode="json")
    payload["reading_order"] = []
    with pytest.raises(ValidationError):
        ContentIR.model_validate(payload)

    payload = _content().model_dump(mode="json")
    payload["reading_order"] = ["text-1", "text-1"]
    with pytest.raises(ValidationError, match="duplicate content reading-order refs"):
        ContentIR.model_validate(payload)


@pytest.mark.parametrize(
    ("field_path", "value"),
    (
        ("rows", True),
        ("columns", "1"),
        ("cells.0.row", False),
        ("cells.0.column", "0"),
        ("cells.0.row_span", True),
        ("cells.0.column_span", "1"),
    ),
)
def test_table_topology_rejects_boolean_and_numeric_coercion(
    field_path: str,
    value: object,
) -> None:
    payload = _table_payload()
    if field_path.startswith("cells.0."):
        payload["cells"][0][field_path.removeprefix("cells.0.")] = value
    else:
        payload[field_path] = value

    with pytest.raises(ValidationError, match="integer without coercion"):
        TableContentNode.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("rows", "columns"),
    (
        (2**63 - 1, 1),
        (10_001, 1),
        (10_000, 10_000),
    ),
)
def test_table_topology_rejects_huge_shape_before_range_allocation(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    columns: int,
) -> None:
    payload = _table_payload()
    payload["rows"] = rows
    payload["columns"] = columns
    payload["cells"][0]["row_span"] = rows
    payload["cells"][0]["column_span"] = columns

    def forbidden_range(*args: int) -> range:
        assert args
        raise AssertionError("table topology attempted an unbounded range allocation")

    monkeypatch.setattr(contract_models, "range", forbidden_range, raising=False)
    with pytest.raises(ValidationError, match="table resource limit"):
        TableContentNode.model_validate_json(json.dumps(payload))


def test_table_contract_schema_exposes_resource_limits() -> None:
    schema = TableContentNode.model_json_schema()
    properties = schema["properties"]
    assert properties["rows"]["maximum"] == 10_000
    assert properties["columns"]["maximum"] == 10_000
    assert properties["cells"]["maxItems"] == 100_000

    cell_properties = schema["$defs"]["TableCell"]["properties"]
    assert cell_properties["row"]["maximum"] == 9_999
    assert cell_properties["column"]["maximum"] == 9_999
    assert cell_properties["row_span"]["maximum"] == 10_000
    assert cell_properties["column_span"]["maximum"] == 10_000


def test_v4_max_table_shape_with_68_merged_cells_remains_valid() -> None:
    payload = _table_payload()
    payload["rows"] = 20
    payload["columns"] = 16
    cells: list[dict[str, object]] = []
    for row in range(3):
        cells.extend(
            {
                "row": row,
                "column": column,
                "text": f"{row},{column}",
                "evidence_refs": ["observation-1"],
            }
            for column in range(16)
        )
    for row in range(3, 6):
        cells.extend(
            {
                "row": row,
                "column": column,
                "column_span": 8,
                "text": f"{row},{column}",
                "evidence_refs": ["observation-1"],
            }
            for column in (0, 8)
        )
    cells.extend(
        {
            "row": row,
            "column": 0,
            "column_span": 16,
            "text": str(row),
            "evidence_refs": ["observation-1"],
        }
        for row in range(6, 20)
    )
    assert len(cells) == 68
    payload["cells"] = cells

    table = TableContentNode.model_validate_json(json.dumps(payload))

    assert (table.rows, table.columns, len(table.cells)) == (20, 16, 68)


@pytest.mark.parametrize(
    ("cells", "message"),
    (
        (
            [
                {
                    "row": 0,
                    "column": 0,
                    "text": "",
                    "evidence_refs": ["observation-1"],
                }
            ],
            "complete table grid",
        ),
        (
            [
                {
                    "row": 0,
                    "column": 0,
                    "row_span": 2,
                    "column_span": 2,
                    "text": "",
                    "evidence_refs": ["observation-1"],
                },
                {
                    "row": 1,
                    "column": 1,
                    "text": "",
                    "evidence_refs": ["observation-1"],
                },
            ],
            "overlap",
        ),
    ),
)
def test_bounded_table_still_rejects_incomplete_or_overlapping_cells(
    cells: list[dict[str, object]],
    message: str,
) -> None:
    payload = _table_payload()
    payload["rows"] = 2
    payload["columns"] = 2
    payload["cells"] = cells

    with pytest.raises(ValidationError, match=message):
        TableContentNode.model_validate_json(json.dumps(payload))


def test_cross_contract_integrity_rejects_unknown_references() -> None:
    _content().assert_evidence_integrity(_evidence())
    _plan().assert_content_integrity(_content())

    content_payload = _content().model_dump(mode="json")
    content_payload["nodes"][0]["evidence_refs"] = ["missing-observation"]
    invalid_content = ContentIR.model_validate(content_payload)
    with pytest.raises(ValueError, match="unknown evidence observation refs"):
        invalid_content.assert_evidence_integrity(_evidence())

    plan_payload = _plan().model_dump(mode="json")
    plan_payload["flow"][0]["content_ref"] = "missing-content"
    invalid_plan = HwpDocumentPlan.model_validate(plan_payload)
    with pytest.raises(ValueError, match="unknown content refs"):
        invalid_plan.assert_content_integrity(_content())


def test_cross_contract_integrity_rejects_stale_parent_bindings() -> None:
    content = _content()
    stale_evidence_payload = _evidence().model_dump(mode="json")
    stale_evidence_payload["source_document_sha256"] = "f" * 64
    stale_evidence = EvidenceIR.model_validate(stale_evidence_payload)
    with pytest.raises(ValueError, match="EvidenceIR digest mismatch"):
        content.assert_evidence_integrity(stale_evidence)

    plan = _plan()
    revised_content_payload = content.model_dump(mode="json")
    revised_content_payload["revision"] = 2
    revised_content = ContentIR.model_validate(revised_content_payload)
    with pytest.raises(ValueError, match="ContentIR revision mismatch"):
        plan.assert_content_integrity(revised_content)

    stale_content_payload = content.model_dump(mode="json")
    stale_content_payload["nodes"][0]["text"] = "same id and revision, different content"
    stale_content = ContentIR.model_validate(stale_content_payload)
    with pytest.raises(ValueError, match="ContentIR digest mismatch"):
        plan.assert_content_integrity(stale_content)


def test_plan_cannot_emit_body_text_raw_xml_or_unknown_styles() -> None:
    payload = _plan().model_dump(mode="json")
    payload["flow"][0]["text"] = "모델이 새로 만든 본문"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HwpDocumentPlan.model_validate(payload)

    payload = _plan().model_dump(mode="json")
    payload["raw_xml"] = "<hp:p/>"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HwpDocumentPlan.model_validate(payload)

    payload = _plan().model_dump(mode="json")
    payload["flow"][0]["style_ref"] = "missing-style"
    with pytest.raises(ValidationError, match="unknown plan style refs"):
        HwpDocumentPlan.model_validate(payload)


def test_plan_rejects_content_kind_mismatch() -> None:
    payload = _plan().model_dump(mode="json")
    payload["flow"][0]["render_as"] = "table"
    mismatched = HwpDocumentPlan.model_validate(payload)

    with pytest.raises(ValueError, match="content kind/render_as mismatch"):
        mismatched.assert_content_integrity(_content())


def test_quality_report_outcome_matches_most_severe_check() -> None:
    payload = _quality_report().model_dump(mode="json")
    payload["checks"][0]["decision"] = "block"

    with pytest.raises(ValidationError, match="quality outcome must be block"):
        QualityReport.model_validate(payload)


def test_auto_pass_requires_the_complete_versioned_check_set() -> None:
    payload = _quality_report().model_dump(mode="json")
    payload["checks"].pop()

    with pytest.raises(ValidationError, match="missing required checks"):
        QualityReport.model_validate(payload)


def test_quality_pass_cannot_contradict_its_numeric_threshold() -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        QualityCheck(
            id="visual-layout",
            decision=CheckDecision.PASS,
            message="Contradictory result.",
            score=0.5,
            threshold=0.9,
            comparator=">=",
        )


def test_quality_report_detects_stale_report_lineage() -> None:
    report = _quality_report()
    report.assert_contract_lineage(_evidence(), _content(), _plan())

    payload = report.model_dump(mode="json")
    payload["content_ir_sha256"] = "e" * 64
    stale_report = QualityReport.model_validate(payload)
    with pytest.raises(ValueError, match="content_ir_sha256"):
        stale_report.assert_contract_lineage(_evidence(), _content(), _plan())


def test_quality_report_rejects_stale_cross_contract_bindings() -> None:
    report_payload = _quality_report().model_dump(mode="json")
    stale_evidence_payload = _evidence().model_dump(mode="json")
    stale_evidence_payload["source_document_sha256"] = "f" * 64
    stale_evidence = EvidenceIR.model_validate(stale_evidence_payload)
    report_payload["evidence_ir_sha256"] = contract_sha256(stale_evidence)
    report = QualityReport.model_validate(report_payload)
    with pytest.raises(ValueError, match="EvidenceIR digest mismatch"):
        report.assert_contract_lineage(stale_evidence, _content(), _plan())

    stale_content_payload = _content().model_dump(mode="json")
    stale_content_payload["nodes"][0]["text"] = "stale content"
    stale_content = ContentIR.model_validate(stale_content_payload)
    report_payload = _quality_report().model_dump(mode="json")
    report_payload["content_ir_sha256"] = contract_sha256(stale_content)
    report = QualityReport.model_validate(report_payload)
    with pytest.raises(ValueError, match="ContentIR digest mismatch"):
        report.assert_contract_lineage(_evidence(), stale_content, _plan())
