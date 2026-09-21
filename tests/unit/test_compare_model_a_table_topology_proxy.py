from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.modeling import compare_model_a_table_topology_proxy as comparison
from scan2hwpx.contracts import (
    BBox,
    ContentIR,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
    TableCell,
    TableContentNode,
    contract_sha256,
)
from scan2hwpx.model_a.table_topology import (
    TABLE_TOPOLOGY_POLICY_SHA256,
    ModelATableTopologyArtifact,
    ModelATableTopologyCell,
    ModelATableTopologyDetectorArtifact,
    ModelATableTopologySourceBinding,
    ModelATableTopologyTable,
)
from scan2hwpx.reference.hwpx import (
    HwpxProjectionSource,
    ProjectedCell,
    ProjectedNode,
    ProjectedPageLayout,
    ProjectedSection,
)


def _bbox(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(
        pixel=(x0 * 100, y0 * 100, x1 * 100, y1 * 100),
        normalized=(x0, y0, x1, y1),
    )


def _reference_fixture() -> tuple[HwpxProjectionSource, ContentIR, EvidenceIR]:
    source_sha = "a" * 64
    evidence_sha = "b" * 64
    page_image_sha = "c" * 64
    locator = "s000/p000000/r000/tbl000"
    source_id = f"{source_sha}:{locator}"
    projected_cells = (
        ProjectedCell(
            locator=f"{locator}/cell-r000-c000",
            row=0,
            column=0,
            row_span=2,
            text="A",
        ),
        ProjectedCell(
            locator=f"{locator}/cell-r000-c001",
            row=0,
            column=1,
            text="B",
        ),
        ProjectedCell(
            locator=f"{locator}/cell-r001-c001",
            row=1,
            column=1,
            text="",
        ),
    )
    projected_table = ProjectedNode(
        id=source_id,
        locator=locator,
        kind="table",
        text="A\tB\n\t",
        rows=2,
        columns=2,
        cells=projected_cells,
        review_reasons=("machine_projected",),
    )
    projection = HwpxProjectionSource(
        source_document_sha256=source_sha,
        hwpx_sha256="d" * 64,
        page_count=1,
        sections=(
            ProjectedSection(
                index=0,
                layout=ProjectedPageLayout(
                    width_hwp=100,
                    height_hwp=100,
                    margin_top_hwp=0,
                    margin_right_hwp=0,
                    margin_bottom_hwp=0,
                    margin_left_hwp=0,
                    margin_header_hwp=0,
                    margin_footer_hwp=0,
                    margin_gutter_hwp=0,
                ),
                node_refs=(source_id,),
            ),
        ),
        nodes=(projected_table,),
        styles=(),
        assets=(),
        issues=(),
        stats={"tables": 1, "table_cells": 3},
    )
    original = EvidenceSource(
        id="input",
        kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
        artifact_ref="artifact://fixture/source",
        producer="fixture",
        sha256=evidence_sha,
    )
    page_image = EvidenceSource(
        id="page-1-image",
        kind=EvidenceSourceKind.PAGE_IMAGE,
        artifact_ref="artifact://fixture/page-1",
        producer="fixture",
        sha256=page_image_sha,
    )
    observations = (
        EvidenceObservation(
            id="t1",
            kind=ObservationKind.TEXT_LINE,
            bbox=_bbox(0.1, 0.1, 0.2, 0.2),
            confidence=1,
            source_refs=("input", "page-1-image"),
            ocr_candidates=(
                OcrCandidate(
                    text="A",
                    provider="fixture",
                    confidence=1,
                    source_ref="input",
                    selected=True,
                ),
            ),
        ),
        EvidenceObservation(
            id="t2",
            kind=ObservationKind.TEXT_LINE,
            bbox=_bbox(0.5, 0.1, 0.6, 0.2),
            confidence=1,
            source_refs=("input", "page-1-image"),
            ocr_candidates=(
                OcrCandidate(
                    text="B",
                    provider="fixture",
                    confidence=1,
                    source_ref="input",
                    selected=True,
                ),
            ),
        ),
        EvidenceObservation(
            id="page-1-region",
            kind=ObservationKind.REGION,
            bbox=_bbox(0, 0, 1, 1),
            confidence=0,
            source_refs=("page-1-image",),
        ),
    )
    evidence = EvidenceIR(
        id="evidence-fixture",
        source_document_sha256=evidence_sha,
        sources=(original, page_image),
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=100,
                height=100,
                image_source_ref="page-1-image",
                observations=observations,
            ),
        ),
    )
    content_id = comparison._projection_content_id(projected_table)
    content = ContentIR(
        id="content-fixture",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        nodes=(
            TableContentNode(
                id=content_id,
                rows=2,
                columns=2,
                cells=(
                    TableCell(
                        row=0,
                        column=0,
                        row_span=2,
                        text="A",
                        evidence_refs=("t1",),
                    ),
                    TableCell(
                        row=0,
                        column=1,
                        text="B",
                        evidence_refs=("t2",),
                    ),
                    TableCell(
                        row=1,
                        column=1,
                        text="",
                        evidence_refs=("page-1-region",),
                    ),
                ),
                evidence_refs=("t1", "t2", "page-1-region"),
                confidence=1,
                needs_review=True,
            ),
        ),
        reading_order=(content_id,),
    )
    return projection, content, evidence


def _prediction_table(
    table_id: str,
    cells: tuple[ModelATableTopologyCell, ...],
    *,
    rows: int,
    columns: int,
) -> ModelATableTopologyTable:
    ordered_ids = tuple(
        dict.fromkeys(
            observation_id for cell in cells for observation_id in cell.ordered_observation_ids
        )
    )
    return ModelATableTopologyTable(
        table_grid_observation_id=table_id,
        rows=rows,
        columns=columns,
        bbox_normalized=(0.05, 0.05, 0.95, 0.95),
        ordered_observation_ids=ordered_ids,
        cells=cells,
    )


def _prediction_cell(
    row: int,
    column: int,
    observation_ids: tuple[str, ...],
    text: str,
    *,
    row_span: int = 1,
    column_span: int = 1,
) -> ModelATableTopologyCell:
    return ModelATableTopologyCell(
        row=row,
        column=column,
        row_span=row_span,
        column_span=column_span,
        bbox_normalized=(0.05, 0.05, 0.95, 0.95),
        ordered_observation_ids=observation_ids,
        text=text,
    )


def test_projection_candidate_binding_preserves_span_empty_and_text_refs() -> None:
    projection, content, evidence = _reference_fixture()

    tables = comparison._build_proxy_tables(projection, content, evidence)

    assert len(tables) == 1
    table = tables[0]
    assert (table.rows, table.columns) == (2, 2)
    assert tuple(cell.key for cell in table.cells) == (
        (0, 0, 2, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
    )
    assert table.ordered_observation_ids == ("t1", "t2")
    assert table.cells[-1].text == ""
    assert table.cells[-1].ordered_observation_ids == ()

    table_node = content.nodes[0]
    forged_cell = table_node.cells[0].model_copy(update={"text": "forged"})
    forged_table = table_node.model_copy(update={"cells": (forged_cell, *table_node.cells[1:])})
    forged_content = content.model_copy(update={"nodes": (forged_table,)})
    with pytest.raises(
        ValueError,
        match="cell text differs from projected HWPX table",
    ):
        comparison._build_proxy_tables(projection, forged_content, evidence)


def test_table_metrics_separate_topology_assignment_text_and_outside_absorption() -> None:
    projection, content, evidence = _reference_fixture()
    reference = comparison._build_proxy_tables(projection, content, evidence)[0]
    prediction = _prediction_table(
        "page-1:grid",
        (
            _prediction_cell(0, 0, ("t1",), "A", row_span=2),
            _prediction_cell(0, 1, ("t2", "outside"), "B"),
            _prediction_cell(1, 1, (), ""),
        ),
        rows=2,
        columns=2,
    )

    metrics = comparison._table_comparison(reference, prediction)

    assert metrics["rows_columns_exact"] is True
    assert metrics["grid_span_exact"] is True
    assert metrics["empty_cell_signature_exact"] is True
    assert metrics["strict_cell_assignment_exact_count"] == 2
    assert metrics["strict_cell_assignment_sample_count"] == 3
    assert metrics["assignment_pair_true_positive_count"] == 2
    assert metrics["extra_assignment_pair_count"] == 1
    assert metrics["outside_absorbed_text_observation_ids"] == ["outside"]
    assert metrics["normalized_cell_text_exact_count"] == 3

    missing = comparison._table_comparison(reference, None)
    assert missing["grid_span_exact"] is False
    assert missing["empty_cell_signature_exact"] is False


def test_shared_line_cells_are_reported_without_losing_exact_assignment() -> None:
    reference = comparison._ProxyTable(
        source_node_id="source",
        content_node_id="content",
        page_id="page-1",
        page_no=1,
        rows=1,
        columns=2,
        bbox_normalized=(0.1, 0.1, 0.9, 0.2),
        cells=(
            comparison._ProxyCell((0, 0, 1, 1), "11.", ("line",)),
            comparison._ProxyCell((0, 1, 1, 1), "①", ("line",)),
        ),
    )
    prediction = _prediction_table(
        "page-1:grid",
        (
            _prediction_cell(0, 0, ("line",), "11."),
            _prediction_cell(0, 1, ("line",), "①"),
        ),
        rows=1,
        columns=2,
    )

    metrics = comparison._table_comparison(reference, prediction)

    assert metrics["reference_ambiguous_text_observation_count"] == 1
    assert metrics["reference_ambiguous_cell_count"] == 2
    assert metrics["shared_text_observation_signature_exact"] is True
    assert metrics["strict_cell_assignment_exact_count"] == 2
    assert metrics["unambiguous_cell_assignment_sample_count"] == 0
    assert metrics["normalized_cell_text_exact_count"] == 2


def test_matching_maximizes_total_overlap_instead_of_greedy_first_match() -> None:
    def proxy(name: str, ids: tuple[str, ...]) -> comparison._ProxyTable:
        return comparison._ProxyTable(
            source_node_id=f"source-{name}",
            content_node_id=f"content-{name}",
            page_id="page-1",
            page_no=1,
            rows=1,
            columns=1,
            bbox_normalized=None,
            cells=(comparison._ProxyCell((0, 0, 1, 1), name, ids),),
        )

    references = (proxy("first", ("a", "b")), proxy("second", ("c",)))
    predictions = (
        _prediction_table(
            "grid-0",
            (_prediction_cell(0, 0, ("a", "c"), "ac"),),
            rows=1,
            columns=1,
        ),
        _prediction_table(
            "grid-1",
            (_prediction_cell(0, 0, ("b",), "b"),),
            rows=1,
            columns=1,
        ),
    )

    assert comparison._maximum_overlap_matching(references, predictions) == (
        (0, 1),
        (1, 0),
    )


def test_grid_promotion_is_ready_but_not_materialized_in_original_evidence() -> None:
    _projection, _content, evidence = _reference_fixture()
    grid_id = f"page-1:table-grid:{TABLE_TOPOLOGY_POLICY_SHA256[:12]}:0000"
    table = _prediction_table(
        grid_id,
        (_prediction_cell(0, 0, ("t1",), "A"),),
        rows=1,
        columns=1,
    )
    artifact = ModelATableTopologyArtifact(
        source=ModelATableTopologySourceBinding(
            evidence_ir_id=evidence.id,
            evidence_ir_contract_sha256=contract_sha256(evidence),
            source_document_sha256=evidence.source_document_sha256,
            target_page_id="page-1",
            target_page_no=1,
            page_image_source_ref="page-1-image",
            page_image_sha256="c" * 64,
            page_width_pixels=100,
            page_height_pixels=100,
        ),
        detector=ModelATableTopologyDetectorArtifact(),
        tables=(table,),
    )

    result = comparison._evidence_promotion_block(
        evidence,
        evidence.pages,
        (SimpleNamespace(artifact=artifact),),
    )

    assert result["deterministic_grid_ids_exact"] is True
    assert result["grid_id_collision_count"] == 0
    assert result["promotion_contract_ready"] is True
    assert result["promotion_materialized_in_input_evidence"] is False
    assert result["current_evidence_ir_model_a_table_compatible"] is False

    layout_source = EvidenceSource(
        id="table-layout",
        kind=EvidenceSourceKind.LAYOUT_PROVIDER,
        artifact_ref="artifact://fixture/table-layout",
        producer="fixture",
    )
    grid_observation = EvidenceObservation(
        id=grid_id,
        kind=ObservationKind.TABLE_GRID,
        bbox=_bbox(0.05, 0.05, 0.95, 0.95),
        confidence=1,
        source_refs=("page-1-image", "table-layout"),
    )
    augmented_page = evidence.pages[0].model_copy(
        update={"observations": (*evidence.pages[0].observations, grid_observation)}
    )
    augmented = evidence.model_copy(
        update={
            "sources": (*evidence.sources, layout_source),
            "pages": (augmented_page,),
        }
    )
    materialized = comparison._evidence_promotion_block(
        augmented,
        augmented.pages,
        (SimpleNamespace(artifact=artifact),),
    )

    assert materialized["grid_id_collision_count"] == 0
    assert materialized["promotion_contract_ready"] is True
    assert materialized["promotion_materialized_in_input_evidence"] is True
    assert materialized["current_evidence_ir_model_a_table_compatible"] is True
