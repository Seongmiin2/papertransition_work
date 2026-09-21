from __future__ import annotations

import hashlib
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from scan2hwpx.contracts import (
    BBox,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
)
from scan2hwpx.model_a import table_topology
from scan2hwpx.model_a.inference import ModelAPageImage
from scan2hwpx.model_a.table_topology import (
    TABLE_TOPOLOGY_POLICY_SHA256,
    ModelATableTopologyCell,
    ModelATableTopologyDetectorArtifact,
    ModelATableTopologyError,
    ModelATableTopologyTable,
    build_model_a_table_topology,
    validate_model_a_table_topology,
)
from scan2hwpx.vision.layout_dataset import TableGrid

_WIDTH = 1_000
_HEIGHT = 1_400


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _page_png(
    *,
    partial_horizontal: bool = False,
    partial_vertical: bool = False,
    marker_grid: bool = False,
) -> bytes:
    image = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    if marker_grid:
        draw.rectangle((300, 200, 800, 800), outline="black", width=5)
        draw.line((550, 200, 550, 800), fill="black", width=5)
        draw.line((300, 500, 800, 500), fill="black", width=5)
    else:
        draw.rectangle((100, 100, 700, 700), outline="black", width=5)
        vertical_top = 400 if partial_vertical else 100
        draw.line((400, vertical_top, 400, 700), fill="black", width=5)
        if partial_horizontal:
            draw.line((400, 400, 700, 400), fill="black", width=5)
        else:
            draw.line((100, 400, 700, 400), fill="black", width=5)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _observation(
    observation_id: str,
    text: str,
    bbox: tuple[float, float, float, float],
) -> EvidenceObservation:
    x0, y0, x1, y1 = bbox
    return EvidenceObservation(
        id=observation_id,
        kind=ObservationKind.TEXT_LINE,
        bbox=BBox(
            pixel=bbox,
            normalized=(x0 / _WIDTH, y0 / _HEIGHT, x1 / _WIDTH, y1 / _HEIGHT),
        ),
        confidence=0.9,
        source_refs=("page-image", "ocr-source"),
        ocr_candidates=(
            OcrCandidate(
                text=text,
                provider="fixture-ocr",
                confidence=0.9,
                source_ref="ocr-source",
                selected=True,
            ),
        ),
    )


def _matrix_observations(
    prefix: str,
    *,
    x_centers: tuple[float, ...],
    y_centers: tuple[float, ...],
    box_width: float = 30.0,
    box_height: float = 20.0,
) -> tuple[EvidenceObservation, ...]:
    return tuple(
        _observation(
            f"{prefix}-{row}-{column}",
            f"{row}-{column}",
            (
                x - box_width / 2,
                y - box_height / 2,
                x + box_width / 2,
                y + box_height / 2,
            ),
        )
        for row, y in enumerate(y_centers)
        for column, x in enumerate(x_centers)
    )


def _disable_ruled_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    horizontal = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    monkeypatch.setattr(
        table_topology,
        "ruled_line_masks_array",
        lambda _rgb: (horizontal, vertical),
    )
    monkeypatch.setattr(table_topology, "detect_table_grids_array", lambda _rgb: [])


def _bound_inputs(
    png: bytes,
    observations: tuple[EvidenceObservation, ...],
    *,
    width: float = _WIDTH,
    extra_sources: tuple[EvidenceSource, ...] = (),
) -> tuple[EvidenceIR, ModelAPageImage]:
    image_sha256 = _sha256(png)
    evidence = EvidenceIR(
        id="evidence-fixture",
        source_document_sha256="a" * 64,
        sources=(
            EvidenceSource(
                id="source-document",
                kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
                artifact_ref="artifact://fixture/source",
                producer="fixture",
                sha256="a" * 64,
            ),
            EvidenceSource(
                id="page-image",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/page",
                producer="fixture",
                sha256=image_sha256,
            ),
            EvidenceSource(
                id="ocr-source",
                kind=EvidenceSourceKind.OCR_PROVIDER,
                artifact_ref="artifact://fixture/ocr",
                producer="fixture-ocr",
            ),
            *extra_sources,
        ),
        pages=(
            EvidencePage(
                id="page-0001",
                page_no=1,
                width=width,
                height=_HEIGHT,
                image_source_ref="page-image",
                observations=observations,
            ),
        ),
    )
    image = ModelAPageImage(
        page_id="page-0001",
        page_no=1,
        image_source_ref="page-image",
        media_type="image/png",
        raw_bytes=png,
        sha256=image_sha256,
    )
    return evidence, image


def _complete_cell(
    *,
    column: int,
    column_span: int = 1,
    observation_ids: tuple[str, ...] = (),
    text: str = "",
) -> ModelATableTopologyCell:
    return ModelATableTopologyCell(
        row=0,
        column=column,
        row_span=1,
        column_span=column_span,
        bbox_normalized=(column / 2, 0.1, (column + column_span) / 2, 0.2),
        ordered_observation_ids=observation_ids,
        text=text,
    )


def test_build_infers_rectangular_row_span_and_copies_selected_ocr() -> None:
    png = _page_png(partial_horizontal=True)
    evidence, image = _bound_inputs(
        png,
        (
            _observation("left", "왼쪽", (150, 220, 250, 260)),
            _observation("right-top", "위", (450, 220, 500, 260)),
            _observation("right-bottom", "아래", (450, 520, 520, 560)),
        ),
    )

    built = build_model_a_table_topology(evidence, page_image=image)

    assert len(built.artifact.tables) == 1
    table = built.artifact.tables[0]
    assert (table.rows, table.columns) == (2, 2)
    assert [(cell.row, cell.column, cell.row_span, cell.column_span) for cell in table.cells] == [
        (0, 0, 2, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
    ]
    assert table.cells[0].ordered_observation_ids == ("left",)
    assert table.cells[0].text == "왼쪽"
    assert table.ordered_observation_ids == ("left", "right-top", "right-bottom")
    assert built.artifact.detector.detector_policy_sha256 == TABLE_TOPOLOGY_POLICY_SHA256
    assert not built.artifact.golden_eligible
    assert not built.artifact.training_eligible
    assert not built.artifact.release_eligible
    assert _sha256(built.artifact_bytes) == built.artifact_sha256


def test_bbox_overlap_can_share_one_ocr_line_across_horizontal_cells() -> None:
    png = _page_png()
    evidence, image = _bound_inputs(
        png,
        (
            _observation("combined", "11. ②", (260, 220, 540, 270)),
            _observation("mismatch", "토큰 세 개", (260, 520, 540, 570)),
        ),
    )

    table = build_model_a_table_topology(evidence, page_image=image).artifact.tables[0]

    assert table.cells[0].ordered_observation_ids == ("combined",)
    assert table.cells[1].ordered_observation_ids == ("combined",)
    assert table.cells[0].text == "11."
    assert table.cells[1].text == "②"
    assert table.cells[2].text == table.cells[3].text == "토큰 세 개"
    assert table.ordered_observation_ids == ("combined", "mismatch")


def test_infers_column_span_only_when_missing_boundaries_form_a_rectangle() -> None:
    png = _page_png(partial_vertical=True)
    evidence, image = _bound_inputs(png, ())

    table = build_model_a_table_topology(evidence, page_image=image).artifact.tables[0]

    assert [(cell.row, cell.column, cell.row_span, cell.column_span) for cell in table.cells] == [
        (0, 0, 1, 2),
        (1, 0, 1, 1),
        (1, 1, 1, 1),
    ]

    l_shape_png = _page_png(partial_horizontal=True, partial_vertical=True)
    l_evidence, l_image = _bound_inputs(l_shape_png, ())
    l_table = build_model_a_table_topology(l_evidence, page_image=l_image).artifact.tables[0]
    assert [(cell.row_span, cell.column_span) for cell in l_table.cells] == [(1, 1)] * 4


def test_enriches_narrow_left_marker_column_from_evidence() -> None:
    png = _page_png(marker_grid=True)
    evidence, image = _bound_inputs(
        png,
        (
            _observation("marker-1", "①", (250, 320, 285, 360)),
            _observation("marker-2", "②", (250, 620, 285, 660)),
            _observation("body", "본문", (360, 320, 440, 360)),
        ),
    )

    table = build_model_a_table_topology(evidence, page_image=image).artifact.tables[0]

    assert (table.rows, table.columns) == (2, 3)
    assert table.bbox_normalized[0] < 0.3
    assert next(cell for cell in table.cells if (cell.row, cell.column) == (0, 0)).text == "①"
    assert next(cell for cell in table.cells if (cell.row, cell.column) == (1, 0)).text == "②"


def test_internal_line_filter_uses_original_candidates_and_preserves_outer_lines() -> None:
    horizontal = np.zeros((180, 120), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    grid = TableGrid(
        bbox=(10.0, 20.0, 90.0, 140.0),
        x_lines=(10.0, 50.0, 90.0),
        y_lines=(20.0, 60.0, 100.0, 140.0),
    )
    vertical[12:29, 46:55] = 255
    horizontal[56:65, 42:59] = 255
    horizontal[56:65, 82:99] = 255

    filtered = table_topology._filter_internal_grid_lines(
        grid,
        horizontal=horizontal,
        vertical=vertical,
    )

    assert filtered == TableGrid(
        bbox=grid.bbox,
        x_lines=(10.0, 90.0),
        y_lines=(20.0, 60.0, 140.0),
    )


def test_build_rejects_grid_below_minimum_after_internal_line_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = _page_png()
    evidence, image = _bound_inputs(png, ())
    horizontal = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    grid = TableGrid(
        bbox=(100.0, 100.0, 700.0, 700.0),
        x_lines=(100.0, 700.0),
        y_lines=(100.0, 400.0, 700.0),
    )
    horizontal[396:405, 92:109] = 255
    monkeypatch.setattr(
        table_topology,
        "ruled_line_masks_array",
        lambda _rgb: (horizontal, vertical),
    )
    monkeypatch.setattr(table_topology, "detect_table_grids_array", lambda _rgb: [grid])

    built = build_model_a_table_topology(evidence, page_image=image)

    assert built.artifact.tables == ()


def test_crop_image_suppresses_only_empty_contained_ruled_grid() -> None:
    png = _page_png()
    crop_source = EvidenceSource(
        id="crop-source",
        kind=EvidenceSourceKind.CROP,
        artifact_ref="artifact://fixture/crop",
        producer="fixture",
        sha256="b" * 64,
    )
    image = EvidenceObservation(
        id="content-image",
        kind=ObservationKind.IMAGE,
        bbox=BBox(
            pixel=(80.0, 80.0, 720.0, 720.0),
            normalized=(0.08, 80 / _HEIGHT, 0.72, 720 / _HEIGHT),
        ),
        confidence=0.9,
        source_refs=("page-image", "crop-source"),
    )

    empty_evidence, page_image = _bound_inputs(
        png,
        (image,),
        extra_sources=(crop_source,),
    )
    assert build_model_a_table_topology(empty_evidence, page_image=page_image).artifact.tables == ()

    text_evidence, page_image = _bound_inputs(
        png,
        (image, _observation("inside", "value", (150, 220, 250, 260))),
        extra_sources=(crop_source,),
    )
    assert len(build_model_a_table_topology(text_evidence, page_image=page_image).artifact.tables) == 1


def test_page_image_source_does_not_suppress_empty_ruled_grid() -> None:
    png = _page_png()
    image = EvidenceObservation(
        id="page-fallback-image",
        kind=ObservationKind.IMAGE,
        bbox=BBox(
            pixel=(0.0, 0.0, float(_WIDTH), float(_HEIGHT)),
            normalized=(0.0, 0.0, 1.0, 1.0),
        ),
        confidence=0.9,
        source_refs=("page-image",),
    )
    evidence, page_image = _bound_inputs(png, (image,))

    assert len(build_model_a_table_topology(evidence, page_image=page_image).artifact.tables) == 1


def test_build_detects_atomic_ocr_geometry_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_ruled_detection(monkeypatch)
    observations = _matrix_observations(
        "matrix",
        x_centers=(200.0, 400.0, 600.0),
        y_centers=(200.0, 250.0, 300.0, 350.0),
    )
    png = _page_png()
    evidence, image = _bound_inputs(png, observations)

    built = build_model_a_table_topology(evidence, page_image=image)

    assert len(built.artifact.tables) == 1
    table = built.artifact.tables[0]
    assert (table.rows, table.columns) == (4, 3)
    assert len(table.cells) == 12
    assert all((cell.row_span, cell.column_span) == (1, 1) for cell in table.cells)
    assert table.ordered_observation_ids == tuple(item.id for item in observations)


def test_build_rejects_ordinary_prose_as_ocr_geometry_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_ruled_detection(monkeypatch)
    observations = tuple(
        _observation(f"line-{index}", f"prose {index}", (100, y, 700, y + 20))
        for index, y in enumerate((200, 250, 300, 350))
    )
    png = _page_png()
    evidence, image = _bound_inputs(png, observations)

    built = build_model_a_table_topology(evidence, page_image=image)

    assert built.artifact.tables == ()


def test_duplicate_ocr_does_not_inflate_matrix_slot_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_ruled_detection(monkeypatch)
    observations = (
        _observation("r0-c0-a", "a", (185, 190, 215, 210)),
        _observation("r0-c0-b", "a", (188, 190, 218, 210)),
        _observation("r0-c1", "b", (385, 190, 415, 210)),
        _observation("r1-c1-a", "c", (385, 240, 415, 260)),
        _observation("r1-c1-b", "c", (388, 240, 418, 260)),
        _observation("r1-c2", "d", (585, 240, 615, 260)),
        _observation("r2-c0", "e", (185, 290, 215, 310)),
        _observation("r2-c2-a", "f", (585, 290, 615, 310)),
        _observation("r2-c2-b", "f", (588, 290, 618, 310)),
    )
    png = _page_png()
    evidence, image = _bound_inputs(png, observations)

    built = build_model_a_table_topology(evidence, page_image=image)

    assert built.artifact.tables == ()


def test_ruled_grid_wins_over_overlapping_ocr_geometry_matrix() -> None:
    observations = _matrix_observations(
        "overlap",
        x_centers=(200.0, 400.0, 600.0),
        y_centers=(200.0, 400.0, 600.0),
        box_height=80.0,
    )
    png = _page_png()
    evidence, image = _bound_inputs(png, observations)

    built = build_model_a_table_topology(evidence, page_image=image)

    assert len(built.artifact.tables) == 1
    assert (built.artifact.tables[0].rows, built.artifact.tables[0].columns) == (2, 2)


def test_independent_half_matrices_do_not_emit_combined_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_ruled_detection(monkeypatch)
    observations = (
        *_matrix_observations(
            "left",
            x_centers=(100.0, 200.0, 300.0),
            y_centers=(200.0, 250.0, 300.0, 350.0, 400.0),
        ),
        *_matrix_observations(
            "right",
            x_centers=(650.0, 750.0, 850.0),
            y_centers=(200.0, 250.0, 300.0, 350.0, 400.0),
        ),
    )
    png = _page_png()
    evidence, image = _bound_inputs(png, observations)

    built = build_model_a_table_topology(evidence, page_image=image)

    assert [(table.rows, table.columns) for table in built.artifact.tables] == [(5, 3), (5, 3)]
    first_ids = set(built.artifact.tables[0].ordered_observation_ids)
    second_ids = set(built.artifact.tables[1].ordered_observation_ids)
    assert not first_ids & second_ids
    assert first_ids | second_ids == {item.id for item in observations}


def test_topology_models_reject_overlap_incomplete_grid_and_duplicate_cell_ref() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ModelATableTopologyTable(
            table_grid_observation_id="grid",
            rows=1,
            columns=2,
            bbox_normalized=(0.0, 0.1, 1.0, 0.2),
            ordered_observation_ids=(),
            cells=(
                _complete_cell(column=0, column_span=2),
                _complete_cell(column=1),
            ),
        )
    with pytest.raises(ValidationError, match="completely cover"):
        ModelATableTopologyTable(
            table_grid_observation_id="grid",
            rows=1,
            columns=2,
            bbox_normalized=(0.0, 0.1, 1.0, 0.2),
            ordered_observation_ids=(),
            cells=(_complete_cell(column=0),),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        _complete_cell(
            column=0,
            observation_ids=("duplicate", "duplicate"),
            text="원문\n원문",
        )


def test_replay_rejects_forged_source_digest_ref_escape_and_text() -> None:
    png = _page_png()
    evidence, image = _bound_inputs(
        png,
        (_observation("inside", "원문", (150, 220, 250, 260)),),
    )
    built = build_model_a_table_topology(evidence, page_image=image)

    forged_source = built.artifact.source.model_copy(
        update={"evidence_ir_contract_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="cannot be reproduced"):
        replace(built, artifact=built.artifact.model_copy(update={"source": forged_source}))

    first_table = built.artifact.tables[0]
    first_cell = first_table.cells[0]
    escaped_cell = first_cell.model_copy(
        update={"ordered_observation_ids": ("outside",), "text": "위조"}
    )
    escaped_table = first_table.model_copy(
        update={
            "cells": (escaped_cell, *first_table.cells[1:]),
            "ordered_observation_ids": ("outside",),
        }
    )
    escaped_artifact = built.artifact.model_copy(update={"tables": (escaped_table,)})
    with pytest.raises(ValueError, match="cannot be reproduced"):
        validate_model_a_table_topology(
            escaped_artifact,
            evidence_ir=evidence,
            page_image=image,
        )

    forged_cell = first_cell.model_copy(update={"text": "새로 만든 문구"})
    forged_table = first_table.model_copy(update={"cells": (forged_cell, *first_table.cells[1:])})
    with pytest.raises(ValueError, match="cannot be reproduced"):
        validate_model_a_table_topology(
            built.artifact.model_copy(update={"tables": (forged_table,)}),
            evidence_ir=evidence,
            page_image=image,
        )


def test_rejects_forged_detector_policy_and_image_binding() -> None:
    with pytest.raises(ValidationError, match="deployed policy"):
        ModelATableTopologyDetectorArtifact(detector_policy_sha256="0" * 64)

    png = _page_png()
    evidence, _image = _bound_inputs(png, ())
    changed_png = _page_png(marker_grid=True)
    changed_image = ModelAPageImage(
        page_id="page-0001",
        page_no=1,
        image_source_ref="page-image",
        media_type="image/png",
        raw_bytes=changed_png,
        sha256=_sha256(changed_png),
    )
    with pytest.raises(ModelATableTopologyError, match="exact page binding"):
        build_model_a_table_topology(evidence, page_image=changed_image)


def test_rejects_decoded_dimensions_that_disagree_with_evidence_page() -> None:
    png = _page_png()
    evidence, image = _bound_inputs(png, (), width=_WIDTH - 1)

    with pytest.raises(ModelATableTopologyError, match="exact page binding"):
        build_model_a_table_topology(evidence, page_image=image)


@pytest.mark.parametrize(
    ("field", "value"),
    (("page_id", "other-page"), ("page_no", 2), ("image_source_ref", "other-source")),
)
def test_rejects_page_image_identity_mismatch(field: str, value: object) -> None:
    png = _page_png()
    evidence, image = _bound_inputs(png, ())
    mismatched = replace(image, **{field: value})

    with pytest.raises(ModelATableTopologyError, match="exact page binding"):
        build_model_a_table_topology(evidence, page_image=mismatched)
