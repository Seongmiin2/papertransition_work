from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from statistics import median
from typing import Annotated, Any, Literal, Self

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ConfigDict, Field, ValidationError, model_validator

from scan2hwpx.contracts import (
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSourceKind,
    ObservationKind,
    contract_sha256,
)
from scan2hwpx.contracts.models import StrictContractModel
from scan2hwpx.vision.layout_dataset import (
    TableGrid,
    detect_table_grids_array,
    ruled_line_masks_array,
)

from .inference import ModelAPageImage

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedText = Annotated[str, Field(min_length=1, max_length=1_024)]
NormalizedBBox = tuple[float, float, float, float]

TABLE_TOPOLOGY_ARTIFACT_VERSION: Literal["model-a-table-topology/1.0"] = (
    "model-a-table-topology/1.0"
)
TABLE_TOPOLOGY_DETECTOR_VERSION: Literal["model-a-table-topology-detector/1.3"] = (
    "model-a-table-topology-detector/1.3"
)

_MAX_TABLES = 1_000
_MAX_TABLE_DIMENSION = 10_000
_MAX_TABLE_CELLS = 100_000
_MAX_REFS_PER_CELL = 20_000
_MAX_CELL_TEXT_CHARS = 1_000_000
_MAX_IMAGE_PIXELS = 50_000_000
_BOUNDARY_BAND_RADIUS = 4
_BOUNDARY_INSET = 3
_BOUNDARY_COVERAGE = 0.55
_INTERNAL_LINE_MIN_INTERSECTIONS = 2
_VERTICAL_INTERSECTION_X_RADIUS = 4
_VERTICAL_INTERSECTION_Y_RADIUS = 8
_HORIZONTAL_INTERSECTION_X_RADIUS = 8
_HORIZONTAL_INTERSECTION_Y_RADIUS = 4
_OCR_MATRIX_ZONES = ("left_half", "right_half", "full_page")
_OCR_MATRIX_MIN_ROWS = 3
_OCR_MATRIX_MIN_COLUMNS = 3
_OCR_MATRIX_MAX_COLUMNS = 10
_OCR_MATRIX_MIN_Y_TOLERANCE = 5.0
_OCR_MATRIX_Y_TOLERANCE_HEIGHT_RATIO = 0.28
_OCR_MATRIX_MIN_ROW_GAP_HEIGHT_RATIO = 0.75
_OCR_MATRIX_MAX_ROW_GAP_HEIGHT_RATIO = 3.2
_OCR_MATRIX_MIN_GAP_MEDIAN_RATIO = 0.65
_OCR_MATRIX_MAX_GAP_MEDIAN_RATIO = 1.35
_OCR_MATRIX_MIN_X_TOLERANCE = 16.0
_OCR_MATRIX_X_TOLERANCE_PAGE_WIDTH_RATIO = 0.0125
_OCR_MATRIX_MAX_ADJACENT_X_GAP_RATIO = 1.75
_OCR_MATRIX_MIN_OCCUPANCY = 0.85
_OCR_MATRIX_MAX_MEDIAN_BOX_WIDTH_RATIO = 0.09
_OCR_MATRIX_MIN_X_SPAN_RATIO = 0.12
_OCR_MATRIX_CANDIDATE_DEDUPE_IOU = 0.55
_OCR_MATRIX_RULED_GRID_MAX_IOU = 0.35
_IMAGE_CONTAINMENT_MIN_GRID_COVERAGE = 0.95
_NORMALIZED_DECIMALS = 8
_MARKER_MIN_COUNT = 2
_MARKER_MAX_WIDTH_RATIO = 0.04
_MARKER_MAX_GAP_RATIO = 0.03
_MARKER_LEFT_PADDING_RATIO = 0.125
_MIN_CELL_OVERLAP_RATIO = 0.05
_MARKER_PATTERN_TEXT = (
    r"(?:[\u2460-\u2473\u3251-\u325f\u3260-\u327b]|[(（][가-힣A-Za-z0-9]{1,3}[)）])"
)
_MARKER_PATTERN = re.compile(rf"^(?:{_MARKER_PATTERN_TEXT})$")


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


_DETECTOR_POLICY = {
    "boundary_detection": {
        "band_radius_pixels": _BOUNDARY_BAND_RADIUS,
        "coverage_threshold": _BOUNDARY_COVERAGE,
        "interior_inset_pixels": _BOUNDARY_INSET,
        "masks": "scan2hwpx.vision.layout_dataset.ruled_line_masks_array",
    },
    "cell_text": {
        "base": "newline_join_preferred_ocr_in_y_x_evidence_order",
        "shared_horizontal_line": (
            "when_whitespace_token_count_equals_grounded_output_cell_count_assign_one_"
            "unchanged_token_per_cell_left_to_right_otherwise_repeat_full_ocr"
        ),
    },
    "detector": "scan2hwpx.vision.layout_dataset.detect_table_grids_array",
    "detector_version": TABLE_TOPOLOGY_DETECTOR_VERSION,
    "internal_line_intersection_filter": {
        "horizontal_mask_window_pixels": {
            "x_radius": _HORIZONTAL_INTERSECTION_X_RADIUS,
            "y_radius": _HORIZONTAL_INTERSECTION_Y_RADIUS,
        },
        "minimum_original_orthogonal_candidates": _INTERNAL_LINE_MIN_INTERSECTIONS,
        "one_pass_uses_original_candidate_sets": True,
        "preserve_outer_lines": True,
        "reject_below_line_counts": {"x": 2, "y": 3},
        "vertical_mask_window_pixels": {
            "x_radius": _VERTICAL_INTERSECTION_X_RADIUS,
            "y_radius": _VERTICAL_INTERSECTION_Y_RADIUS,
        },
    },
    "image_containment_filter": {
        "image_binding": "image_observation_with_hashed_crop_source",
        "minimum_grid_area_coverage": _IMAGE_CONTAINMENT_MIN_GRID_COVERAGE,
        "require_zero_assignable_text_lines": True,
        "scope": "ruled_grids_after_marker_column_enrichment",
    },
    "ocr_geometry_matrix_head": {
        "atomic_cells_only": True,
        "cell_assignment": "candidate_member_text_lines_only",
        "candidate_dedupe_iou": _OCR_MATRIX_CANDIDATE_DEDUPE_IOU,
        "candidate_members_must_be_disjoint": True,
        "column_count": {
            "maximum": _OCR_MATRIX_MAX_COLUMNS,
            "minimum": _OCR_MATRIX_MIN_COLUMNS,
        },
        "occupancy": {
            "minimum": _OCR_MATRIX_MIN_OCCUPANCY,
            "numerator": "unique_nearest_row_column_slots",
        },
        "candidate_selection_order": [
            "occupied_slot_count_desc",
            "occupancy_desc",
            "bbox_y_x",
            "zone_index",
            "member_ids",
        ],
        "infer_boundaries": "midpoints_between_cluster_means_with_symmetric_outer_edges",
        "marker_column_enrichment": False,
        "max_median_box_width_page_ratio": _OCR_MATRIX_MAX_MEDIAN_BOX_WIDTH_RATIO,
        "min_horizontal_span_page_ratio": _OCR_MATRIX_MIN_X_SPAN_RATIO,
        "minimum_rows_per_run": _OCR_MATRIX_MIN_ROWS,
        "minimum_text_lines_per_row": 2,
        "row_gap_height_ratio": {
            "maximum": _OCR_MATRIX_MAX_ROW_GAP_HEIGHT_RATIO,
            "minimum": _OCR_MATRIX_MIN_ROW_GAP_HEIGHT_RATIO,
        },
        "row_gap_median_ratio": {
            "maximum": _OCR_MATRIX_MAX_GAP_MEDIAN_RATIO,
            "minimum": _OCR_MATRIX_MIN_GAP_MEDIAN_RATIO,
        },
        "ruled_grid_max_iou_exclusive": _OCR_MATRIX_RULED_GRID_MAX_IOU,
        "ruled_grid_member_ids_must_be_disjoint": True,
        "typical_height": "median_valid_text_line_height_per_zone",
        "valid_bbox": "finite_positive_and_within_page",
        "x_clustering": {
            "maximum_adjacent_gap_ratio": _OCR_MATRIX_MAX_ADJACENT_X_GAP_RATIO,
            "method": "sorted_greedy_distance_from_current_group_mean",
            "tolerance": {
                "minimum_pixels": _OCR_MATRIX_MIN_X_TOLERANCE,
                "page_width_ratio": _OCR_MATRIX_X_TOLERANCE_PAGE_WIDTH_RATIO,
            },
        },
        "y_clustering": {
            "method": "sorted_greedy_distance_from_current_group_mean",
            "tolerance": {
                "height_ratio": _OCR_MATRIX_Y_TOLERANCE_HEIGHT_RATIO,
                "minimum_pixels": _OCR_MATRIX_MIN_Y_TOLERANCE,
            },
        },
        "zones": list(_OCR_MATRIX_ZONES),
    },
    "line_assignment": {
        "column": "bbox_overlap_at_least_five_percent_of_observation_width",
        "row": "bbox_vertical_center",
        "shared_refs": "one_ocr_line_may_ground_multiple_horizontal_cells",
    },
    "marker_column_enrichment": {
        "left_padding_ratio_of_median_marker_width": _MARKER_LEFT_PADDING_RATIO,
        "marker_pattern": _MARKER_PATTERN_TEXT,
        "max_gap_page_width_ratio": _MARKER_MAX_GAP_RATIO,
        "max_marker_page_width_ratio": _MARKER_MAX_WIDTH_RATIO,
        "minimum_distinct_rows": _MARKER_MIN_COUNT,
        "synthetic_marker_cells_never_merge_vertically": True,
    },
    "merge_policy": (
        "union_adjacent_atomic_cells_only_across_absent_local_rule_and_emit_merge_only_"
        "for_complete_rectangular_components_otherwise_keep_component_atomic"
    ),
    "normalization_decimal_places": _NORMALIZED_DECIMALS,
    "observation_kinds": [ObservationKind.TEXT_LINE.value],
    "table_grid_observation_id": "{page_id}:table-grid:{policy_sha256_prefix_12}:{index_4d}",
}
TABLE_TOPOLOGY_POLICY_BYTES = _canonical_json_bytes(_DETECTOR_POLICY)
TABLE_TOPOLOGY_POLICY_SHA256 = _sha256(TABLE_TOPOLOGY_POLICY_BYTES)


class ModelATableTopologyError(ValueError):
    """Raised when a topology cannot bind to one exact EvidenceIR page image."""


class _StrictModel(StrictContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class _NonEligibleBoundary(_StrictModel):
    research_only: Literal[True] = True
    human_review_required: Literal[True] = True
    verification_status: Literal["unverified"] = "unverified"
    rights_status: Literal["unverified"] = "unverified"
    identity_assurance: Literal["self_asserted_untrusted"] = "self_asserted_untrusted"
    golden_eligible: Literal[False] = False
    training_eligible: Literal[False] = False
    release_eligible: Literal[False] = False


class ModelATableTopologyDetectorArtifact(_StrictModel):
    detector_id: Literal["scan2hwpx.model_a.table_topology"] = "scan2hwpx.model_a.table_topology"
    detector_version: Literal["model-a-table-topology-detector/1.3"] = (
        TABLE_TOPOLOGY_DETECTOR_VERSION
    )
    detector_policy_sha256: Sha256 = TABLE_TOPOLOGY_POLICY_SHA256

    @model_validator(mode="after")
    def validate_deployed_policy(self) -> Self:
        if self.detector_policy_sha256 != TABLE_TOPOLOGY_POLICY_SHA256:
            raise ValueError("table topology detector digest is not the deployed policy")
        return self


class ModelATableTopologySourceBinding(_StrictModel):
    evidence_ir_id: BoundedText
    evidence_ir_contract_sha256: Sha256
    source_document_sha256: Sha256
    target_page_id: BoundedText
    target_page_no: int = Field(ge=1)
    page_image_source_ref: BoundedText
    page_image_sha256: Sha256
    page_width_pixels: int = Field(ge=1)
    page_height_pixels: int = Field(ge=1)


class ModelATableTopologyCell(_StrictModel):
    row: int = Field(ge=0, lt=_MAX_TABLE_DIMENSION)
    column: int = Field(ge=0, lt=_MAX_TABLE_DIMENSION)
    row_span: int = Field(ge=1, le=_MAX_TABLE_DIMENSION)
    column_span: int = Field(ge=1, le=_MAX_TABLE_DIMENSION)
    bbox_normalized: NormalizedBBox
    ordered_observation_ids: tuple[BoundedText, ...] = Field(max_length=_MAX_REFS_PER_CELL)
    text: str = Field(max_length=_MAX_CELL_TEXT_CHARS)

    @model_validator(mode="after")
    def validate_cell(self) -> Self:
        _require_normalized_bbox(self.bbox_normalized, "cell")
        if len(self.ordered_observation_ids) != len(set(self.ordered_observation_ids)):
            raise ValueError("table cell observation ids must be unique")
        if not self.ordered_observation_ids and self.text:
            raise ValueError("empty table cell references require empty text")
        if self.ordered_observation_ids and not self.text:
            raise ValueError("referenced table cell requires OCR-derived text")
        return self


class ModelATableTopologyTable(_StrictModel):
    table_grid_observation_id: BoundedText
    rows: int = Field(ge=1, le=_MAX_TABLE_DIMENSION)
    columns: int = Field(ge=1, le=_MAX_TABLE_DIMENSION)
    bbox_normalized: NormalizedBBox
    ordered_observation_ids: tuple[BoundedText, ...] = Field(max_length=_MAX_TABLE_CELLS)
    cells: tuple[ModelATableTopologyCell, ...] = Field(
        min_length=1,
        max_length=_MAX_TABLE_CELLS,
    )

    @model_validator(mode="after")
    def validate_complete_topology(self) -> Self:
        _require_normalized_bbox(self.bbox_normalized, "table")
        if self.rows * self.columns > _MAX_TABLE_CELLS:
            raise ValueError("table atomic grid area exceeds limit")
        addresses = tuple((cell.row, cell.column) for cell in self.cells)
        if addresses != tuple(sorted(addresses)):
            raise ValueError("table cells must be in canonical row-major order")
        occupied: set[tuple[int, int]] = set()
        referenced_observation_ids: list[str] = []
        tx0, ty0, tx1, ty1 = self.bbox_normalized
        for cell in self.cells:
            if cell.row + cell.row_span > self.rows:
                raise ValueError("table cell row span escapes the grid")
            if cell.column + cell.column_span > self.columns:
                raise ValueError("table cell column span escapes the grid")
            cx0, cy0, cx1, cy1 = cell.bbox_normalized
            if cx0 < tx0 or cy0 < ty0 or cx1 > tx1 or cy1 > ty1:
                raise ValueError("table cell bbox escapes table bbox")
            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(cell.column, cell.column + cell.column_span):
                    address = (row, column)
                    if address in occupied:
                        raise ValueError("table cells overlap")
                    occupied.add(address)
            referenced_observation_ids.extend(cell.ordered_observation_ids)
        if len(occupied) != self.rows * self.columns:
            raise ValueError("table cells do not completely cover the atomic grid")
        if len(self.ordered_observation_ids) != len(set(self.ordered_observation_ids)):
            raise ValueError("table observation union must be unique")
        if set(self.ordered_observation_ids) != set(referenced_observation_ids):
            raise ValueError("table observation union must match cell references")
        return self


class ModelATableTopologyArtifact(_NonEligibleBoundary):
    schema_version: Literal["model-a-table-topology/1.0"] = TABLE_TOPOLOGY_ARTIFACT_VERSION
    artifact_role: Literal["deterministic_page_table_topology"] = (
        "deterministic_page_table_topology"
    )
    source: ModelATableTopologySourceBinding
    detector: ModelATableTopologyDetectorArtifact
    tables: tuple[ModelATableTopologyTable, ...] = Field(max_length=_MAX_TABLES)

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        table_order = tuple(
            (table.bbox_normalized[1], table.bbox_normalized[0]) for table in self.tables
        )
        if table_order != tuple(sorted(table_order)):
            raise ValueError("tables must be in canonical page order")
        grid_ids = tuple(table.table_grid_observation_id for table in self.tables)
        if len(grid_ids) != len(set(grid_ids)):
            raise ValueError("table grid observation ids must be unique")
        observation_ids = [
            observation_id
            for table in self.tables
            for observation_id in table.ordered_observation_ids
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("an observation cannot appear in multiple detected tables")
        return self


@dataclass(frozen=True, slots=True)
class BuiltModelATableTopology:
    artifact: ModelATableTopologyArtifact
    artifact_bytes: bytes
    artifact_sha256: str
    evidence_ir: EvidenceIR
    page_image: ModelAPageImage

    def __post_init__(self) -> None:
        if type(self.artifact) is not ModelATableTopologyArtifact:
            raise TypeError("artifact must be ModelATableTopologyArtifact")
        if type(self.evidence_ir) is not EvidenceIR:
            raise TypeError("evidence_ir must be EvidenceIR")
        if type(self.page_image) is not ModelAPageImage:
            raise TypeError("page_image must be ModelAPageImage")
        expected = _build_artifact(self.evidence_ir, self.page_image)
        if self.artifact != expected:
            raise ValueError("table topology cannot be reproduced from bound inputs")
        expected_bytes = _canonical_json_bytes(expected.model_dump(mode="json"))
        if self.artifact_bytes != expected_bytes:
            raise ValueError("table topology bytes are not canonical artifact bytes")
        if _sha256(self.artifact_bytes) != self.artifact_sha256:
            raise ValueError("table topology artifact digest mismatch")


@dataclass(frozen=True, slots=True)
class _BoundPage:
    page: EvidencePage
    rgb: np.ndarray[Any, Any]
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _WorkingGrid:
    x_lines: tuple[float, ...]
    y_lines: tuple[float, ...]
    marker_column_enriched: bool
    atomic_cells: bool = False
    allowed_observation_ids: frozenset[str] | None = None

    @property
    def rows(self) -> int:
        return len(self.y_lines) - 1

    @property
    def columns(self) -> int:
        return len(self.x_lines) - 1


@dataclass(frozen=True, slots=True)
class _CellRect:
    row: int
    column: int
    row_span: int
    column_span: int


@dataclass(frozen=True, slots=True)
class _OcrMatrixCandidate:
    grid: TableGrid
    observation_ids: frozenset[str]
    occupied_slots: frozenset[tuple[int, int]]
    zone_index: int
    occupancy: float


def build_model_a_table_topology(
    evidence_ir: EvidenceIR,
    *,
    page_image: ModelAPageImage,
) -> BuiltModelATableTopology:
    """Detect and bind deterministic table topology for exactly one page."""

    try:
        artifact = _build_artifact(evidence_ir, page_image)
    except (TypeError, ValueError, ValidationError, UnidentifiedImageError) as exc:
        raise ModelATableTopologyError(
            "table topology input is not a valid exact page binding"
        ) from exc
    artifact_bytes = _canonical_json_bytes(artifact.model_dump(mode="json"))
    return BuiltModelATableTopology(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        artifact_sha256=_sha256(artifact_bytes),
        evidence_ir=evidence_ir,
        page_image=page_image,
    )


def validate_model_a_table_topology(
    artifact: ModelATableTopologyArtifact,
    *,
    evidence_ir: EvidenceIR,
    page_image: ModelAPageImage,
) -> BuiltModelATableTopology:
    """Replay the detector and bind a loaded artifact to exact source bytes."""

    if type(artifact) is not ModelATableTopologyArtifact:
        raise TypeError("artifact must be ModelATableTopologyArtifact")
    artifact_bytes = _canonical_json_bytes(artifact.model_dump(mode="json"))
    return BuiltModelATableTopology(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        artifact_sha256=_sha256(artifact_bytes),
        evidence_ir=evidence_ir,
        page_image=page_image,
    )


def _build_artifact(
    evidence_ir: EvidenceIR,
    page_image: ModelAPageImage,
) -> ModelATableTopologyArtifact:
    bound = _bind_page(evidence_ir, page_image)
    horizontal, vertical = ruled_line_masks_array(bound.rgb)
    ruled_detected = tuple(
        filtered
        for grid in detect_table_grids_array(bound.rgb)
        if (
            filtered := _filter_internal_grid_lines(
                grid,
                horizontal=horizontal,
                vertical=vertical,
            )
        )
        is not None
    )
    observations = tuple(
        observation
        for observation in bound.page.observations
        if observation.kind == ObservationKind.TEXT_LINE
    )
    ruled_grids_unfiltered = tuple(
        _enrich_marker_column(grid, observations, bound.width) for grid in ruled_detected
    )
    hashed_crop_source_ids = frozenset(
        source.id
        for source in evidence_ir.sources
        if source.kind == EvidenceSourceKind.CROP and source.sha256 is not None
    )
    crop_images = tuple(
        observation
        for observation in bound.page.observations
        if observation.kind == ObservationKind.IMAGE
        and any(source_ref in hashed_crop_source_ids for source_ref in observation.source_refs)
    )
    ruled_grids = tuple(
        grid
        for grid in ruled_grids_unfiltered
        if not _empty_grid_is_contained_by_image(
            grid,
            text_observations=observations,
            image_observations=crop_images,
        )
    )
    ocr_grids = tuple(
        _WorkingGrid(
            x_lines=candidate.grid.x_lines,
            y_lines=candidate.grid.y_lines,
            marker_column_enriched=False,
            atomic_cells=True,
            allowed_observation_ids=candidate.observation_ids,
        )
        for candidate in _select_ocr_matrix_candidates(
            _detect_ocr_matrix_candidates(
                observations,
                width=bound.width,
                height=bound.height,
            ),
            ruled_grids=ruled_grids,
            observations=observations,
        )
    )
    grids = tuple(
        sorted(
            (*ruled_grids, *ocr_grids),
            key=lambda grid: (grid.y_lines[0], grid.x_lines[0]),
        )
    )
    tables = tuple(
        _build_table(
            grid,
            table_index=table_index,
            page_id=bound.page.id,
            observations=observations,
            horizontal=horizontal,
            vertical=vertical,
            width=bound.width,
            height=bound.height,
        )
        for table_index, grid in enumerate(grids)
    )
    existing_observation_ids = {observation.id for observation in bound.page.observations}
    if any(table.table_grid_observation_id in existing_observation_ids for table in tables):
        raise ValueError("promoted table grid observation id collides with EvidenceIR")
    return ModelATableTopologyArtifact(
        source=ModelATableTopologySourceBinding(
            evidence_ir_id=evidence_ir.id,
            evidence_ir_contract_sha256=contract_sha256(evidence_ir),
            source_document_sha256=evidence_ir.source_document_sha256,
            target_page_id=bound.page.id,
            target_page_no=bound.page.page_no,
            page_image_source_ref=bound.page.image_source_ref,
            page_image_sha256=page_image.sha256,
            page_width_pixels=bound.width,
            page_height_pixels=bound.height,
        ),
        detector=ModelATableTopologyDetectorArtifact(),
        tables=tables,
    )


def _bind_page(evidence_ir: EvidenceIR, page_image: ModelAPageImage) -> _BoundPage:
    if type(evidence_ir) is not EvidenceIR:
        raise TypeError("evidence_ir must be EvidenceIR")
    if type(page_image) is not ModelAPageImage:
        raise TypeError("page_image must be ModelAPageImage")
    _require_strict_evidence(evidence_ir)
    ModelAPageImage.__post_init__(page_image)
    matches = tuple(page for page in evidence_ir.pages if page.id == page_image.page_id)
    if len(matches) != 1:
        raise ValueError("page image page_id must identify exactly one EvidenceIR page")
    page = matches[0]
    if page.page_no != page_image.page_no or page.image_source_ref != page_image.image_source_ref:
        raise ValueError("page image identity does not match target EvidencePage")
    sources = {source.id: source for source in evidence_ir.sources}
    image_source = sources[page.image_source_ref]
    if image_source.kind != EvidenceSourceKind.PAGE_IMAGE:
        raise ValueError("target page image source must have kind page_image")
    if image_source.sha256 is None or image_source.sha256 != page_image.sha256:
        raise ValueError("page image bytes do not match EvidenceSource.sha256")
    if not any(
        source.kind == EvidenceSourceKind.ORIGINAL_DOCUMENT
        and source.sha256 == evidence_ir.source_document_sha256
        for source in evidence_ir.sources
    ):
        raise ValueError("EvidenceIR source document digest has no bound original source")
    with Image.open(BytesIO(page_image.raw_bytes)) as opened:
        transposed = ImageOps.exif_transpose(opened)
        width, height = transposed.size
        if width * height > _MAX_IMAGE_PIXELS:
            raise ValueError("page image pixel count exceeds topology limit")
        transposed.load()
        rgb = np.asarray(transposed.convert("RGB"), dtype=np.uint8)
    if page.width != float(width) or page.height != float(height):
        raise ValueError("decoded image dimensions do not match EvidencePage")
    if rgb.shape != (height, width, 3):
        raise ValueError("decoded page image must be an exact three-channel RGB array")
    return _BoundPage(page=page, rgb=rgb, width=width, height=height)


def _require_strict_evidence(evidence_ir: EvidenceIR) -> None:
    raw = _canonical_json_bytes(evidence_ir.model_dump(mode="json"))
    restored = EvidenceIR.model_validate_json(raw, strict=True)
    if restored != evidence_ir:
        raise ValueError("EvidenceIR is not canonical")


def _filter_internal_grid_lines(
    grid: TableGrid,
    *,
    horizontal: np.ndarray[Any, Any],
    vertical: np.ndarray[Any, Any],
) -> TableGrid | None:
    x_lines = _canonical_lines(grid.x_lines, "x")
    y_lines = _canonical_lines(grid.y_lines, "y")
    filtered_x = (
        x_lines[0],
        *(
            x
            for x in x_lines[1:-1]
            if sum(
                _mask_window_has_ink(
                    vertical,
                    x=x,
                    y=y,
                    x_radius=_VERTICAL_INTERSECTION_X_RADIUS,
                    y_radius=_VERTICAL_INTERSECTION_Y_RADIUS,
                )
                for y in y_lines
            )
            >= _INTERNAL_LINE_MIN_INTERSECTIONS
        ),
        x_lines[-1],
    )
    filtered_y = (
        y_lines[0],
        *(
            y
            for y in y_lines[1:-1]
            if sum(
                _mask_window_has_ink(
                    horizontal,
                    x=x,
                    y=y,
                    x_radius=_HORIZONTAL_INTERSECTION_X_RADIUS,
                    y_radius=_HORIZONTAL_INTERSECTION_Y_RADIUS,
                )
                for x in x_lines
            )
            >= _INTERNAL_LINE_MIN_INTERSECTIONS
        ),
        y_lines[-1],
    )
    if len(filtered_x) < 2 or len(filtered_y) < 3:
        return None
    return TableGrid(bbox=grid.bbox, x_lines=filtered_x, y_lines=filtered_y)


def _mask_window_has_ink(
    mask: np.ndarray[Any, Any],
    *,
    x: float,
    y: float,
    x_radius: int,
    y_radius: int,
) -> bool:
    height, width = mask.shape
    center_x = round(x)
    center_y = round(y)
    x0 = max(0, center_x - x_radius)
    x1 = min(width, center_x + x_radius + 1)
    y0 = max(0, center_y - y_radius)
    y1 = min(height, center_y + y_radius + 1)
    return bool(np.any(mask[y0:y1, x0:x1]))


def _detect_ocr_matrix_candidates(
    observations: tuple[EvidenceObservation, ...],
    *,
    width: int,
    height: int,
) -> tuple[_OcrMatrixCandidate, ...]:
    valid = tuple(
        observation
        for observation in observations
        if _valid_page_observation(observation, width=width, height=height)
    )
    candidates: list[_OcrMatrixCandidate] = []
    for zone_index, zone in enumerate(_OCR_MATRIX_ZONES):
        zoned = tuple(
            observation
            for observation in valid
            if _observation_in_zone(observation, zone=zone, width=width)
        )
        if len(zoned) < _OCR_MATRIX_MIN_ROWS * _OCR_MATRIX_MIN_COLUMNS:
            continue
        typical_height = median(
            observation.bbox.pixel[3] - observation.bbox.pixel[1]
            for observation in zoned
        )
        y_tolerance = max(
            _OCR_MATRIX_MIN_Y_TOLERANCE,
            typical_height * _OCR_MATRIX_Y_TOLERANCE_HEIGHT_RATIO,
        )
        rows = tuple(
            row
            for row in _cluster_observations(zoned, axis="y", tolerance=y_tolerance)
            if len(row) >= 2
        )
        for run in _regular_row_runs(rows, typical_height=typical_height):
            members = tuple(observation for row in run for observation in row)
            box_widths = tuple(
                observation.bbox.pixel[2] - observation.bbox.pixel[0]
                for observation in members
            )
            if median(box_widths) > width * _OCR_MATRIX_MAX_MEDIAN_BOX_WIDTH_RATIO:
                continue
            x_centers = tuple(_observation_center(observation, axis="x") for observation in members)
            if max(x_centers) - min(x_centers) < width * _OCR_MATRIX_MIN_X_SPAN_RATIO:
                continue
            columns = _cluster_observations(
                members,
                axis="x",
                tolerance=max(
                    _OCR_MATRIX_MIN_X_TOLERANCE,
                    width * _OCR_MATRIX_X_TOLERANCE_PAGE_WIDTH_RATIO,
                ),
            )
            if not _OCR_MATRIX_MIN_COLUMNS <= len(columns) <= _OCR_MATRIX_MAX_COLUMNS:
                continue
            column_centers = tuple(
                _cluster_center(column, axis="x") for column in columns
            )
            column_gaps = tuple(right - left for left, right in pairwise(column_centers))
            if max(column_gaps) / min(column_gaps) > _OCR_MATRIX_MAX_ADJACENT_X_GAP_RATIO:
                continue
            row_centers = tuple(_cluster_center(row, axis="y") for row in run)
            row_by_id = {
                observation.id: _nearest_center_index(
                    _observation_center(observation, axis="y"), row_centers
                )
                for observation in members
            }
            column_by_id = {
                observation.id: _nearest_center_index(
                    _observation_center(observation, axis="x"), column_centers
                )
                for observation in members
            }
            occupied = {
                (row_by_id[observation.id], column_by_id[observation.id])
                for observation in members
            }
            occupancy = len(occupied) / (len(run) * len(columns))
            if occupancy < _OCR_MATRIX_MIN_OCCUPANCY:
                continue
            x_lines = _midpoint_boundaries(
                tuple(_cluster_center(column, axis="x") for column in columns),
                limit=width,
            )
            y_lines = _midpoint_boundaries(
                row_centers,
                limit=height,
            )
            candidates.append(
                _OcrMatrixCandidate(
                    grid=TableGrid(
                        bbox=(x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]),
                        x_lines=x_lines,
                        y_lines=y_lines,
                    ),
                    observation_ids=frozenset(observation.id for observation in members),
                    occupied_slots=frozenset(occupied),
                    zone_index=zone_index,
                    occupancy=occupancy,
                )
            )
    return tuple(candidates)


def _valid_page_observation(
    observation: EvidenceObservation,
    *,
    width: int,
    height: int,
) -> bool:
    x0, y0, x1, y1 = observation.bbox.pixel
    return all(math.isfinite(value) for value in (x0, y0, x1, y1)) and (
        0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
    )


def _observation_in_zone(
    observation: EvidenceObservation,
    *,
    zone: str,
    width: int,
) -> bool:
    center_x = _observation_center(observation, axis="x")
    if zone == "left_half":
        return center_x < width / 2
    if zone == "right_half":
        return center_x >= width / 2
    return zone == "full_page"


def _cluster_observations(
    observations: tuple[EvidenceObservation, ...],
    *,
    axis: Literal["x", "y"],
    tolerance: float,
) -> tuple[tuple[EvidenceObservation, ...], ...]:
    groups: list[list[EvidenceObservation]] = []
    for observation in sorted(
        observations,
        key=lambda item: (_observation_center(item, axis=axis), item.id),
    ):
        value = _observation_center(observation, axis=axis)
        if groups and value - _cluster_center(groups[-1], axis=axis) <= tolerance:
            groups[-1].append(observation)
        else:
            groups.append([observation])
    return tuple(tuple(group) for group in groups)


def _regular_row_runs(
    rows: tuple[tuple[EvidenceObservation, ...], ...],
    *,
    typical_height: float,
) -> tuple[tuple[tuple[EvidenceObservation, ...], ...], ...]:
    runs: list[tuple[tuple[EvidenceObservation, ...], ...]] = []
    start = 0
    while start < len(rows):
        end = start + 1
        gaps: list[float] = []
        while end < len(rows):
            gap = _cluster_center(rows[end], axis="y") - _cluster_center(
                rows[end - 1], axis="y"
            )
            trial = [*gaps, gap]
            middle = median(trial)
            if (
                gap < typical_height * _OCR_MATRIX_MIN_ROW_GAP_HEIGHT_RATIO
                or gap > typical_height * _OCR_MATRIX_MAX_ROW_GAP_HEIGHT_RATIO
                or min(trial) < middle * _OCR_MATRIX_MIN_GAP_MEDIAN_RATIO
                or max(trial) > middle * _OCR_MATRIX_MAX_GAP_MEDIAN_RATIO
            ):
                break
            gaps = trial
            end += 1
        if end - start >= _OCR_MATRIX_MIN_ROWS:
            runs.append(rows[start:end])
        start = max(start + 1, end)
    return tuple(runs)


def _observation_center(
    observation: EvidenceObservation,
    *,
    axis: Literal["x", "y"],
) -> float:
    x0, y0, x1, y1 = observation.bbox.pixel
    return (x0 + x1) / 2 if axis == "x" else (y0 + y1) / 2


def _cluster_center(
    observations: list[EvidenceObservation] | tuple[EvidenceObservation, ...],
    *,
    axis: Literal["x", "y"],
) -> float:
    return sum(_observation_center(item, axis=axis) for item in observations) / len(
        observations
    )


def _midpoint_boundaries(centers: tuple[float, ...], *, limit: int) -> tuple[float, ...]:
    inner = tuple((left + right) / 2 for left, right in pairwise(centers))
    return (
        max(0.0, centers[0] - (centers[1] - centers[0]) / 2),
        *inner,
        min(float(limit), centers[-1] + (centers[-1] - centers[-2]) / 2),
    )


def _nearest_center_index(value: float, centers: tuple[float, ...]) -> int:
    return min(range(len(centers)), key=lambda index: (abs(value - centers[index]), index))


def _select_ocr_matrix_candidates(
    candidates: tuple[_OcrMatrixCandidate, ...],
    *,
    ruled_grids: tuple[_WorkingGrid, ...],
    observations: tuple[EvidenceObservation, ...],
) -> tuple[_OcrMatrixCandidate, ...]:
    ruled_members = tuple(
        frozenset(
            observation.id
            for observation in observations
            if _observation_assigns_to_grid(observation, grid)
        )
        for grid in ruled_grids
    )
    selected: list[_OcrMatrixCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -len(item.occupied_slots),
            -item.occupancy,
            item.grid.bbox[1],
            item.grid.bbox[0],
            item.zone_index,
            tuple(sorted(item.observation_ids)),
        ),
    ):
        if any(
            _bbox_iou(candidate.grid.bbox, _working_grid_bbox(grid))
            >= _OCR_MATRIX_RULED_GRID_MAX_IOU
            or bool(candidate.observation_ids & member_ids)
            for grid, member_ids in zip(ruled_grids, ruled_members, strict=True)
        ):
            continue
        if any(
            candidate.observation_ids & existing.observation_ids
            or _bbox_iou(candidate.grid.bbox, existing.grid.bbox)
            >= _OCR_MATRIX_CANDIDATE_DEDUPE_IOU
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(selected)


def _observation_assigns_to_grid(
    observation: EvidenceObservation,
    grid: _WorkingGrid,
) -> bool:
    x0, y0, x1, y1 = observation.bbox.pixel
    if _line_interval(grid.y_lines, (y0 + y1) / 2) is None:
        return False
    width = x1 - x0
    return any(
        max(0.0, min(x1, right) - max(x0, left)) / width >= _MIN_CELL_OVERLAP_RATIO
        for left, right in pairwise(grid.x_lines)
    )


def _working_grid_bbox(grid: _WorkingGrid) -> tuple[float, float, float, float]:
    return (grid.x_lines[0], grid.y_lines[0], grid.x_lines[-1], grid.y_lines[-1])


def _empty_grid_is_contained_by_image(
    grid: _WorkingGrid,
    *,
    text_observations: tuple[EvidenceObservation, ...],
    image_observations: tuple[EvidenceObservation, ...],
) -> bool:
    if any(_observation_assigns_to_grid(observation, grid) for observation in text_observations):
        return False
    grid_bbox = _working_grid_bbox(grid)
    return any(
        _bbox_coverage(grid_bbox, observation.bbox.pixel)
        >= _IMAGE_CONTAINMENT_MIN_GRID_COVERAGE
        for observation in image_observations
    )


def _bbox_coverage(
    subject: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> float:
    x0 = max(subject[0], container[0])
    y0 = max(subject[1], container[1])
    x1 = min(subject[2], container[2])
    y1 = min(subject[3], container[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    subject_area = max(0.0, subject[2] - subject[0]) * max(
        0.0, subject[3] - subject[1]
    )
    return intersection / max(1.0, subject_area)


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def _enrich_marker_column(
    grid: TableGrid,
    observations: tuple[EvidenceObservation, ...],
    page_width: int,
) -> _WorkingGrid:
    x_lines = _canonical_lines(grid.x_lines, "x")
    y_lines = _canonical_lines(grid.y_lines, "y")
    left = x_lines[0]
    max_gap = page_width * _MARKER_MAX_GAP_RATIO
    candidates: list[tuple[EvidenceObservation, int]] = []
    for observation in observations:
        text = _preferred_ocr_text(observation).strip()
        x0, y0, x1, y1 = observation.bbox.pixel
        marker_width = x1 - x0
        center_y = (y0 + y1) / 2
        gap = left - x1
        if not _MARKER_PATTERN.fullmatch(text):
            continue
        if marker_width > page_width * _MARKER_MAX_WIDTH_RATIO or not 0 <= gap <= max_gap:
            continue
        row = _line_interval(y_lines, center_y)
        if row is not None:
            candidates.append((observation, row))
    rows = {row for _observation, row in candidates}
    if len(candidates) < _MARKER_MIN_COUNT or len(rows) != len(candidates):
        return _WorkingGrid(x_lines=x_lines, y_lines=y_lines, marker_column_enriched=False)
    marker_widths = [
        observation.bbox.pixel[2] - observation.bbox.pixel[0] for observation, _row in candidates
    ]
    marker_centers = [
        (observation.bbox.pixel[0] + observation.bbox.pixel[2]) / 2
        for observation, _row in candidates
    ]
    if max(marker_centers) - min(marker_centers) > max(marker_widths):
        return _WorkingGrid(x_lines=x_lines, y_lines=y_lines, marker_column_enriched=False)
    inferred_left = min(observation.bbox.pixel[0] for observation, _row in candidates) - max(
        2.0,
        median(marker_widths) * _MARKER_LEFT_PADDING_RATIO,
    )
    inferred_left = max(0.0, inferred_left)
    if inferred_left >= left:
        return _WorkingGrid(x_lines=x_lines, y_lines=y_lines, marker_column_enriched=False)
    return _WorkingGrid(
        x_lines=(inferred_left, *x_lines),
        y_lines=y_lines,
        marker_column_enriched=True,
    )


def _canonical_lines(values: tuple[float, ...], axis: str) -> tuple[float, ...]:
    lines = tuple(sorted(float(value) for value in values))
    if len(lines) < 2 or any(not math.isfinite(value) for value in lines):
        raise ValueError(f"detected {axis} grid lines are invalid")
    if any(second - first < 1.0 for first, second in pairwise(lines)):
        raise ValueError(f"detected {axis} grid lines are not distinct")
    return lines


def _build_table(
    grid: _WorkingGrid,
    *,
    table_index: int,
    page_id: str,
    observations: tuple[EvidenceObservation, ...],
    horizontal: np.ndarray[Any, Any],
    vertical: np.ndarray[Any, Any],
    width: int,
    height: int,
) -> ModelATableTopologyTable:
    if grid.rows * grid.columns > _MAX_TABLE_CELLS:
        raise ValueError("detected table grid area exceeds limit")
    cell_rects = (
        tuple(
            _CellRect(row=row, column=column, row_span=1, column_span=1)
            for row in range(grid.rows)
            for column in range(grid.columns)
        )
        if grid.atomic_cells
        else _infer_cell_rects(grid, horizontal=horizontal, vertical=vertical)
    )
    observation_order = {observation.id: index for index, observation in enumerate(observations)}
    assigned: dict[tuple[int, int], list[EvidenceObservation]] = defaultdict(list)
    for observation in observations:
        if (
            grid.allowed_observation_ids is not None
            and observation.id not in grid.allowed_observation_ids
        ):
            continue
        x0, y0, x1, y1 = observation.bbox.pixel
        row = _line_interval(grid.y_lines, (y0 + y1) / 2)
        if row is None:
            continue
        observation_width = x1 - x0
        for column in range(grid.columns):
            overlap = max(
                0.0,
                min(x1, grid.x_lines[column + 1]) - max(x0, grid.x_lines[column]),
            )
            if overlap / observation_width >= _MIN_CELL_OVERLAP_RATIO:
                assigned[(row, column)].append(observation)
    observations_by_cell: list[list[EvidenceObservation]] = []
    for rect in cell_rects:
        by_id = {
            observation.id: observation
            for row in range(rect.row, rect.row + rect.row_span)
            for column in range(rect.column, rect.column + rect.column_span)
            for observation in assigned.get((row, column), ())
        }
        observations_by_cell.append(
            sorted(
                by_id.values(),
                key=lambda observation: (
                    observation.bbox.pixel[1],
                    observation.bbox.pixel[0],
                    observation_order[observation.id],
                ),
            )
        )
    cell_indexes_by_observation: dict[str, list[int]] = defaultdict(list)
    for cell_index, cell_observations in enumerate(observations_by_cell):
        for observation in cell_observations:
            cell_indexes_by_observation[observation.id].append(cell_index)
    assigned_ids = set(cell_indexes_by_observation)
    cells: list[ModelATableTopologyCell] = []
    for cell_index, (rect, cell_observations) in enumerate(
        zip(cell_rects, observations_by_cell, strict=True)
    ):
        cells.append(
            ModelATableTopologyCell(
                row=rect.row,
                column=rect.column,
                row_span=rect.row_span,
                column_span=rect.column_span,
                bbox_normalized=_normalized_bbox(
                    grid.x_lines[rect.column],
                    grid.y_lines[rect.row],
                    grid.x_lines[rect.column + rect.column_span],
                    grid.y_lines[rect.row + rect.row_span],
                    width,
                    height,
                ),
                ordered_observation_ids=tuple(observation.id for observation in cell_observations),
                text="\n".join(
                    _cell_ocr_fragment(
                        observation,
                        cell_index=cell_index,
                        cell_indexes_by_observation=cell_indexes_by_observation,
                    )
                    for observation in cell_observations
                ),
            )
        )
    ordered_observation_ids = tuple(
        observation.id for observation in observations if observation.id in assigned_ids
    )
    return ModelATableTopologyTable(
        table_grid_observation_id=(
            f"{page_id}:table-grid:{TABLE_TOPOLOGY_POLICY_SHA256[:12]}:{table_index:04d}"
        ),
        rows=grid.rows,
        columns=grid.columns,
        bbox_normalized=_normalized_bbox(
            grid.x_lines[0],
            grid.y_lines[0],
            grid.x_lines[-1],
            grid.y_lines[-1],
            width,
            height,
        ),
        ordered_observation_ids=ordered_observation_ids,
        cells=tuple(cells),
    )


def _infer_cell_rects(
    grid: _WorkingGrid,
    *,
    horizontal: np.ndarray[Any, Any],
    vertical: np.ndarray[Any, Any],
) -> tuple[_CellRect, ...]:
    parent = list(range(grid.rows * grid.columns))

    def address(row: int, column: int) -> int:
        return row * grid.columns + column

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    vertical_present: dict[tuple[int, int], bool] = {}
    for row in range(grid.rows):
        for column in range(1, grid.columns):
            present = _vertical_boundary_present(
                vertical,
                x=grid.x_lines[column],
                y0=grid.y_lines[row],
                y1=grid.y_lines[row + 1],
            )
            vertical_present[(row, column)] = present
            if not present:
                union(address(row, column - 1), address(row, column))

    horizontal_present: dict[tuple[int, int], bool] = {}
    for row in range(1, grid.rows):
        for column in range(grid.columns):
            present = grid.marker_column_enriched and column == 0
            if not present:
                present = _horizontal_boundary_present(
                    horizontal,
                    y=grid.y_lines[row],
                    x0=grid.x_lines[column],
                    x1=grid.x_lines[column + 1],
                )
            horizontal_present[(row, column)] = present
            if not present:
                union(address(row - 1, column), address(row, column))

    components: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in range(grid.rows):
        for column in range(grid.columns):
            components[find(address(row, column))].add((row, column))

    rects: list[_CellRect] = []
    for component in sorted(components.values(), key=min):
        component_rows = [row for row, _column in component]
        component_columns = [column for _row, column in component]
        row0, row1 = min(component_rows), max(component_rows) + 1
        column0, column1 = min(component_columns), max(component_columns) + 1
        expected = {
            (row, column) for row in range(row0, row1) for column in range(column0, column1)
        }
        rectangular = component == expected and _rectangle_has_no_internal_boundaries(
            row0,
            row1,
            column0,
            column1,
            vertical_present=vertical_present,
            horizontal_present=horizontal_present,
        )
        if rectangular:
            rects.append(
                _CellRect(
                    row=row0,
                    column=column0,
                    row_span=row1 - row0,
                    column_span=column1 - column0,
                )
            )
        else:
            rects.extend(
                _CellRect(row=row, column=column, row_span=1, column_span=1)
                for row, column in sorted(component)
            )
    return tuple(sorted(rects, key=lambda cell: (cell.row, cell.column)))


def _rectangle_has_no_internal_boundaries(
    row0: int,
    row1: int,
    column0: int,
    column1: int,
    *,
    vertical_present: dict[tuple[int, int], bool],
    horizontal_present: dict[tuple[int, int], bool],
) -> bool:
    return not any(
        vertical_present[(row, column)]
        for row in range(row0, row1)
        for column in range(column0 + 1, column1)
    ) and not any(
        horizontal_present[(row, column)]
        for row in range(row0 + 1, row1)
        for column in range(column0, column1)
    )


def _vertical_boundary_present(
    mask: np.ndarray[Any, Any],
    *,
    x: float,
    y0: float,
    y1: float,
) -> bool:
    top, bottom = _inset_interval(y0, y1, mask.shape[0])
    center = round(x)
    left = max(0, center - _BOUNDARY_BAND_RADIUS)
    right = min(mask.shape[1], center + _BOUNDARY_BAND_RADIUS + 1)
    band = mask[top:bottom, left:right]
    if not band.size:
        return False
    coverage = float(np.mean(np.max(band, axis=1) > 0))
    return coverage >= _BOUNDARY_COVERAGE


def _horizontal_boundary_present(
    mask: np.ndarray[Any, Any],
    *,
    y: float,
    x0: float,
    x1: float,
) -> bool:
    left, right = _inset_interval(x0, x1, mask.shape[1])
    center = round(y)
    top = max(0, center - _BOUNDARY_BAND_RADIUS)
    bottom = min(mask.shape[0], center + _BOUNDARY_BAND_RADIUS + 1)
    band = mask[top:bottom, left:right]
    if not band.size:
        return False
    coverage = float(np.mean(np.max(band, axis=0) > 0))
    return coverage >= _BOUNDARY_COVERAGE


def _inset_interval(start: float, end: float, limit: int) -> tuple[int, int]:
    inset = min(_BOUNDARY_INSET, max(0, int((end - start) // 4)))
    lower = max(0, round(start) + inset)
    upper = min(limit, round(end) - inset)
    if upper <= lower:
        lower = max(0, round(start))
        upper = min(limit, round(end))
    return lower, upper


def _line_interval(lines: tuple[float, ...], value: float) -> int | None:
    if value < lines[0] or value > lines[-1]:
        return None
    index = bisect_right(lines, value) - 1
    if index == len(lines) - 1:
        index -= 1
    return index if 0 <= index < len(lines) - 1 else None


def _preferred_ocr_text(observation: EvidenceObservation) -> str:
    selected = next(
        (candidate for candidate in observation.ocr_candidates if candidate.selected),
        None,
    )
    if selected is not None:
        return selected.text
    return max(observation.ocr_candidates, key=lambda candidate: candidate.confidence).text


def _cell_ocr_fragment(
    observation: EvidenceObservation,
    *,
    cell_index: int,
    cell_indexes_by_observation: dict[str, list[int]],
) -> str:
    text = _preferred_ocr_text(observation)
    cell_indexes = cell_indexes_by_observation[observation.id]
    tokens = text.split()
    if len(cell_indexes) > 1 and len(tokens) == len(cell_indexes):
        return tokens[cell_indexes.index(cell_index)]
    return text


def _normalized_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> NormalizedBBox:
    return (
        round(max(0.0, min(1.0, x0 / width)), _NORMALIZED_DECIMALS),
        round(max(0.0, min(1.0, y0 / height)), _NORMALIZED_DECIMALS),
        round(max(0.0, min(1.0, x1 / width)), _NORMALIZED_DECIMALS),
        round(max(0.0, min(1.0, y1 / height)), _NORMALIZED_DECIMALS),
    )


def _require_normalized_bbox(bbox: NormalizedBBox, label: str) -> None:
    x0, y0, x1, y1 = bbox
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in bbox):
        raise ValueError(f"{label} bbox must contain finite normalized coordinates")
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"{label} bbox coordinates must be ordered")
