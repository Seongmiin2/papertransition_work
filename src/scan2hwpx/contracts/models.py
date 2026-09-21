from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

EVIDENCE_IR_VERSION: Literal["evidence-ir/1.0"] = "evidence-ir/1.0"
CONTENT_IR_VERSION: Literal["content-ir/1.0"] = "content-ir/1.0"
HWP_DOCUMENT_PLAN_VERSION: Literal["hwp-document-plan/1.0"] = "hwp-document-plan/1.0"
QUALITY_REPORT_VERSION: Literal["quality-report/1.0"] = "quality-report/1.0"
_MAX_TABLE_DIMENSION = 10_000
_MAX_TABLE_CELLS = 100_000
_MAX_TABLE_GRID_AREA = 100_000

REQUIRED_DOCUMENT_QUALITY_CHECKS_V1 = frozenset(
    {
        "schema-reference-integrity",
        "content-conservation",
        "reading-order",
        "table-topology",
        "formula-grounding",
        "visual-layout",
        "hwpx-package-valid",
        "dvc-validation",
        "editable-objects",
        "hancom-round-trip",
    }
)

NonEmpty = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Confidence = Annotated[float, Field(ge=0, le=1)]
LogicalArtifactRef = Annotated[
    str,
    Field(
        min_length=1,
        pattern=(
            r"^(?:artifact|blob)://[A-Za-z0-9][A-Za-z0-9._-]*"
            r"(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
        ),
    ),
]


class StrictContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class BBox(StrictContractModel):
    pixel: tuple[float, float, float, float]
    normalized: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        x0, y0, x1, y1 = self.pixel
        if min(self.pixel) < 0 or x0 >= x1 or y0 >= y1:
            raise ValueError("pixel bbox must be non-negative and ordered x0,y0,x1,y1")
        nx0, ny0, nx1, ny1 = self.normalized
        if not all(0 <= value <= 1 for value in self.normalized):
            raise ValueError("normalized bbox values must be between 0 and 1")
        if nx0 >= nx1 or ny0 >= ny1:
            raise ValueError("normalized bbox must be ordered x0,y0,x1,y1")
        return self


class EvidenceSourceKind(StrEnum):
    ORIGINAL_DOCUMENT = "original_document"
    PAGE_IMAGE = "page_image"
    CROP = "crop"
    OCR_PROVIDER = "ocr_provider"
    LAYOUT_PROVIDER = "layout_provider"
    FORMULA_PROVIDER = "formula_provider"


class EvidenceSource(StrictContractModel):
    id: NonEmpty
    kind: EvidenceSourceKind
    artifact_ref: LogicalArtifactRef
    producer: NonEmpty
    sha256: Sha256 | None = None


class OcrCandidate(StrictContractModel):
    text: NonEmpty
    provider: NonEmpty
    confidence: Confidence
    quality: Confidence | None = None
    source_ref: NonEmpty
    selected: bool = False


class ObservationKind(StrEnum):
    TEXT_LINE = "text_line"
    REGION = "region"
    TABLE_GRID = "table_grid"
    IMAGE = "image"
    FORMULA = "formula"


class EvidenceObservation(StrictContractModel):
    id: NonEmpty
    kind: ObservationKind
    bbox: BBox
    confidence: Confidence
    source_refs: tuple[NonEmpty, ...] = Field(min_length=1)
    ocr_candidates: tuple[OcrCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_local_references(self) -> Self:
        _reject_duplicates(self.source_refs, "observation source refs")
        if sum(candidate.selected for candidate in self.ocr_candidates) > 1:
            raise ValueError("observation cannot have more than one selected OCR candidate")
        if self.kind == ObservationKind.TEXT_LINE and not self.ocr_candidates:
            raise ValueError("text_line observation requires at least one OCR candidate")
        unknown = sorted(
            {candidate.source_ref for candidate in self.ocr_candidates} - set(self.source_refs)
        )
        if unknown:
            raise ValueError("OCR candidates reference unlisted sources: " + ", ".join(unknown))
        return self


class EvidencePage(StrictContractModel):
    id: NonEmpty
    page_no: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    rotation: int = Field(default=0)
    image_source_ref: NonEmpty
    observations: tuple[EvidenceObservation, ...] = ()

    @model_validator(mode="after")
    def validate_observation_ids(self) -> Self:
        _reject_duplicates(
            (observation.id for observation in self.observations),
            f"observation ids on page {self.page_no}",
        )
        return self


class EvidenceIR(StrictContractModel):
    schema_version: Literal["evidence-ir/1.0"] = EVIDENCE_IR_VERSION
    id: NonEmpty
    source_document_sha256: Sha256
    sources: tuple[EvidenceSource, ...] = Field(min_length=1)
    pages: tuple[EvidencePage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        _reject_duplicates((source.id for source in self.sources), "evidence source ids")
        _reject_duplicates((page.id for page in self.pages), "evidence page ids")
        _reject_duplicates((str(page.page_no) for page in self.pages), "evidence page numbers")
        observations = [
            observation for page in self.pages for observation in page.observations
        ]
        _reject_duplicates(
            (observation.id for observation in observations),
            "evidence observation ids",
        )

        source_ids = {source.id for source in self.sources}
        referenced_sources = {
            page.image_source_ref for page in self.pages
        } | {
            source_ref
            for observation in observations
            for source_ref in observation.source_refs
        }
        unknown = sorted(referenced_sources - source_ids)
        if unknown:
            raise ValueError("unknown evidence source refs: " + ", ".join(unknown))
        return self


class ContentRole(StrEnum):
    TITLE = "title"
    INSTRUCTION = "instruction"
    PASSAGE = "passage"
    QUESTION = "question"
    CHOICE = "choice"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class ContentNodeBase(StrictContractModel):
    id: NonEmpty
    evidence_refs: tuple[NonEmpty, ...] = Field(min_length=1)
    confidence: Confidence
    needs_review: bool = False

    @model_validator(mode="after")
    def validate_evidence_refs(self) -> Self:
        _reject_duplicates(self.evidence_refs, f"evidence refs for content node {self.id}")
        return self


class TextContentNode(ContentNodeBase):
    kind: Literal["text"] = "text"
    role: ContentRole
    text: NonEmpty


def _bounded_table_integer(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(  # noqa: TRY004 - Pydantic does not wrap TypeError.
            f"{label} must be an integer without coercion"
        )
    if value < minimum or value > maximum:
        raise ValueError(f"{label} exceeds the table resource limit")
    return value


class TableCell(StrictContractModel):
    row: int = Field(ge=0, le=_MAX_TABLE_DIMENSION - 1, strict=True)
    column: int = Field(ge=0, le=_MAX_TABLE_DIMENSION - 1, strict=True)
    row_span: int = Field(default=1, ge=1, le=_MAX_TABLE_DIMENSION, strict=True)
    column_span: int = Field(default=1, ge=1, le=_MAX_TABLE_DIMENSION, strict=True)
    text: str = ""
    evidence_refs: tuple[NonEmpty, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_refs(self) -> Self:
        _reject_duplicates(self.evidence_refs, "table cell evidence refs")
        return self


class TableContentNode(ContentNodeBase):
    kind: Literal["table"] = "table"
    role: Literal[ContentRole.TABLE] = ContentRole.TABLE
    rows: int = Field(ge=1, le=_MAX_TABLE_DIMENSION, strict=True)
    columns: int = Field(ge=1, le=_MAX_TABLE_DIMENSION, strict=True)
    cells: tuple[TableCell, ...] = Field(
        min_length=1,
        max_length=_MAX_TABLE_CELLS,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_topology_resource_bounds(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if not isinstance(value, Mapping):
            return value
        if "rows" not in value or "columns" not in value:
            return value
        rows = _bounded_table_integer(
            value["rows"],
            "table rows",
            minimum=1,
            maximum=_MAX_TABLE_DIMENSION,
        )
        columns = _bounded_table_integer(
            value["columns"],
            "table columns",
            minimum=1,
            maximum=_MAX_TABLE_DIMENSION,
        )
        if rows * columns > _MAX_TABLE_GRID_AREA:
            raise ValueError("table grid area exceeds the table resource limit")

        cells = value.get("cells")
        if not isinstance(cells, (list, tuple)):
            return value
        if len(cells) > _MAX_TABLE_CELLS:
            raise ValueError("table cell count exceeds the table resource limit")
        for cell in cells:
            if isinstance(cell, TableCell):
                row_value: object = cell.row
                column_value: object = cell.column
                row_span_value: object = cell.row_span
                column_span_value: object = cell.column_span
            elif isinstance(cell, Mapping):
                if "row" not in cell or "column" not in cell:
                    continue
                row_value = cell["row"]
                column_value = cell["column"]
                row_span_value = cell.get("row_span", 1)
                column_span_value = cell.get("column_span", 1)
            else:
                continue
            row = _bounded_table_integer(
                row_value,
                "table cell row",
                minimum=0,
                maximum=_MAX_TABLE_DIMENSION - 1,
            )
            column = _bounded_table_integer(
                column_value,
                "table cell column",
                minimum=0,
                maximum=_MAX_TABLE_DIMENSION - 1,
            )
            row_span = _bounded_table_integer(
                row_span_value,
                "table cell row_span",
                minimum=1,
                maximum=_MAX_TABLE_DIMENSION,
            )
            column_span = _bounded_table_integer(
                column_span_value,
                "table cell column_span",
                minimum=1,
                maximum=_MAX_TABLE_DIMENSION,
            )
            if row + row_span > rows or column + column_span > columns:
                raise ValueError("table cell span exceeds table bounds")
        if info.mode == "json":
            normalized = dict(value)
            evidence_refs = normalized.get("evidence_refs")
            if isinstance(evidence_refs, list):
                normalized["evidence_refs"] = tuple(evidence_refs)
            normalized_cells: list[object] = []
            for cell in cells:
                if isinstance(cell, Mapping):
                    normalized_cell = dict(cell)
                    cell_refs = normalized_cell.get("evidence_refs")
                    if isinstance(cell_refs, list):
                        normalized_cell["evidence_refs"] = tuple(cell_refs)
                    normalized_cells.append(normalized_cell)
                else:
                    normalized_cells.append(cell)
            normalized["cells"] = tuple(normalized_cells)
            return normalized
        return value

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if cell.row + cell.row_span > self.rows:
                raise ValueError("table cell row span exceeds table bounds")
            if cell.column + cell.column_span > self.columns:
                raise ValueError("table cell column span exceeds table bounds")
            positions = {
                (row, column)
                for row in range(cell.row, cell.row + cell.row_span)
                for column in range(cell.column, cell.column + cell.column_span)
            }
            if occupied & positions:
                raise ValueError("table cells overlap")
            occupied.update(positions)
        expected = {(row, column) for row in range(self.rows) for column in range(self.columns)}
        if occupied != expected:
            raise ValueError("table cells must cover the complete table grid")
        return self


class ImageContentNode(ContentNodeBase):
    kind: Literal["image"] = "image"
    role: Literal[ContentRole.IMAGE] = ContentRole.IMAGE
    asset_ref: NonEmpty


class FormulaFormat(StrEnum):
    LATEX = "latex"
    MATHML = "mathml"


class FormulaContentNode(ContentNodeBase):
    kind: Literal["formula"] = "formula"
    role: Literal[ContentRole.FORMULA] = ContentRole.FORMULA
    expression: NonEmpty
    format: FormulaFormat


ContentNode = Annotated[
    TextContentNode | TableContentNode | ImageContentNode | FormulaContentNode,
    Field(discriminator="kind"),
]


class ContentIR(StrictContractModel):
    schema_version: Literal["content-ir/1.0"] = CONTENT_IR_VERSION
    id: NonEmpty
    evidence_ir_id: NonEmpty
    evidence_ir_sha256: Sha256
    revision: int = Field(default=1, ge=1)
    nodes: tuple[ContentNode, ...] = Field(min_length=1)
    reading_order: tuple[NonEmpty, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_node_order(self) -> Self:
        node_ids = tuple(node.id for node in self.nodes)
        _reject_duplicates(node_ids, "content node ids")
        _reject_duplicates(self.reading_order, "content reading-order refs")
        if set(self.reading_order) != set(node_ids):
            raise ValueError("reading_order must reference every content node exactly once")
        return self

    def assert_evidence_integrity(self, evidence: EvidenceIR) -> None:
        if self.evidence_ir_id != evidence.id:
            raise ValueError("ContentIR does not belong to the supplied EvidenceIR")
        if self.evidence_ir_sha256 != contract_sha256(evidence):
            raise ValueError("ContentIR EvidenceIR digest mismatch")
        observation_ids = {
            observation.id for page in evidence.pages for observation in page.observations
        }
        source_ids = {source.id for source in evidence.sources}
        refs: set[str] = set()
        for node in self.nodes:
            refs.update(node.evidence_refs)
            if isinstance(node, TableContentNode):
                refs.update(ref for cell in node.cells for ref in cell.evidence_refs)
            if isinstance(node, ImageContentNode) and node.asset_ref not in source_ids:
                raise ValueError(f"unknown image asset ref: {node.asset_ref}")
        unknown = sorted(refs - observation_ids)
        if unknown:
            raise ValueError("unknown evidence observation refs: " + ", ".join(unknown))


class TextAlignment(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class StyleIntent(StrictContractModel):
    id: NonEmpty
    semantic_role: NonEmpty
    font_family: NonEmpty | None = None
    font_size_pt: float | None = Field(default=None, gt=0, le=200)
    bold: bool = False
    italic: bool = False
    alignment: TextAlignment = TextAlignment.LEFT
    line_spacing: float | None = Field(default=None, ge=0.5, le=5)


class PageLayoutIntent(StrictContractModel):
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    margin_top_mm: float = Field(ge=0)
    margin_right_mm: float = Field(ge=0)
    margin_bottom_mm: float = Field(ge=0)
    margin_left_mm: float = Field(ge=0)
    columns: int = Field(default=1, ge=1, le=4)
    column_gap_mm: float = Field(default=0, ge=0)


class LayoutIntent(StrictContractModel):
    width_fraction: float | None = Field(default=None, gt=0, le=1)
    column_span: int = Field(default=1, ge=1, le=4)
    keep_with_next: bool = False


class ContentPlanItem(StrictContractModel):
    id: NonEmpty
    kind: Literal["content"] = "content"
    render_as: Literal["paragraph", "table", "image", "formula"]
    content_ref: NonEmpty
    style_ref: NonEmpty | None = None
    layout: LayoutIntent = Field(default_factory=LayoutIntent)


class PageBreakPlanItem(StrictContractModel):
    id: NonEmpty
    kind: Literal["page_break"] = "page_break"


class ColumnBreakPlanItem(StrictContractModel):
    id: NonEmpty
    kind: Literal["column_break"] = "column_break"


PlanFlowItem = Annotated[
    ContentPlanItem | PageBreakPlanItem | ColumnBreakPlanItem,
    Field(discriminator="kind"),
]


class HwpDocumentPlan(StrictContractModel):
    schema_version: Literal["hwp-document-plan/1.0"] = HWP_DOCUMENT_PLAN_VERSION
    id: NonEmpty
    content_ir_id: NonEmpty
    content_ir_revision: int = Field(ge=1)
    content_ir_sha256: Sha256
    capability_profile_id: NonEmpty
    design_profile_id: NonEmpty
    official_spec_refs: tuple[NonEmpty, ...] = Field(min_length=1)
    page_layout: PageLayoutIntent
    styles: tuple[StyleIntent, ...] = ()
    flow: tuple[PlanFlowItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_local_references(self) -> Self:
        _reject_duplicates(self.official_spec_refs, "official spec refs")
        _reject_duplicates((style.id for style in self.styles), "plan style ids")
        _reject_duplicates((item.id for item in self.flow), "plan flow item ids")
        content_items = [item for item in self.flow if isinstance(item, ContentPlanItem)]
        _reject_duplicates(
            (item.content_ref for item in content_items),
            "plan content refs",
        )
        style_ids = {style.id for style in self.styles}
        unknown_styles = sorted(
            {
                item.style_ref
                for item in content_items
                if item.style_ref is not None and item.style_ref not in style_ids
            }
        )
        if unknown_styles:
            raise ValueError("unknown plan style refs: " + ", ".join(unknown_styles))
        return self

    def assert_content_integrity(self, content: ContentIR) -> None:
        if self.content_ir_id != content.id:
            raise ValueError("HwpDocumentPlan does not belong to the supplied ContentIR")
        if self.content_ir_revision != content.revision:
            raise ValueError("HwpDocumentPlan ContentIR revision mismatch")
        if self.content_ir_sha256 != contract_sha256(content):
            raise ValueError("HwpDocumentPlan ContentIR digest mismatch")
        content_by_id = {node.id: node for node in content.nodes}
        planned = [item for item in self.flow if isinstance(item, ContentPlanItem)]
        planned_refs = {item.content_ref for item in planned}
        unknown = sorted(planned_refs - set(content_by_id))
        missing = sorted(set(content_by_id) - planned_refs)
        if unknown:
            raise ValueError("unknown content refs: " + ", ".join(unknown))
        if missing:
            raise ValueError("unplanned content refs: " + ", ".join(missing))

        expected_renderers = {
            TextContentNode: "paragraph",
            TableContentNode: "table",
            ImageContentNode: "image",
            FormulaContentNode: "formula",
        }
        mismatches = sorted(
            item.content_ref
            for item in planned
            if item.content_ref in content_by_id
            and item.render_as != expected_renderers[type(content_by_id[item.content_ref])]
        )
        if mismatches:
            raise ValueError("content kind/render_as mismatch: " + ", ".join(mismatches))


class CheckDecision(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    BLOCK = "block"


class QualityOutcome(StrEnum):
    AUTO_PASS = "auto_pass"
    REVIEW = "review"
    BLOCK = "block"


class QualityCheck(StrictContractModel):
    id: NonEmpty
    decision: CheckDecision
    message: NonEmpty
    score: float | None = None
    threshold: float | None = None
    comparator: Literal["<=", ">=", "=="] | None = None
    evidence_refs: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def validate_check(self) -> Self:
        _reject_duplicates(self.evidence_refs, f"quality evidence refs for {self.id}")
        numeric_values_present = self.score is not None or self.threshold is not None
        if numeric_values_present and (self.score is None or self.threshold is None):
            raise ValueError("score and threshold must be supplied together")
        if numeric_values_present != (self.comparator is not None):
            raise ValueError("numeric quality checks require a comparator")
        if (
            self.decision == CheckDecision.PASS
            and self.comparator is not None
            and not _comparison_passes(self.score, self.threshold, self.comparator)
        ):
            raise ValueError("pass decision contradicts score, comparator, and threshold")
        return self


class QualityReport(StrictContractModel):
    schema_version: Literal["quality-report/1.0"] = QUALITY_REPORT_VERSION
    id: NonEmpty
    document_id: NonEmpty
    evidence_ir_id: NonEmpty
    evidence_ir_sha256: Sha256
    content_ir_id: NonEmpty
    content_ir_revision: int = Field(ge=1)
    content_ir_sha256: Sha256
    hwp_document_plan_id: NonEmpty
    hwp_document_plan_sha256: Sha256
    compiled_artifact_ref: LogicalArtifactRef
    compiled_artifact_sha256: Sha256
    compiler_version: NonEmpty
    model_bundle_id: NonEmpty
    check_set_id: Literal["document-quality-v1"]
    outcome: QualityOutcome
    checks: tuple[QualityCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        _reject_duplicates((check.id for check in self.checks), "quality check ids")
        decisions = {check.decision for check in self.checks}
        expected = (
            QualityOutcome.BLOCK
            if CheckDecision.BLOCK in decisions
            else QualityOutcome.REVIEW
            if CheckDecision.REVIEW in decisions
            else QualityOutcome.AUTO_PASS
        )
        if self.outcome != expected:
            raise ValueError(f"quality outcome must be {expected.value}")
        if self.outcome == QualityOutcome.AUTO_PASS:
            check_ids = {check.id for check in self.checks}
            missing = sorted(REQUIRED_DOCUMENT_QUALITY_CHECKS_V1 - check_ids)
            if missing:
                raise ValueError(
                    "auto-pass report is missing required checks: " + ", ".join(missing)
                )
        return self

    def assert_contract_lineage(
        self,
        evidence: EvidenceIR,
        content: ContentIR,
        plan: HwpDocumentPlan,
    ) -> None:
        content.assert_evidence_integrity(evidence)
        plan.assert_content_integrity(content)
        expected = {
            "evidence_ir_id": evidence.id,
            "evidence_ir_sha256": contract_sha256(evidence),
            "content_ir_id": content.id,
            "content_ir_revision": content.revision,
            "content_ir_sha256": contract_sha256(content),
            "hwp_document_plan_id": plan.id,
            "hwp_document_plan_sha256": contract_sha256(plan),
        }
        mismatches = sorted(
            field for field, value in expected.items() if getattr(self, field) != value
        )
        if mismatches:
            raise ValueError("quality report lineage mismatch: " + ", ".join(mismatches))


def contract_sha256(model: StrictContractModel) -> str:
    """Return a stable digest for binding one immutable contract to the next."""
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _comparison_passes(
    score: float | None,
    threshold: float | None,
    comparator: Literal["<=", ">=", "=="],
) -> bool:
    if score is None or threshold is None:
        return False
    if comparator == "<=":
        return score <= threshold
    if comparator == ">=":
        return score >= threshold
    return score == threshold


def _reject_duplicates(values: Iterable[str], label: str) -> None:
    materialized: tuple[str, ...] = tuple(values)
    duplicates = sorted(value for value, count in Counter(materialized).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: " + ", ".join(duplicates))
