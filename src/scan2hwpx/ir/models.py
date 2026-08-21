from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlockKind(StrEnum):
    TITLE = "title"
    INSTRUCTION = "instruction"
    PASSAGE = "passage"
    QUESTION = "question"
    CHOICE = "choice"
    BOX = "box"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    UNKNOWN = "unknown"


class FormulaFormat(StrEnum):
    LATEX = "latex"
    MATHML = "mathml"


class FormulaStatus(StrEnum):
    RECOGNIZED = "recognized"
    NEEDS_REVIEW = "needs_review"
    IMAGE_FALLBACK = "image_fallback"


class AnnotationState(StrEnum):
    PRINTED = "printed"
    ANNOTATION = "annotation"
    UNCERTAIN = "uncertain_annotation"


class BBox(StrictModel):
    pixel: tuple[float, float, float, float]
    normalized: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_coordinates(self) -> BBox:
        if self.pixel[0] > self.pixel[2] or self.pixel[1] > self.pixel[3]:
            raise ValueError("pixel bbox must be ordered x0,y0,x1,y1")
        if not all(0 <= value <= 1 for value in self.normalized):
            raise ValueError("normalized bbox values must be between 0 and 1")
        return self


class Span(StrictModel):
    text: str
    bbox: BBox | None = None
    confidence: float = Field(ge=0, le=1)


class Block(StrictModel):
    id: str
    kind: BlockKind
    bbox: BBox
    reading_order: int = Field(ge=0)
    text: str
    spans: list[Span] = Field(default_factory=list)
    style: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    source_provider: str
    source_payload_ref: str
    annotation_state: AnnotationState = AnnotationState.PRINTED


class Column(StrictModel):
    index: int = Field(ge=0)
    bbox: BBox


class PageQuality(StrictModel):
    score: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class FormulaValidation(StrictModel):
    syntax_valid: bool = False
    render_similarity: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class Formula(StrictModel):
    id: str
    bbox: BBox
    source_image: str
    expression: str = ""
    format: FormulaFormat = FormulaFormat.LATEX
    confidence: float = Field(ge=0, le=1)
    source_provider: str
    status: FormulaStatus = FormulaStatus.NEEDS_REVIEW
    validation: FormulaValidation = Field(default_factory=FormulaValidation)


class Page(StrictModel):
    page_no: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    rotation: int = 0
    columns: list[Column] = Field(default_factory=list)
    blocks: list[Block]
    formulas: list[Formula] = Field(default_factory=list)
    quality: PageQuality


class Choice(StrictModel):
    label: str
    blocks: list[str]
    confidence: float = Field(ge=0, le=1)


class Question(StrictModel):
    number: int = Field(ge=1)
    score: float | None = Field(default=None, ge=0)
    instruction_blocks: list[str] = Field(default_factory=list)
    passage_blocks: list[str] = Field(default_factory=list)
    prompt_blocks: list[str] = Field(default_factory=list)
    choices: list[Choice] = Field(default_factory=list)
    material_blocks: list[str] = Field(default_factory=list)
    source_blocks: list[str] = Field(default_factory=list)
    page_refs: list[int]
    confidence: float = Field(ge=0, le=1)


class Asset(StrictModel):
    id: str
    media_type: str
    path: str


class ReprocessRecord(StrictModel):
    block_id: str
    action: str
    reason: str


class QAReport(StrictModel):
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    low_confidence_blocks: list[str] = Field(default_factory=list)
    auto_reprocess_history: list[ReprocessRecord] = Field(default_factory=list)


class Document(StrictModel):
    schema_version: str = "0.2.0"
    id: str
    source_hash: str
    page_size: tuple[float, float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    pages: list[Page]
    questions: list[Question] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    qa: QAReport = Field(default_factory=QAReport)
