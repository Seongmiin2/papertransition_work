from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self, TypeAlias, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from scan2hwpx.contracts import (
    ContentIR,
    ContentPlanItem,
    ContentRole,
    EvidenceIR,
    EvidenceSourceKind,
    FormulaContentNode,
    HwpDocumentPlan,
    ImageContentNode,
    ObservationKind,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.hwpx.assets import (
    MAX_IMAGE_ASSET_BYTES,
    MAX_IMAGE_ASSET_TOTAL_BYTES,
    MAX_IMAGE_ASSETS,
    ImageAssetBundle,
    ImageAssetBundleError,
    PngImageAsset,
    build_image_asset_bundle,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str,
    Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
IssueCode = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]*$")]
Split = Literal["train", "validation", "test"]
ContractT = TypeVar("ContractT", EvidenceIR, ContentIR, HwpDocumentPlan)
ContentNodeT: TypeAlias = (
    TextContentNode | TableContentNode | ImageContentNode | FormulaContentNode
)
TextContentRole = Literal[
    ContentRole.TITLE,
    ContentRole.INSTRUCTION,
    ContentRole.PASSAGE,
    ContentRole.QUESTION,
    ContentRole.CHOICE,
    ContentRole.CAPTION,
    ContentRole.HEADER,
    ContentRole.FOOTER,
    ContentRole.OTHER,
]
_MAX_VIEW_NODES = 5_000
_MAX_VIEW_CELLS = 100_000
_MAX_VIEW_ISSUES = 1_000
_MAX_PATCH_OPERATIONS = 1_000
# Electron/JavaScript measures string length in UTF-16 code units. These
# boundary limits deliberately use that same metric in Python.
_MAX_VIEW_FIELD_CHARS = 1_000_000
_MAX_VIEW_TOTAL_CHARS = 10_000_000
_MAX_DRAFT_REVISION = 1_000_000
_MAX_JS_SAFE_INTEGER = (1 << 53) - 1
_MAX_PATCH_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_CANDIDATE_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_REVIEW_DRAFT_BYTES = 64 * 1024 * 1024
_MAX_CONTRACT_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_METADATA_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_BINARY_ARTIFACT_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES_PER_DOCUMENT = 2 * 1024 * 1024 * 1024
_MAX_DOCUMENTS = 10_000
_MAX_PAGES_PER_DOCUMENT = 10_000
_MAX_IMAGE_CROPS_PER_DOCUMENT = 100_000
_PAGE_IMAGE_NAME = re.compile(r"^page-(\d{4,})\.png$")


def _utf16_code_units(value: str) -> int:
    """Return the value JavaScript exposes as ``string.length``."""
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("review strings cannot contain unpaired UTF-16 surrogates")
    return len(value.encode("utf-16-le")) // 2


def _bound_view_id_utf16(value: str) -> str:
    if _utf16_code_units(value) > 1_024:
        raise ValueError("review view id exceeds 1024 UTF-16 code units")
    return value


def _bound_view_text_utf16(value: str) -> str:
    if _utf16_code_units(value) > _MAX_VIEW_FIELD_CHARS:
        raise ValueError("review view text exceeds the UTF-16 field size limit")
    return value


_ViewId = Annotated[
    str,
    Field(min_length=1, max_length=1_024),
    AfterValidator(_bound_view_id_utf16),
]
_ViewText = Annotated[
    str,
    Field(max_length=_MAX_VIEW_FIELD_CHARS),
    AfterValidator(_bound_view_text_utf16),
]
_ViewNonEmptyText = Annotated[
    str,
    Field(min_length=1, max_length=_MAX_VIEW_FIELD_CHARS),
    AfterValidator(_bound_view_text_utf16),
]
_ViewIssueCode = Annotated[
    str,
    Field(min_length=1, max_length=256, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
_ViewDocumentId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
_SafeNonNegativeInteger = Annotated[
    int,
    Field(ge=0, le=_MAX_JS_SAFE_INTEGER),
]


class ReviewDraftVerificationError(ValueError):
    """Raised when a review draft is not bound to an intact candidate bundle."""


class ReviewDraftPatchError(ValueError):
    """Raised when a typed review patch cannot be applied safely."""


class ReviewDraftCompletionError(ValueError):
    """Raised when a review draft cannot be completed safely."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class _NonEligibleBoundary(_StrictModel):
    golden_eligible: Literal[False] = False
    training_eligible: Literal[False] = False
    release_eligible: Literal[False] = False
    rights_status: Literal["unverified"] = "unverified"
    identity_assurance: Literal["self_asserted_untrusted"] = "self_asserted_untrusted"


class CandidateArtifactBinding(_StrictModel):
    path: str = Field(min_length=1)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        parts = value.split("/")
        if (
            "\\" in value
            or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or any(
                not part[0].isalnum()
                or any(not (character.isalnum() or character in "._-") for character in part)
                for part in parts
            )
        ):
            raise ValueError("candidate artifact path must be a safe relative path")
        return value


class CandidateContractArtifactBinding(CandidateArtifactBinding):
    contract_sha256: Sha256


class IssueDisposition(_StrictModel):
    issue_code: IssueCode
    disposition: Literal["pending", "resolved", "accepted_limitation"]
    note: str = ""

    @model_validator(mode="after")
    def require_accepted_limitation_reason(self) -> Self:
        if self.disposition == "accepted_limitation" and not self.note.strip():
            raise ValueError("accepted limitation requires a note")
        return self


class SetTextPatchOperation(_StrictModel):
    op: Literal["set_text"]
    node_id: _ViewId
    text: _ViewNonEmptyText


class SetRolePatchOperation(_StrictModel):
    op: Literal["set_role"]
    node_id: _ViewId
    role: TextContentRole


class SetTableCellTextPatchOperation(_StrictModel):
    op: Literal["set_table_cell_text"]
    node_id: _ViewId
    row: int = Field(ge=0, le=_MAX_JS_SAFE_INTEGER)
    column: int = Field(ge=0, le=_MAX_JS_SAFE_INTEGER)
    row_span: int = Field(ge=1, le=_MAX_JS_SAFE_INTEGER)
    column_span: int = Field(ge=1, le=_MAX_JS_SAFE_INTEGER)
    text: _ViewText


class SetNeedsReviewPatchOperation(_StrictModel):
    op: Literal["set_needs_review"]
    node_id: _ViewId
    needs_review: bool


class SetImageGroundingPatchOperation(_StrictModel):
    op: Literal["set_image_grounding"]
    node_id: _ViewId
    asset_ref: _ViewId
    observation_ref: _ViewId


class DropImageNodePatchOperation(_StrictModel):
    op: Literal["drop_image_node"]
    node_id: _ViewId


class SetIssueDispositionPatchOperation(_StrictModel):
    op: Literal["set_issue_disposition"]
    issue_code: _ViewIssueCode
    disposition: Literal["pending", "resolved", "accepted_limitation"]
    note: _ViewText = ""

    @model_validator(mode="after")
    def require_accepted_limitation_reason(self) -> Self:
        if self.disposition == "accepted_limitation" and not self.note.strip():
            raise ValueError("accepted limitation requires a note")
        return self


CandidateReviewPatchOperation: TypeAlias = Annotated[
    SetTextPatchOperation
    | SetRolePatchOperation
    | SetTableCellTextPatchOperation
    | SetNeedsReviewPatchOperation
    | SetImageGroundingPatchOperation
    | DropImageNodePatchOperation
    | SetIssueDispositionPatchOperation,
    Field(discriminator="op"),
]


class CandidateReviewPatchRequest(_StrictModel):
    """Strict renderer-safe patch request with no arbitrary document mutation path."""

    schema_version: Literal["candidate-review-patch/1.1"] = "candidate-review-patch/1.1"
    expected_draft_revision: int = Field(ge=1, le=_MAX_DRAFT_REVISION)
    operations: tuple[CandidateReviewPatchOperation, ...] = Field(
        min_length=1,
        max_length=_MAX_PATCH_OPERATIONS,
    )

    @model_validator(mode="after")
    def bound_total_text(self) -> Self:
        total_chars = 0
        for operation in self.operations:
            total_chars += _utf16_code_units(getattr(operation, "node_id", ""))
            total_chars += _utf16_code_units(getattr(operation, "asset_ref", ""))
            total_chars += _utf16_code_units(getattr(operation, "observation_ref", ""))
            total_chars += _utf16_code_units(getattr(operation, "issue_code", ""))
            total_chars += _utf16_code_units(getattr(operation, "text", ""))
            total_chars += _utf16_code_units(getattr(operation, "note", ""))
            if total_chars > _MAX_VIEW_TOTAL_CHARS:
                raise ValueError("review patch text exceeds total size limit")
        if len(self.model_dump_json().encode("utf-8")) > _MAX_PATCH_REQUEST_BYTES:
            raise ValueError("review patch request exceeds serialized size limit")
        return self


class CandidateReviewCompletionRequest(_StrictModel):
    """Strict request to complete one exact review draft revision."""

    schema_version: Literal["candidate-review-completion/1.0"] = (
        "candidate-review-completion/1.0"
    )
    expected_draft_revision: int = Field(ge=1, le=_MAX_DRAFT_REVISION)


class CandidateReviewDraft(_NonEligibleBoundary):
    """Untrusted, non-promotable edits over one immutable candidate document."""

    schema_version: Literal["candidate-review-draft/1.0"] = "candidate-review-draft/1.0"
    status: Literal["in_progress", "complete"]
    draft_revision: int = Field(ge=1, le=_MAX_DRAFT_REVISION)
    reviewer_label: Identifier
    updated_at: str = Field(min_length=1)
    candidate_manifest_sha256: Sha256
    document_id: Identifier
    lineage_id: Sha256
    source_pdf_sha256: Sha256
    evidence_ir: CandidateContractArtifactBinding
    base_content_ir: CandidateContractArtifactBinding
    base_hwp_document_plan: CandidateContractArtifactBinding
    reviewed_content_ir: ContentIR
    reviewed_content_ir_contract_sha256: Sha256
    reviewed_hwp_document_plan: HwpDocumentPlan
    reviewed_hwp_document_plan_contract_sha256: Sha256
    issue_dispositions: tuple[IssueDisposition, ...] = ()

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("updated_at must be an ISO-8601 datetime") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("updated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_embedded_review(self) -> Self:
        disposition_codes = [item.issue_code for item in self.issue_dispositions]
        if len(set(disposition_codes)) != len(disposition_codes):
            raise ValueError("duplicate issue dispositions")
        if self.status == "complete" and any(
            item.disposition == "pending" for item in self.issue_dispositions
        ):
            raise ValueError("complete draft cannot contain pending issue dispositions")
        if contract_sha256(self.reviewed_content_ir) != self.reviewed_content_ir_contract_sha256:
            raise ValueError("reviewed ContentIR contract digest mismatch")
        if (
            contract_sha256(self.reviewed_hwp_document_plan)
            != self.reviewed_hwp_document_plan_contract_sha256
        ):
            raise ValueError("reviewed HwpDocumentPlan contract digest mismatch")
        try:
            self.reviewed_hwp_document_plan.assert_content_integrity(self.reviewed_content_ir)
        except ValueError as exc:
            raise ValueError(f"reviewed ContentIR/Plan lineage is invalid: {exc}") from exc
        return self


@dataclass(frozen=True, slots=True)
class VerifiedCandidateCompileInputs:
    """Compiler inputs materialized from one complete, verified review revision."""

    candidate_review: CandidateReviewDraft
    evidence_ir: EvidenceIR
    image_assets: ImageAssetBundle | None


class _ReviewTextNodeView(_StrictModel):
    id: _ViewId
    kind: Literal["text"] = "text"
    text: _ViewNonEmptyText
    role: ContentRole
    needs_review: bool


class _ReviewTableCellView(_StrictModel):
    row: int = Field(ge=0, le=_MAX_JS_SAFE_INTEGER)
    column: int = Field(ge=0, le=_MAX_JS_SAFE_INTEGER)
    row_span: int = Field(ge=1, le=_MAX_JS_SAFE_INTEGER)
    column_span: int = Field(ge=1, le=_MAX_JS_SAFE_INTEGER)
    text: _ViewText

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.row, self.column, self.row_span, self.column_span)


class _ReviewTableNodeView(_StrictModel):
    id: _ViewId
    kind: Literal["table"] = "table"
    needs_review: bool
    cells: tuple[_ReviewTableCellView, ...] = Field(
        min_length=1,
        max_length=_MAX_VIEW_CELLS,
    )

    @model_validator(mode="after")
    def reject_duplicate_cell_keys(self) -> Self:
        keys = [cell.key for cell in self.cells]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate review view table cell keys")
        return self


class _ReviewImageNodeView(_StrictModel):
    id: _ViewId
    kind: Literal["image"] = "image"
    needs_review: bool


class _ReviewFormulaNodeView(_StrictModel):
    id: _ViewId
    kind: Literal["formula"] = "formula"
    needs_review: bool


_ReviewNodeView: TypeAlias = Annotated[
    _ReviewTextNodeView | _ReviewTableNodeView | _ReviewImageNodeView | _ReviewFormulaNodeView,
    Field(discriminator="kind"),
]


class _ReviewIssueDispositionView(_StrictModel):
    issue_code: _ViewIssueCode
    disposition: Literal["pending", "resolved", "accepted_limitation"]
    note: _ViewText = ""

    @model_validator(mode="after")
    def require_accepted_limitation_reason(self) -> Self:
        if self.disposition == "accepted_limitation" and not self.note.strip():
            raise ValueError("accepted limitation requires a note")
        return self


class CandidateReviewDraftView(_NonEligibleBoundary):
    """Bounded Electron-safe projection of a verified review draft."""

    schema_version: Literal["candidate-review-view/1.0"] = "candidate-review-view/1.0"
    document_id: _ViewDocumentId
    status: Literal["in_progress", "complete"]
    draft_revision: int = Field(ge=1, le=_MAX_DRAFT_REVISION)
    updated_at: str = Field(min_length=1, max_length=64)
    reviewed_content_ir_contract_sha256: Sha256
    nodes: tuple[_ReviewNodeView, ...] = Field(
        min_length=1,
        max_length=_MAX_VIEW_NODES,
    )
    issue_dispositions: tuple[_ReviewIssueDispositionView, ...] = Field(
        max_length=_MAX_VIEW_ISSUES,
    )

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("updated_at must be an ISO-8601 datetime") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("updated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_view_inventory(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("duplicate review view node ids")
        issue_codes = [item.issue_code for item in self.issue_dispositions]
        if len(set(issue_codes)) != len(issue_codes):
            raise ValueError("duplicate review view issue dispositions")
        cell_count = sum(
            len(node.cells) for node in self.nodes if isinstance(node, _ReviewTableNodeView)
        )
        if cell_count > _MAX_VIEW_CELLS:
            raise ValueError("review view contains too many table cells")
        text_chars = sum(
            _utf16_code_units(node.text)
            if isinstance(node, _ReviewTextNodeView)
            else sum(_utf16_code_units(cell.text) for cell in node.cells)
            if isinstance(node, _ReviewTableNodeView)
            else 0
            for node in self.nodes
        ) + sum(_utf16_code_units(item.note) for item in self.issue_dispositions)
        if text_chars > _MAX_VIEW_TOTAL_CHARS:
            raise ValueError("review view contains too much text")
        return self


class _CandidateSourceArtifact(_StrictModel):
    sha256: Sha256


class _CandidateSourceArtifacts(_StrictModel):
    hwp: _CandidateSourceArtifact
    hwpx: _CandidateSourceArtifact
    pdf: _CandidateSourceArtifact


class _CandidateArtifacts(_StrictModel):
    projection_source: CandidateArtifactBinding
    evidence_ir: CandidateContractArtifactBinding
    content_ir_candidate: CandidateContractArtifactBinding
    hwp_document_plan_candidate: CandidateContractArtifactBinding
    report: CandidateArtifactBinding
    page_images: tuple[CandidateArtifactBinding, ...] = Field(
        max_length=_MAX_PAGES_PER_DOCUMENT
    )
    image_crops: tuple[CandidateArtifactBinding, ...] = Field(
        max_length=_MAX_IMAGE_CROPS_PER_DOCUMENT
    )

    def bindings(self) -> Iterator[CandidateArtifactBinding]:
        yield self.projection_source
        yield self.evidence_ir
        yield self.content_ir_candidate
        yield self.hwp_document_plan_candidate
        yield self.report
        yield from self.page_images
        yield from self.image_crops


class _CandidateDocument(_StrictModel):
    document_id: Identifier
    lineage_id: Sha256
    split: Split
    page_count: int = Field(gt=0, le=_MAX_PAGES_PER_DOCUMENT)
    status: Literal["needs_human_review"]
    source_artifacts: _CandidateSourceArtifacts
    artifacts: _CandidateArtifacts
    issue_codes: tuple[IssueCode, ...]

    @model_validator(mode="after")
    def validate_artifact_paths(self) -> Self:
        paths = [binding.path for binding in self.artifacts.bindings()]
        normalized = [path.casefold() for path in paths]
        if len(set(normalized)) != len(normalized):
            raise ValueError("candidate document contains duplicate artifact paths")
        outside = sorted(
            path
            for path in paths
            if not PurePosixPath(path).parts or PurePosixPath(path).parts[0] != self.lineage_id
        )
        if outside:
            raise ValueError("candidate artifact path is outside its lineage directory")
        if len(set(self.issue_codes)) != len(self.issue_codes):
            raise ValueError("candidate document contains duplicate issue codes")
        page_numbers: set[int] = set()
        for binding in self.artifacts.page_images:
            parts = PurePosixPath(binding.path).parts
            match = _PAGE_IMAGE_NAME.fullmatch(parts[-1]) if parts else None
            if len(parts) != 3 or parts[1] != "pages" or match is None:
                raise ValueError("candidate page image path is invalid")
            page_number = int(match.group(1))
            if page_number < 1 or page_number > self.page_count:
                raise ValueError("candidate page image number is outside the document")
            if page_number in page_numbers:
                raise ValueError("candidate document contains duplicate page images")
            page_numbers.add(page_number)
        if page_numbers != set(range(1, self.page_count + 1)):
            raise ValueError("candidate page image inventory does not match page_count")
        return self


class _CandidateRightsManifest(_StrictModel):
    status: Literal["missing"]


class _CandidateSourceBundle(_StrictModel):
    schema_version: Literal["hwp-hwpx-projection-pairs/1.0"]
    manifest_sha256: Sha256
    source_dataset_report_sha256: Sha256
    source_inventory_sha256: Sha256
    producer: dict[str, str]


class _CandidateProducer(_StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    pymupdf_version: str = Field(min_length=1)
    dpi: int = Field(gt=0, le=_MAX_JS_SAFE_INTEGER)
    pdf_text_order: Literal["native"]


class _CandidateCapabilityProfile(_StrictModel):
    id: str = Field(min_length=1)
    sha256: Sha256


class _CandidateContractVersions(_StrictModel):
    projection_source: Literal["hwpx-projection-source/1.0"]
    evidence_ir: Literal["evidence-ir/1.0"]
    content_ir: Literal["content-ir/1.0"]
    hwp_document_plan: Literal["hwp-document-plan/1.0"]


class _CandidateSplitCount(_StrictModel):
    documents: _SafeNonNegativeInteger
    pages: _SafeNonNegativeInteger


class _CandidateManifest(_StrictModel):
    schema_version: Literal["hwp-human-review-candidates/1.0"]
    artifact_role: Literal["human_review_candidate"]
    research_only: Literal[True]
    human_review_required: Literal[True]
    golden_eligible: Literal[False]
    training_eligible: Literal[False]
    release_eligible: Literal[False]
    rights_manifest: _CandidateRightsManifest
    source_bundle: _CandidateSourceBundle
    producer: _CandidateProducer
    capability_profile: _CandidateCapabilityProfile
    knowledge_corpus_sha256: Sha256
    contract_versions: _CandidateContractVersions
    document_count: int = Field(gt=0, le=_MAX_DOCUMENTS)
    page_count: int = Field(gt=0, le=_MAX_DOCUMENTS * _MAX_PAGES_PER_DOCUMENT)
    splits: dict[Split, _CandidateSplitCount]
    issue_counts: dict[IssueCode, _SafeNonNegativeInteger]
    documents: tuple[_CandidateDocument, ...] = Field(
        min_length=1,
        max_length=_MAX_DOCUMENTS,
    )

    @model_validator(mode="after")
    def validate_inventory_metadata(self) -> Self:
        if self.document_count != len(self.documents):
            raise ValueError("candidate document_count does not match documents")
        if self.page_count != sum(document.page_count for document in self.documents):
            raise ValueError("candidate page_count does not match documents")
        if set(self.splits) != {"train", "validation", "test"}:
            raise ValueError("candidate splits must contain train, validation, and test")
        for split in ("train", "validation", "test"):
            documents = [document for document in self.documents if document.split == split]
            expected = _CandidateSplitCount(
                documents=len(documents),
                pages=sum(document.page_count for document in documents),
            )
            if self.splits[split] != expected:
                raise ValueError(f"candidate split counts do not match for {split}")
        expected_issues = Counter(
            issue for document in self.documents for issue in document.issue_codes
        )
        if self.issue_counts != dict(expected_issues):
            raise ValueError("candidate issue_counts do not match documents")
        document_ids = [document.document_id.casefold() for document in self.documents]
        lineages = [document.lineage_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("candidate manifest contains duplicate document ids")
        if len(set(lineages)) != len(lineages):
            raise ValueError("candidate manifest contains duplicate lineages")
        paths = [
            binding.path.casefold()
            for document in self.documents
            for binding in document.artifacts.bindings()
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("candidate manifest contains duplicate artifact paths")
        return self


def load_candidate_review_draft(path: Path | str) -> CandidateReviewDraft:
    """Load only the untrusted draft contract; candidate verification is separate."""
    draft_path = _required_regular_file(Path(path), "review draft")
    try:
        return CandidateReviewDraft.model_validate_json(
            _read_bounded_file(
                draft_path,
                _MAX_REVIEW_DRAFT_BYTES,
                "review draft",
            ),
            strict=True,
        )
    except ValidationError as exc:
        raise ReviewDraftVerificationError("review draft is not valid strict JSON") from exc


def build_candidate_review_draft_view(
    verified_draft: CandidateReviewDraft,
) -> CandidateReviewDraftView:
    """Project an already candidate-verified draft into a sanitized bounded view."""
    if not isinstance(verified_draft, CandidateReviewDraft):
        raise TypeError("verified_draft must be a CandidateReviewDraft")
    try:
        verified_draft = CandidateReviewDraft.model_validate(
            verified_draft.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ReviewDraftVerificationError("verified review draft is not valid") from exc

    nodes: list[_ReviewNodeView] = []
    for node in verified_draft.reviewed_content_ir.nodes:
        if isinstance(node, TextContentNode):
            nodes.append(
                _ReviewTextNodeView(
                    id=node.id,
                    text=node.text,
                    role=node.role,
                    needs_review=node.needs_review,
                )
            )
        elif isinstance(node, TableContentNode):
            nodes.append(
                _ReviewTableNodeView(
                    id=node.id,
                    needs_review=node.needs_review,
                    cells=tuple(
                        _ReviewTableCellView(
                            row=cell.row,
                            column=cell.column,
                            row_span=cell.row_span,
                            column_span=cell.column_span,
                            text=cell.text,
                        )
                        for cell in node.cells
                    ),
                )
            )
        elif isinstance(node, ImageContentNode):
            nodes.append(
                _ReviewImageNodeView(
                    id=node.id,
                    needs_review=node.needs_review,
                )
            )
        elif isinstance(node, FormulaContentNode):
            nodes.append(
                _ReviewFormulaNodeView(
                    id=node.id,
                    needs_review=node.needs_review,
                )
            )

    return CandidateReviewDraftView(
        document_id=verified_draft.document_id,
        status=verified_draft.status,
        draft_revision=verified_draft.draft_revision,
        updated_at=verified_draft.updated_at,
        reviewed_content_ir_contract_sha256=(verified_draft.reviewed_content_ir_contract_sha256),
        nodes=tuple(nodes),
        issue_dispositions=tuple(
            _ReviewIssueDispositionView(
                issue_code=item.issue_code,
                disposition=item.disposition,
                note=item.note,
            )
            for item in verified_draft.issue_dispositions
        ),
    )


def start_candidate_review_draft(
    candidate_root: Path | str,
    document_id: str,
    reviewer_label: str,
    output_path: Path | str,
    *,
    updated_at: datetime | None = None,
    expected_manifest_sha256: str | None = None,
    expected_lineage_id: str | None = None,
) -> CandidateReviewDraft:
    """Create and atomically publish an untrusted review draft outside a candidate.

    The candidate bundle is read-only. This initializer copies the registered base
    ContentIR at the next revision, rebinds a copy of its Plan, marks every registered
    issue pending, verifies the complete draft, and only then creates ``output_path``.
    Existing outputs are never replaced.
    """
    root = _required_directory(Path(candidate_root), "candidate root")
    destination = _outside_new_output(root, Path(output_path))
    (
        manifest_sha256,
        document,
        _evidence,
        base_content,
        base_plan,
    ) = _load_candidate_base(
        root,
        document_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_lineage_id=expected_lineage_id,
    )

    reviewed_content_value = base_content.model_dump(mode="python")
    reviewed_content_value["revision"] = base_content.revision + 1
    reviewed_content = ContentIR.model_validate(
        reviewed_content_value,
        strict=True,
    )
    reviewed_plan = _rebind_reviewed_plan(base_plan, reviewed_content)
    timestamp = _review_timestamp(updated_at)

    draft = CandidateReviewDraft(
        status="in_progress",
        draft_revision=1,
        reviewer_label=reviewer_label,
        updated_at=timestamp.isoformat(),
        candidate_manifest_sha256=manifest_sha256,
        document_id=document.document_id,
        lineage_id=document.lineage_id,
        source_pdf_sha256=document.source_artifacts.pdf.sha256,
        evidence_ir=document.artifacts.evidence_ir,
        base_content_ir=document.artifacts.content_ir_candidate,
        base_hwp_document_plan=document.artifacts.hwp_document_plan_candidate,
        reviewed_content_ir=reviewed_content,
        reviewed_content_ir_contract_sha256=contract_sha256(reviewed_content),
        reviewed_hwp_document_plan=reviewed_plan,
        reviewed_hwp_document_plan_contract_sha256=contract_sha256(reviewed_plan),
        issue_dispositions=tuple(
            IssueDisposition(issue_code=code, disposition="pending")
            for code in document.issue_codes
        ),
    )
    verified = verify_candidate_review_draft(draft, candidate_root=root)
    build_candidate_review_draft_view(verified)
    _atomic_create_review_draft(root, destination, verified)
    return verified


def apply_candidate_review_patch(
    candidate_root: Path | str,
    draft_path: Path | str,
    request: CandidateReviewPatchRequest,
    output_path: Path | str,
    *,
    updated_at: datetime | None = None,
    expected_manifest_sha256: str | None = None,
    expected_lineage_id: str | None = None,
) -> CandidateReviewDraft:
    """Apply only typed human-review edits and publish a new immutable revision."""
    if not isinstance(request, CandidateReviewPatchRequest):
        raise TypeError("request must be a CandidateReviewPatchRequest")

    root = _required_directory(Path(candidate_root), "candidate root")
    current = load_candidate_review_draft(draft_path)
    current = verify_candidate_review_draft(current, candidate_root=root)
    _require_expected_candidate_binding(
        current.candidate_manifest_sha256,
        current.lineage_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_lineage_id=expected_lineage_id,
    )
    if current.status != "in_progress":
        raise ReviewDraftPatchError("completed review draft cannot be patched")
    try:
        request = CandidateReviewPatchRequest.model_validate(
            request.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ReviewDraftPatchError("review patch request is not valid") from exc
    if current.draft_revision != request.expected_draft_revision:
        raise ReviewDraftPatchError(
            "stale review draft revision: expected "
            f"{request.expected_draft_revision}, found {current.draft_revision}"
        )
    destination = _outside_new_output(root, Path(output_path))
    (
        _manifest_sha256,
        candidate_document,
        evidence,
        _base_content,
        _base_plan,
    ) = _load_candidate_base(
        root,
        current.document_id,
        expected_manifest_sha256=current.candidate_manifest_sha256,
        expected_lineage_id=current.lineage_id,
    )

    reviewed_content = current.reviewed_content_ir
    reviewed_plan = current.reviewed_hwp_document_plan
    issue_dispositions = current.issue_dispositions
    for operation in request.operations:
        if isinstance(operation, SetIssueDispositionPatchOperation):
            issue_dispositions = _apply_issue_disposition(
                issue_dispositions,
                operation,
            )
        elif isinstance(operation, SetImageGroundingPatchOperation):
            reviewed_content = _apply_image_grounding_operation(
                reviewed_content,
                operation,
                evidence=evidence,
                candidate_document=candidate_document,
            )
        elif isinstance(operation, DropImageNodePatchOperation):
            reviewed_content, reviewed_plan = _apply_drop_image_node_operation(
                reviewed_content,
                reviewed_plan,
                operation,
            )
        else:
            reviewed_content = _apply_content_operation(
                reviewed_content,
                operation,
            )
    reviewed_plan = _rebind_reviewed_plan(
        reviewed_plan,
        reviewed_content,
    )
    timestamp = _review_timestamp(updated_at)
    patched = CandidateReviewDraft(
        status="in_progress",
        draft_revision=current.draft_revision + 1,
        reviewer_label=current.reviewer_label,
        updated_at=timestamp.isoformat(),
        candidate_manifest_sha256=current.candidate_manifest_sha256,
        document_id=current.document_id,
        lineage_id=current.lineage_id,
        source_pdf_sha256=current.source_pdf_sha256,
        evidence_ir=current.evidence_ir,
        base_content_ir=current.base_content_ir,
        base_hwp_document_plan=current.base_hwp_document_plan,
        reviewed_content_ir=reviewed_content,
        reviewed_content_ir_contract_sha256=contract_sha256(reviewed_content),
        reviewed_hwp_document_plan=reviewed_plan,
        reviewed_hwp_document_plan_contract_sha256=contract_sha256(reviewed_plan),
        issue_dispositions=issue_dispositions,
    )
    verified = verify_candidate_review_draft(patched, candidate_root=root)
    build_candidate_review_draft_view(verified)
    _atomic_create_review_draft(root, destination, verified)
    return verified


def complete_candidate_review_draft(
    candidate_root: Path | str,
    draft_path: Path | str,
    request: CandidateReviewCompletionRequest,
    output_path: Path | str,
    *,
    updated_at: datetime | None = None,
    expected_manifest_sha256: str | None = None,
    expected_lineage_id: str | None = None,
) -> CandidateReviewDraft:
    """Publish an immutable complete revision without changing reviewed data."""
    if not isinstance(request, CandidateReviewCompletionRequest):
        raise TypeError("request must be a CandidateReviewCompletionRequest")

    root = _required_directory(Path(candidate_root), "candidate root")
    current = load_candidate_review_draft(draft_path)
    current = verify_candidate_review_draft(current, candidate_root=root)
    _require_expected_candidate_binding(
        current.candidate_manifest_sha256,
        current.lineage_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_lineage_id=expected_lineage_id,
    )
    if current.status != "in_progress":
        raise ReviewDraftCompletionError("review draft is already complete")
    try:
        request = CandidateReviewCompletionRequest.model_validate(
            request.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ReviewDraftCompletionError("review completion request is not valid") from exc
    if current.draft_revision != request.expected_draft_revision:
        raise ReviewDraftCompletionError(
            "stale review draft revision: expected "
            f"{request.expected_draft_revision}, found {current.draft_revision}"
        )
    destination = _outside_new_output(root, Path(output_path))
    timestamp = _review_timestamp(updated_at)
    completed = current.model_copy(
        update={
            "status": "complete",
            "draft_revision": current.draft_revision + 1,
            "updated_at": timestamp.isoformat(),
        }
    )
    verified = verify_candidate_review_draft(completed, candidate_root=root)
    build_candidate_review_draft_view(verified)
    _atomic_create_review_draft(root, destination, verified)
    return verified


def _apply_content_operation(
    content: ContentIR,
    operation: SetTextPatchOperation
    | SetRolePatchOperation
    | SetTableCellTextPatchOperation
    | SetNeedsReviewPatchOperation,
) -> ContentIR:
    node_positions = {node.id: index for index, node in enumerate(content.nodes)}
    position = node_positions.get(operation.node_id)
    if position is None:
        raise ReviewDraftPatchError(f"review patch node does not exist: {operation.node_id}")
    node = content.nodes[position]
    node_value = node.model_dump(mode="python")

    if isinstance(operation, SetTextPatchOperation):
        if not isinstance(node, TextContentNode):
            raise ReviewDraftPatchError("set_text requires a text node")
        node_value["text"] = operation.text
    elif isinstance(operation, SetRolePatchOperation):
        if not isinstance(node, TextContentNode):
            raise ReviewDraftPatchError("set_role requires a text node")
        node_value["role"] = operation.role
    elif isinstance(operation, SetTableCellTextPatchOperation):
        if not isinstance(node, TableContentNode):
            raise ReviewDraftPatchError("set_table_cell_text requires a table node")
        key = (
            operation.row,
            operation.column,
            operation.row_span,
            operation.column_span,
        )
        cell_values = [cell.model_dump(mode="python") for cell in node.cells]
        matches = [
            index
            for index, cell in enumerate(node.cells)
            if (cell.row, cell.column, cell.row_span, cell.column_span) == key
        ]
        if len(matches) != 1:
            raise ReviewDraftPatchError("set_table_cell_text cell key does not exist exactly")
        cell_value = node.cells[matches[0]].model_dump(mode="python")
        cell_value["text"] = operation.text
        cell_values[matches[0]] = TableCell.model_validate(
            cell_value,
            strict=True,
        ).model_dump(mode="python")
        node_value["cells"] = tuple(cell_values)
    else:
        node_value["needs_review"] = operation.needs_review

    node_values = [item.model_dump(mode="python") for item in content.nodes]
    node_values[position] = node_value
    content_value = content.model_dump(mode="python")
    content_value["nodes"] = tuple(node_values)
    return ContentIR.model_validate(content_value, strict=True)


def _apply_image_grounding_operation(
    content: ContentIR,
    operation: SetImageGroundingPatchOperation,
    *,
    evidence: EvidenceIR,
    candidate_document: _CandidateDocument,
) -> ContentIR:
    node_positions = {node.id: index for index, node in enumerate(content.nodes)}
    position = node_positions.get(operation.node_id)
    if position is None:
        raise ReviewDraftPatchError(
            f"review patch node does not exist: {operation.node_id}"
        )
    node = content.nodes[position]
    if not isinstance(node, ImageContentNode):
        raise ReviewDraftPatchError("set_image_grounding requires an image node")

    node_value = node.model_dump(mode="python")
    node_value.update(
        asset_ref=operation.asset_ref,
        evidence_refs=(operation.observation_ref,),
    )
    rebound = ImageContentNode.model_validate(node_value, strict=True)
    error = _exact_image_grounding_error(
        rebound,
        evidence,
        registered_source_ids=_registered_image_source_ids(
            evidence,
            candidate_document,
        ),
    )
    if error is not None:
        raise ReviewDraftPatchError(error)

    node_values = [item.model_dump(mode="python") for item in content.nodes]
    node_values[position] = rebound.model_dump(mode="python")
    content_value = content.model_dump(mode="python")
    content_value["nodes"] = tuple(node_values)
    return ContentIR.model_validate(content_value, strict=True)


def _apply_drop_image_node_operation(
    content: ContentIR,
    plan: HwpDocumentPlan,
    operation: DropImageNodePatchOperation,
) -> tuple[ContentIR, HwpDocumentPlan]:
    node_by_id = {node.id: node for node in content.nodes}
    node = node_by_id.get(operation.node_id)
    if node is None:
        raise ReviewDraftPatchError(
            f"review patch node does not exist: {operation.node_id}"
        )
    if not isinstance(node, ImageContentNode):
        raise ReviewDraftPatchError("drop_image_node requires an image node")
    if len(content.nodes) == 1:
        raise ReviewDraftPatchError("drop_image_node cannot remove the final content node")

    content_value = content.model_dump(mode="python")
    content_value["nodes"] = tuple(
        item for item in content_value["nodes"] if item["id"] != operation.node_id
    )
    content_value["reading_order"] = tuple(
        node_id for node_id in content.reading_order if node_id != operation.node_id
    )
    reviewed_content = ContentIR.model_validate(content_value, strict=True)

    matching_flow = tuple(
        item
        for item in plan.flow
        if isinstance(item, ContentPlanItem) and item.content_ref == operation.node_id
    )
    if len(matching_flow) != 1:
        raise ReviewDraftPatchError(
            "drop_image_node requires exactly one matching plan content flow item"
        )
    plan_value = plan.model_dump(mode="python")
    plan_value["flow"] = tuple(
        item
        for item in plan_value["flow"]
        if not (
            item["kind"] == "content"
            and item.get("content_ref") == operation.node_id
        )
    )
    reviewed_plan = HwpDocumentPlan.model_validate(plan_value, strict=True)
    return reviewed_content, reviewed_plan


def _apply_issue_disposition(
    current: tuple[IssueDisposition, ...],
    operation: SetIssueDispositionPatchOperation,
) -> tuple[IssueDisposition, ...]:
    replacement = IssueDisposition(
        issue_code=operation.issue_code,
        disposition=operation.disposition,
        note=operation.note,
    )
    values = list(current)
    for index, item in enumerate(values):
        if item.issue_code == operation.issue_code:
            values[index] = replacement
            break
    else:
        values.append(replacement)
    return tuple(values)


def _rebind_reviewed_plan(
    plan: HwpDocumentPlan,
    content: ContentIR,
) -> HwpDocumentPlan:
    value = plan.model_dump(mode="python")
    value.update(
        {
            "content_ir_id": content.id,
            "content_ir_revision": content.revision,
            "content_ir_sha256": contract_sha256(content),
        }
    )
    return HwpDocumentPlan.model_validate(value, strict=True)


def _review_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("updated_at must include a timezone")
    return timestamp


def _load_candidate_base(
    root: Path,
    document_id: str,
    *,
    expected_manifest_sha256: str | None,
    expected_lineage_id: str | None,
) -> tuple[str, _CandidateDocument, EvidenceIR, ContentIR, HwpDocumentPlan]:
    manifest_path = _resolve_registered_file(root, "manifest.json", "candidate manifest")
    manifest_payload = _read_bounded_file(
        manifest_path,
        _MAX_CANDIDATE_MANIFEST_BYTES,
        "candidate manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    try:
        manifest = _CandidateManifest.model_validate_json(manifest_payload, strict=True)
    except ValidationError as exc:
        raise ReviewDraftVerificationError("candidate manifest is not valid strict JSON") from exc
    matching = [document for document in manifest.documents if document.document_id == document_id]
    if len(matching) != 1:
        raise ReviewDraftVerificationError(
            "review draft document is not uniquely registered in candidate manifest"
        )
    document = matching[0]
    _require_expected_candidate_binding(
        manifest_sha256,
        document.lineage_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_lineage_id=expected_lineage_id,
    )
    payloads = _read_verified_candidate_artifacts(root, document)
    evidence = _parse_contract(
        EvidenceIR,
        payloads[document.artifacts.evidence_ir.path],
        "EvidenceIR",
    )
    base_content = _parse_contract(
        ContentIR,
        payloads[document.artifacts.content_ir_candidate.path],
        "base ContentIR",
    )
    base_plan = _parse_contract(
        HwpDocumentPlan,
        payloads[document.artifacts.hwp_document_plan_candidate.path],
        "base HwpDocumentPlan",
    )
    _require_contract_digest(evidence, document.artifacts.evidence_ir, "EvidenceIR")
    _require_contract_digest(
        base_content,
        document.artifacts.content_ir_candidate,
        "base ContentIR",
    )
    _require_contract_digest(
        base_plan,
        document.artifacts.hwp_document_plan_candidate,
        "base HwpDocumentPlan",
    )
    if evidence.source_document_sha256 != document.source_artifacts.pdf.sha256:
        raise ReviewDraftVerificationError("EvidenceIR is not bound to the candidate PDF")
    if len(evidence.pages) != document.page_count:
        raise ReviewDraftVerificationError("EvidenceIR page count does not match candidate")
    try:
        base_content.assert_evidence_integrity(evidence)
        base_plan.assert_content_integrity(base_content)
    except ValueError as exc:
        raise ReviewDraftVerificationError(f"candidate contract lineage is invalid: {exc}") from exc
    return manifest_sha256, document, evidence, base_content, base_plan


def _outside_new_output(root: Path, output_path: Path) -> Path:
    if output_path.is_symlink():
        raise ValueError("review draft output must not be a symlink")
    destination = output_path.resolve(strict=False)
    if destination.is_relative_to(root):
        raise ValueError("review draft output must be outside candidate root")
    if destination.exists():
        raise FileExistsError(f"review draft output already exists: {destination}")
    return destination


def _atomic_create_review_draft(
    candidate_root: Path,
    destination: Path,
    draft: CandidateReviewDraft,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _required_directory(destination.parent, "review draft output directory")
    destination = parent / destination.name
    if destination.is_relative_to(candidate_root):
        raise ValueError("review draft output must be outside candidate root")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"review draft output already exists: {destination}")
    payload = (
        json.dumps(
            draft.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"review draft output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_expected_candidate_binding(
    manifest_sha256: str,
    lineage_id: str,
    *,
    expected_manifest_sha256: str | None,
    expected_lineage_id: str | None,
) -> None:
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise ReviewDraftVerificationError(
            "candidate manifest does not match the selected snapshot"
        )
    if expected_lineage_id is not None and lineage_id != expected_lineage_id:
        raise ReviewDraftVerificationError(
            "candidate lineage does not match the selected snapshot"
        )


def verify_candidate_review_draft(
    draft: CandidateReviewDraft,
    *,
    candidate_root: Path | str,
) -> CandidateReviewDraft:
    """Verify a review draft without granting rights, trust, or training eligibility.

    A successful return proves only that the supplied edits are structurally bound to
    one intact candidate document. It does not create a Golden Dataset manifest or a
    human/rights attestation, and the returned contract remains non-promotable.
    """
    draft = _revalidate_candidate_review_contract(draft)

    root = _required_directory(Path(candidate_root), "candidate root")
    manifest_path = _resolve_registered_file(root, "manifest.json", "candidate manifest")
    manifest_payload = _read_bounded_file(
        manifest_path,
        _MAX_CANDIDATE_MANIFEST_BYTES,
        "candidate manifest",
    )
    actual_manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if actual_manifest_sha256 != draft.candidate_manifest_sha256:
        raise ReviewDraftVerificationError("candidate manifest SHA-256 mismatch")
    try:
        manifest = _CandidateManifest.model_validate_json(
            manifest_payload,
            strict=True,
        )
    except ValidationError as exc:
        raise ReviewDraftVerificationError("candidate manifest is not valid strict JSON") from exc

    matching = [
        document for document in manifest.documents if document.document_id == draft.document_id
    ]
    if len(matching) != 1:
        raise ReviewDraftVerificationError(
            "review draft document is not uniquely registered in candidate manifest"
        )
    document = matching[0]
    if document.lineage_id != draft.lineage_id:
        raise ReviewDraftVerificationError("review draft lineage does not match candidate")
    if document.source_artifacts.pdf.sha256 != draft.source_pdf_sha256:
        raise ReviewDraftVerificationError("review draft PDF digest does not match candidate")

    payloads = _read_verified_candidate_artifacts(root, document)
    _require_binding(
        draft.evidence_ir,
        document.artifacts.evidence_ir,
        "EvidenceIR",
    )
    _require_binding(
        draft.base_content_ir,
        document.artifacts.content_ir_candidate,
        "base ContentIR",
    )
    _require_binding(
        draft.base_hwp_document_plan,
        document.artifacts.hwp_document_plan_candidate,
        "base HwpDocumentPlan",
    )

    evidence = _parse_contract(
        EvidenceIR,
        payloads[document.artifacts.evidence_ir.path],
        "EvidenceIR",
    )
    base_content = _parse_contract(
        ContentIR,
        payloads[document.artifacts.content_ir_candidate.path],
        "base ContentIR",
    )
    base_plan = _parse_contract(
        HwpDocumentPlan,
        payloads[document.artifacts.hwp_document_plan_candidate.path],
        "base HwpDocumentPlan",
    )
    _require_contract_digest(evidence, document.artifacts.evidence_ir, "EvidenceIR")
    _require_contract_digest(
        base_content,
        document.artifacts.content_ir_candidate,
        "base ContentIR",
    )
    _require_contract_digest(
        base_plan,
        document.artifacts.hwp_document_plan_candidate,
        "base HwpDocumentPlan",
    )

    if evidence.source_document_sha256 != document.source_artifacts.pdf.sha256:
        raise ReviewDraftVerificationError("EvidenceIR is not bound to the candidate PDF")
    if len(evidence.pages) != document.page_count:
        raise ReviewDraftVerificationError("EvidenceIR page count does not match candidate")
    try:
        base_content.assert_evidence_integrity(evidence)
        base_plan.assert_content_integrity(base_content)
        draft.reviewed_content_ir.assert_evidence_integrity(evidence)
        draft.reviewed_hwp_document_plan.assert_content_integrity(draft.reviewed_content_ir)
    except ValueError as exc:
        raise ReviewDraftVerificationError(f"candidate contract lineage is invalid: {exc}") from exc

    _validate_reviewed_content(
        base_content,
        draft.reviewed_content_ir,
        evidence,
        candidate_document=document,
        require_exact_image_grounding=draft.status == "complete",
    )
    _validate_reviewed_plan(
        base_plan,
        draft.reviewed_hwp_document_plan,
        base_content,
        draft.reviewed_content_ir,
    )
    _validate_issue_dispositions(draft, document)
    return draft


def materialize_verified_candidate_compile_inputs(
    draft: CandidateReviewDraft,
    *,
    candidate_root: Path | str,
) -> VerifiedCandidateCompileInputs:
    """Materialize compiler inputs from one complete, intact candidate review."""
    draft = _revalidate_candidate_review_contract(draft)
    if draft.status != "complete":
        raise ReviewDraftVerificationError(
            "candidate review must be complete before compile inputs are materialized"
        )

    requested_asset_refs = tuple(
        sorted(
            {
                node.asset_ref
                for node in draft.reviewed_content_ir.nodes
                if isinstance(node, ImageContentNode)
            }
        )
    )
    if len(requested_asset_refs) > MAX_IMAGE_ASSETS:
        raise ReviewDraftVerificationError(
            "candidate review image asset count exceeds the bundle limit"
        )

    candidate_review = verify_candidate_review_draft(
        draft,
        candidate_root=candidate_root,
    )
    root = _required_directory(Path(candidate_root), "candidate root")
    (
        _manifest_sha256,
        document,
        evidence,
        _base_content,
        _base_plan,
    ) = _load_candidate_base(
        root,
        candidate_review.document_id,
        expected_manifest_sha256=candidate_review.candidate_manifest_sha256,
        expected_lineage_id=candidate_review.lineage_id,
    )

    image_nodes = tuple(
        node
        for node in candidate_review.reviewed_content_ir.nodes
        if isinstance(node, ImageContentNode)
    )
    if not image_nodes:
        return VerifiedCandidateCompileInputs(
            candidate_review=candidate_review,
            evidence_ir=evidence,
            image_assets=None,
        )

    registered_source_ids = _registered_image_source_ids(evidence, document)
    for image in image_nodes:
        error = _exact_image_grounding_error(
            image,
            evidence,
            registered_source_ids=registered_source_ids,
        )
        if error is not None:
            raise ReviewDraftVerificationError(f"{error}: {image.id}")

    source_by_id = {source.id: source for source in evidence.sources}
    bindings_by_kind = {
        EvidenceSourceKind.PAGE_IMAGE: tuple(
            sorted(document.artifacts.page_images, key=lambda binding: binding.path)
        ),
        EvidenceSourceKind.CROP: tuple(
            sorted(document.artifacts.image_crops, key=lambda binding: binding.path)
        ),
    }
    selected: list[tuple[str, CandidateArtifactBinding, Path, int]] = []
    total_declared_bytes = 0
    for asset_ref in requested_asset_refs:
        source = source_by_id[asset_ref]
        if source.sha256 is None or source.kind not in bindings_by_kind:
            raise ReviewDraftVerificationError(
                f"reviewed image asset source is not materializable: {asset_ref}"
            )
        matches = tuple(
            binding
            for binding in bindings_by_kind[source.kind]
            if binding.sha256 == source.sha256
        )
        if not matches:
            raise ReviewDraftVerificationError(
                f"reviewed image asset source SHA is not registered: {asset_ref}"
            )
        binding = matches[0]
        path = _resolve_registered_file(
            root,
            binding.path,
            "candidate image asset",
        )
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ReviewDraftVerificationError(
                f"candidate image asset cannot be inspected: {binding.path}"
            ) from exc
        if size > MAX_IMAGE_ASSET_BYTES:
            raise ReviewDraftVerificationError(
                f"candidate image asset exceeds the per-asset byte limit: {binding.path}"
            )
        total_declared_bytes += size
        if total_declared_bytes > MAX_IMAGE_ASSET_TOTAL_BYTES:
            raise ReviewDraftVerificationError(
                "candidate image assets exceed the bundle total byte limit"
            )
        selected.append((asset_ref, binding, path, size))

    assets: list[PngImageAsset] = []
    total_read_bytes = 0
    for asset_ref, binding, _path, _size in selected:
        payload, size = _read_and_verify_registered_artifact(
            root,
            binding,
            max_bytes=min(
                MAX_IMAGE_ASSET_BYTES,
                MAX_IMAGE_ASSET_TOTAL_BYTES - total_read_bytes,
            ),
            retain_payload=True,
        )
        if payload is None:
            raise ReviewDraftVerificationError(
                f"candidate image asset payload was not retained: {binding.path}"
            )
        total_read_bytes += size
        try:
            assets.append(
                PngImageAsset(
                    asset_ref=asset_ref,
                    media_type="image/png",
                    sha256=binding.sha256,
                    payload=payload,
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ReviewDraftVerificationError(
                f"candidate image asset is not a valid PNG: {binding.path}"
            ) from exc

    try:
        image_assets = build_image_asset_bundle(
            evidence,
            candidate_review.reviewed_content_ir,
            tuple(assets),
        )
    except (ImageAssetBundleError, TypeError, ValueError, ValidationError) as exc:
        raise ReviewDraftVerificationError(
            "candidate image asset bundle is not valid"
        ) from exc
    return VerifiedCandidateCompileInputs(
        candidate_review=candidate_review,
        evidence_ir=evidence,
        image_assets=image_assets,
    )


def _revalidate_candidate_review_contract(
    draft: CandidateReviewDraft,
) -> CandidateReviewDraft:
    if not isinstance(draft, CandidateReviewDraft):
        raise TypeError("draft must be a CandidateReviewDraft")
    try:
        return CandidateReviewDraft.model_validate(
            draft.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ReviewDraftVerificationError("review draft contract is not valid") from exc


def _parse_contract(
    model_type: type[ContractT],
    payload: bytes,
    label: str,
) -> ContractT:
    try:
        return model_type.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ReviewDraftVerificationError(f"{label} is not valid strict JSON") from exc


def _require_contract_digest(
    contract: EvidenceIR | ContentIR | HwpDocumentPlan,
    binding: CandidateContractArtifactBinding,
    label: str,
) -> None:
    if contract_sha256(contract) != binding.contract_sha256:
        raise ReviewDraftVerificationError(f"{label} contract SHA-256 mismatch")


def _require_binding(
    draft_binding: CandidateContractArtifactBinding,
    manifest_binding: CandidateContractArtifactBinding,
    label: str,
) -> None:
    if draft_binding != manifest_binding:
        raise ReviewDraftVerificationError(
            f"review draft {label} binding does not match candidate manifest"
        )


def _validate_reviewed_content(
    base: ContentIR,
    reviewed: ContentIR,
    evidence: EvidenceIR,
    *,
    candidate_document: _CandidateDocument,
    require_exact_image_grounding: bool,
) -> None:
    if reviewed.id != base.id:
        raise ReviewDraftVerificationError("reviewed ContentIR id must be preserved")
    if reviewed.revision != base.revision + 1:
        raise ReviewDraftVerificationError(
            "reviewed ContentIR revision must increment base revision by one"
        )
    if (
        reviewed.evidence_ir_id != base.evidence_ir_id
        or reviewed.evidence_ir_sha256 != base.evidence_ir_sha256
    ):
        raise ReviewDraftVerificationError(
            "reviewed ContentIR EvidenceIR binding must be preserved"
        )

    base_by_id = {node.id: node for node in base.nodes}
    reviewed_by_id = {node.id: node for node in reviewed.nodes}
    unknown_ids = set(reviewed_by_id) - set(base_by_id)
    removed_ids = set(base_by_id) - set(reviewed_by_id)
    if unknown_ids or any(
        not isinstance(base_by_id[node_id], ImageContentNode)
        for node_id in removed_ids
    ):
        raise ReviewDraftVerificationError("reviewed ContentIR node ids must be preserved")
    expected_node_order = tuple(
        node.id for node in base.nodes if node.id not in removed_ids
    )
    if tuple(node.id for node in reviewed.nodes) != expected_node_order:
        raise ReviewDraftVerificationError("reviewed ContentIR node order must be preserved")
    base_metadata = base.model_dump(
        mode="json",
        exclude={"nodes", "reading_order", "revision"},
    )
    reviewed_metadata = reviewed.model_dump(
        mode="json",
        exclude={"nodes", "reading_order", "revision"},
    )
    expected_reading_order = tuple(
        node_id for node_id in base.reading_order if node_id not in removed_ids
    )
    if reviewed_metadata != base_metadata or reviewed.reading_order != expected_reading_order:
        raise ReviewDraftVerificationError(
            "reviewed ContentIR reading order and metadata must be preserved"
        )
    registered_source_ids = _registered_image_source_ids(evidence, candidate_document)
    for node_id, reviewed_node in reviewed_by_id.items():
        base_node = base_by_id[node_id]
        if base_node.kind != reviewed_node.kind:
            raise ReviewDraftVerificationError(
                f"reviewed ContentIR node kind must be preserved: {node_id}"
            )
        if isinstance(base_node, ImageContentNode):
            if not isinstance(reviewed_node, ImageContentNode):
                raise ReviewDraftVerificationError(
                    f"reviewed image node kind must be preserved: {node_id}"
                )
            binding_changed = (
                base_node.evidence_refs != reviewed_node.evidence_refs
                or base_node.asset_ref != reviewed_node.asset_ref
            )
            if binding_changed or require_exact_image_grounding:
                error = _exact_image_grounding_error(
                    reviewed_node,
                    evidence,
                    registered_source_ids=registered_source_ids,
                )
                if error is not None:
                    raise ReviewDraftVerificationError(f"{error}: {node_id}")
            else:
                _validate_image_regrounding(node_id, reviewed_node, evidence)
        elif base_node.evidence_refs != reviewed_node.evidence_refs:
            raise ReviewDraftVerificationError(
                f"reviewed ContentIR evidence refs must be preserved: {node_id}"
            )
        if isinstance(base_node, TableContentNode):
            if not isinstance(reviewed_node, TableContentNode):
                raise ReviewDraftVerificationError(
                    f"reviewed table node kind must be preserved: {node_id}"
                )
            _validate_table_preservation(node_id, base_node, reviewed_node)
        _validate_only_typed_node_edits(node_id, base_node, reviewed_node)


def _validate_only_typed_node_edits(
    node_id: str,
    base: ContentNodeT,
    reviewed: ContentNodeT,
) -> None:
    normalized: ContentNodeT
    if isinstance(base, TextContentNode) and isinstance(reviewed, TextContentNode):
        normalized = reviewed.model_copy(
            update={
                "text": base.text,
                "role": base.role,
                "needs_review": base.needs_review,
            }
        )
    elif isinstance(base, TableContentNode) and isinstance(reviewed, TableContentNode):
        if len(base.cells) != len(reviewed.cells):
            raise ReviewDraftVerificationError(
                f"reviewed table topology must be preserved: {node_id}"
            )
        normalized_cells = tuple(
            reviewed_cell.model_copy(update={"text": base_cell.text})
            for base_cell, reviewed_cell in zip(base.cells, reviewed.cells, strict=True)
        )
        normalized = reviewed.model_copy(
            update={
                "cells": normalized_cells,
                "needs_review": base.needs_review,
            }
        )
    elif isinstance(base, ImageContentNode) and isinstance(reviewed, ImageContentNode):
        normalized = reviewed.model_copy(
            update={
                "asset_ref": base.asset_ref,
                "evidence_refs": base.evidence_refs,
                "needs_review": base.needs_review,
            }
        )
    elif isinstance(base, FormulaContentNode) and isinstance(reviewed, FormulaContentNode):
        normalized = reviewed.model_copy(
            update={"needs_review": base.needs_review}
        )
    else:
        raise ReviewDraftVerificationError(
            f"reviewed ContentIR node kind must be preserved: {node_id}"
        )
    if normalized != base:
        raise ReviewDraftVerificationError(
            f"reviewed ContentIR may only change typed review fields: {node_id}"
        )


def _validate_image_regrounding(
    node_id: str,
    reviewed: ImageContentNode,
    evidence: EvidenceIR,
) -> None:
    source_ids = {source.id for source in evidence.sources}
    if reviewed.asset_ref not in source_ids:
        raise ReviewDraftVerificationError(
            f"reviewed image asset is not an EvidenceIR source: {node_id}"
        )
    observations = {
        observation.id: observation for page in evidence.pages for observation in page.observations
    }
    grounded = any(
        observation_id in observations
        and observations[observation_id].kind in {ObservationKind.IMAGE, ObservationKind.REGION}
        and reviewed.asset_ref in observations[observation_id].source_refs
        for observation_id in reviewed.evidence_refs
    )
    if not grounded:
        raise ReviewDraftVerificationError(
            "reviewed image must reference an image or region observation that "
            f"directly contains its asset: {node_id}"
        )


def _registered_image_source_ids(
    evidence: EvidenceIR,
    candidate_document: _CandidateDocument,
) -> frozenset[str]:
    registered_sha256 = {
        EvidenceSourceKind.PAGE_IMAGE: {
            binding.sha256 for binding in candidate_document.artifacts.page_images
        },
        EvidenceSourceKind.CROP: {
            binding.sha256 for binding in candidate_document.artifacts.image_crops
        },
    }
    return frozenset(
        source.id
        for source in evidence.sources
        if source.kind in (EvidenceSourceKind.PAGE_IMAGE, EvidenceSourceKind.CROP)
        and source.sha256 is not None
        and source.sha256 in registered_sha256[source.kind]
    )


def _exact_image_grounding_error(
    image: ImageContentNode,
    evidence: EvidenceIR,
    *,
    registered_source_ids: frozenset[str],
) -> str | None:
    sources = {source.id: source for source in evidence.sources}
    source = sources.get(image.asset_ref)
    if source is None:
        return f"reviewed image asset is not an EvidenceIR source: {image.asset_ref}"
    if source.kind not in (EvidenceSourceKind.PAGE_IMAGE, EvidenceSourceKind.CROP):
        return f"reviewed image asset source must be PAGE_IMAGE or CROP: {image.asset_ref}"
    if source.sha256 is None:
        return f"reviewed image asset source requires sha256: {image.asset_ref}"
    if source.id not in registered_source_ids:
        return f"reviewed image asset source SHA is not registered: {image.asset_ref}"

    observations = {
        observation.id: observation
        for page in evidence.pages
        for observation in page.observations
    }
    image_groundings = tuple(
        observations[observation_ref]
        for observation_ref in image.evidence_refs
        if observation_ref in observations
        and observations[observation_ref].kind == ObservationKind.IMAGE
        and image.asset_ref in observations[observation_ref].source_refs
    )
    if len(image_groundings) != 1:
        return (
            "reviewed image must be directly grounded by exactly one IMAGE "
            f"observation for source {image.asset_ref}"
        )
    return None


def _validate_reviewed_plan(
    base: HwpDocumentPlan,
    reviewed: HwpDocumentPlan,
    base_content: ContentIR,
    reviewed_content: ContentIR,
) -> None:
    base_value = base.model_dump(mode="json")
    reviewed_value = reviewed.model_dump(mode="json")
    reviewed_node_ids = {node.id for node in reviewed_content.nodes}
    dropped_image_ids = {
        node.id
        for node in base_content.nodes
        if isinstance(node, ImageContentNode) and node.id not in reviewed_node_ids
    }
    base_value["flow"] = [
        item
        for item in base_value["flow"]
        if not (
            item["kind"] == "content"
            and item.get("content_ref") in dropped_image_ids
        )
    ]
    for field_name in ("content_ir_revision", "content_ir_sha256"):
        base_value.pop(field_name)
        reviewed_value.pop(field_name)
    if reviewed_value != base_value:
        raise ReviewDraftVerificationError(
            "reviewed HwpDocumentPlan may only change its ContentIR revision and digest"
        )


def _validate_table_preservation(
    node_id: str,
    base: TableContentNode,
    reviewed: TableContentNode,
) -> None:
    if (base.rows, base.columns) != (reviewed.rows, reviewed.columns):
        raise ReviewDraftVerificationError(
            f"reviewed table topology must be preserved (dimensions): {node_id}"
        )
    base_cells = {
        (cell.row, cell.column, cell.row_span, cell.column_span): cell for cell in base.cells
    }
    reviewed_cells = {
        (cell.row, cell.column, cell.row_span, cell.column_span): cell for cell in reviewed.cells
    }
    if set(base_cells) != set(reviewed_cells):
        raise ReviewDraftVerificationError(f"reviewed table topology must be preserved: {node_id}")
    changed_evidence = sorted(
        str(key)
        for key, base_cell in base_cells.items()
        if base_cell.evidence_refs != reviewed_cells[key].evidence_refs
    )
    if changed_evidence:
        raise ReviewDraftVerificationError(
            f"reviewed table cell evidence refs must be preserved: {node_id}"
        )


def _validate_issue_dispositions(
    draft: CandidateReviewDraft,
    document: _CandidateDocument,
) -> None:
    expected = set(document.issue_codes)
    actual = {item.issue_code for item in draft.issue_dispositions}
    unknown = sorted(actual - expected)
    if unknown:
        raise ReviewDraftVerificationError(
            "review draft contains unknown issue dispositions: " + ", ".join(unknown)
        )
    if draft.status != "complete":
        return
    missing = sorted(expected - actual)
    if missing:
        raise ReviewDraftVerificationError(
            "complete review draft is missing issue dispositions: " + ", ".join(missing)
        )
    unresolved_nodes = sorted(
        node.id for node in draft.reviewed_content_ir.nodes if node.needs_review
    )
    if unresolved_nodes:
        raise ReviewDraftVerificationError(
            "complete review draft contains nodes that still need review: "
            + ", ".join(unresolved_nodes)
        )


def _read_verified_candidate_artifacts(
    root: Path,
    document: _CandidateDocument,
) -> dict[str, bytes]:
    retained_paths = {
        document.artifacts.evidence_ir.path,
        document.artifacts.content_ir_candidate.path,
        document.artifacts.hwp_document_plan_candidate.path,
    }
    binary_paths = {
        binding.path
        for binding in (*document.artifacts.page_images, *document.artifacts.image_crops)
    }
    retained: dict[str, bytes] = {}
    total_bytes = 0
    for binding in document.artifacts.bindings():
        per_file_limit = (
            _MAX_BINARY_ARTIFACT_BYTES
            if binding.path in binary_paths
            else _MAX_CONTRACT_ARTIFACT_BYTES
            if binding.path in retained_paths
            else _MAX_METADATA_ARTIFACT_BYTES
        )
        remaining = _MAX_TOTAL_ARTIFACT_BYTES_PER_DOCUMENT - total_bytes
        if remaining <= 0:
            raise ReviewDraftVerificationError(
                "candidate document artifacts exceed the total size limit"
            )
        payload, size = _read_and_verify_registered_artifact(
            root,
            binding,
            max_bytes=min(per_file_limit, remaining),
            retain_payload=binding.path in retained_paths,
        )
        total_bytes += size
        if payload is not None:
            retained[binding.path] = payload
    return retained


def _read_and_verify_registered_artifact(
    root: Path,
    binding: CandidateArtifactBinding,
    *,
    max_bytes: int,
    retain_payload: bool,
) -> tuple[bytes | None, int]:
    path = _resolve_registered_file(root, binding.path, "candidate artifact")
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] | None = [] if retain_payload else None
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                if size > max_bytes:
                    raise ReviewDraftVerificationError(
                        f"candidate artifact exceeds the size limit: {binding.path}"
                    )
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
    except OSError as exc:
        raise ReviewDraftVerificationError(
            f"candidate artifact cannot be read: {binding.path}"
        ) from exc
    if digest.hexdigest() != binding.sha256:
        raise ReviewDraftVerificationError(f"candidate artifact SHA-256 mismatch: {binding.path}")
    return (b"".join(chunks) if chunks is not None else None), size


def _read_bounded_file(path: Path, max_bytes: int, label: str) -> bytes:
    payload = bytearray()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise ReviewDraftVerificationError(f"{label} exceeds the size limit")
    except OSError as exc:
        raise ReviewDraftVerificationError(f"{label} cannot be read") from exc
    return bytes(payload)


def _required_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReviewDraftVerificationError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ReviewDraftVerificationError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise ReviewDraftVerificationError(f"{label} must be a directory")
    return resolved


def _required_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReviewDraftVerificationError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ReviewDraftVerificationError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise ReviewDraftVerificationError(f"{label} must be a regular file")
    return resolved


def _resolve_registered_file(root: Path, relative: str, label: str) -> Path:
    try:
        binding = CandidateArtifactBinding(path=relative, sha256="0" * 64)
    except ValidationError as exc:
        raise ReviewDraftVerificationError(f"{label} path is invalid") from exc
    current = root
    for part in PurePosixPath(binding.path).parts:
        current = current / part
        if current.is_symlink():
            raise ReviewDraftVerificationError(f"{label} must not contain symlinks")
    try:
        resolved = current.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ReviewDraftVerificationError(f"{label} is missing: {relative}") from exc
    if not resolved.is_relative_to(root):
        raise ReviewDraftVerificationError(f"{label} escapes candidate root")
    if not resolved.is_file():
        raise ReviewDraftVerificationError(f"{label} must be a regular file")
    return resolved
