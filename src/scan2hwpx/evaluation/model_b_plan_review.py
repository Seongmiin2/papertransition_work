from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Self, TypeAlias, TypeVar

from pydantic import ConfigDict, Field, ValidationError, model_validator

from scan2hwpx.contracts import (
    ContentIR,
    ContentPlanItem,
    HwpDocumentPlan,
    LayoutIntent,
    PageLayoutIntent,
    contract_sha256,
)
from scan2hwpx.contracts.models import StrictContractModel, TextAlignment
from scan2hwpx.evaluation.safe_artifact_io import (
    SafeArtifactIOError,
    publish_file_create_only,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_BoundedText = Annotated[str, Field(min_length=1, max_length=1_024)]
_MAX_PATCH_OPERATIONS = 4
_MAX_STYLES = 5_000
_MAX_FLOW_ITEMS = 100_000
_MAX_SPEC_REFS = 1_000
_MAX_PATCH_TEXT_CHARS = 10_000_000
_MAX_PATCH_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_COMPLETION_REQUEST_BYTES = 16 * 1024
_MAX_SOURCE_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_REVIEW_DRAFT_BYTES = 64 * 1024 * 1024

ContractT = TypeVar("ContractT", bound=StrictContractModel)


class ModelBPlanReviewPatchError(ValueError):
    """Raised when a typed Model B plan patch cannot be applied safely."""


class ModelBPlanReviewCompletionError(ValueError):
    """Raised when an exact Model B review revision cannot be completed."""


class ModelBPlanReviewVerificationError(ValueError):
    """Raised when a persisted review is not bound to its source contracts."""


class _StrictModel(StrictContractModel):
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


class ModelBOfficialSpecRefEvidence(_StrictModel):
    """One retrieved spec reference bound to immutable source and chunk bytes."""

    official_spec_ref: _BoundedText
    source_document_sha256: Sha256
    retrieved_chunk_sha256: Sha256


class ModelBPlanGroundingEvidence(_StrictModel):
    """Reproducible, still-untrusted evidence boundary for a Model B review."""

    schema_version: Literal["model-b-plan-grounding-evidence/1.0"] = (
        "model-b-plan-grounding-evidence/1.0"
    )
    knowledge_corpus_sha256: Sha256
    retrieval_artifact_sha256: Sha256
    capability_profile_id: _BoundedText
    capability_profile_sha256: Sha256
    official_spec_refs: tuple[ModelBOfficialSpecRefEvidence, ...] = Field(
        min_length=1,
        max_length=_MAX_SPEC_REFS,
    )

    @model_validator(mode="after")
    def reject_duplicate_refs(self) -> Self:
        _reject_duplicate_ids(
            (item.official_spec_ref for item in self.official_spec_refs),
            "official spec ref evidence",
        )
        return self


class ModelBPlanReviewCandidateBinding(_StrictModel):
    """Canonical identity of the immutable ContentIR and Model B plan candidate."""

    schema_version: Literal["model-b-plan-review-candidate-binding/1.0"] = (
        "model-b-plan-review-candidate-binding/1.0"
    )
    document_id: _BoundedText
    content_ir_contract_sha256: Sha256
    base_hwp_document_plan_contract_sha256: Sha256


class ModelBPlanReviewInferenceBinding(_StrictModel):
    """Immutable provenance for the exact Model B result seeded into review."""

    schema_version: Literal["model-b-plan-review-inference-binding/1.0"] = (
        "model-b-plan-review-inference-binding/1.0"
    )
    handoff_artifact_sha256: Sha256
    request_sha256: Sha256
    result_sha256: Sha256
    raw_response_sha256: Sha256
    inferred_hwp_document_plan_contract_sha256: Sha256
    model_id: _BoundedText
    model_revision: _BoundedText
    model_artifact_sha256: Sha256
    handoff_source_candidate_contract_sha256: Sha256


class ModelBPlanReviewStyle(_StrictModel):
    """Bounded patch representation of the current StyleIntent schema."""

    id: _BoundedText
    semantic_role: _BoundedText
    font_family: _BoundedText | None = None
    font_size_pt: float | None = Field(default=None, gt=0, le=200)
    bold: bool = False
    italic: bool = False
    alignment: TextAlignment = TextAlignment.LEFT
    line_spacing: float | None = Field(default=None, ge=0.5, le=5)


class ModelBPlanReviewContentFlowItem(_StrictModel):
    """Editable flow fields; content_ref is deliberately absent."""

    id: _BoundedText
    kind: Literal["content"] = "content"
    render_as: Literal["paragraph", "table", "image", "formula"]
    style_ref: _BoundedText | None = None
    layout: LayoutIntent = Field(default_factory=LayoutIntent)


class ModelBPlanReviewPageBreakFlowItem(_StrictModel):
    id: _BoundedText
    kind: Literal["page_break"] = "page_break"


class ModelBPlanReviewColumnBreakFlowItem(_StrictModel):
    id: _BoundedText
    kind: Literal["column_break"] = "column_break"


ModelBPlanReviewFlowItem: TypeAlias = Annotated[
    ModelBPlanReviewContentFlowItem
    | ModelBPlanReviewPageBreakFlowItem
    | ModelBPlanReviewColumnBreakFlowItem,
    Field(discriminator="kind"),
]


class SetModelBPageLayoutPatchOperation(_StrictModel):
    op: Literal["set_page_layout"]
    page_layout: PageLayoutIntent


class ReplaceModelBStylesPatchOperation(_StrictModel):
    op: Literal["replace_styles"]
    styles: tuple[ModelBPlanReviewStyle, ...] = Field(max_length=_MAX_STYLES)

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> Self:
        _reject_duplicate_ids((style.id for style in self.styles), "style")
        return self


class ReplaceModelBFlowPatchOperation(_StrictModel):
    op: Literal["replace_flow"]
    flow: tuple[ModelBPlanReviewFlowItem, ...] = Field(
        min_length=1,
        max_length=_MAX_FLOW_ITEMS,
    )

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> Self:
        _reject_duplicate_ids((item.id for item in self.flow), "flow item")
        return self


class SetModelBOfficialSpecRefsPatchOperation(_StrictModel):
    op: Literal["set_official_spec_refs"]
    official_spec_refs: tuple[_BoundedText, ...] = Field(
        min_length=1,
        max_length=_MAX_SPEC_REFS,
    )

    @model_validator(mode="after")
    def reject_duplicate_refs(self) -> Self:
        _reject_duplicate_ids(self.official_spec_refs, "official spec ref")
        return self


ModelBPlanReviewPatchOperation: TypeAlias = Annotated[
    SetModelBPageLayoutPatchOperation
    | ReplaceModelBStylesPatchOperation
    | ReplaceModelBFlowPatchOperation
    | SetModelBOfficialSpecRefsPatchOperation,
    Field(discriminator="op"),
]


class ModelBPlanReviewPatchRequest(_StrictModel):
    """Typed design-only patch request with no paths or arbitrary JSON pointers."""

    schema_version: Literal["model-b-plan-review-patch/1.3"] = "model-b-plan-review-patch/1.3"
    expected_draft_revision: int = Field(ge=1)
    expected_source_candidate_contract_sha256: Sha256
    expected_grounding_evidence_contract_sha256: Sha256
    expected_reviewed_hwp_document_plan_contract_sha256: Sha256
    operations: tuple[ModelBPlanReviewPatchOperation, ...] = Field(
        min_length=1,
        max_length=_MAX_PATCH_OPERATIONS,
    )

    @model_validator(mode="after")
    def validate_request_bounds(self) -> Self:
        operation_names = [operation.op for operation in self.operations]
        if len(set(operation_names)) != len(operation_names):
            raise ValueError("duplicate Model B plan patch operation")
        if _count_text_characters(self.model_dump(mode="json")) > _MAX_PATCH_TEXT_CHARS:
            raise ValueError("Model B plan patch text exceeds total size limit")
        return self


class ModelBPlanReviewCompletionRequest(_StrictModel):
    """CAS request to complete one exact Model B review revision."""

    schema_version: Literal["model-b-plan-review-completion/1.0"] = (
        "model-b-plan-review-completion/1.0"
    )
    expected_draft_revision: int = Field(ge=1)
    expected_source_candidate_contract_sha256: Sha256
    expected_grounding_evidence_contract_sha256: Sha256
    expected_reviewed_hwp_document_plan_contract_sha256: Sha256


class ModelBPlanReviewDraft(_NonEligibleBoundary):
    """Untrusted, non-promotable immutable revision of one Model B plan review."""

    schema_version: Literal[
        "model-b-plan-review-draft/1.2",
        "model-b-plan-review-draft/1.3",
    ] = "model-b-plan-review-draft/1.2"
    status: Literal["in_progress", "complete"] = "in_progress"
    draft_revision: int = Field(ge=1)
    reviewer_label: _BoundedText
    updated_at: Annotated[str, Field(min_length=1, max_length=64)]
    document_id: _BoundedText
    source_candidate_contract_sha256: Sha256
    content_ir: ContentIR
    content_ir_contract_sha256: Sha256
    base_hwp_document_plan: HwpDocumentPlan
    base_hwp_document_plan_contract_sha256: Sha256
    reviewed_hwp_document_plan: HwpDocumentPlan
    reviewed_hwp_document_plan_contract_sha256: Sha256
    grounding_evidence: ModelBPlanGroundingEvidence
    grounding_evidence_contract_sha256: Sha256
    model_b_inference_binding: ModelBPlanReviewInferenceBinding | None = None

    @model_validator(mode="after")
    def validate_review_boundary(self) -> Self:
        _require_timezone(self.updated_at)
        expected_candidate_sha256 = model_b_plan_review_candidate_contract_sha256(
            self.content_ir,
            self.base_hwp_document_plan,
            document_id=self.document_id,
        )
        if expected_candidate_sha256 != self.source_candidate_contract_sha256:
            raise ValueError("Model B source candidate contract digest mismatch")
        if contract_sha256(self.grounding_evidence) != self.grounding_evidence_contract_sha256:
            raise ValueError("Model B grounding evidence contract digest mismatch")
        if (
            self.schema_version == "model-b-plan-review-draft/1.2"
            and self.model_b_inference_binding is not None
        ):
            raise ValueError("legacy Model B review draft cannot contain inference binding")
        if (
            self.schema_version == "model-b-plan-review-draft/1.3"
            and self.model_b_inference_binding is None
        ):
            raise ValueError("Model B review draft 1.3 requires inference binding")
        if (
            self.model_b_inference_binding is not None
            and self.model_b_inference_binding.inferred_hwp_document_plan_contract_sha256
            != self.base_hwp_document_plan_contract_sha256
        ):
            raise ValueError("Model B inference plan binding does not match review base plan")
        if contract_sha256(self.content_ir) != self.content_ir_contract_sha256:
            raise ValueError("ContentIR contract digest mismatch")
        if (
            contract_sha256(self.base_hwp_document_plan)
            != self.base_hwp_document_plan_contract_sha256
        ):
            raise ValueError("base HwpDocumentPlan contract digest mismatch")
        if (
            contract_sha256(self.reviewed_hwp_document_plan)
            != self.reviewed_hwp_document_plan_contract_sha256
        ):
            raise ValueError("reviewed HwpDocumentPlan contract digest mismatch")
        try:
            self.base_hwp_document_plan.assert_content_integrity(self.content_ir)
            self.reviewed_hwp_document_plan.assert_content_integrity(self.content_ir)
        except ValueError as exc:
            raise ValueError(f"Model B plan review lineage is invalid: {exc}") from exc
        _require_plan_identity_preserved(
            self.base_hwp_document_plan,
            self.reviewed_hwp_document_plan,
        )
        _require_content_refs_preserved(
            self.base_hwp_document_plan,
            self.reviewed_hwp_document_plan,
        )
        _require_plan_bounds(self.base_hwp_document_plan)
        _require_plan_bounds(self.reviewed_hwp_document_plan)
        if (
            self.grounding_evidence.capability_profile_id
            != self.base_hwp_document_plan.capability_profile_id
        ):
            raise ValueError("grounding evidence capability profile does not match the base plan")
        evidence_refs = {
            item.official_spec_ref for item in self.grounding_evidence.official_spec_refs
        }
        unknown_refs = sorted(
            set(self.reviewed_hwp_document_plan.official_spec_refs) - evidence_refs
        )
        if unknown_refs:
            raise ValueError(
                "reviewed HwpDocumentPlan contains ungrounded official spec refs: "
                + ", ".join(unknown_refs)
            )
        return self


def load_model_b_plan_review_patch_request(
    stream: BinaryIO,
    *,
    max_bytes: int = _MAX_PATCH_REQUEST_BYTES,
) -> ModelBPlanReviewPatchRequest:
    """Read one strict patch request without accepting unbounded input."""
    try:
        payload = _read_stream_bounded(
            stream,
            "Model B plan review patch request",
            max_bytes=max_bytes,
        )
        _assert_strict_json(payload, "Model B plan review patch request")
    except ModelBPlanReviewVerificationError as exc:
        raise ModelBPlanReviewPatchError(str(exc)) from exc
    try:
        return ModelBPlanReviewPatchRequest.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ModelBPlanReviewPatchError(
            "Model B plan review patch request is not valid strict JSON"
        ) from exc


def load_model_b_plan_review_completion_request(
    stream: BinaryIO,
    *,
    max_bytes: int = _MAX_COMPLETION_REQUEST_BYTES,
) -> ModelBPlanReviewCompletionRequest:
    """Read one strict completion request without accepting unbounded input."""
    try:
        payload = _read_stream_bounded(
            stream,
            "Model B plan review completion request",
            max_bytes=max_bytes,
        )
        _assert_strict_json(payload, "Model B plan review completion request")
    except ModelBPlanReviewVerificationError as exc:
        raise ModelBPlanReviewCompletionError(str(exc)) from exc
    try:
        return ModelBPlanReviewCompletionRequest.model_validate_json(
            payload,
            strict=True,
        )
    except ValidationError as exc:
        raise ModelBPlanReviewCompletionError(
            "Model B plan review completion request is not valid strict JSON"
        ) from exc


class ModelBPlanReviewDraftView(_NonEligibleBoundary):
    """Bounded design-only view of a source-verified Model B review revision."""

    schema_version: Literal["model-b-plan-review-view/1.0"] = "model-b-plan-review-view/1.0"
    status: Literal["in_progress", "complete"] = "in_progress"
    document_id: _BoundedText
    draft_revision: int = Field(ge=1)
    updated_at: Annotated[str, Field(min_length=1, max_length=64)]
    source_candidate_contract_sha256: Sha256
    grounding_evidence_contract_sha256: Sha256
    reviewed_hwp_document_plan_contract_sha256: Sha256
    available_official_spec_refs: tuple[_BoundedText, ...] = Field(
        min_length=1,
        max_length=_MAX_SPEC_REFS,
    )
    reviewed_hwp_document_plan: HwpDocumentPlan

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        _require_timezone(self.updated_at)
        _reject_duplicate_ids(
            self.available_official_spec_refs,
            "available official spec ref",
        )
        if (
            contract_sha256(self.reviewed_hwp_document_plan)
            != self.reviewed_hwp_document_plan_contract_sha256
        ):
            raise ValueError("reviewed HwpDocumentPlan view digest mismatch")
        if not set(self.reviewed_hwp_document_plan.official_spec_refs).issubset(
            self.available_official_spec_refs
        ):
            raise ValueError("review view contains unavailable official spec refs")
        _require_plan_bounds(self.reviewed_hwp_document_plan)
        return self


def start_model_b_plan_review_draft(
    content_ir: ContentIR,
    base_plan: HwpDocumentPlan,
    *,
    document_id: str,
    reviewer_label: str,
    grounding_evidence: ModelBPlanGroundingEvidence,
    inference_binding: ModelBPlanReviewInferenceBinding | None = None,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Start a noneligible plan-only review without altering ContentIR."""
    if not isinstance(content_ir, ContentIR):
        raise TypeError("content_ir must be a ContentIR")
    if not isinstance(base_plan, HwpDocumentPlan):
        raise TypeError("base_plan must be a HwpDocumentPlan")
    try:
        base_plan.assert_content_integrity(content_ir)
    except ValueError as exc:
        raise ModelBPlanReviewPatchError(f"base plan lineage is invalid: {exc}") from exc
    timestamp = _review_timestamp(updated_at)
    try:
        return ModelBPlanReviewDraft(
            schema_version=(
                "model-b-plan-review-draft/1.3"
                if inference_binding is not None
                else "model-b-plan-review-draft/1.2"
            ),
            draft_revision=1,
            reviewer_label=reviewer_label,
            updated_at=timestamp.isoformat(),
            document_id=document_id,
            source_candidate_contract_sha256=(
                model_b_plan_review_candidate_contract_sha256(
                    content_ir,
                    base_plan,
                    document_id=document_id,
                )
            ),
            content_ir=content_ir,
            content_ir_contract_sha256=contract_sha256(content_ir),
            base_hwp_document_plan=base_plan,
            base_hwp_document_plan_contract_sha256=contract_sha256(base_plan),
            reviewed_hwp_document_plan=base_plan,
            reviewed_hwp_document_plan_contract_sha256=contract_sha256(base_plan),
            grounding_evidence=grounding_evidence,
            grounding_evidence_contract_sha256=contract_sha256(grounding_evidence),
            model_b_inference_binding=inference_binding,
        )
    except ValidationError as exc:
        raise ModelBPlanReviewPatchError("Model B plan review draft is not valid") from exc


def apply_model_b_plan_review_patch(
    current: ModelBPlanReviewDraft,
    request: ModelBPlanReviewPatchRequest,
    *,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Return a new design-only revision; the current revision is never mutated."""
    if not isinstance(current, ModelBPlanReviewDraft):
        raise TypeError("current must be a ModelBPlanReviewDraft")
    if not isinstance(request, ModelBPlanReviewPatchRequest):
        raise TypeError("request must be a ModelBPlanReviewPatchRequest")
    try:
        current = ModelBPlanReviewDraft.model_validate(
            current.model_dump(mode="python"),
            strict=True,
        )
        request = ModelBPlanReviewPatchRequest.model_validate(
            request.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ModelBPlanReviewPatchError("Model B plan review patch input is not valid") from exc
    if current.status != "in_progress":
        raise ModelBPlanReviewPatchError("completed Model B plan review draft cannot be patched")
    if current.draft_revision != request.expected_draft_revision:
        raise ModelBPlanReviewPatchError(
            "stale Model B plan review revision: expected "
            f"{request.expected_draft_revision}, found {current.draft_revision}"
        )
    if (
        current.source_candidate_contract_sha256
        != request.expected_source_candidate_contract_sha256
    ):
        raise ModelBPlanReviewPatchError("Model B plan review source candidate binding changed")
    if (
        current.grounding_evidence_contract_sha256
        != request.expected_grounding_evidence_contract_sha256
    ):
        raise ModelBPlanReviewPatchError("Model B plan review grounding evidence binding changed")
    if (
        current.reviewed_hwp_document_plan_contract_sha256
        != request.expected_reviewed_hwp_document_plan_contract_sha256
    ):
        raise ModelBPlanReviewPatchError(
            "Model B plan review current reviewed plan binding changed"
        )

    plan_value = current.reviewed_hwp_document_plan.model_dump(mode="python")
    for operation in request.operations:
        if isinstance(operation, SetModelBPageLayoutPatchOperation):
            plan_value["page_layout"] = operation.page_layout.model_dump(mode="python")
        elif isinstance(operation, ReplaceModelBStylesPatchOperation):
            plan_value["styles"] = tuple(
                style.model_dump(mode="python") for style in operation.styles
            )
        elif isinstance(operation, ReplaceModelBFlowPatchOperation):
            plan_value["flow"] = _rebuild_flow_without_content_refs(
                current.reviewed_hwp_document_plan,
                operation,
            )
        else:
            plan_value["official_spec_refs"] = operation.official_spec_refs

    try:
        reviewed_plan = HwpDocumentPlan.model_validate(plan_value, strict=True)
        reviewed_plan.assert_content_integrity(current.content_ir)
    except (ValidationError, ValueError) as exc:
        raise ModelBPlanReviewPatchError(f"patched HwpDocumentPlan is invalid: {exc}") from exc

    timestamp = _review_timestamp(updated_at)
    try:
        return ModelBPlanReviewDraft(
            schema_version=current.schema_version,
            status="in_progress",
            draft_revision=current.draft_revision + 1,
            reviewer_label=current.reviewer_label,
            updated_at=timestamp.isoformat(),
            document_id=current.document_id,
            source_candidate_contract_sha256=(current.source_candidate_contract_sha256),
            content_ir=current.content_ir,
            content_ir_contract_sha256=current.content_ir_contract_sha256,
            base_hwp_document_plan=current.base_hwp_document_plan,
            base_hwp_document_plan_contract_sha256=(current.base_hwp_document_plan_contract_sha256),
            reviewed_hwp_document_plan=reviewed_plan,
            reviewed_hwp_document_plan_contract_sha256=contract_sha256(reviewed_plan),
            grounding_evidence=current.grounding_evidence,
            grounding_evidence_contract_sha256=(current.grounding_evidence_contract_sha256),
            model_b_inference_binding=current.model_b_inference_binding,
        )
    except ValidationError as exc:
        raise ModelBPlanReviewPatchError(
            f"patched Model B plan review draft is invalid: {exc}"
        ) from exc


def complete_model_b_plan_review_draft(
    current: ModelBPlanReviewDraft,
    request: ModelBPlanReviewCompletionRequest,
    *,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Seal one exact revision without promoting or changing reviewed artifacts."""
    if not isinstance(current, ModelBPlanReviewDraft):
        raise TypeError("current must be a ModelBPlanReviewDraft")
    if not isinstance(request, ModelBPlanReviewCompletionRequest):
        raise TypeError("request must be a ModelBPlanReviewCompletionRequest")
    try:
        current = ModelBPlanReviewDraft.model_validate(
            current.model_dump(mode="python"),
            strict=True,
        )
        request = ModelBPlanReviewCompletionRequest.model_validate(
            request.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ModelBPlanReviewCompletionError(
            "Model B plan review completion input is not valid"
        ) from exc
    if current.status != "in_progress":
        raise ModelBPlanReviewCompletionError("Model B plan review draft is already complete")
    if current.draft_revision != request.expected_draft_revision:
        raise ModelBPlanReviewCompletionError(
            "stale Model B plan review revision: expected "
            f"{request.expected_draft_revision}, found {current.draft_revision}"
        )
    if (
        current.source_candidate_contract_sha256
        != request.expected_source_candidate_contract_sha256
    ):
        raise ModelBPlanReviewCompletionError(
            "Model B plan review source candidate binding changed"
        )
    if (
        current.grounding_evidence_contract_sha256
        != request.expected_grounding_evidence_contract_sha256
    ):
        raise ModelBPlanReviewCompletionError(
            "Model B plan review grounding evidence binding changed"
        )
    if (
        current.reviewed_hwp_document_plan_contract_sha256
        != request.expected_reviewed_hwp_document_plan_contract_sha256
    ):
        raise ModelBPlanReviewCompletionError(
            "Model B plan review current reviewed plan binding changed"
        )

    timestamp = _review_timestamp(updated_at)
    value = current.model_dump(mode="python")
    value.update(
        status="complete",
        draft_revision=current.draft_revision + 1,
        updated_at=timestamp.isoformat(),
    )
    try:
        return ModelBPlanReviewDraft.model_validate(value, strict=True)
    except ValidationError as exc:
        raise ModelBPlanReviewCompletionError(
            "completed Model B plan review draft is not valid"
        ) from exc


def model_b_plan_review_candidate_contract_sha256(
    content_ir: ContentIR,
    base_plan: HwpDocumentPlan,
    *,
    document_id: str,
) -> str:
    """Return the canonical identity of one immutable Model B source candidate."""
    return contract_sha256(
        ModelBPlanReviewCandidateBinding(
            document_id=document_id,
            content_ir_contract_sha256=contract_sha256(content_ir),
            base_hwp_document_plan_contract_sha256=contract_sha256(base_plan),
        )
    )


def load_model_b_plan_review_draft(path: Path | str) -> ModelBPlanReviewDraft:
    """Load one strict, bounded review revision without trusting its source bindings."""
    return _load_strict_contract(
        path,
        ModelBPlanReviewDraft,
        "Model B plan review draft",
        max_bytes=_MAX_REVIEW_DRAFT_BYTES,
    )


def verify_model_b_plan_review_draft(
    draft: ModelBPlanReviewDraft,
    *,
    content_ir: ContentIR,
    base_plan: HwpDocumentPlan,
    grounding_evidence: ModelBPlanGroundingEvidence,
) -> ModelBPlanReviewDraft:
    """Bind a draft to the supplied immutable candidate and grounding contracts."""
    if not isinstance(draft, ModelBPlanReviewDraft):
        raise TypeError("draft must be a ModelBPlanReviewDraft")
    if not isinstance(content_ir, ContentIR):
        raise TypeError("content_ir must be a ContentIR")
    if not isinstance(base_plan, HwpDocumentPlan):
        raise TypeError("base_plan must be a HwpDocumentPlan")
    if not isinstance(grounding_evidence, ModelBPlanGroundingEvidence):
        raise TypeError("grounding_evidence must be ModelBPlanGroundingEvidence")
    try:
        verified = ModelBPlanReviewDraft.model_validate(
            draft.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ModelBPlanReviewVerificationError(
            "Model B plan review draft contract is not valid"
        ) from exc

    expected_candidate_sha256 = model_b_plan_review_candidate_contract_sha256(
        content_ir,
        base_plan,
        document_id=verified.document_id,
    )
    if (
        verified.source_candidate_contract_sha256 != expected_candidate_sha256
        or verified.content_ir != content_ir
        or verified.base_hwp_document_plan != base_plan
    ):
        raise ModelBPlanReviewVerificationError(
            "Model B plan review source candidate binding does not match"
        )
    if (
        verified.grounding_evidence_contract_sha256 != contract_sha256(grounding_evidence)
        or verified.grounding_evidence != grounding_evidence
    ):
        raise ModelBPlanReviewVerificationError(
            "Model B plan review grounding evidence binding does not match"
        )
    return verified


def build_model_b_plan_review_draft_view(
    draft: ModelBPlanReviewDraft,
) -> ModelBPlanReviewDraftView:
    """Build a bounded design-only projection after source verification."""
    if not isinstance(draft, ModelBPlanReviewDraft):
        raise TypeError("draft must be a ModelBPlanReviewDraft")
    return ModelBPlanReviewDraftView(
        status=draft.status,
        document_id=draft.document_id,
        draft_revision=draft.draft_revision,
        updated_at=draft.updated_at,
        source_candidate_contract_sha256=draft.source_candidate_contract_sha256,
        grounding_evidence_contract_sha256=(draft.grounding_evidence_contract_sha256),
        reviewed_hwp_document_plan_contract_sha256=(
            draft.reviewed_hwp_document_plan_contract_sha256
        ),
        available_official_spec_refs=tuple(
            item.official_spec_ref for item in draft.grounding_evidence.official_spec_refs
        ),
        reviewed_hwp_document_plan=draft.reviewed_hwp_document_plan,
    )


def start_persisted_model_b_plan_review_draft(
    content_ir_path: Path | str,
    base_plan_path: Path | str,
    grounding_evidence_path: Path | str,
    *,
    document_id: str,
    reviewer_label: str,
    output_path: Path | str,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Create revision 1 atomically from three strict source artifacts."""
    content_ir, base_plan, grounding_evidence = _load_model_b_review_sources(
        content_ir_path,
        base_plan_path,
        grounding_evidence_path,
    )
    draft = start_model_b_plan_review_draft(
        content_ir,
        base_plan,
        document_id=document_id,
        reviewer_label=reviewer_label,
        grounding_evidence=grounding_evidence,
        updated_at=updated_at,
    )
    verified = verify_model_b_plan_review_draft(
        draft,
        content_ir=content_ir,
        base_plan=base_plan,
        grounding_evidence=grounding_evidence,
    )
    build_model_b_plan_review_draft_view(verified)
    _atomic_create_model_b_review_draft(output_path, verified)
    return verified


def reopen_model_b_plan_review_draft(
    draft_path: Path | str,
    content_ir_path: Path | str,
    base_plan_path: Path | str,
    grounding_evidence_path: Path | str,
) -> ModelBPlanReviewDraft:
    """Reopen and rebind a saved immutable revision to all source artifacts."""
    content_ir, base_plan, grounding_evidence = _load_model_b_review_sources(
        content_ir_path,
        base_plan_path,
        grounding_evidence_path,
    )
    draft = load_model_b_plan_review_draft(draft_path)
    return verify_model_b_plan_review_draft(
        draft,
        content_ir=content_ir,
        base_plan=base_plan,
        grounding_evidence=grounding_evidence,
    )


def reopen_model_b_plan_review_draft_view(
    draft_path: Path | str,
    content_ir_path: Path | str,
    base_plan_path: Path | str,
    grounding_evidence_path: Path | str,
) -> ModelBPlanReviewDraftView:
    """Reopen against all source artifacts before exposing a review view."""
    verified = reopen_model_b_plan_review_draft(
        draft_path,
        content_ir_path,
        base_plan_path,
        grounding_evidence_path,
    )
    return build_model_b_plan_review_draft_view(verified)


def apply_persisted_model_b_plan_review_patch(
    draft_path: Path | str,
    content_ir_path: Path | str,
    base_plan_path: Path | str,
    grounding_evidence_path: Path | str,
    request: ModelBPlanReviewPatchRequest,
    output_path: Path | str,
    *,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Verify, patch, preflight, and atomically save one new immutable revision."""
    content_ir, base_plan, grounding_evidence = _load_model_b_review_sources(
        content_ir_path,
        base_plan_path,
        grounding_evidence_path,
    )
    current = verify_model_b_plan_review_draft(
        load_model_b_plan_review_draft(draft_path),
        content_ir=content_ir,
        base_plan=base_plan,
        grounding_evidence=grounding_evidence,
    )
    patched = apply_model_b_plan_review_patch(
        current,
        request,
        updated_at=updated_at,
    )
    verified = verify_model_b_plan_review_draft(
        patched,
        content_ir=content_ir,
        base_plan=base_plan,
        grounding_evidence=grounding_evidence,
    )
    build_model_b_plan_review_draft_view(verified)
    _atomic_create_model_b_review_draft(output_path, verified)
    return verified


def complete_persisted_model_b_plan_review_draft(
    draft_path: Path | str,
    content_ir_path: Path | str,
    base_plan_path: Path | str,
    grounding_evidence_path: Path | str,
    request: ModelBPlanReviewCompletionRequest,
    output_path: Path | str,
    *,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Verify, seal, preflight, and atomically save one complete revision."""
    content_ir, base_plan, grounding_evidence = _load_model_b_review_sources(
        content_ir_path,
        base_plan_path,
        grounding_evidence_path,
    )
    current = verify_model_b_plan_review_draft(
        load_model_b_plan_review_draft(draft_path),
        content_ir=content_ir,
        base_plan=base_plan,
        grounding_evidence=grounding_evidence,
    )
    completed = complete_model_b_plan_review_draft(
        current,
        request,
        updated_at=updated_at,
    )
    verified = verify_model_b_plan_review_draft(
        completed,
        content_ir=content_ir,
        base_plan=base_plan,
        grounding_evidence=grounding_evidence,
    )
    build_model_b_plan_review_draft_view(verified)
    _atomic_create_model_b_review_draft(output_path, verified)
    return verified


def _rebuild_flow_without_content_refs(
    current_plan: HwpDocumentPlan,
    operation: ReplaceModelBFlowPatchOperation,
) -> tuple[dict[str, object], ...]:
    content_refs_by_item_id = {
        item.id: item.content_ref for item in current_plan.flow if isinstance(item, ContentPlanItem)
    }
    requested_content_ids = {
        item.id for item in operation.flow if isinstance(item, ModelBPlanReviewContentFlowItem)
    }
    if requested_content_ids != set(content_refs_by_item_id):
        raise ModelBPlanReviewPatchError(
            "replace_flow must preserve every content flow item id exactly"
        )

    rebuilt: list[dict[str, object]] = []
    for item in operation.flow:
        value = item.model_dump(mode="python")
        if isinstance(item, ModelBPlanReviewContentFlowItem):
            value["content_ref"] = content_refs_by_item_id[item.id]
        rebuilt.append(value)
    return tuple(rebuilt)


def _require_plan_identity_preserved(
    base: HwpDocumentPlan,
    reviewed: HwpDocumentPlan,
) -> None:
    immutable_fields = (
        "schema_version",
        "id",
        "content_ir_id",
        "content_ir_revision",
        "content_ir_sha256",
        "capability_profile_id",
        "design_profile_id",
    )
    changed = [
        field_name
        for field_name in immutable_fields
        if getattr(base, field_name) != getattr(reviewed, field_name)
    ]
    if changed:
        raise ValueError("reviewed HwpDocumentPlan changed immutable fields: " + ", ".join(changed))


def _require_content_refs_preserved(
    base: HwpDocumentPlan,
    reviewed: HwpDocumentPlan,
) -> None:
    def bindings(plan: HwpDocumentPlan) -> dict[str, str]:
        return {
            item.id: item.content_ref for item in plan.flow if isinstance(item, ContentPlanItem)
        }

    if bindings(base) != bindings(reviewed):
        raise ValueError("reviewed HwpDocumentPlan content_ref bindings must be preserved")


def _require_plan_bounds(plan: HwpDocumentPlan) -> None:
    if len(plan.styles) > _MAX_STYLES:
        raise ValueError("Model B plan styles exceed review size limit")
    if len(plan.flow) > _MAX_FLOW_ITEMS:
        raise ValueError("Model B plan flow exceeds review size limit")
    if len(plan.official_spec_refs) > _MAX_SPEC_REFS:
        raise ValueError("Model B plan official spec refs exceed review size limit")
    if _count_text_characters(plan.model_dump(mode="json")) > _MAX_PATCH_TEXT_CHARS:
        raise ValueError("Model B plan text exceeds review size limit")


def _load_model_b_review_sources(
    content_ir_path: Path | str,
    base_plan_path: Path | str,
    grounding_evidence_path: Path | str,
) -> tuple[ContentIR, HwpDocumentPlan, ModelBPlanGroundingEvidence]:
    content_ir = _load_strict_contract(content_ir_path, ContentIR, "ContentIR")
    base_plan = _load_strict_contract(
        base_plan_path,
        HwpDocumentPlan,
        "base HwpDocumentPlan",
    )
    grounding_evidence = _load_strict_contract(
        grounding_evidence_path,
        ModelBPlanGroundingEvidence,
        "Model B grounding evidence",
    )
    return content_ir, base_plan, grounding_evidence


def _load_strict_contract(
    path: Path | str,
    model_type: type[ContractT],
    label: str,
    *,
    max_bytes: int = _MAX_SOURCE_ARTIFACT_BYTES,
) -> ContractT:
    payload = _read_bounded_regular_file(path, label, max_bytes=max_bytes)
    _assert_strict_json(payload, label)
    try:
        return model_type.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ModelBPlanReviewVerificationError(f"{label} is not valid strict JSON") from exc


def _read_bounded_regular_file(
    path: Path | str,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    source = Path(path)
    descriptor: int | None = None
    try:
        path_metadata = source.lstat()
        _require_regular_non_reparse_file(path_metadata, label)
        if path_metadata.st_size > max_bytes:
            raise ModelBPlanReviewVerificationError(f"{label} exceeds size limit")

        flags = os.O_RDONLY
        for flag_name in (
            "O_BINARY",
            "O_CLOEXEC",
            "O_NOINHERIT",
            "O_NOFOLLOW",
            "O_NONBLOCK",
        ):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(source, flags)
        opened_metadata = os.fstat(descriptor)
        _require_regular_non_reparse_file(opened_metadata, label)
        if opened_metadata.st_size > max_bytes:
            raise ModelBPlanReviewVerificationError(f"{label} exceeds size limit")
        _require_same_file_snapshot(path_metadata, opened_metadata, label)

        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = _read_stream_bounded(stream, label, max_bytes=max_bytes)
            read_metadata = os.fstat(stream.fileno())
        _require_same_file_snapshot(opened_metadata, read_metadata, label)

        final_path_metadata = source.lstat()
        _require_regular_non_reparse_file(final_path_metadata, label)
        _require_same_file_snapshot(read_metadata, final_path_metadata, label)
    except ModelBPlanReviewVerificationError:
        raise
    except OSError as exc:
        raise ModelBPlanReviewVerificationError(f"{label} is missing or unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload


def _read_stream_bounded(
    stream: BinaryIO,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    payload = bytearray()
    while True:
        remaining = max_bytes + 1 - len(payload)
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise ModelBPlanReviewVerificationError(f"{label} exceeds size limit")
    return bytes(payload)


def _assert_strict_json(payload: bytes, label: str) -> None:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite number: {value}")

    try:
        json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ModelBPlanReviewVerificationError(f"{label} is not valid strict JSON") from exc


def _require_regular_non_reparse_file(metadata: os.stat_result, label: str) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(file_attributes & reparse_attribute)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ModelBPlanReviewVerificationError(f"{label} must be a regular non-symlink file")


def _require_same_file_snapshot(
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    unchanged = (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )
    if not unchanged:
        raise ModelBPlanReviewVerificationError(f"{label} changed while it was read")


def _atomic_create_model_b_review_draft(
    output_path: Path | str,
    draft: ModelBPlanReviewDraft,
) -> None:
    requested = Path(output_path)
    if requested.name in {"", ".", ".."}:
        raise ValueError("Model B review output must name a file")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Model B review output already exists")
    payload = (
        json.dumps(
            draft.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_REVIEW_DRAFT_BYTES:
        raise ModelBPlanReviewVerificationError("Model B plan review draft exceeds size limit")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            publish_file_create_only(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError("Model B review output already exists") from exc
        except SafeArtifactIOError as exc:
            raise ModelBPlanReviewVerificationError(
                "Model B review output changed during publication"
            ) from exc
        committed = True
    finally:
        if not committed:
            temporary.unlink(missing_ok=True)


def _review_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("updated_at must include a timezone")
    return timestamp


def _require_timezone(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("updated_at must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("updated_at must include a timezone")


def _reject_duplicate_ids(values: Iterable[str], label: str) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"duplicate {label} ids")


def _count_text_characters(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(
            _count_text_characters(key) + _count_text_characters(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_text_characters(item) for item in value)
    return 0
