from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from pydantic import ValidationError

from scan2hwpx.contracts import contract_sha256
from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    VerifiedCandidateToModelBHandoff,
)
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBPlanReviewDraft,
    ModelBPlanReviewInferenceBinding,
    start_model_b_plan_review_draft,
)
from scan2hwpx.knowledge.hancom import HancomChunk

from .inference import (
    DEFAULT_MAX_RESPONSE_BYTES,
    BuiltModelBInferenceRequest,
    BuiltModelBInferenceResult,
    ModelBModelArtifact,
    build_model_b_inference_request,
    validate_model_b_inference_response,
)


class ModelBResponseProvider(Protocol):
    """Provider-neutral boundary for one raw Model B response."""

    def generate(self, request: BuiltModelBInferenceRequest, /) -> bytes: ...


class ModelBInferenceReviewError(ValueError):
    """Raised when an inference result cannot seed the human review boundary."""


def execute_model_b_plan(
    verified_handoff: VerifiedCandidateToModelBHandoff,
    model: ModelBModelArtifact,
    capability_profile_bytes: bytes,
    retrieved_chunks: Sequence[HancomChunk],
    provider: ModelBResponseProvider,
    *,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> BuiltModelBInferenceResult:
    """Build, execute, and validate exactly one Model B planning attempt.

    Every local input and the resulting canonical request is validated before the
    provider is invoked. The raw response is neither repaired nor retried; invalid
    model output is returned through the inference validator's blocked result.
    """

    response_byte_limit = _require_response_byte_limit(max_response_bytes)
    request = build_model_b_inference_request(
        verified_handoff,
        model=model,
        capability_profile_bytes=capability_profile_bytes,
        retrieved_chunks=retrieved_chunks,
    )
    if type(request) is not BuiltModelBInferenceRequest:
        raise TypeError("Model B request builder must return an exact built request")
    BuiltModelBInferenceRequest.__post_init__(request)

    raw_response = provider.generate(request)
    if type(raw_response) is not bytes:
        raise TypeError("Model B provider response must be exact bytes")
    return validate_model_b_inference_response(
        request,
        raw_response,
        max_response_bytes=response_byte_limit,
    )


def start_model_b_plan_review_from_inference(
    result: BuiltModelBInferenceResult,
    *,
    updated_at: datetime | None = None,
) -> ModelBPlanReviewDraft:
    """Seed one typed human-review revision from an exact valid Model B result."""

    if type(result) is not BuiltModelBInferenceResult:
        raise TypeError("result must be an exact BuiltModelBInferenceResult")
    try:
        BuiltModelBInferenceResult.__post_init__(result)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ModelBInferenceReviewError("Model B inference result is not valid") from exc
    artifact = result.artifact
    plan = artifact.hwp_document_plan
    if artifact.outcome != "valid_unverified" or plan is None:
        raise ModelBInferenceReviewError("blocked Model B inference cannot seed review")

    request = result.request
    handoff = request.verified_handoff
    source = request.artifact.source
    if (
        artifact.request_sha256 != request.raw_request_sha256
        or artifact.source != source
        or artifact.model != request.artifact.model
        or source.handoff_artifact_sha256 != handoff.artifact_sha256
        or artifact.hwp_document_plan_contract_sha256 != contract_sha256(plan)
    ):
        raise ModelBInferenceReviewError("Model B inference lineage is inconsistent")

    seed = handoff.envelope.model_b_plan_review_draft
    if seed.status != "in_progress" or seed.model_b_inference_binding is not None:
        raise ModelBInferenceReviewError("Model B handoff review seed is not pristine")
    binding = ModelBPlanReviewInferenceBinding(
        handoff_artifact_sha256=handoff.artifact_sha256,
        request_sha256=request.raw_request_sha256,
        result_sha256=result.result_sha256,
        raw_response_sha256=artifact.raw_response_sha256,
        inferred_hwp_document_plan_contract_sha256=contract_sha256(plan),
        model_id=artifact.model.model_id,
        model_revision=artifact.model.model_revision,
        model_artifact_sha256=artifact.model.artifact_sha256,
        handoff_source_candidate_contract_sha256=(seed.source_candidate_contract_sha256),
    )
    try:
        return start_model_b_plan_review_draft(
            request.content_ir,
            plan,
            document_id=handoff.envelope.document_id,
            reviewer_label=seed.reviewer_label,
            grounding_evidence=request.grounding_evidence,
            inference_binding=binding,
            updated_at=updated_at,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ModelBInferenceReviewError(
            "validated Model B plan cannot seed the review draft"
        ) from exc


def _require_response_byte_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_response_bytes must be a positive integer")
    if value > DEFAULT_MAX_RESPONSE_BYTES:
        raise ValueError("max_response_bytes cannot exceed the hard response byte limit")
    return value
