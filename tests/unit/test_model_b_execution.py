from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import cast

import pytest

import scan2hwpx.model_b.execution as execution_module
from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    VerifiedCandidateToModelBHandoff,
)
from scan2hwpx.knowledge.hancom import HancomChunk
from scan2hwpx.model_b import (
    BuiltModelBInferenceRequest,
    BuiltModelBInferenceResult,
    ModelBInferenceIssueCode,
    ModelBInferenceReviewError,
    ModelBModelArtifact,
    build_model_b_inference_request,
    execute_model_b_plan,
    start_model_b_plan_review_from_inference,
)
from tests.unit.test_model_b_inference import (
    _json_bytes,
    _response_plan,
    _verified_handoff,
)


def _model() -> ModelBModelArtifact:
    return ModelBModelArtifact(
        model_id="fixture-model-b",
        model_revision="revision-1",
        artifact_sha256="9" * 64,
    )


class _RecordingProvider:
    def __init__(self, raw_response: bytes | None = None) -> None:
        self.raw_response = raw_response
        self.calls: list[BuiltModelBInferenceRequest] = []
        self.returned: list[bytes] = []

    def generate(self, request: BuiltModelBInferenceRequest, /) -> bytes:
        self.calls.append(request)
        raw_response = self.raw_response
        if raw_response is None:
            raw_response = _json_bytes(_response_plan(request).model_dump(mode="json"))
        self.returned.append(raw_response)
        return raw_response


def _execute(
    handoff: VerifiedCandidateToModelBHandoff,
    capability_bytes: bytes,
    chunk: HancomChunk,
    provider: _RecordingProvider,
    *,
    model: ModelBModelArtifact | None = None,
    max_response_bytes: int = 16 * 1024 * 1024,
) -> BuiltModelBInferenceResult:
    return execute_model_b_plan(
        handoff,
        model or _model(),
        capability_bytes,
        (chunk,),
        provider,
        max_response_bytes=max_response_bytes,
    )


def test_executes_provider_once_and_preserves_reproducible_artifacts() -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    provider = _RecordingProvider()

    result = _execute(handoff, capability_bytes, chunk, provider)

    assert len(provider.calls) == 1
    assert result.request is provider.calls[0]
    assert result.artifact.outcome == "valid_unverified"
    assert result.raw_response_bytes == provider.returned[0]
    assert hashlib.sha256(result.request.raw_request_bytes).hexdigest() == (
        result.request.raw_request_sha256
    )
    assert hashlib.sha256(result.raw_response_bytes).hexdigest() == (
        result.artifact.raw_response_sha256
    )
    assert hashlib.sha256(result.result_bytes).hexdigest() == result.result_sha256


def test_valid_execution_result_starts_inference_bound_plan_review() -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    result = _execute(handoff, capability_bytes, chunk, _RecordingProvider())

    draft = start_model_b_plan_review_from_inference(result)

    plan = result.artifact.hwp_document_plan
    assert plan is not None
    assert draft.schema_version == "model-b-plan-review-draft/1.3"
    assert draft.draft_revision == 1
    assert draft.base_hwp_document_plan == plan
    assert draft.reviewed_hwp_document_plan == plan
    binding = draft.model_b_inference_binding
    assert binding is not None
    assert binding.handoff_artifact_sha256 == handoff.artifact_sha256
    assert binding.request_sha256 == result.request.raw_request_sha256
    assert binding.result_sha256 == result.result_sha256
    assert binding.raw_response_sha256 == result.artifact.raw_response_sha256
    assert binding.model_id == result.artifact.model.model_id
    assert binding.model_revision == result.artifact.model.model_revision
    assert binding.model_artifact_sha256 == result.artifact.model.artifact_sha256
    assert binding.handoff_source_candidate_contract_sha256 == (
        handoff.envelope.model_b_plan_review_draft.source_candidate_contract_sha256
    )


def test_invalid_model_response_is_returned_as_the_existing_blocked_result() -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    raw_response = b'{"id":'
    provider = _RecordingProvider(raw_response)

    result = _execute(handoff, capability_bytes, chunk, provider)

    assert len(provider.calls) == 1
    assert result.raw_response_bytes == raw_response
    assert result.artifact.outcome == "blocked"
    assert result.artifact.hwp_document_plan is None
    assert tuple(issue.code for issue in result.artifact.issues) == (
        ModelBInferenceIssueCode.RESPONSE_MALFORMED_JSON,
    )
    with pytest.raises(ModelBInferenceReviewError, match="blocked"):
        start_model_b_plan_review_from_inference(result)


def test_tampered_execution_result_cannot_start_review() -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    result = _execute(handoff, capability_bytes, chunk, _RecordingProvider())
    object.__setattr__(result, "result_sha256", "0" * 64)

    with pytest.raises(ModelBInferenceReviewError, match="not valid"):
        start_model_b_plan_review_from_inference(result)


@pytest.mark.parametrize(
    "failure",
    ("handoff", "model", "capability", "chunk"),
)
def test_invalid_inference_inputs_are_rejected_before_provider_call(failure: str) -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    model = _model()
    if failure == "handoff":
        handoff = replace(handoff, artifact_sha256="0" * 64)
    elif failure == "model":
        model = model.model_copy(update={"artifact_sha256": "not-a-digest"})
    elif failure == "capability":
        capability_bytes += b" "
    else:
        chunk = replace(chunk, text=chunk.text + " tampered")
    provider = _RecordingProvider()

    with pytest.raises((TypeError, ValueError)):
        _execute(
            handoff,
            capability_bytes,
            chunk,
            provider,
            model=model,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "max_response_bytes",
    (0, -1, True, 16 * 1024 * 1024 + 1),
)
def test_invalid_response_limit_is_rejected_before_provider_call(
    max_response_bytes: object,
) -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    provider = _RecordingProvider()

    with pytest.raises(ValueError):
        _execute(
            handoff,
            capability_bytes,
            chunk,
            provider,
            max_response_bytes=cast(int, max_response_bytes),
        )

    assert provider.calls == []


def test_built_request_is_revalidated_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    request = build_model_b_inference_request(
        handoff,
        model=_model(),
        capability_profile_bytes=capability_bytes,
        retrieved_chunks=(chunk,),
    )
    object.__setattr__(request, "raw_request_sha256", "0" * 64)
    monkeypatch.setattr(
        execution_module,
        "build_model_b_inference_request",
        lambda *args, **kwargs: request,
    )
    provider = _RecordingProvider()

    with pytest.raises(ValueError, match="raw request digest"):
        _execute(handoff, capability_bytes, chunk, provider)

    assert provider.calls == []


def test_provider_response_must_be_exact_bytes() -> None:
    class BytesSubclass(bytes):
        pass

    handoff, capability_bytes, chunk = _verified_handoff()
    provider = _RecordingProvider(BytesSubclass(b"{}"))

    with pytest.raises(TypeError, match="exact bytes"):
        _execute(handoff, capability_bytes, chunk, provider)

    assert len(provider.calls) == 1


def test_configured_response_limit_is_preserved_in_blocked_result() -> None:
    handoff, capability_bytes, chunk = _verified_handoff()
    raw_response = b"{}" * 6
    provider = _RecordingProvider(raw_response)

    result = _execute(
        handoff,
        capability_bytes,
        chunk,
        provider,
        max_response_bytes=10,
    )

    assert len(provider.calls) == 1
    assert result.raw_response_bytes == raw_response
    assert result.artifact.response_byte_limit == 10
    assert tuple(issue.code for issue in result.artifact.issues) == (
        ModelBInferenceIssueCode.RESPONSE_TOO_LARGE,
    )
