from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from scan2hwpx.contracts import (
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    ObservationKind,
    OcrCandidate,
    contract_sha256,
)
from scan2hwpx.model_a.block_role_vector import (
    MAX_TARGET_BLOCKS_PER_SEGMENT,
    BuiltModelABlockRoleVectorRequest,
    ModelABlockExpandedLineDecision,
    ModelABlockRoleVectorIssueCode,
    ModelABlockRoleVectorPageIssueCode,
    ModelABlockRoleVectorPageResult,
    build_model_a_block_role_vector_requests,
    compile_model_a_block_role_vector_responses,
    execute_model_a_block_role_vector_page,
)
from scan2hwpx.model_a.classification_blocks import (
    ModelAClassificationBlockFormation,
    ModelAClassificationRoleSource,
)
from scan2hwpx.model_a.inference import ModelAPromptMessage
from tests.unit.test_model_a_role_vector import (
    _bbox,
    _evidence,
    _image,
    _json_bytes,
    _model,
)


def _text(observation_id: str, text: str, x: float, y: float) -> EvidenceObservation:
    return EvidenceObservation(
        id=observation_id,
        kind=ObservationKind.TEXT_LINE,
        bbox=_bbox(x, y),
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


def _evidence_with_observations(
    evidence_id: str,
    observations: tuple[EvidenceObservation, ...],
) -> EvidenceIR:
    base = _evidence()
    return EvidenceIR(
        id=evidence_id,
        source_document_sha256=base.source_document_sha256,
        sources=base.sources,
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=100,
                height=100,
                image_source_ref="page-image",
                observations=observations,
            ),
        ),
    )


def _numbered_question_evidence(count: int = 70) -> EvidenceIR:
    observations = tuple(
        _text(
            f"question-{index:03d}",
            f"{index + 1}. Question?",
            0.1 if index < 40 else 0.6,
            0.1 + (index % 40) * 0.018,
        )
        for index in range(count)
    )
    return _evidence_with_observations("evidence-many-blocks", observations)


def _answer_key_evidence() -> EvidenceIR:
    return _evidence_with_observations(
        "evidence-answer-key",
        (
            _text("answer-header", "[정답] ①②③", 0.10, 0.10),
            _text("answer-choice", "①", 0.12, 0.20),
            _text("answer-number", "1.", 0.12, 0.30),
            _text("answer-pair", "2. ①", 0.12, 0.40),
            _text("answer-pair-2", "11. ②", 0.12, 0.50),
            _text("footer", "- 8 -", 0.45, 0.94),
        ),
    )


def _prompt_payload(request: BuiltModelABlockRoleVectorRequest) -> dict[str, object]:
    content = request.artifact.messages[1].content
    return json.loads(content[content.index("{") :])


def _valid_responses(
    requests: tuple[BuiltModelABlockRoleVectorRequest, ...],
    *,
    role: str = "passage",
) -> tuple[bytes, ...]:
    return tuple(
        _json_bytes({"roles": [role] * len(request.artifact.target_blocks)}) for request in requests
    )


def test_requests_partition_max_32_blocks_with_four_whole_block_contexts() -> None:
    requests = build_model_a_block_role_vector_requests(
        _numbered_question_evidence(),
        page_image=_image(),
        model=_model(),
    )

    assert len(requests[0].classification_plan.blocks) == 70
    assert [len(request.artifact.target_blocks) for request in requests] == [32, 32, 6]
    assert [
        tuple(block.block_index for block in request.artifact.target_blocks) for request in requests
    ] == [tuple(range(32)), tuple(range(32, 64)), tuple(range(64, 70))]
    middle = requests[1]
    assert tuple(block.block_index for block in middle.artifact.context_before_blocks) == tuple(
        range(28, 32)
    )
    assert tuple(block.block_index for block in middle.artifact.context_after_blocks) == tuple(
        range(64, 68)
    )
    assert all(
        len(block.ordered_text_observation_ids) == 1
        for block in (
            middle.artifact.context_before_blocks
            + middle.artifact.target_blocks
            + middle.artifact.context_after_blocks
        )
    )
    schema = json.loads(middle.artifact.response_schema_json)
    assert schema["properties"]["roles"]["minItems"] == MAX_TARGET_BLOCKS_PER_SEGMENT
    assert schema["properties"]["roles"]["maxItems"] == MAX_TARGET_BLOCKS_PER_SEGMENT
    prompt = _prompt_payload(middle)
    assert [block["target_position"] for block in prompt["target_blocks"]] == list(range(32))
    assert [block["block_index"] for block in prompt["non_output_context_before_blocks"]] == list(
        range(28, 32)
    )
    assert [block["block_index"] for block in prompt["non_output_context_after_blocks"]] == list(
        range(64, 68)
    )


def test_answer_key_formation_is_serialized_and_forces_table_resolution() -> None:
    requests = build_model_a_block_role_vector_requests(
        _answer_key_evidence(),
        page_image=_image(),
        model=_model(),
    )
    plan = requests[0].classification_plan

    assert plan.blocks[0].formation == ModelAClassificationBlockFormation.ANSWER_KEY_TABLE
    assert plan.blocks[0].forced_role == "table"
    prompt = _prompt_payload(requests[0])
    assert prompt["target_blocks"][0]["formation"] == "answer_key_table"
    assert prompt["target_blocks"][0]["forced_role"] == "table"

    result = compile_model_a_block_role_vector_responses(
        requests,
        _valid_responses(requests),
    ).artifact
    first_resolution = result.resolved_role_plan.block_resolutions[0]
    assert first_resolution.model_role == "passage"
    assert first_resolution.resolved_role == "table"
    assert first_resolution.role_source == ModelAClassificationRoleSource.DETERMINISTIC
    assert first_resolution.conflict is True


def test_valid_page_preserves_raw_and_resolved_roles_and_one_line_per_node() -> None:
    requests = build_model_a_block_role_vector_requests(
        _evidence(),
        page_image=_image(),
        model=_model(),
    )
    built = compile_model_a_block_role_vector_responses(
        requests,
        _valid_responses(requests),
    )
    result = built.artifact

    assert result.outcome == "valid_unverified"
    assert result.raw_combined_block_decision.roles == ("passage", "passage", "passage")
    assert tuple(
        resolution.role_source for resolution in result.resolved_role_plan.block_resolutions
    ) == (
        ModelAClassificationRoleSource.MODEL,
        ModelAClassificationRoleSource.MODEL,
        ModelAClassificationRoleSource.DETERMINISTIC,
    )
    assert result.expanded_line_decision.roles == ("passage", "passage", "footer")
    text_nodes = tuple(node for node in result.compiled_page_analysis.nodes if node.kind == "text")
    assert len(text_nodes) == len(result.classification_plan.ordered_text_observation_ids)
    assert tuple(node.evidence_refs[0] for node in text_nodes) == (
        result.classification_plan.ordered_text_observation_ids
    )
    assert all(len(node.evidence_refs) == 1 and node.needs_review for node in text_nodes)
    assert built.compiled_page_analysis_bytes is not None


def test_one_bad_segment_blocks_page_preserves_all_traces_and_never_retries() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[BuiltModelABlockRoleVectorRequest] = []

        def generate(self, request: BuiltModelABlockRoleVectorRequest) -> bytes:
            self.calls.append(request)
            if len(self.calls) == 2:
                return _json_bytes({"roles": ["passage"]})
            return _json_bytes({"roles": ["passage"] * len(request.artifact.target_blocks)})

    provider = Provider()
    built = execute_model_a_block_role_vector_page(
        _numbered_question_evidence(),
        page_image=_image(),
        model=_model(),
        provider=provider,
    )
    result = built.artifact

    assert len(provider.calls) == 3
    assert len(result.segments) == 3
    assert tuple(trace.outcome for trace in result.segments) == (
        "valid_unverified",
        "blocked",
        "valid_unverified",
    )
    assert result.segments[1].issues[0].code == (ModelABlockRoleVectorIssueCode.ROLE_COUNT_MISMATCH)
    assert result.segments[1].raw_decision is not None
    assert result.segments[1].raw_decision.roles == ("passage",)
    assert result.outcome == "blocked"
    assert tuple(issue.code for issue in result.issues) == (
        ModelABlockRoleVectorPageIssueCode.SEGMENT_RESPONSE_BLOCKED,
    )
    assert result.raw_combined_block_decision is None
    assert result.resolved_role_plan is None
    assert result.expanded_line_decision is None
    assert result.compiled_page_analysis is None
    assert len(built.raw_role_response_bytes) == 3


def test_reordered_segments_are_reported_as_exact_coverage_failure() -> None:
    requests = build_model_a_block_role_vector_requests(
        _numbered_question_evidence(),
        page_image=_image(),
        model=_model(),
    )
    reordered = tuple(reversed(requests))
    result = compile_model_a_block_role_vector_responses(
        reordered,
        _valid_responses(reordered),
    ).artifact

    assert result.outcome == "blocked"
    assert tuple(trace.segment_index for trace in result.segments) == (0, 1, 2)
    assert tuple(issue.code for issue in result.issues) == (
        ModelABlockRoleVectorPageIssueCode.SEGMENT_COVERAGE_MISMATCH,
    )
    assert result.raw_combined_block_decision is None


def test_standalone_result_rejects_forged_expansion_and_plan_digest() -> None:
    requests = build_model_a_block_role_vector_requests(
        _evidence(),
        page_image=_image(),
        model=_model(),
    )
    result = compile_model_a_block_role_vector_responses(
        requests,
        _valid_responses(requests),
    ).artifact
    payload = result.model_dump(mode="python")
    expanded_payload = dict(payload["expanded_line_decision"])
    expanded_payload["roles"] = (
        "title",
        *expanded_payload["roles"][1:],
    )
    forged_expansion = ModelABlockExpandedLineDecision.model_validate(expanded_payload)
    payload["expanded_line_decision"] = forged_expansion.model_dump(mode="python")
    payload["expanded_line_decision_sha256"] = contract_sha256(forged_expansion)

    with pytest.raises(ValidationError, match="expanded line decision"):
        ModelABlockRoleVectorPageResult.model_validate(payload)

    digest_payload = result.model_dump(mode="python")
    digest_payload["classification_plan_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="classification plan digest"):
        ModelABlockRoleVectorPageResult.model_validate(digest_payload)


def test_built_request_replays_prompt_from_bound_evidence() -> None:
    request = build_model_a_block_role_vector_requests(
        _evidence(),
        page_image=_image(),
        model=_model(),
    )[0]
    messages = request.artifact.messages
    forged_messages = (
        messages[0],
        ModelAPromptMessage(
            role="user",
            content=messages[1].content.replace("passage", "title", 1),
        ),
    )
    forged_artifact = request.artifact.model_copy(update={"messages": forged_messages})

    with pytest.raises(ValueError, match="cannot be reproduced"):
        replace(request, artifact=forged_artifact)


def test_zero_text_page_uses_one_empty_segment_and_preserves_image() -> None:
    image_observation = next(
        observation
        for observation in _evidence().pages[0].observations
        if observation.kind == ObservationKind.IMAGE
    )
    evidence = _evidence_with_observations("evidence-image-only", (image_observation,))
    requests = build_model_a_block_role_vector_requests(
        evidence,
        page_image=_image(),
        model=_model(),
    )

    assert len(requests) == 1
    assert requests[0].artifact.target_blocks == ()
    built = compile_model_a_block_role_vector_responses(
        requests,
        (_json_bytes({"roles": []}),),
    )
    assert built.artifact.outcome == "valid_unverified"
    assert built.artifact.expanded_line_decision.roles == ()
    assert tuple(node.kind for node in built.artifact.compiled_page_analysis.nodes) == ("image",)
