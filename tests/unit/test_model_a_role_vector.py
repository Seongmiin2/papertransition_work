from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import scan2hwpx.model_a.ollama as ollama_module
from scan2hwpx.contracts import (
    BBox,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
    contract_sha256,
)
from scan2hwpx.model_a import (
    BuiltModelARoleVectorRequest,
    ModelAModelArtifact,
    ModelAPageAnalysis,
    ModelAPageImage,
    ModelARoleVectorChunkedPageResult,
    ModelARoleVectorDecision,
    ModelARoleVectorIssueCode,
    ModelARoleVectorPageIssueCode,
    ModelARoleVectorRequest,
    ModelARoleVectorRequestError,
    ModelARoleVectorResult,
    OllamaModelARoleVectorProvider,
    build_model_a_role_vector_request,
    build_model_a_role_vector_segment_requests,
    compile_model_a_role_vector_chunked_responses,
    compile_model_a_role_vector_response,
    execute_model_a_role_vector_chunked_page,
    execute_model_a_role_vector_page,
)

_PNG = b"\x89PNG\r\n\x1a\nrole-vector-fixture"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _bbox(x: float, y: float) -> BBox:
    return BBox(
        pixel=(x * 100, y * 100, x * 100 + 10, y * 100 + 5),
        normalized=(x, y, x + 0.1, y + 0.05),
    )


def _model() -> ModelAModelArtifact:
    return ModelAModelArtifact(
        model_id="qwen3.5:4b-q4_K_M",
        model_revision="ollama-Q4_K_M-fixture",
        artifact_sha256="9" * 64,
    )


def _evidence(
    *,
    unsupported_kind: ObservationKind | None = None,
    image_has_bound_asset: bool = True,
) -> EvidenceIR:
    sources = (
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
            sha256=_sha256(_PNG),
        ),
        EvidenceSource(
            id="crop-image",
            kind=EvidenceSourceKind.CROP,
            artifact_ref="artifact://fixture/crop",
            producer="fixture",
            sha256="b" * 64 if image_has_bound_asset else None,
        ),
        EvidenceSource(
            id="ocr-source",
            kind=EvidenceSourceKind.OCR_PROVIDER,
            artifact_ref="artifact://fixture/ocr",
            producer="fixture-ocr",
        ),
        EvidenceSource(
            id="layout-source",
            kind=EvidenceSourceKind.LAYOUT_PROVIDER,
            artifact_ref="artifact://fixture/layout",
            producer="fixture-layout",
        ),
    )
    text_footer = EvidenceObservation(
        id="text-footer",
        kind=ObservationKind.TEXT_LINE,
        bbox=_bbox(0.45, 0.95),
        confidence=0.9,
        source_refs=("page-image", "ocr-source"),
        ocr_candidates=(
            OcrCandidate(
                text="- 1 -",
                provider="fixture-ocr",
                confidence=0.9,
                source_ref="ocr-source",
                selected=True,
            ),
        ),
    )
    image_refs = (
        ("crop-image", "layout-source")
        if not image_has_bound_asset
        else ("page-image", "crop-image", "layout-source")
    )
    image = EvidenceObservation(
        id="image-1",
        kind=ObservationKind.IMAGE,
        bbox=_bbox(0.2, 0.5),
        confidence=0.8,
        source_refs=image_refs,
    )
    text_passage = EvidenceObservation(
        id="text-passage",
        kind=ObservationKind.TEXT_LINE,
        bbox=_bbox(0.1, 0.3),
        confidence=0.7,
        source_refs=("page-image", "ocr-source"),
        ocr_candidates=(
            OcrCandidate(
                text="선택된 본문",
                provider="fixture-ocr",
                confidence=0.4,
                source_ref="ocr-source",
                selected=True,
            ),
            OcrCandidate(
                text="선택되지 않은 후보",
                provider="fixture-ocr",
                confidence=0.99,
                source_ref="ocr-source",
            ),
        ),
    )
    text_header = EvidenceObservation(
        id="text-header",
        kind=ObservationKind.TEXT_LINE,
        bbox=_bbox(0.1, 0.05),
        confidence=0.95,
        source_refs=("page-image", "ocr-source"),
        ocr_candidates=(
            OcrCandidate(
                text="시험 이름",
                provider="fixture-ocr",
                confidence=0.95,
                source_ref="ocr-source",
                selected=True,
            ),
        ),
    )
    region = EvidenceObservation(
        id="region-1",
        kind=ObservationKind.REGION,
        bbox=_bbox(0.0, 0.0),
        confidence=0.5,
        source_refs=("layout-source",),
    )
    observations = [text_footer, text_header, text_passage, image, region]
    if unsupported_kind is not None:
        observations.append(
            EvidenceObservation(
                id="unsupported-1",
                kind=unsupported_kind,
                bbox=_bbox(0.3, 0.4),
                confidence=0.8,
                source_refs=("layout-source",),
            )
        )
    page = EvidencePage(
        id="page-1",
        page_no=1,
        width=100,
        height=100,
        image_source_ref="page-image",
        observations=tuple(observations),
    )
    return EvidenceIR(
        id="evidence-1",
        source_document_sha256="a" * 64,
        sources=sources,
        pages=(page,),
    )


def _image() -> ModelAPageImage:
    return ModelAPageImage(
        page_id="page-1",
        page_no=1,
        image_source_ref="page-image",
        media_type="image/png",
        raw_bytes=_PNG,
        sha256=_sha256(_PNG),
    )


def _many_text_evidence(count: int = 70) -> EvidenceIR:
    observations = []
    for index in range(count):
        column_index = index % 35
        x = 0.1 if index < 35 else 0.6
        y = 0.1 + column_index * 0.02
        text = {
            10: "3. numbered question",
            11: "\u2460 circled choice",
        }.get(index, f"line {index}")
        observations.append(
            EvidenceObservation(
                id=f"text-{index:03d}",
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
        )
    return EvidenceIR(
        id="evidence-many",
        source_document_sha256="a" * 64,
        sources=_evidence().sources,
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=100,
                height=100,
                image_source_ref="page-image",
                observations=tuple(observations),
            ),
        ),
    )


def _request() -> BuiltModelARoleVectorRequest:
    return build_model_a_role_vector_request(
        _evidence(),
        page_image=_image(),
        model=_model(),
    )


def test_request_binds_spatial_text_order_and_exact_response_shape() -> None:
    request = _request()
    schema = json.loads(request.artifact.response_schema_json)
    user_content = request.artifact.messages[1].content
    prompt = json.loads(user_content[user_content.index("{") :])

    assert request.artifact.ordered_text_observation_ids == (
        "text-header",
        "text-passage",
        "text-footer",
    )
    assert schema["properties"]["roles"]["minItems"] == 3
    assert schema["properties"]["roles"]["maxItems"] == 3
    assert schema["additionalProperties"] is False
    assert [line["text"] for line in prompt["lines_in_required_output_order"]] == [
        "시험 이름",
        "선택된 본문",
        "- 1 -",
    ]
    assert [line["anchor_hint"] for line in prompt["lines_in_required_output_order"]] == [
        "top_margin_header_candidate",
        "none",
        "bare_page_number_footer",
    ]


def test_request_rejects_noncanonical_response_schema_with_matching_digest() -> None:
    payload = _request().artifact.model_dump(mode="json")
    payload["response_schema_json"] = "{}"
    payload["response_schema_sha256"] = _sha256(b"{}")

    with pytest.raises(ValueError, match="not canonical for the bound role count"):
        ModelARoleVectorRequest.model_validate_json(_json_bytes(payload), strict=True)


def test_role_vector_compiles_exact_evidence_and_keeps_provenance_separate() -> None:
    raw_response = _json_bytes({"roles": ["header", "passage", "footer"]})

    result = compile_model_a_role_vector_response(_request(), raw_response)

    assert result.artifact.outcome == "valid_unverified"
    assert result.artifact.issues == ()
    assert result.artifact.compiled_validation_issues == ()
    assert result.artifact.raw_role_response_sha256 == _sha256(raw_response)
    assert result.artifact.raw_role_response_sha256 != (
        result.artifact.compiled_page_analysis_sha256
    )
    assert result.artifact.raw_decision is not None
    assert result.artifact.resolved_decision is not None
    assert result.artifact.raw_decision.roles == result.artifact.resolved_decision.roles
    assert len(result.artifact.deterministic_anchor_resolutions) == 1
    assert result.artifact.deterministic_anchor_resolutions[0].overrode_model is False
    analysis = result.artifact.compiled_page_analysis
    assert analysis is not None
    assert result.compiled_page_analysis_bytes is not None
    assert tuple(node.kind for node in analysis.nodes) == (
        "text",
        "text",
        "image",
        "text",
    )
    assert tuple(node.role for node in analysis.nodes) == (
        ContentRole.HEADER,
        ContentRole.PASSAGE,
        ContentRole.IMAGE,
        ContentRole.FOOTER,
    )
    assert analysis.nodes[1].text == "선택된 본문"
    assert analysis.nodes[2].asset_ref == "crop-image"
    assert tuple(node.evidence_refs for node in analysis.nodes) == (
        ("text-header",),
        ("text-passage",),
        ("image-1",),
        ("text-footer",),
    )
    assert all(node.confidence == 0.0 for node in analysis.nodes)
    assert all(node.needs_review for node in analysis.nodes)
    assert result.artifact.review_required_node_ids == analysis.reading_order


def test_content_images_follow_the_same_margin_order_as_text() -> None:
    evidence = _evidence()
    observations = {item.id: item for item in evidence.pages[0].observations}
    bottom_image = observations["image-1"].model_copy(
        update={"bbox": _bbox(0.20, 0.93)}
    )
    top_image = observations["image-1"].model_copy(
        update={"id": "image-top", "bbox": _bbox(0.20, 0.02)}
    )
    page = evidence.pages[0].model_copy(
        update={
            "observations": (
                bottom_image,
                observations["text-passage"],
                top_image,
                observations["text-header"],
                observations["text-footer"],
                observations["region-1"],
            )
        }
    )
    ordered_evidence = evidence.model_copy(update={"pages": (page,)})
    request = build_model_a_role_vector_request(
        ordered_evidence,
        page_image=_image(),
        model=_model(),
    )

    result = compile_model_a_role_vector_response(
        request,
        _json_bytes({"roles": ["header", "passage", "footer"]}),
    )

    analysis = result.artifact.compiled_page_analysis
    assert analysis is not None
    assert tuple(node.evidence_refs[0] for node in analysis.nodes) == (
        "image-top",
        "text-header",
        "text-passage",
        "image-1",
        "text-footer",
    )


@pytest.mark.parametrize(
    ("raw_response", "expected_code"),
    (
        (
            b"{'roles':['header'],'roles':['footer']}",
            ModelARoleVectorIssueCode.RESPONSE_JSON_INVALID,
        ),
        (
            _json_bytes(
                {
                    "roles": ["header", "passage", "footer"],
                    "extra": True,
                }
            ),
            ModelARoleVectorIssueCode.RESPONSE_SCHEMA_INVALID,
        ),
        (
            _json_bytes({"roles": ["header"]}),
            ModelARoleVectorIssueCode.ROLE_COUNT_MISMATCH,
        ),
    ),
)
def test_invalid_role_responses_are_blocked_without_repair(
    raw_response: bytes,
    expected_code: ModelARoleVectorIssueCode,
) -> None:
    result = compile_model_a_role_vector_response(_request(), raw_response)

    assert result.artifact.outcome == "blocked"
    assert tuple(issue.code for issue in result.artifact.issues) == (expected_code,)
    assert result.artifact.compiled_page_analysis is None
    assert result.compiled_page_analysis_bytes is None


def test_oversized_role_response_is_blocked_before_parsing() -> None:
    result = compile_model_a_role_vector_response(
        _request(),
        b"{}" * 8,
        max_response_bytes=10,
    )

    assert tuple(issue.code for issue in result.artifact.issues) == (
        ModelARoleVectorIssueCode.RESPONSE_TOO_LARGE,
    )


@pytest.mark.parametrize(
    "kind",
    (ObservationKind.TABLE_GRID, ObservationKind.FORMULA),
)
def test_table_and_formula_are_blocked_before_provider_call(
    kind: ObservationKind,
) -> None:
    with pytest.raises(ModelARoleVectorRequestError):
        build_model_a_role_vector_request(
            _evidence(unsupported_kind=kind),
            page_image=_image(),
            model=_model(),
        )


def test_image_without_hashed_crop_or_page_source_is_blocked_preflight() -> None:
    with pytest.raises(ModelARoleVectorRequestError):
        build_model_a_role_vector_request(
            _evidence(image_has_bound_asset=False),
            page_image=_image(),
            model=_model(),
        )


def test_executor_calls_provider_once_and_never_repairs() -> None:
    class Provider:
        calls = 0

        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            self.calls += 1
            return b"{}"

    provider = Provider()
    result = execute_model_a_role_vector_page(
        _evidence(),
        page_image=_image(),
        model=_model(),
        provider=provider,
    )

    assert provider.calls == 1
    assert result.artifact.outcome == "blocked"


def test_chunk_requests_cover_page_once_with_bounded_context() -> None:
    requests = build_model_a_role_vector_segment_requests(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        max_lines_per_segment=32,
    )

    assert tuple(len(request.artifact.ordered_text_observation_ids) for request in requests) == (
        32,
        32,
        6,
    )
    assert tuple(
        observation_id
        for request in requests
        for observation_id in request.artifact.ordered_text_observation_ids
    ) == tuple(f"text-{index:03d}" for index in range(70))

    second_prompt_text = requests[1].artifact.messages[1].content
    second_prompt = json.loads(second_prompt_text[second_prompt_text.index("{") :])
    assert [line["global_index"] for line in second_prompt["lines_in_required_output_order"]] == (
        list(range(32, 64))
    )
    assert [line["observation_id"] for line in second_prompt["lines_in_required_output_order"]] == [
        f"text-{index:03d}" for index in range(32, 64)
    ]
    assert [line["global_index"] for line in second_prompt["non_output_context_before"]] == [
        28,
        29,
        30,
        31,
    ]
    assert [line["global_index"] for line in second_prompt["non_output_context_after"]] == [
        64,
        65,
        66,
        67,
    ]
    assert all(
        line["layout_zone"] in {"left_body", "right_body"}
        for line in second_prompt["lines_in_required_output_order"]
    )
    first_prompt_text = requests[0].artifact.messages[1].content
    first_prompt = json.loads(first_prompt_text[first_prompt_text.index("{") :])
    assert first_prompt["lines_in_required_output_order"][10]["anchor_hint"] == (
        "numbered_question_start"
    )
    assert first_prompt["lines_in_required_output_order"][11]["anchor_hint"] == (
        "circled_choice_start"
    )
    assert "never emit roles for those entries" in requests[1].artifact.messages[0].content


def test_chunk_executor_compiles_all_segments_with_separate_provenance() -> None:
    class Provider:
        calls = 0

        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            first_roles = ("title", "question", "footer")
            roles = ["passage"] * len(request.artifact.ordered_text_observation_ids)
            roles[0] = first_roles[self.calls]
            if self.calls == 0:
                roles[10] = "question"
            self.calls += 1
            return _json_bytes({"roles": roles})

    provider = Provider()
    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=provider,
        max_lines_per_segment=32,
    )

    assert provider.calls == 3
    assert result.artifact.outcome == "valid_unverified"
    assert len(result.artifact.segments) == 3
    assert len({segment.raw_role_response_sha256 for segment in result.artifact.segments}) == 3
    assert result.artifact.raw_combined_decision is not None
    assert result.artifact.raw_combined_decision.roles[0] == "title"
    assert result.artifact.raw_combined_decision.roles[10] == "question"
    assert result.artifact.raw_combined_decision.roles[11] == "passage"
    assert result.artifact.resolved_decision is not None
    assert result.artifact.resolved_decision.roles[10] == "question"
    assert result.artifact.resolved_decision.roles[11] == "choice"
    assert result.artifact.resolved_decision.roles[32] == "question"
    assert result.artifact.resolved_decision.roles[64] == "footer"
    assert len(result.artifact.deterministic_anchor_resolutions) == 2
    assert tuple(
        resolution.overrode_model for resolution in result.artifact.deterministic_anchor_resolutions
    ) == (False, True)
    analysis = result.artifact.compiled_page_analysis
    assert analysis is not None
    assert len(analysis.nodes) == 70
    assert result.artifact.review_required_node_ids == analysis.reading_order


def test_chunk_executor_blocks_page_when_one_segment_is_invalid() -> None:
    class Provider:
        calls = 0

        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            self.calls += 1
            if self.calls == 2:
                return b"not-json"
            return _json_bytes(
                {"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)}
            )

    provider = Provider()
    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=provider,
        max_lines_per_segment=32,
    )

    assert provider.calls == 3
    assert result.artifact.outcome == "blocked"
    assert tuple(issue.code for issue in result.artifact.issues) == (
        ModelARoleVectorPageIssueCode.SEGMENT_RESPONSE_BLOCKED,
    )
    assert result.artifact.compiled_page_analysis is None
    assert result.compiled_page_analysis_bytes is None


def test_chunk_compiler_rejects_reordered_segment_coverage() -> None:
    requests = build_model_a_role_vector_segment_requests(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        max_lines_per_segment=32,
    )
    reordered = tuple(reversed(requests))
    responses = tuple(
        _json_bytes({"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)})
        for request in reordered
    )

    result = compile_model_a_role_vector_chunked_responses(
        reordered,
        responses,
        max_lines_per_segment=32,
    )

    assert result.artifact.outcome == "blocked"
    assert tuple(issue.code for issue in result.artifact.issues) == (
        ModelARoleVectorPageIssueCode.SEGMENT_COVERAGE_MISMATCH,
    )
    assert result.artifact.compiled_page_analysis is None


def test_chunk_compiler_rejects_limit_that_does_not_match_request_partition() -> None:
    requests = build_model_a_role_vector_segment_requests(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        max_lines_per_segment=32,
    )
    responses = tuple(
        _json_bytes({"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)})
        for request in requests
    )

    result = compile_model_a_role_vector_chunked_responses(
        requests,
        responses,
        max_lines_per_segment=64,
    )

    assert result.artifact.outcome == "blocked"
    assert tuple(issue.code for issue in result.artifact.issues) == (
        ModelARoleVectorPageIssueCode.SEGMENT_COVERAGE_MISMATCH,
    )


def test_chunk_artifact_rejects_forged_anchor_binding() -> None:
    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            return _json_bytes(
                {"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)}
            )

    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )
    payload = result.artifact.model_dump(mode="json")
    payload["deterministic_anchor_resolutions"][0]["global_index"] = 9

    with pytest.raises(ValueError, match="does not bind its observation"):
        ModelARoleVectorChunkedPageResult.model_validate_json(
            _json_bytes(payload),
            strict=True,
        )


def test_chunk_artifact_rejects_forged_segment_schema_digest() -> None:
    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            return _json_bytes(
                {"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)}
            )

    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )
    payload = result.artifact.model_dump(mode="json")
    payload["segments"][0]["response_schema_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="schema does not match its bound observations"):
        ModelARoleVectorChunkedPageResult.model_validate_json(
            _json_bytes(payload),
            strict=True,
        )


def test_chunk_artifact_rejects_forged_declared_partition_limit() -> None:
    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            return _json_bytes(
                {"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)}
            )

    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )
    payload = result.artifact.model_dump(mode="json")
    payload["segment_line_limit"] = 64

    with pytest.raises(ValueError, match="declared canonical line partition"):
        ModelARoleVectorChunkedPageResult.model_validate_json(
            _json_bytes(payload),
            strict=True,
        )


def test_single_artifact_rejects_forged_anchor_observation_id() -> None:
    result = compile_model_a_role_vector_response(
        _request(),
        _json_bytes({"roles": ["header", "passage", "footer"]}),
    )
    payload = result.artifact.model_dump(mode="json")
    payload["deterministic_anchor_resolutions"][0]["observation_id"] = "forged-id"

    with pytest.raises(ValueError, match="does not bind its compiled observation"):
        ModelARoleVectorResult.model_validate_json(_json_bytes(payload), strict=True)


def test_chunk_artifact_rejects_compiled_source_page_mismatch() -> None:
    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            return _json_bytes(
                {"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)}
            )

    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )
    payload = result.artifact.model_dump(mode="json")
    analysis_payload = payload["compiled_page_analysis"]
    analysis_payload["page_no"] = 999
    forged_analysis = ModelAPageAnalysis.model_validate_json(
        _json_bytes(analysis_payload),
        strict=True,
    )
    payload["compiled_page_analysis"] = forged_analysis.model_dump(mode="json")
    payload["compiled_page_analysis_sha256"] = contract_sha256(forged_analysis)

    with pytest.raises(ValueError, match="does not match its source binding"):
        ModelARoleVectorChunkedPageResult.model_validate_json(
            _json_bytes(payload),
            strict=True,
        )


def test_chunk_artifact_rejects_compiled_text_evidence_mismatch() -> None:
    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            return _json_bytes(
                {"roles": ["passage"] * len(request.artifact.ordered_text_observation_ids)}
            )

    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )
    payload = result.artifact.model_dump(mode="json")
    analysis_payload = payload["compiled_page_analysis"]
    forged_observation_id = "forged-observation"
    forged_node_id = (
        f"{payload['source']['evidence_ir_id']}:content:"
        f"{payload['source']['target_page_id']}:{forged_observation_id}"
    )
    analysis_payload["nodes"][0]["evidence_refs"] = [forged_observation_id]
    analysis_payload["nodes"][0]["id"] = forged_node_id
    analysis_payload["reading_order"][0] = forged_node_id
    payload["review_required_node_ids"][0] = forged_node_id
    forged_analysis = ModelAPageAnalysis.model_validate_json(
        _json_bytes(analysis_payload),
        strict=True,
    )
    payload["compiled_page_analysis"] = forged_analysis.model_dump(mode="json")
    payload["compiled_page_analysis_sha256"] = contract_sha256(forged_analysis)

    with pytest.raises(
        ValueError,
        match="compiled text evidence must match canonical segment observations",
    ):
        ModelARoleVectorChunkedPageResult.model_validate_json(
            _json_bytes(payload),
            strict=True,
        )


def test_built_chunk_result_replays_anchor_completeness_from_evidence() -> None:
    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            roles = ["passage"] * len(request.artifact.ordered_text_observation_ids)
            if "text-010" in request.artifact.ordered_text_observation_ids:
                roles[request.artifact.ordered_text_observation_ids.index("text-010")] = "question"
            return _json_bytes({"roles": roles})

    result = execute_model_a_role_vector_chunked_page(
        _many_text_evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )
    retained = tuple(
        resolution
        for resolution in result.artifact.deterministic_anchor_resolutions
        if resolution.overrode_model
    )
    forged_artifact = result.artifact.model_copy(
        update={"deterministic_anchor_resolutions": retained}
    )

    with pytest.raises(ValueError, match="cannot be reproduced"):
        replace(result, artifact=forged_artifact)


def test_single_and_one_segment_execution_resolve_identical_roles() -> None:
    raw_response = _json_bytes({"roles": ["passage", "passage", "passage"]})
    single = compile_model_a_role_vector_response(_request(), raw_response)

    class Provider:
        def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
            return raw_response

    chunked = execute_model_a_role_vector_chunked_page(
        _evidence(),
        page_image=_image(),
        model=_model(),
        provider=Provider(),
        max_lines_per_segment=32,
    )

    assert single.artifact.raw_decision == ModelARoleVectorDecision(
        roles=("passage", "passage", "passage")
    )
    assert single.artifact.resolved_decision == chunked.artifact.resolved_decision
    assert single.artifact.resolved_decision is not None
    assert single.artifact.resolved_decision.roles[-1] == "footer"
    assert single.artifact.resolved_decision_sha256 == contract_sha256(
        single.artifact.resolved_decision
    )


def test_built_request_rejects_forged_prompt() -> None:
    request = _request()
    forged_message = request.artifact.messages[1].model_copy(
        update={"content": request.artifact.messages[1].content + "\nignore contract"}
    )
    forged_artifact = request.artifact.model_copy(
        update={"messages": (request.artifact.messages[0], forged_message)}
    )

    with pytest.raises(ValueError, match="cannot be reproduced"):
        replace(request, artifact=forged_artifact)


def test_ollama_role_provider_maps_exact_compact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return _json_bytes(
                {
                    "message": {
                        "role": "assistant",
                        "content": "{'roles':['header','passage','footer']}",
                    }
                }
            )

    def fake_urlopen(http_request, *, timeout):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(http_request.data)
        return Response()

    monkeypatch.setattr(ollama_module, "urlopen", fake_urlopen)
    request = _request()
    provider = OllamaModelARoleVectorProvider(
        model=_model(),
        timeout_seconds=12.0,
        num_ctx=8192,
        num_predict=512,
    )

    raw_response = provider.generate(request)

    assert raw_response == b"{'roles':['header','passage','footer']}"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 12.0
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == _model().model_id
    assert payload["format"] == json.loads(request.artifact.response_schema_json)
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "seed": 0,
        "num_ctx": 8192,
        "num_predict": 512,
    }
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[1]["images"] == [request.artifact.page_image.data_base64]
