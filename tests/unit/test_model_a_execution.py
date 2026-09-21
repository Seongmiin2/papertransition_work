from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Self

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
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.model_a import (
    BuiltModelAInferenceRequest,
    ModelAModelArtifact,
    ModelAPageAnalysis,
    ModelAPageImage,
    OllamaModelAProvider,
    OllamaModelAProviderError,
    build_model_a_inference_request,
    execute_model_a_document,
    resolve_ollama_model_artifact,
)

_PAGE_BYTES = {
    1: b"\x89PNG\r\n\x1a\nmodel-a-execution-page-one",
    2: b"\x89PNG\r\n\x1a\nmodel-a-execution-page-two",
}
_PAGE_TEXT = {1: "첫 페이지", 2: "둘째 페이지"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _evidence() -> EvidenceIR:
    sources = [
        EvidenceSource(
            id="source-document",
            kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
            artifact_ref="artifact://fixture/source",
            producer="fixture",
            sha256="a" * 64,
        ),
        EvidenceSource(
            id="ocr-source",
            kind=EvidenceSourceKind.OCR_PROVIDER,
            artifact_ref="artifact://fixture/ocr",
            producer="fixture-ocr",
        ),
    ]
    pages: list[EvidencePage] = []
    for page_no in (2, 1):
        image_source_id = f"page-image-{page_no}"
        sources.append(
            EvidenceSource(
                id=image_source_id,
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref=f"artifact://fixture/pages/{page_no}",
                producer="fixture-rasterizer",
                sha256=_sha256(_PAGE_BYTES[page_no]),
            )
        )
        pages.append(
            EvidencePage(
                id=f"page-{page_no}",
                page_no=page_no,
                width=100,
                height=100,
                image_source_ref=image_source_id,
                observations=(
                    EvidenceObservation(
                        id=f"text-{page_no}",
                        kind=ObservationKind.TEXT_LINE,
                        bbox=BBox(
                            pixel=(0.0, 0.0, 10.0, 10.0),
                            normalized=(0.0, 0.0, 0.1, 0.1),
                        ),
                        confidence=0.99,
                        source_refs=("source-document", image_source_id, "ocr-source"),
                        ocr_candidates=(
                            OcrCandidate(
                                text=_PAGE_TEXT[page_no],
                                provider="fixture-ocr",
                                confidence=0.99,
                                quality=0.99,
                                source_ref="ocr-source",
                                selected=True,
                            ),
                        ),
                    ),
                ),
            )
        )
    return EvidenceIR(
        id="evidence-execution",
        source_document_sha256="a" * 64,
        sources=tuple(sources),
        pages=tuple(pages),
    )


def _page_image(page_no: int) -> ModelAPageImage:
    payload = _PAGE_BYTES[page_no]
    return ModelAPageImage(
        page_id=f"page-{page_no}",
        page_no=page_no,
        image_source_ref=f"page-image-{page_no}",
        media_type="image/png",
        raw_bytes=payload,
        sha256=_sha256(payload),
    )


def _model() -> ModelAModelArtifact:
    return ModelAModelArtifact(
        model_id="fixture-model-a",
        model_revision="revision-1",
        artifact_sha256="9" * 64,
    )


class _RecordingProvider:
    def __init__(self, *, invalid_page: int | None = None) -> None:
        self.invalid_page = invalid_page
        self.calls: list[int] = []

    def generate(self, request: BuiltModelAInferenceRequest, /) -> bytes:
        page_no = request.artifact.source.target_page_no
        self.calls.append(page_no)
        if page_no == self.invalid_page:
            return b"{}"
        page_id = request.artifact.source.target_page_id
        node = TextContentNode(
            id=f"{request.evidence_ir.id}:content:{page_id}:text",
            role=ContentRole.PASSAGE,
            text=_PAGE_TEXT[page_no],
            evidence_refs=(f"text-{page_no}",),
            confidence=0.99,
        )
        analysis = ModelAPageAnalysis(
            id=request.artifact.source.expected_page_analysis_id,
            evidence_ir_id=request.evidence_ir.id,
            evidence_ir_sha256=contract_sha256(request.evidence_ir),
            page_id=page_id,
            page_no=page_no,
            nodes=(node,),
            reading_order=(node.id,),
        )
        return _json_bytes(analysis.model_dump(mode="json"))


def test_valid_multi_page_execution_preserves_reproducible_artifacts() -> None:
    provider = _RecordingProvider()

    assembly = execute_model_a_document(
        _evidence(),
        (_page_image(2), _page_image(1)),
        _model(),
        provider,
    )

    assert provider.calls == [1, 2]
    assert assembly.artifact.outcome == "valid_unverified"
    assert assembly.artifact.content_ir is not None
    assert [result.artifact.source.target_page_no for result in assembly.page_results] == [1, 2]
    for result in assembly.page_results:
        assert _sha256(result.request.raw_request_bytes) == result.request.raw_request_sha256
        assert _sha256(result.raw_response_bytes) == result.artifact.raw_response_sha256
        assert _sha256(result.result_bytes) == result.result_sha256
    assert _sha256(assembly.result_bytes) == assembly.result_sha256


def test_page_image_input_order_does_not_change_execution_artifacts() -> None:
    evidence = _evidence()
    first = execute_model_a_document(
        evidence,
        (_page_image(1), _page_image(2)),
        _model(),
        _RecordingProvider(),
    )
    second = execute_model_a_document(
        evidence,
        (_page_image(2), _page_image(1)),
        _model(),
        _RecordingProvider(),
    )

    assert tuple(result.request.raw_request_sha256 for result in first.page_results) == tuple(
        result.request.raw_request_sha256 for result in second.page_results
    )
    assert tuple(result.artifact.raw_response_sha256 for result in first.page_results) == tuple(
        result.artifact.raw_response_sha256 for result in second.page_results
    )
    assert tuple(result.result_sha256 for result in first.page_results) == tuple(
        result.result_sha256 for result in second.page_results
    )
    assert first.result_bytes == second.result_bytes
    assert first.result_sha256 == second.result_sha256


@pytest.mark.parametrize("failure", ["missing", "duplicate", "sha_tamper"])
def test_page_image_preflight_rejects_before_provider_call(failure: str) -> None:
    first = _page_image(1)
    second = _page_image(2)
    images: tuple[ModelAPageImage, ...]
    if failure == "missing":
        images = (first,)
    elif failure == "duplicate":
        images = (first, first, second)
    else:
        object.__setattr__(second, "sha256", "f" * 64)
        images = (first, second)
    provider = _RecordingProvider()

    with pytest.raises(ValueError):
        execute_model_a_document(_evidence(), images, _model(), provider)

    assert provider.calls == []


def test_one_invalid_page_blocks_document_assembly() -> None:
    provider = _RecordingProvider(invalid_page=2)

    assembly = execute_model_a_document(
        _evidence(),
        (_page_image(2), _page_image(1)),
        _model(),
        provider,
    )

    assert provider.calls == [1, 2]
    assert [result.artifact.outcome for result in assembly.page_results] == [
        "valid_unverified",
        "blocked",
    ]
    assert assembly.artifact.outcome == "blocked"
    assert assembly.artifact.content_ir is None


def test_provider_is_called_exactly_once_per_canonical_page() -> None:
    provider = _RecordingProvider()

    assembly = execute_model_a_document(
        _evidence(),
        (_page_image(2), _page_image(1)),
        _model(),
        provider,
    )

    assert provider.calls == [1, 2]
    assert Counter(provider.calls) == Counter({1: 1, 2: 1})
    assert [result.request.artifact.source.target_page_no for result in assembly.page_results] == [
        1,
        2,
    ]


def test_invalid_response_limit_is_rejected_before_provider_call() -> None:
    provider = _RecordingProvider()

    with pytest.raises(ValueError, match="positive integer"):
        execute_model_a_document(
            _evidence(),
            (_page_image(1), _page_image(2)),
            _model(),
            provider,
            max_response_bytes=0,
        )

    assert provider.calls == []


class _FakeOllamaResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_ollama_provider_maps_the_canonical_request_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    model = _model()
    request = build_model_a_inference_request(
        evidence,
        page_image=_page_image(1),
        model=model,
    )
    raw_content = "{sentinel:true}"
    envelope = _json_bytes({"message": {"role": "assistant", "content": raw_content}})
    captured: dict[str, object] = {}

    def fake_urlopen(http_request: object, timeout: float) -> _FakeOllamaResponse:
        captured["request"] = http_request
        captured["timeout"] = timeout
        return _FakeOllamaResponse(envelope)

    monkeypatch.setattr(ollama_module, "urlopen", fake_urlopen)
    provider = OllamaModelAProvider(model=model)

    response = provider.generate(request)

    assert response == raw_content.encode("utf-8")
    assert captured["timeout"] == 600.0
    http_request = captured["request"]
    assert http_request.full_url == "http://127.0.0.1:11434/api/chat"
    assert http_request.get_method() == "POST"
    payload = json.loads(http_request.data)
    assert payload["model"] == model.model_id
    assert payload["messages"][0] == {
        "role": request.artifact.messages[0].role,
        "content": request.artifact.messages[0].content,
    }
    assert payload["messages"][1] == {
        "role": request.artifact.messages[1].role,
        "content": request.artifact.messages[1].content,
        "images": [request.artifact.page_image.data_base64],
    }
    assert payload["format"] == json.loads(request.artifact.response_schema_json)
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "seed": 0,
        "num_ctx": 8192,
        "num_predict": 4096,
    }


def test_ollama_provider_does_not_retry_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def failing_urlopen(http_request: object, timeout: float) -> _FakeOllamaResponse:
        nonlocal calls
        calls += 1
        raise ollama_module.URLError("offline")

    monkeypatch.setattr(ollama_module, "urlopen", failing_urlopen)
    model = _model()
    request = build_model_a_inference_request(
        _evidence(),
        page_image=_page_image(1),
        model=model,
    )

    with pytest.raises(OllamaModelAProviderError, match="local Ollama request failed"):
        OllamaModelAProvider(model=model).generate(request)

    assert calls == 1


def test_ollama_model_resolver_binds_exact_vision_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    envelope = _json_bytes(
        {
            "models": [
                {
                    "name": "fixture-vision:4b",
                    "model": "fixture-vision:4b",
                    "digest": digest,
                    "capabilities": ["completion", "vision"],
                    "details": {"quantization_level": "Q4_K_M"},
                }
            ]
        }
    )
    captured: dict[str, object] = {}

    def fake_urlopen(http_request: object, timeout: float) -> _FakeOllamaResponse:
        captured["request"] = http_request
        captured["timeout"] = timeout
        return _FakeOllamaResponse(envelope)

    monkeypatch.setattr(ollama_module, "urlopen", fake_urlopen)

    artifact = resolve_ollama_model_artifact("fixture-vision:4b")

    assert artifact.model_id == "fixture-vision:4b"
    assert artifact.model_revision == "ollama-Q4_K_M-aaaaaaaaaaaa"
    assert artifact.artifact_sha256 == digest
    assert captured["timeout"] == 10.0
    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:11434/api/tags"
    assert request.get_method() == "GET"


def test_ollama_model_resolver_rejects_nonvision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _json_bytes(
        {
            "models": [
                {
                    "name": "fixture-text:4b",
                    "digest": "a" * 64,
                    "capabilities": ["completion"],
                }
            ]
        }
    )
    monkeypatch.setattr(
        ollama_module,
        "urlopen",
        lambda http_request, timeout: _FakeOllamaResponse(envelope),
    )

    with pytest.raises(OllamaModelAProviderError, match="no vision capability"):
        resolve_ollama_model_artifact("fixture-text:4b")


def test_ollama_provider_rejects_nonlocal_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback HTTP origin"):
        OllamaModelAProvider(model=_model(), base_url="https://example.com:11434")
