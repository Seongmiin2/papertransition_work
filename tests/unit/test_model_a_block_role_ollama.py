from __future__ import annotations

import json

import pytest

from scan2hwpx.model_a import ollama as ollama_module
from scan2hwpx.model_a.block_role_vector import (
    build_model_a_block_role_vector_requests,
)
from scan2hwpx.model_a.ollama import OllamaModelABlockRoleVectorProvider
from tests.unit.test_model_a_role_vector import _evidence, _image, _json_bytes, _model


def test_ollama_block_role_provider_defaults_to_observed_working_context() -> None:
    provider = OllamaModelABlockRoleVectorProvider(model=_model())

    assert provider.num_ctx == 32768


def test_ollama_block_role_provider_uses_exact_wire_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = build_model_a_block_role_vector_requests(
        _evidence(),
        page_image=_image(),
        model=_model(),
    )[0]
    provider = OllamaModelABlockRoleVectorProvider(
        model=_model(),
        base_url="http://localhost:11434/",
        timeout_seconds=12.5,
        num_ctx=8192,
        num_predict=512,
    )
    expected_body = provider.wire_payload_bytes(request)
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
                        "content": '{"roles":["passage","passage","footer"]}',
                    }
                }
            )

    def fake_urlopen(http_request, *, timeout):
        captured["request"] = http_request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(ollama_module, "urlopen", fake_urlopen)

    raw_response = provider.generate(request)

    assert raw_response == b'{"roles":["passage","passage","footer"]}'
    assert captured["timeout"] == 12.5
    http_request = captured["request"]
    assert http_request.full_url == "http://localhost:11434/api/chat"
    assert http_request.get_method() == "POST"
    assert http_request.get_header("Content-type") == "application/json"
    assert http_request.get_header("Accept") == "application/json"
    assert http_request.data == expected_body

    payload = json.loads(expected_body)
    assert payload == {
        "model": request.artifact.model.model_id,
        "messages": [
            {
                "role": request.artifact.messages[0].role,
                "content": request.artifact.messages[0].content,
            },
            {
                "role": request.artifact.messages[1].role,
                "content": request.artifact.messages[1].content,
                "images": [request.artifact.page_image.data_base64],
            },
        ],
        "format": json.loads(request.artifact.response_schema_json),
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 8192,
            "num_predict": 512,
        },
    }
