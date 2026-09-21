from __future__ import annotations

import json
import math
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .block_role_vector import BuiltModelABlockRoleVectorRequest
from .inference import BuiltModelAInferenceRequest, ModelAModelArtifact
from .role_vector import BuiltModelARoleVectorRequest


class OllamaModelAProviderError(RuntimeError):
    """Raised when the local Ollama boundary cannot return Model A JSON."""


@dataclass(frozen=True, slots=True)
class OllamaModelAProvider:
    """Local Ollama adapter for the provider-neutral Model A execution boundary."""

    model: ModelAModelArtifact
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 600.0
    num_ctx: int = 8192
    num_predict: int = 4096

    def __post_init__(self) -> None:
        if type(self.model) is not ModelAModelArtifact:
            raise TypeError("model must be an exact ModelAModelArtifact")
        _require_local_base_url(self.base_url)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if (
            isinstance(self.num_ctx, bool)
            or not isinstance(self.num_ctx, int)
            or self.num_ctx < 1024
        ):
            raise ValueError("num_ctx must be an integer of at least 1024")
        if (
            isinstance(self.num_predict, bool)
            or not isinstance(self.num_predict, int)
            or self.num_predict < 1
            or self.num_predict >= self.num_ctx
        ):
            raise ValueError("num_predict must be a positive integer smaller than num_ctx")

    def generate(self, request: BuiltModelAInferenceRequest, /) -> bytes:
        if type(request) is not BuiltModelAInferenceRequest:
            raise TypeError("request must be an exact BuiltModelAInferenceRequest")
        BuiltModelAInferenceRequest.__post_init__(request)
        if request.artifact.model != self.model:
            raise ValueError("request model artifact does not match the Ollama provider model")

        payload = _chat_payload(request, num_ctx=self.num_ctx, num_predict=self.num_predict)
        return _post_chat(
            self.base_url.removesuffix("/") + "/api/chat",
            payload,
            timeout_seconds=float(self.timeout_seconds),
        )


@dataclass(frozen=True, slots=True)
class OllamaModelARoleVectorProvider:
    """Local Ollama adapter for the compact Model A semantic boundary."""

    model: ModelAModelArtifact
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 600.0
    num_ctx: int = 8192
    num_predict: int = 1024

    def __post_init__(self) -> None:
        if type(self.model) is not ModelAModelArtifact:
            raise TypeError("model must be an exact ModelAModelArtifact")
        _require_local_base_url(self.base_url)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if (
            isinstance(self.num_ctx, bool)
            or not isinstance(self.num_ctx, int)
            or self.num_ctx < 1024
        ):
            raise ValueError("num_ctx must be an integer of at least 1024")
        if (
            isinstance(self.num_predict, bool)
            or not isinstance(self.num_predict, int)
            or self.num_predict < 1
            or self.num_predict >= self.num_ctx
        ):
            raise ValueError("num_predict must be a positive integer smaller than num_ctx")

    def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
        if type(request) is not BuiltModelARoleVectorRequest:
            raise TypeError("request must be an exact BuiltModelARoleVectorRequest")
        BuiltModelARoleVectorRequest.__post_init__(request)
        if request.artifact.model != self.model:
            raise ValueError("request model artifact does not match the Ollama provider model")
        payload = _chat_payload(
            request,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        return _post_chat(
            self.base_url.removesuffix("/") + "/api/chat",
            payload,
            timeout_seconds=float(self.timeout_seconds),
        )


@dataclass(frozen=True, slots=True)
class OllamaModelABlockRoleVectorProvider:
    """Local Ollama adapter for the block-based Model A v2 boundary."""

    model: ModelAModelArtifact
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 600.0
    num_ctx: int = 32768
    num_predict: int = 1024

    def __post_init__(self) -> None:
        if type(self.model) is not ModelAModelArtifact:
            raise TypeError("model must be an exact ModelAModelArtifact")
        _require_local_base_url(self.base_url)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if (
            isinstance(self.num_ctx, bool)
            or not isinstance(self.num_ctx, int)
            or self.num_ctx < 1024
        ):
            raise ValueError("num_ctx must be an integer of at least 1024")
        if (
            isinstance(self.num_predict, bool)
            or not isinstance(self.num_predict, int)
            or self.num_predict < 1
            or self.num_predict >= self.num_ctx
        ):
            raise ValueError("num_predict must be a positive integer smaller than num_ctx")

    def generate(self, request: BuiltModelABlockRoleVectorRequest, /) -> bytes:
        body = self.wire_payload_bytes(request)
        return _post_chat_body(
            self.base_url.removesuffix("/") + "/api/chat",
            body,
            timeout_seconds=float(self.timeout_seconds),
        )

    def wire_payload_bytes(self, request: BuiltModelABlockRoleVectorRequest, /) -> bytes:
        """Return the exact deterministic JSON body used by ``generate``."""

        if type(request) is not BuiltModelABlockRoleVectorRequest:
            raise TypeError("request must be an exact BuiltModelABlockRoleVectorRequest")
        BuiltModelABlockRoleVectorRequest.__post_init__(request)
        if request.artifact.model != self.model:
            raise ValueError("request model artifact does not match the Ollama provider model")
        payload = _chat_payload(
            request,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        return _chat_payload_bytes(payload)


def resolve_ollama_model_artifact(
    model_id: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 10.0,
) -> ModelAModelArtifact:
    """Bind a vision-capable local Ollama tag to its exact manifest digest."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    _require_local_base_url(base_url)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    request = Request(
        base_url.removesuffix("/") + "/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    envelope = _request_json(request, timeout_seconds=float(timeout_seconds))
    models = envelope.get("models")
    if not isinstance(models, list):
        raise OllamaModelAProviderError("local Ollama tags response has no models list")
    matches = [
        value
        for value in models
        if isinstance(value, dict)
        and (value.get("name") == model_id or value.get("model") == model_id)
    ]
    if len(matches) != 1:
        raise OllamaModelAProviderError(
            f"local Ollama model tag must resolve exactly once: {model_id}"
        )
    metadata = matches[0]
    capabilities = metadata.get("capabilities")
    if not isinstance(capabilities, list) or "vision" not in capabilities:
        raise OllamaModelAProviderError(f"local Ollama model has no vision capability: {model_id}")
    digest = metadata.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise OllamaModelAProviderError("local Ollama model digest is invalid")
    details = metadata.get("details")
    quantization = (
        details.get("quantization_level")
        if isinstance(details, dict) and isinstance(details.get("quantization_level"), str)
        else "unknown"
    )
    return ModelAModelArtifact(
        model_id=model_id,
        model_revision=f"ollama-{quantization}-{digest[:12]}",
        artifact_sha256=digest,
    )


def _chat_payload(
    request: (
        BuiltModelAInferenceRequest
        | BuiltModelARoleVectorRequest
        | BuiltModelABlockRoleVectorRequest
    ),
    *,
    num_ctx: int,
    num_predict: int,
) -> dict[str, object]:
    messages = [
        {
            "role": request.artifact.messages[0].role,
            "content": request.artifact.messages[0].content,
        },
        {
            "role": request.artifact.messages[1].role,
            "content": request.artifact.messages[1].content,
            "images": [request.artifact.page_image.data_base64],
        },
    ]
    return {
        "model": request.artifact.model.model_id,
        "messages": messages,
        "format": json.loads(request.artifact.response_schema_json),
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }


def _request_json(http_request: Request, *, timeout_seconds: float) -> dict[str, object]:
    try:
        with urlopen(http_request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise OllamaModelAProviderError(f"local Ollama returned HTTP {exc.code}{suffix}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaModelAProviderError(f"local Ollama request failed: {exc}") from exc
    try:
        envelope = json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaModelAProviderError("local Ollama returned malformed JSON") from exc
    if not isinstance(envelope, dict):
        raise OllamaModelAProviderError("local Ollama response must be a JSON object")
    if isinstance(envelope.get("error"), str) and envelope["error"]:
        error = envelope["error"]
        raise OllamaModelAProviderError(f"local Ollama error: {error}")
    return envelope


def _post_chat(url: str, payload: dict[str, object], *, timeout_seconds: float) -> bytes:
    return _post_chat_body(
        url,
        _chat_payload_bytes(payload),
        timeout_seconds=timeout_seconds,
    )


def _chat_payload_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _post_chat_body(url: str, body: bytes, *, timeout_seconds: float) -> bytes:
    http_request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise OllamaModelAProviderError(f"local Ollama returned HTTP {exc.code}{suffix}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaModelAProviderError(f"local Ollama request failed: {exc}") from exc

    try:
        envelope = json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaModelAProviderError("local Ollama returned malformed JSON") from exc
    if not isinstance(envelope, dict):
        raise OllamaModelAProviderError("local Ollama response must be a JSON object")
    if isinstance(envelope.get("error"), str) and envelope["error"]:
        error = envelope["error"]
        raise OllamaModelAProviderError(f"local Ollama error: {error}")
    message = envelope.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise OllamaModelAProviderError("local Ollama response has no assistant content")
    return message["content"].encode("utf-8")


def _require_local_base_url(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("base_url must be a nonempty string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url is invalid") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an explicit loopback HTTP origin")
