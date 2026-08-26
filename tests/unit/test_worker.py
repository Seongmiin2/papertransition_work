from __future__ import annotations

import io
import json
from pathlib import Path

from scan2hwpx import worker
from scan2hwpx.worker import PROTOCOL_VERSION, emit


def test_protocol_version_is_stable() -> None:
    assert PROTOCOL_VERSION == "1.0"


def test_emit_is_single_line_ndjson(monkeypatch: object) -> None:
    target = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", target)  # type: ignore[attr-defined]
    emit("request-1", "heartbeat", {"ok": True})
    lines = target.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "heartbeat"


def test_provider_is_reused_for_matching_worker_settings(monkeypatch: object) -> None:
    created: list[object] = []

    class FakeProvider:
        def __init__(self, **_kwargs: object) -> None:
            created.append(self)

    monkeypatch.setattr("scan2hwpx.ocr.providers.paddle.PaddlePdfOcrProvider", FakeProvider)
    monkeypatch.setattr(worker, "_PROVIDER", None)
    monkeypatch.setattr(worker, "_PROVIDER_SETTINGS", None)

    first = worker._provider_for(120, "fast", "auto", Path("ocr-model"), Path("model.npz"))
    second = worker._provider_for(120, "fast", "auto", Path("ocr-model"), Path("model.npz"))

    assert first is second
    assert len(created) == 1
