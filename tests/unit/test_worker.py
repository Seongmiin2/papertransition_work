from __future__ import annotations

import io
import json

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
