from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scan2hwpx import batch


def test_batch_rejects_more_than_ten_pdfs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(11):
        (input_dir / f"exam-{index}.pdf").write_bytes(b"pdf")

    with pytest.raises(ValueError, match="at most 10"):
        batch.convert_directory(input_dir, tmp_path / "output")


def test_cpu_batch_keeps_single_worker_and_skips_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "exam.pdf").write_bytes(b"pdf")
    calls: list[bool] = []

    class FakeProvider:
        device = "cpu"

        def __init__(self, **_kwargs: Any) -> None:
            pass

    def fake_convert(_source: Path, output: Path, **kwargs: Any) -> dict[str, object]:
        calls.append(bool(kwargs["write_diagnostics"]))
        output.write_bytes(b"hwpx")
        return {"pages": 1}

    monkeypatch.setattr(batch, "PaddlePdfOcrProvider", FakeProvider)
    monkeypatch.setattr(batch, "convert_pdf", fake_convert)

    report = batch.convert_directory(input_dir, tmp_path / "output", device="cpu")

    assert calls == [False]
    assert report["workers"] == 1
    assert report["summary"]["completed"] == 1

