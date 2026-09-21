from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scan2hwpx import batch
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider
from scan2hwpx.preprocess import preprocess_for_ocr


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


def test_stage_jobs_round_trip_through_document_ir_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipelined multi-file path hands documents between stages as JSON on
    disk (each stage runs in its own worker pool/process); this exercises
    that exact round trip without paying for a real ProcessPoolExecutor."""
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.width = 248
    page.height = 351
    for block in page.blocks:
        block.bbox.pixel = tuple(value / 10 for value in block.bbox.pixel)
    image = np.full((351, 248, 3), 255, dtype=np.uint8)
    preprocessed = preprocess_for_ocr(image)

    class FakeProvider:
        device = "cpu"

        def convert(
            self, _path: Path, _progress: object = None, *, page_callback: object = None
        ) -> object:
            assert callable(page_callback)
            page_callback(0, 1, page, image, preprocessed)  # type: ignore[operator]
            return document

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = job_dir / "result.hwpx"
    monkeypatch.setattr(batch, "_WORKER_PROVIDER", FakeProvider())

    ocr_result = batch._run_ocr_stage_job(tmp_path / "input.pdf", job_dir, 120, "fidelity")
    assert ocr_result["status"] == "ok"
    assert (job_dir / "document_ir.json").is_file()

    render_result = batch._run_render_stage_job(
        job_dir,
        output_path,
        "fidelity",
        "input.pdf",
        "hash123",
        {"dpi": 120},
        ocr_result["seconds"],
    )

    assert render_result["status"] == "completed"
    assert render_result["render_engine"] == "fidelity-page-image-hwpx"
    assert output_path.is_file()


def test_run_ocr_stage_job_requires_an_initialized_worker(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        batch._run_ocr_stage_job(tmp_path / "input.pdf", tmp_path, 120, "fidelity")

