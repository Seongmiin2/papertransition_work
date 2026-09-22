from __future__ import annotations

from pathlib import Path

import numpy as np

from scan2hwpx import pipeline
from scan2hwpx.hwpx import validate_hwpx
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider
from scan2hwpx.preprocess import preprocess_for_ocr


def test_convert_pdf_streams_page_artifacts_without_full_document_cache(
    tmp_path: Path, monkeypatch: object
) -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.width = 248
    page.height = 351
    for block in page.blocks:
        block.bbox.pixel = tuple(value / 10 for value in block.bbox.pixel)
    image = np.full((351, 248, 3), 255, dtype=np.uint8)
    preprocessed = preprocess_for_ocr(image)
    input_path = tmp_path / "input.pdf"
    output = tmp_path / "result.hwpx"

    def fake_convert(
        _self: object,
        _path: Path,
        _progress: object = None,
        *,
        page_callback: object = None,
    ) -> object:
        assert callable(page_callback)
        page_callback(0, 1, page, image, preprocessed)
        return document

    def reject_fallback(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("streamed pages must not be rasterized again")

    monkeypatch.setattr(pipeline.PaddlePdfOcrProvider, "convert", fake_convert)  # type: ignore[attr-defined]
    monkeypatch.setattr(pipeline, "_write_debug_images", reject_fallback)  # type: ignore[attr-defined]

    summary = pipeline.convert_pdf(input_path, output, renderer="fidelity")

    assert summary["render_engine"] == "fidelity-page-image-hwpx"
    assert validate_hwpx(output).valid
    assert (tmp_path / "debug" / "original" / "page-1.png").is_file()
    assert (tmp_path / "debug" / "ocr_overlay" / "page-1.png").is_file()
    assert (tmp_path / "review_crops" / "b-mark.png").is_file()


def test_convert_pdf_can_skip_batch_diagnostics(tmp_path: Path, monkeypatch: object) -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.width = 248
    page.height = 351
    for block in page.blocks:
        block.bbox.pixel = tuple(value / 10 for value in block.bbox.pixel)
    image = np.full((351, 248, 3), 255, dtype=np.uint8)
    preprocessed = preprocess_for_ocr(image)

    def fake_convert(
        _self: object,
        _path: Path,
        _progress: object = None,
        *,
        page_callback: object = None,
    ) -> object:
        assert callable(page_callback)
        page_callback(0, 1, page, image, preprocessed)
        return document

    def reject_review(*_args: object) -> None:
        raise AssertionError("review must be skipped")

    monkeypatch.setattr(pipeline.PaddlePdfOcrProvider, "convert", fake_convert)  # type: ignore[attr-defined]
    monkeypatch.setattr(pipeline, "write_review", reject_review)  # type: ignore[attr-defined]

    output = tmp_path / "result.hwpx"
    summary = pipeline.convert_pdf(
        tmp_path / "input.pdf",
        output,
        renderer="fidelity",
        write_diagnostics=False,
    )

    assert summary["render_engine"] == "fidelity-page-image-hwpx"
    assert (tmp_path / "debug" / "original" / "page-1.png").is_file()
    assert not (tmp_path / "debug" / "preprocessed").exists()
    assert not (tmp_path / "debug" / "ocr_overlay").exists()
    assert not (tmp_path / "review.html").exists()
