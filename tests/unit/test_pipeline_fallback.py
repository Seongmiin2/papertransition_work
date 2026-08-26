from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from lxml import etree

from scan2hwpx import pipeline
from scan2hwpx.hwpx import validate_hwpx
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider
from scan2hwpx.preprocess import preprocess_for_ocr


def test_convert_pdf_falls_back_and_atomically_replaces_output(
    tmp_path: Path, monkeypatch: object
) -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    document.pages[0].page_no = 2
    input_path = tmp_path / "input.pdf"
    output = tmp_path / "result.hwpx"
    output.write_bytes(b"previous result")

    def fake_convert(*_args: object, **_kwargs: object) -> object:
        return document

    def fail_hancom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Hancom is unavailable")

    monkeypatch.setattr(pipeline.PaddlePdfOcrProvider, "convert", fake_convert)  # type: ignore[attr-defined]
    monkeypatch.setattr(pipeline, "render_clean_with_hancom", fail_hancom)  # type: ignore[attr-defined]
    monkeypatch.setattr(pipeline, "_write_debug_images", lambda *_args: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(pipeline, "write_review", lambda *_args: None)  # type: ignore[attr-defined]

    summary = pipeline.convert_pdf(input_path, output, renderer="hancom")

    assert summary["render_engine"] == "hancom-template-hwpx"
    assert summary["hancom_reopened"] is False
    assert validate_hwpx(output).valid
    assert not (tmp_path / ".result.candidate.hwpx").exists()
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
    text = "".join(section.itertext())
    assert "1. 다음 글의 내용" in text
    assert "채점 표시" not in text


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


def test_clean_renderer_skips_page_images_without_diagnostics(
    tmp_path: Path, monkeypatch: object
) -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.page_no = 2
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

    monkeypatch.setattr(pipeline.PaddlePdfOcrProvider, "convert", fake_convert)  # type: ignore[attr-defined]
    output = tmp_path / "result.hwpx"

    pipeline.convert_pdf(
        tmp_path / "input.pdf",
        output,
        renderer="portable",
        write_diagnostics=False,
    )

    assert output.is_file()
    assert not (tmp_path / "debug").exists()
