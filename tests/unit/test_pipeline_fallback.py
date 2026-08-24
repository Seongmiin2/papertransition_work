from __future__ import annotations

import zipfile
from pathlib import Path

from lxml import etree

from scan2hwpx import pipeline
from scan2hwpx.hwpx import validate_hwpx
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider


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
