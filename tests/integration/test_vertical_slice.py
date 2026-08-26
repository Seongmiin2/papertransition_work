import zipfile
from pathlib import Path

from lxml import etree

from scan2hwpx.hwpx import render_hwpx, validate_hwpx
from scan2hwpx.ocr.providers import FixtureOcrProvider


def test_fixture_to_ir_to_hwpx(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/ocr_page_1.json")
    template = tmp_path / "template.hwpx"
    output = tmp_path / "output.hwpx"
    document = FixtureOcrProvider().convert(fixture)
    render_hwpx(document, template, output)
    assert validate_hwpx(output).valid
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
    text = "".join(section.itertext())
    assert "3학년 국어 기말고사" in text
    assert "① 첫 번째 선택지" in text
    assert "채점 표시" not in text


def test_renderer_marks_source_page_boundaries(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/ocr_page_1.json")
    document = FixtureOcrProvider().convert(fixture)
    second_page = document.pages[0].model_copy(deep=True)
    second_page.page_no = 2
    document.pages.append(second_page)
    output = tmp_path / "pages.hwpx"
    render_hwpx(document, tmp_path / "template.hwpx", output)
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
    assert section.xpath("count(//*[local-name()='p'][@pageBreak='1'])") == 1
