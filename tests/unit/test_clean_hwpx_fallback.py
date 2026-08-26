import zipfile
from pathlib import Path

from lxml import etree

from scan2hwpx.hwpx import render_clean_hwpx, validate_hwpx
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider


def test_clean_hwpx_fallback_writes_valid_package(tmp_path: Path) -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    output = tmp_path / "fallback.hwpx"

    render_clean_hwpx(document, output)

    assert validate_hwpx(output).valid
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        header = etree.fromstring(archive.read("Contents/header.xml"))
    text = "".join(section.itertext())
    assert etree.QName(section).namespace == "http://www.hancom.co.kr/hwpml/2011/section"
    assert header.xpath("count(./*[local-name()='refList'])") == 1
    assert "1. 다음 글의 내용" in text
    assert "채점 표시" not in text
