from pathlib import Path

from scan2hwpx.clean_layout import build_clean_items
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider


def test_clean_layout_keeps_question_and_drops_no_content_fixture() -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    document.pages[0].page_no = 2
    items = build_clean_items(document)
    text = "\n".join(item.text for item in items)
    assert "1. 다음 글의 내용" in text
    assert "① 첫 번째 선택지" in text
    assert "채점 표시" not in text
