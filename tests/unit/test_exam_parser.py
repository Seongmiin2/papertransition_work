from pathlib import Path

from scan2hwpx.exam_parser import parse_questions
from scan2hwpx.ir.models import BBox, Block, BlockKind, Page
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider


def test_parse_question_choice_and_score() -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    questions = parse_questions(document.pages)
    assert len(questions) == 1
    assert questions[0].number == 1
    assert questions[0].score == 3
    assert questions[0].choices[0].label == "①"


def test_short_outline_number_is_not_a_question() -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    document.pages[0].blocks.insert(
        0,
        Block(
            id="outline",
            kind=BlockKind.UNKNOWN,
            bbox=BBox(pixel=(0, 0, 10, 10), normalized=(0, 0, 0.1, 0.1)),
            reading_order=0,
            text="1) 처음",
            confidence=0.99,
            source_provider="fixture",
            source_payload_ref="fixture#outline",
        ),
    )
    questions = parse_questions(document.pages)
    assert [question.number for question in questions] == [1]


def _page_with_texts(texts: list[str]) -> list[Page]:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.blocks = [
        Block(
            id=f"b{index}",
            kind=BlockKind.UNKNOWN,
            bbox=BBox(pixel=(0, index, 10, index + 1), normalized=(0, 0, 0.1, 0.1)),
            reading_order=index,
            text=text,
            confidence=0.99,
            source_provider="fixture",
            source_payload_ref=f"fixture#b{index}",
        )
        for index, text in enumerate(texts)
    ]
    return [page]


def test_undetected_question_number_does_not_drop_later_questions() -> None:
    pages = _page_with_texts(["1. 첫 문항은?", "본문", "3. 셋째 문항은?", "4. 넷째 문항은?"])
    questions = parse_questions(pages)
    assert [question.number for question in questions] == [1, 3, 4]


def test_list_item_cannot_skip_ahead_of_a_later_real_question() -> None:
    pages = _page_with_texts(
        ["1. 첫 문항은?", "3. 지문 속 목록", "2. 둘째 문항은?", "3. 셋째 문항은?"]
    )
    questions = parse_questions(pages)
    assert [question.number for question in questions] == [1, 2, 3]
    assert "b1" in questions[0].prompt_blocks
