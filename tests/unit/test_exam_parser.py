from pathlib import Path

from scan2hwpx.exam_parser import parse_questions
from scan2hwpx.ir.models import BBox, Block, BlockKind
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
