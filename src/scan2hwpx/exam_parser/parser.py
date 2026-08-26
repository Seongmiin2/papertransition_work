from __future__ import annotations

import re

from scan2hwpx.ir.models import BlockKind, Choice, Page, Question

QUESTION = re.compile(r"^\s*(\d{1,3})([.)])\s*(.*)$")
SCORE = re.compile(r"\[(\d+(?:\.\d+)?)점\]")
CHOICE = re.compile(r"^\s*([①②③④⑤])\s*(.*)$")


def parse_questions(pages: list[Page]) -> list[Question]:
    questions: list[Question] = []
    current_number: int | None = None
    current_score: float | None = None
    prompt_ids: list[str] = []
    choices: list[Choice] = []
    page_refs: list[int] = []
    confidences: list[float] = []
    expected_number = 1

    def flush() -> None:
        nonlocal current_number, current_score, prompt_ids, choices, page_refs, confidences
        if current_number is not None:
            questions.append(
                Question(
                    number=current_number,
                    score=current_score,
                    prompt_blocks=prompt_ids,
                    choices=choices,
                    page_refs=sorted(set(page_refs)),
                    confidence=sum(confidences) / max(1, len(confidences)),
                )
            )
        current_number = None
        current_score = None
        prompt_ids = []
        choices = []
        page_refs = []
        confidences = []

    for page in pages:
        for block in sorted(page.blocks, key=lambda item: item.reading_order):
            question_match = QUESTION.match(block.text)
            if (
                question_match
                and question_match.group(2) == ")"
                and len(question_match.group(3).strip()) < 8
            ):
                question_match = None
            if question_match and int(question_match.group(1)) == expected_number:
                flush()
                current_number = int(question_match.group(1))
                expected_number += 1
                current_score = _score(block.text)
                prompt_ids.append(block.id)
                page_refs.append(page.page_no)
                confidences.append(block.confidence)
                block.kind = BlockKind.QUESTION
                continue
            if current_number is None:
                continue
            choice_match = CHOICE.match(block.text)
            if choice_match:
                choices.append(
                    Choice(
                        label=choice_match.group(1),
                        blocks=[block.id],
                        confidence=block.confidence,
                    )
                )
                block.kind = BlockKind.CHOICE
            else:
                prompt_ids.append(block.id)
                if current_score is None:
                    current_score = _score(block.text)
            page_refs.append(page.page_no)
            confidences.append(block.confidence)
    flush()
    return questions


def _score(text: str) -> float | None:
    match = SCORE.search(text)
    return float(match.group(1)) if match else None
