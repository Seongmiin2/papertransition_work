from __future__ import annotations

from scan2hwpx.blueprint import model_a
from scan2hwpx.ir.models import BlockKind, Choice, Page, Question


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
            question_number = model_a.match_question_number(block.text)
            if question_number is not None and question_number == expected_number:
                flush()
                current_number = question_number
                expected_number += 1
                current_score = model_a.match_score(block.text)
                prompt_ids.append(block.id)
                page_refs.append(page.page_no)
                confidences.append(block.confidence)
                block.kind = BlockKind.QUESTION
                block.role = BlockKind.QUESTION
                block.role_source = model_a.ROLE_SOURCE
                block.group_id = f"q{current_number}"
                continue
            if current_number is None:
                continue
            choice_label = model_a.match_choice(block.text)
            if choice_label:
                choices.append(
                    Choice(
                        label=choice_label,
                        blocks=[block.id],
                        confidence=block.confidence,
                    )
                )
                block.kind = BlockKind.CHOICE
                block.role = BlockKind.CHOICE
            else:
                prompt_ids.append(block.id)
                if current_score is None:
                    current_score = model_a.match_score(block.text)
                block.role = model_a.classify_role(block.text, question_started=True)
            block.role_source = model_a.ROLE_SOURCE
            block.group_id = f"q{current_number}"
            page_refs.append(page.page_no)
            confidences.append(block.confidence)
    flush()
    return questions
