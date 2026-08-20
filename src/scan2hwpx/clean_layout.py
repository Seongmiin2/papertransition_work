from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from scan2hwpx.ir.models import AnnotationState, Block, BlockKind, Document


class CleanKind(StrEnum):
    INSTRUCTION = "instruction"
    PASSAGE = "passage"
    QUESTION = "question"
    CHOICE = "choice"
    SOURCE = "source"


@dataclass
class CleanItem:
    page_no: int
    column: int
    kind: CleanKind
    text: str
    confidence: float
    last_y: float


DROP_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"홍천중학교",
        r"2026학년도",
        r"정기시험",
        r"3학년\s*국어\s*\d+/8",
        r"^(과목|과목코드|국어과|3학년|문항지|실시일|쪽수)$",
        r"^7월\s*1일",
        r"^(25문항|100점|총\s*문항수|총5문항수|01|2026년)$",
        r"답안지에\s*인적사항",
        r"답안을\s*작성하시오",
        r"선다형\s*답안",
        r"^(선다형|1번~25번)$",
    )
]
QUESTION_RE = re.compile(r"^\s*\d{1,2}\.\s*")
CHOICE_RE = re.compile(r"^\s*[①②③④⑤]\s*")
NUMBER_CHOICE_RE = re.compile(r"^\s*([1-5])\s*$")
RANGE_RE = re.compile(r"^\s*\[?\d+\s*[~～-]\s*\d+\]?\s*")
PASSAGE_HEAD_RE = re.compile(r"^\s*\([가-힣A-Z]\)\s*")
SOURCE_RE = re.compile(r"^\s*[-―].*[-―]\s*$")
CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}


def build_clean_items(document: Document) -> list[CleanItem]:
    items: list[CleanItem] = []
    question_started = False
    for page in document.pages:
        for block in sorted(page.blocks, key=lambda value: value.reading_order):
            if _drop(block, page.page_no):
                continue
            text = _clean_text(block.text)
            if not text:
                continue
            column = _column(block)
            kind = _kind(text, block.kind, question_started)
            if kind == CleanKind.QUESTION:
                question_started = True
            number_choice = NUMBER_CHOICE_RE.match(text)
            if number_choice and question_started:
                text = CIRCLED[number_choice.group(1)]
                kind = CleanKind.CHOICE
            y = block.bbox.normalized[1]
            if _can_merge(items, page.page_no, column, kind, y):
                previous = items[-1]
                previous.text = _join(previous.text, text)
                previous.confidence = min(previous.confidence, block.confidence)
                previous.last_y = y
            else:
                items.append(
                    CleanItem(
                        page_no=page.page_no,
                        column=column,
                        kind=kind,
                        text=text,
                        confidence=block.confidence,
                        last_y=y,
                    )
                )
    return items


def _drop(block: Block, page_no: int) -> bool:
    text = block.text.strip()
    if block.annotation_state != AnnotationState.PRINTED:
        return True
    if block.kind in {BlockKind.HEADER, BlockKind.FOOTER, BlockKind.PAGE_NUMBER}:
        return True
    if text in {"×", "X", "○", "O", "(X)", "(0)", "(O)"}:
        return True
    if any(pattern.search(text) for pattern in DROP_PATTERNS):
        return True
    x0, y0, x1, _ = block.bbox.normalized
    center = (x0 + x1) / 2
    if page_no == 1 and center < 0.5 and y0 < 0.32:
        return True
    return page_no == 1 and center >= 0.5 and y0 < 0.21


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^([①②③④⑤])(?=\S)", r"\1 ", text)
    text = re.sub(r"^(\d{1,2}\.)\s*", r"\1 ", text)
    return text


def _column(block: Block) -> int:
    center = (block.bbox.normalized[0] + block.bbox.normalized[2]) / 2
    return 0 if center < 0.5 else 1


def _kind(text: str, original: BlockKind, question_started: bool) -> CleanKind:
    if QUESTION_RE.match(text) or original == BlockKind.QUESTION:
        return CleanKind.QUESTION
    if question_started and (CHOICE_RE.match(text) or original == BlockKind.CHOICE):
        return CleanKind.CHOICE
    if RANGE_RE.match(text):
        return CleanKind.INSTRUCTION
    if SOURCE_RE.match(text):
        return CleanKind.SOURCE
    if PASSAGE_HEAD_RE.match(text):
        return CleanKind.PASSAGE
    return CleanKind.PASSAGE


def _can_merge(
    items: list[CleanItem], page_no: int, column: int, kind: CleanKind, y: float
) -> bool:
    if not items or kind in {CleanKind.QUESTION, CleanKind.INSTRUCTION, CleanKind.SOURCE}:
        return False
    previous = items[-1]
    if previous.page_no != page_no or previous.column != column or y - previous.last_y > 0.035:
        return False
    if kind == CleanKind.CHOICE:
        return False
    return previous.kind in {CleanKind.PASSAGE, CleanKind.QUESTION, CleanKind.CHOICE}


def _join(left: str, right: str) -> str:
    if left.endswith(("-", "·")):
        return left + right
    return left + " " + right
