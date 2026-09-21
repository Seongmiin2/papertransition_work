from __future__ import annotations

import re

from scan2hwpx.ir.models import BlockKind

QUESTION_RE = re.compile(r"^\s*(\d{1,3})([.)])\s*(.*)$")
SCORE_RE = re.compile(r"\[(\d+(?:\.\d+)?)점\]")
CHOICE_RE = re.compile(r"^\s*([①②③④⑤])\s*(.*)$")
RANGE_RE = re.compile(r"^\s*\[?\d+\s*[~～-]\s*\d+\]?\s*")
PASSAGE_HEAD_RE = re.compile(r"^\s*\([가-힣A-Z]\)\s*")
SOURCE_RE = re.compile(r"^\s*[-―].*[-―]\s*$")

ROLE_SOURCE = "rule_based_v0"


def match_question_number(text: str) -> int | None:
    """Question number if `text` starts a new question, else None.

    A ")"-style match (e.g. "1)") with under 8 characters of trailing text is
    treated as a short list item rather than a question stem (e.g. "1) 가").
    """
    match = QUESTION_RE.match(text)
    if not match:
        return None
    if match.group(2) == ")" and len(match.group(3).strip()) < 8:
        return None
    return int(match.group(1))


def match_choice(text: str) -> str | None:
    match = CHOICE_RE.match(text)
    return match.group(1) if match else None


def match_score(text: str) -> float | None:
    match = SCORE_RE.search(text)
    return float(match.group(1)) if match else None


def classify_role(text: str, *, question_started: bool = False) -> BlockKind:
    """Best-effort structural role for a block, from its text alone.

    This is the single canonical pattern set behind block-role classification
    (a rule-based stand-in for a future trained "Model A"), shared by
    exam_parser.parse_questions and clean_layout.build_clean_items so the two
    renderers stop disagreeing on what counts as a question or choice.
    """
    if match_question_number(text) is not None:
        return BlockKind.QUESTION
    if question_started and match_choice(text) is not None:
        return BlockKind.CHOICE
    if RANGE_RE.match(text):
        return BlockKind.INSTRUCTION
    if SOURCE_RE.match(text):
        return BlockKind.SOURCE
    if PASSAGE_HEAD_RE.match(text):
        return BlockKind.PASSAGE
    return BlockKind.UNKNOWN
