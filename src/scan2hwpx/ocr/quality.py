from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_EXPECTED = re.compile(r"[0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,?!:;()\[\]{}<>《》〈〉「」『』·…'\"%+\-/=①-⑳ⓐ-ⓩ㉠-㉻]")
_PREFIX = re.compile(r"^(\s*(?:\d{1,3}[.)]|[①-⑳]|[㉠-㉻]|[ⓐ-ⓩ])\s*)")


@dataclass(frozen=True)
class RecognitionCandidate:
    engine: str
    text: str
    confidence: float
    quality: float


@dataclass(frozen=True)
class RecognitionDecision:
    selected: RecognitionCandidate
    candidates: tuple[RecognitionCandidate, ...]
    review_reasons: tuple[str, ...]


class KoreanQualityRouter:
    def __init__(self, lexicon_path: Path | None = None) -> None:
        self.lexicon = _load_lexicon(lexicon_path)

    def score(self, text: str, confidence: float) -> float:
        stripped = text.strip()
        if not stripped:
            return 0.0
        visible = [character for character in stripped if not character.isspace()]
        expected_ratio = sum(bool(_EXPECTED.fullmatch(character)) for character in visible) / max(
            1, len(visible)
        )
        controls = sum(unicodedata.category(character) in {"Cc", "Cs", "Co"} for character in visible)
        replacement = stripped.count("�")
        suspicious_questions = len(re.findall(r"(?<=[가-힣])\?(?=[가-힣])|\?{2,}", stripped))
        bracket_penalty = 0.0
        for opening, closing in (("(", ")"), ("[", "]"), ("<", ">")):
            if abs(stripped.count(opening) - stripped.count(closing)) > 1:
                bracket_penalty += 0.03
        tokens = _TOKEN.findall(stripped)
        known_ratio = (
            sum(token in self.lexicon for token in tokens) / len(tokens)
            if self.lexicon and tokens
            else 0.5
        )
        score = (
            max(0.0, min(1.0, confidence)) * 0.52
            + expected_ratio * 0.28
            + known_ratio * 0.12
            + min(1.0, len(visible) / 8) * 0.08
            - controls * 0.25
            - replacement * 0.3
            - suspicious_questions * 0.12
            - bracket_penalty
        )
        return max(0.0, min(1.0, score))

    def decide(
        self,
        paddle_text: str,
        paddle_confidence: float,
        windows_text: str | None,
        windows_confidence: float = 0.88,
    ) -> RecognitionDecision:
        paddle = RecognitionCandidate(
            engine="paddleocr",
            text=paddle_text.strip(),
            confidence=paddle_confidence,
            quality=self.score(paddle_text, paddle_confidence),
        )
        if not windows_text or len(windows_text.strip()) < 2:
            reasons = tuple(reason for reason in _intrinsic_reasons(paddle.text, paddle.confidence))
            return RecognitionDecision(paddle, (paddle,), reasons)
        aligned_windows = preserve_exam_prefix(paddle.text, windows_text.strip())
        windows = RecognitionCandidate(
            engine="windows-ocr-ko",
            text=aligned_windows,
            confidence=windows_confidence,
            quality=self.score(aligned_windows, windows_confidence),
        )
        similarity = normalized_similarity(paddle.text, windows.text)
        windows_not_truncated = len(windows.text) >= max(2, int(len(paddle.text) * 0.58))
        use_windows = windows_not_truncated and (
            windows.quality >= paddle.quality + 0.08
            or (paddle.confidence < 0.68 and windows.quality > paddle.quality)
        )
        selected = windows if use_windows else paddle
        selected_reasons = list(_intrinsic_reasons(selected.text, selected.confidence))
        if similarity < 0.92:
            selected_reasons.append("engine_disagreement")
        return RecognitionDecision(
            selected, (paddle, windows), tuple(dict.fromkeys(selected_reasons))
        )


def needs_secondary_recognition(text: str, confidence: float) -> bool:
    if confidence < 0.88:
        return True
    return bool(_intrinsic_reasons(text, confidence))


def normalized_similarity(left: str, right: str) -> float:
    a = re.sub(r"\s+", "", left)
    b = re.sub(r"\s+", "", right)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for row, left_character in enumerate(a, start=1):
        current = [row]
        for column, right_character in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def preserve_exam_prefix(paddle_text: str, windows_text: str) -> str:
    prefix = _PREFIX.match(paddle_text)
    if prefix is None:
        return windows_text
    without_prefix = _PREFIX.sub("", windows_text, count=1).lstrip()
    return f"{prefix.group(1)}{without_prefix}".strip()


def _intrinsic_reasons(text: str, confidence: float) -> list[str]:
    reasons: list[str] = []
    if confidence < 0.75:
        reasons.append("low_ocr_confidence")
    if "�" in text or any(unicodedata.category(character) in {"Cc", "Cs", "Co"} for character in text):
        reasons.append("invalid_character")
    if re.search(r"(?<=[가-힣])\?(?=[가-힣])|\?{2,}", text):
        reasons.append("suspicious_character_sequence")
    return reasons


def _load_lexicon(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload.get("words", []) if isinstance(payload, dict) else []
    return {
        str(item["text"])
        for item in words
        if isinstance(item, dict) and int(item.get("count", 0)) >= 2 and item.get("text")
    }
