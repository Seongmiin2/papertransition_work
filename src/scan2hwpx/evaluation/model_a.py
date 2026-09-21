from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Final, TypeVar

_Item = TypeVar("_Item")

MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID: Final = "model-a-text-structure/v1"
MODEL_A_METRIC_DEFINITION_IDS: Final[dict[str, str]] = {
    "model_a.overall_cer": "cer/nfc-whitespace-sensitive-max-length-corpus/v1",
    "model_a.critical_token_error_rate": "critical-token-edit-rate/corpus/v1",
    "model_a.role_macro_f1": "role-macro-f1/observed-label-union/v1",
    "model_a.reading_order_exact_match": "reading-order/document-exact-match-rate/v1",
    "model_a.reading_order_kendall_tau": "reading-order/normalized-kendall-mean/v1",
}

_CRITICAL_TOKEN_PATTERN = re.compile(
    r"(?P<score>[\[(]?\s*(?P<score_value>\d+(?:\.\d+)?)\s*점\s*[\])]?)"
    r"|(?P<question>(?<!\d)(?P<question_value>\d{1,3})\s*(?:\.(?!\d)|\)))"
    r"|(?P<circled>[①-⑳])"
    r"|(?P<hangul>[㉠-㉻])"
)


def levenshtein_distance(left: Sequence[_Item], right: Sequence[_Item]) -> int:
    """Return exact edit distance using memory proportional to the shorter input."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def corpus_cer(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Return the Model A whitespace-sensitive CER for aligned texts.

    NFC normalization is applied, while whitespace and punctuation remain significant.
    Edit distances and ``max(reference length, prediction length)`` are summed across
    the corpus before division, keeping the result in 0..1. This definition is
    versioned in ``MODEL_A_METRIC_DEFINITION_IDS`` and is not the whitespace-insensitive
    OCR recognizer benchmark CER. Empty corpora and corpora with no characters on
    either side are unmeasured and rejected.
    """
    _require_aligned(reference, prediction, "text corpus")
    errors = 0
    characters = 0
    for expected, actual in zip(reference, prediction, strict=True):
        expected = unicodedata.normalize("NFC", expected)
        actual = unicodedata.normalize("NFC", actual)
        errors += levenshtein_distance(expected, actual)
        characters += max(len(expected), len(actual))
    if not characters:
        raise ValueError("text corpus must contain at least one character")
    return errors / characters


def extract_critical_tokens(text: str) -> tuple[str, ...]:
    """Extract canonical question, score, circled-number, and enclosed-Hangul tokens."""
    tokens: list[str] = []
    for match in _CRITICAL_TOKEN_PATTERN.finditer(unicodedata.normalize("NFC", text)):
        if match.group("score") is not None:
            tokens.append("score:" + match.group("score_value"))
        elif match.group("question") is not None:
            tokens.append("question:" + match.group("question_value"))
        elif match.group("circled") is not None:
            tokens.append("circled:" + match.group("circled"))
        else:
            tokens.append("hangul:" + match.group("hangul"))
    return tuple(tokens)


def critical_token_error_rate(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Return normalized edit error over extracted critical-token sequences.

    Distances and maximum token counts are summed over aligned texts. A corpus with
    no critical tokens on either side is unmeasured and rejected; a token present
    on only one side scores 1.0.
    """
    _require_aligned(reference, prediction, "text corpus")
    errors = 0
    token_count = 0
    for expected, actual in zip(reference, prediction, strict=True):
        expected_tokens = extract_critical_tokens(expected)
        actual_tokens = extract_critical_tokens(actual)
        errors += levenshtein_distance(expected_tokens, actual_tokens)
        token_count += max(len(expected_tokens), len(actual_tokens))
    if not token_count:
        raise ValueError("text corpus must contain at least one critical token")
    return errors / token_count


def role_macro_f1(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Return unweighted F1 over labels present in either non-empty aligned sequence."""
    _require_aligned(reference, prediction, "role labels")
    labels = set(reference) | set(prediction)
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(reference, prediction, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(reference, prediction, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(reference, prediction, strict=True)
        )
        scores.append(
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
        )
    return sum(scores) / len(scores)


def reading_order_exact_match(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Return one document's exact-order score for valid, non-empty permutations."""
    _validate_permutations(reference, prediction)
    return float(tuple(reference) == tuple(prediction))


def reading_order_exact_match_rate(
    reference: Sequence[Sequence[str]], prediction: Sequence[Sequence[str]]
) -> float:
    """Return the document-level exact-match rate for a non-empty corpus."""
    _require_aligned(reference, prediction, "reading-order corpus")
    matches = sum(
        reading_order_exact_match(expected, actual)
        for expected, actual in zip(reference, prediction, strict=True)
    )
    return matches / len(reference)


def reading_order_kendall_tau(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Return normalized Kendall agreement for two non-empty permutations.

    The conventional no-ties tau is mapped from -1..1 to 0..1, so identical order
    scores 1.0 and a full reversal scores 0.0. Duplicate IDs or different ID sets
    raise ``ValueError``. A one-item permutation scores 1.0.
    """
    _validate_permutations(reference, prediction)
    if len(reference) < 2:
        return 1.0
    predicted_position = {item: index for index, item in enumerate(prediction)}
    positions = [predicted_position[item] for item in reference]
    discordant = sum(
        positions[left] > positions[right]
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )
    pairs = len(positions) * (len(positions) - 1) // 2
    return 1.0 - discordant / pairs


def reading_order_kendall_tau_mean(
    reference: Sequence[Sequence[str]], prediction: Sequence[Sequence[str]]
) -> float:
    """Return mean normalized Kendall agreement for a non-empty document corpus."""
    _require_aligned(reference, prediction, "reading-order corpus")
    scores = [
        reading_order_kendall_tau(expected, actual)
        for expected, actual in zip(reference, prediction, strict=True)
    ]
    return sum(scores) / len(scores)


def evaluate_model_a_text_structure(
    reference_texts: Sequence[str],
    predicted_texts: Sequence[str],
    reference_roles: Sequence[str],
    predicted_roles: Sequence[str],
    reference_reading_orders: Sequence[Sequence[str]],
    predicted_reading_orders: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Return release-policy measurements from non-empty text and order corpora."""
    return {
        "model_a.overall_cer": corpus_cer(reference_texts, predicted_texts),
        "model_a.critical_token_error_rate": critical_token_error_rate(
            reference_texts, predicted_texts
        ),
        "model_a.role_macro_f1": role_macro_f1(reference_roles, predicted_roles),
        "model_a.reading_order_exact_match": reading_order_exact_match_rate(
            reference_reading_orders, predicted_reading_orders
        ),
        "model_a.reading_order_kendall_tau": reading_order_kendall_tau_mean(
            reference_reading_orders, predicted_reading_orders
        ),
    }


def _require_aligned(
    reference: Sequence[object], prediction: Sequence[object], name: str
) -> None:
    if len(reference) != len(prediction):
        raise ValueError(
            f"{name} must be aligned: {len(reference)} reference items, "
            f"{len(prediction)} prediction items"
        )
    if not reference:
        raise ValueError(f"{name} must contain at least one evaluation item")


def _validate_permutations(reference: Sequence[str], prediction: Sequence[str]) -> None:
    if not reference and not prediction:
        raise ValueError("reading order must contain at least one ID")
    if len(set(reference)) != len(reference):
        raise ValueError("reference reading order contains duplicate IDs")
    if len(set(prediction)) != len(prediction):
        raise ValueError("prediction reading order contains duplicate IDs")
    reference_ids = set(reference)
    prediction_ids = set(prediction)
    if reference_ids != prediction_ids:
        missing = sorted(reference_ids - prediction_ids)
        unexpected = sorted(prediction_ids - reference_ids)
        raise ValueError(
            "reading-order ID sets differ: "
            f"missing from prediction={missing}, unexpected in prediction={unexpected}"
        )
