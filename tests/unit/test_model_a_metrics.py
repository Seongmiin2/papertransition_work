from __future__ import annotations

from collections.abc import Callable

import pytest

from scan2hwpx.evaluation.model_a import (
    MODEL_A_METRIC_DEFINITION_IDS,
    MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID,
    corpus_cer,
    critical_token_error_rate,
    evaluate_model_a_text_structure,
    extract_critical_tokens,
    reading_order_exact_match,
    reading_order_exact_match_rate,
    reading_order_kendall_tau,
    reading_order_kendall_tau_mean,
    role_macro_f1,
)


def test_identical_text_and_critical_tokens_have_zero_error() -> None:
    text = "1. 다음 문항은 [3점] ① ㉠"

    assert corpus_cer([text], [text]) == 0.0
    assert critical_token_error_rate([text], [text]) == 0.0
    assert extract_critical_tokens(text) == (
        "question:1",
        "score:3",
        "circled:①",
        "hangul:㉠",
    )


def test_corpus_cer_sums_characters_before_normalizing() -> None:
    assert corpus_cer(["abc", "가"], ["adc", "가"]) == pytest.approx(0.25)
    assert corpus_cer([""], ["가"]) == 1.0


def test_critical_token_substitutions_are_token_edit_errors() -> None:
    reference = ["1. 다음 문항은 [3점] ① ㉠"]
    prediction = ["1. 다음 문항은 [2점] ② ㉠"]

    assert critical_token_error_rate(reference, prediction) == 0.5


@pytest.mark.parametrize(
    "measurement",
    [
        lambda: corpus_cer([], []),
        lambda: corpus_cer([""], [""]),
        lambda: critical_token_error_rate([], []),
        lambda: critical_token_error_rate(["본문"], ["본문"]),
        lambda: role_macro_f1([], []),
        lambda: reading_order_exact_match([], []),
        lambda: reading_order_exact_match_rate([], []),
        lambda: reading_order_kendall_tau([], []),
        lambda: reading_order_kendall_tau_mean([], []),
    ],
)
def test_empty_or_unmeasured_input_is_rejected(measurement: Callable[[], float]) -> None:
    with pytest.raises(ValueError, match="at least one|critical token"):
        measurement()


def test_role_macro_f1_uses_union_of_observed_labels() -> None:
    reference = ["title", "question", "question", "choice"]
    prediction = ["title", "question", "choice", "choice"]

    assert role_macro_f1(reference, prediction) == pytest.approx(7 / 9)


def test_reading_order_exact_match_and_partial_ordering() -> None:
    reference = ["a", "b", "c"]

    assert reading_order_exact_match(reference, ["a", "b", "c"]) == 1.0
    assert reading_order_exact_match(reference, ["a", "c", "b"]) == 0.0
    assert reading_order_kendall_tau(reference, ["a", "c", "b"]) == pytest.approx(2 / 3)


def test_reading_order_corpus_rate_is_distinct_from_single_document_exact() -> None:
    reference = [["a", "b", "c"], ["d", "e"]]
    prediction = [["a", "b", "c"], ["e", "d"]]

    assert reading_order_exact_match_rate(reference, prediction) == 0.5
    assert reading_order_kendall_tau_mean(reference, prediction) == 0.5


def test_reversed_reading_order_has_zero_normalized_kendall_score() -> None:
    assert reading_order_kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == 0.0


@pytest.mark.parametrize(
    "measurement",
    [reading_order_exact_match, reading_order_kendall_tau],
)
def test_order_metrics_reject_mismatched_or_duplicate_ids(
    measurement: Callable[[list[str], list[str]], float],
) -> None:
    with pytest.raises(ValueError, match="ID sets differ"):
        measurement(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="duplicate IDs"):
        measurement(["a", "a"], ["a", "a"])


def test_aggregate_uses_release_gate_metric_ids_and_corpus_order_rates() -> None:
    metrics = evaluate_model_a_text_structure(
        ["1. 문제 [2점] ①", "2. 문제 [3점] ②"],
        ["1. 문제 [2점] ①", "2. 문제 [3점] ②"],
        ["question", "choice"],
        ["question", "choice"],
        [["q1", "c1"], ["q2", "c2"]],
        [["q1", "c1"], ["c2", "q2"]],
    )

    assert metrics == {
        "model_a.overall_cer": 0.0,
        "model_a.critical_token_error_rate": 0.0,
        "model_a.role_macro_f1": 1.0,
        "model_a.reading_order_exact_match": 0.5,
        "model_a.reading_order_kendall_tau": 0.5,
    }


def test_release_metric_definitions_version_whitespace_sensitive_cer() -> None:
    assert MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID == "model-a-text-structure/v1"
    assert (
        MODEL_A_METRIC_DEFINITION_IDS["model_a.overall_cer"]
        == "cer/nfc-whitespace-sensitive-max-length-corpus/v1"
    )
    assert corpus_cer(["가 나"], ["가나"]) == pytest.approx(1 / 3)


def test_release_aggregate_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_model_a_text_structure([], [], [], [], [], [])


def test_aligned_metrics_reject_different_item_counts() -> None:
    with pytest.raises(ValueError, match="must be aligned"):
        corpus_cer(["a"], [])
    with pytest.raises(ValueError, match="must be aligned"):
        role_macro_f1(["question"], [])
