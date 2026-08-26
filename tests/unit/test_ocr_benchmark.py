from __future__ import annotations

from scan2hwpx.training.ocr_benchmark import (
    compare_ocr_benchmarks,
    levenshtein_distance,
    normalize_benchmark_text,
)


def test_levenshtein_distance_is_exact() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("국어 시험", "국어시험") == 1
    assert levenshtein_distance("", "시험") == 2


def test_normalize_benchmark_text_ignores_layout_whitespace() -> None:
    assert normalize_benchmark_text("국어\n 시 험") == "국어시험"


def test_candidate_requires_better_cer_and_coverage() -> None:
    baseline = {
        "character_error_rate": 0.2,
        "predicted_characters": 100,
        "source_pages": 8,
    }
    better = {
        "character_error_rate": 0.15,
        "predicted_characters": 99,
        "source_pages": 8,
    }
    truncated = {
        "character_error_rate": 0.1,
        "predicted_characters": 80,
        "source_pages": 8,
    }

    assert compare_ocr_benchmarks(baseline, better)["promoted"] is True
    assert compare_ocr_benchmarks(baseline, truncated)["promoted"] is False
