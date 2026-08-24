from __future__ import annotations

from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.ocr.quality import KoreanQualityRouter, normalized_similarity


def _paddle(text: str, confidence: float) -> dict[str, object]:
    return {
        "rec_texts": [text],
        "rec_scores": [confidence],
        "rec_polys": [[[0, 0], [300, 0], [300, 50], [0, 50]]],
    }


def test_fusion_keeps_paddle_question_number_and_uses_clearly_better_body() -> None:
    windows = [((0, 0, 300, 50), "7. 컴퓨터용 사인펜", 0.9)]

    fused = PaddlePdfOcrProvider._fuse_recognition(_paddle("1. 컴퓨터욜 사인펱", 0.6), windows)

    assert fused["rec_texts"] == ["1. 컴퓨터용 사인펜"]
    assert fused["rec_review_reasons"] == [["engine_disagreement"]]


def test_fusion_does_not_replace_high_confidence_paddle_on_disagreement() -> None:
    windows = [((0, 0, 300, 50), "전혀 다른 문장입니다", 0.88)]

    fused = PaddlePdfOcrProvider._fuse_recognition(_paddle("정확한 원문입니다", 0.97), windows)

    assert fused["rec_texts"] == ["정확한 원문입니다"]
    assert "engine_disagreement" in fused["rec_review_reasons"][0]


def test_quality_router_marks_suspicious_characters() -> None:
    decision = KoreanQualityRouter().decide("문?장이??", 0.95, None)

    assert "suspicious_character_sequence" in decision.review_reasons


def test_normalized_similarity_ignores_whitespace() -> None:
    assert normalized_similarity("다음 글", "다음글") == 1.0
