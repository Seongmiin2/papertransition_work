from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf

from scan2hwpx.ir.models import BlockKind, Page
from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.ocr.quality import KoreanQualityRouter, normalized_similarity
from scan2hwpx.preprocess import PreprocessResult
from scan2hwpx.vision.layout_dataset import LayoutAnnotation
from scan2hwpx.vision.page_layout import PageLayout


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


def test_pdf_provider_emits_each_completed_page_to_callback(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = tmp_path / "one-page.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=200, height=300)
    page.insert_text((20, 30), "fixture")
    pdf.save(source)
    pdf.close()

    result = SimpleNamespace(
        json={
            "res": {
                "rec_texts": ["1. 테스트 문제"],
                "rec_scores": [0.99],
                "rec_polys": [[[10, 10], [100, 10], [100, 30], [10, 30]]],
            }
        }
    )

    provider = PaddlePdfOcrProvider(dpi=72, device="cpu")
    monkeypatch.setattr(provider, "_predict", lambda _image: [result])  # type: ignore[attr-defined]
    received: list[tuple[int, int, int]] = []

    def receive_page(
        page_index: int,
        total: int,
        parsed_page: Page,
        image: np.ndarray,
        preprocessed: PreprocessResult,
    ) -> None:
        assert parsed_page.page_no == 1
        assert image.shape[:2] == (300, 200)
        assert preprocessed.image.shape[:2] == (300, 200)
        received.append((page_index, total, parsed_page.page_no))

    document = provider.convert(source, page_callback=receive_page)

    assert received == [(0, 1, 1)]
    assert document.pages[0].blocks[0].text == "1. 테스트 문제"
    assert len(document.pages[0].columns) == 1
    assert document.pages[0].blocks[0].style["layout_column"] == 0
    assert document.metadata["page_layouts"][0]["columns"] == 1


def test_page_layout_controls_block_order_columns_and_region_kind() -> None:
    layout = PageLayout(
        width=1000,
        height=1400,
        column_boxes=((0, 0, 500, 1400), (500, 0, 1000, 1400)),
        regions=(
            LayoutAnnotation(
                label="table",
                bbox=(80, 400, 450, 650),
                confidence=0.8,
                source="test",
                source_label="table",
            ),
        ),
        gutter_x=500,
        column_confidence=0.9,
    )
    data = {
        "rec_texts": ["오른쪽 본문", "왼쪽 표 내용", "시험 제목"],
        "rec_scores": [0.9, 0.9, 0.99],
        "rec_polys": [
            [[600, 100], [900, 100], [900, 140], [600, 140]],
            [[100, 500], [300, 500], [300, 540], [100, 540]],
            [[100, 20], [900, 20], [900, 60], [100, 60]],
        ],
    }

    page = PaddlePdfOcrProvider(dpi=72, device="cpu")._page_from_result(
        1, 1000, 1400, 0, data, "fixture.pdf", layout
    )

    assert [block.text for block in page.blocks] == ["시험 제목", "왼쪽 표 내용", "오른쪽 본문"]
    assert [column.index for column in page.columns] == [0, 1]
    assert page.blocks[1].kind == BlockKind.TABLE
    assert page.blocks[1].style["layout_region"] == "table"
    assert page.blocks[2].style["layout_column"] == 1


def test_single_column_keeps_provider_block_order() -> None:
    layout = PageLayout(
        width=1000,
        height=1400,
        column_boxes=((0, 0, 1000, 1400),),
        regions=(),
        gutter_x=None,
        column_confidence=0.0,
    )
    data = {
        "rec_texts": ["공급자 첫 번째", "공급자 두 번째"],
        "rec_scores": [0.9, 0.9],
        "rec_polys": [
            [[100, 500], [400, 500], [400, 540], [100, 540]],
            [[100, 100], [400, 100], [400, 140], [100, 140]],
        ],
    }

    page = PaddlePdfOcrProvider(dpi=72, device="cpu")._page_from_result(
        1, 1000, 1400, 0, data, "fixture.pdf", layout
    )

    assert [block.text for block in page.blocks] == ["공급자 첫 번째", "공급자 두 번째"]


def test_exam_phrase_normalization_is_limited_to_clear_ocr_slips() -> None:
    layout = PageLayout(
        width=1000,
        height=1400,
        column_boxes=((0, 0, 1000, 1400),),
        regions=(),
        gutter_x=None,
        column_confidence=0.0,
    )
    data = {
        "rec_texts": ["것혼?[4.3점]", "보기>를 참고할 때", "것혼 이야기", "것? [4점]"],
        "rec_scores": [0.9, 0.9, 0.9, 0.9],
        "rec_polys": [
            [[100, 100], [300, 100], [300, 130], [100, 130]],
            [[100, 150], [300, 150], [300, 180], [100, 180]],
            [[100, 200], [300, 200], [300, 230], [100, 230]],
            [[100, 250], [300, 250], [300, 280], [100, 280]],
        ],
    }

    page = PaddlePdfOcrProvider(dpi=72, device="cpu")._page_from_result(
        1, 1000, 1400, 0, data, "fixture.pdf", layout
    )

    assert [block.text for block in page.blocks] == [
        "것은?[4.3점]",
        "<보기>를 참고할 때",
        "것혼 이야기",
        "것은? [4점]",
    ]
