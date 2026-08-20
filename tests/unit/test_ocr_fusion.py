from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider


def test_fusion_keeps_paddle_question_number_and_uses_windows_body() -> None:
    paddle = {
        "rec_texts": ["1. 기달이는 손님"],
        "rec_scores": [0.8],
        "rec_polys": [[[0, 0], [300, 0], [300, 50], [0, 50]]],
    }
    windows = [((0, 0, 300, 50), "7. 기다리는 손님", 0.9)]
    fused = PaddlePdfOcrProvider._fuse_recognition(paddle, windows)
    assert fused["rec_texts"] == ["1. 기다리는 손님"]


def test_fusion_keeps_choice_symbol() -> None:
    paddle = {
        "rec_texts": ["① @: 선비"],
        "rec_scores": [0.8],
        "rec_polys": [[[0, 0], [300, 0], [300, 50], [0, 50]]],
    }
    windows = [((0, 0, 300, 50), "㉠: 선비", 0.9)]
    fused = PaddlePdfOcrProvider._fuse_recognition(paddle, windows)
    assert fused["rec_texts"] == ["① ㉠: 선비"]
