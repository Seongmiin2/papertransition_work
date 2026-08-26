import numpy as np

from scan2hwpx.preprocess import preprocess_for_ocr


def test_preprocess_removes_saturated_red_but_preserves_black() -> None:
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    image[4:8, 4:8] = [255, 0, 0]
    image[12:16, 12:16] = [0, 0, 0]
    result = preprocess_for_ocr(image)
    assert np.all(result.image[4:8, 4:8] == 255)
    assert np.all(result.image[12:16, 12:16] == 0)
    assert result.annotation_ratio > 0
