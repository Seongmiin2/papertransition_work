from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PreprocessResult:
    image: NDArray[np.uint8]
    annotation_mask: NDArray[np.uint8]
    annotation_ratio: float
    text_overlap_risk: float


def preprocess_for_ocr(image_rgb: NDArray[np.uint8]) -> PreprocessResult:
    """Remove strongly saturated red/blue strokes and report preservation risk."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (((hue <= 12) | (hue >= 170)) & (saturation >= 75) & (value >= 55)).astype(np.uint8)
    blue = ((hue >= 90) & (hue <= 135) & (saturation >= 65) & (value >= 50)).astype(np.uint8)
    mask = ((red | blue) * 255).astype(np.uint8)
    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    print_ink = gray < 165
    expanded = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    overlap = float(np.logical_and(expanded, print_ink).sum()) / max(1, int(expanded.sum()))
    cleaned = image_rgb.copy()
    # Also clear the anti-aliased edge around pen strokes.  Leaving that one-pixel
    # halo was enough for OCR to turn a red circle beside a question into "7.".
    cleaned[expanded] = 255
    return PreprocessResult(
        image=cleaned,
        annotation_mask=mask,
        annotation_ratio=float(np.count_nonzero(mask)) / mask.size,
        text_overlap_risk=overlap,
    )
