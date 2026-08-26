from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from scan2hwpx.preprocess import preprocess_for_ocr
from scan2hwpx.vision.layout_dataset import LayoutAnnotation, detect_ruled_regions_array

BBoxTuple = tuple[float, float, float, float]


@dataclass(frozen=True)
class PageLayout:
    width: int
    height: int
    column_boxes: tuple[BBoxTuple, ...]
    regions: tuple[LayoutAnnotation, ...]
    gutter_x: float | None
    column_confidence: float

    def column_index(self, bbox: BBoxTuple) -> int | None:
        if len(self.column_boxes) == 1 or self.gutter_x is None:
            return 0
        x0, _y0, x1, _y1 = bbox
        margin = self.width * 0.01
        if x0 < self.gutter_x - margin and x1 > self.gutter_x + margin:
            return None
        return 0 if (x0 + x1) / 2 < self.gutter_x else 1

    def region_for(self, bbox: BBoxTuple) -> LayoutAnnotation | None:
        matches = [
            (_intersection_ratio(bbox, region.bbox), region)
            for region in self.regions
        ]
        score, region = max(matches, default=(0.0, None), key=lambda item: item[0])
        return region if score >= 0.5 else None

    def reading_key(self, bbox: BBoxTuple) -> tuple[int, int, float, float]:
        x0, y0, _x1, y1 = bbox
        column = self.column_index(bbox)
        if len(self.column_boxes) == 1:
            return 1, 0, y0, x0
        if column is not None:
            return 1, column, y0, x0
        if y1 <= self.height * 0.18:
            return 0, 0, y0, x0
        if y0 >= self.height * 0.85:
            return 3, 0, y0, x0
        return 2, 0, y0, x0

    def block_style(self, bbox: BBoxTuple) -> dict[str, Any]:
        region = self.region_for(bbox)
        return {
            "layout_column": self.column_index(bbox),
            "layout_region": region.label if region is not None else None,
            "layout_region_source": region.source if region is not None else None,
            "layout_region_confidence": region.confidence if region is not None else None,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "columns": len(self.column_boxes),
            "gutter_x": self.gutter_x,
            "column_confidence": round(self.column_confidence, 6),
            "regions": [asdict(region) for region in self.regions],
        }


def analyze_page_layout(
    image_rgb: NDArray[np.uint8], *, already_preprocessed: bool = False
) -> PageLayout:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("page layout analysis requires an RGB image")
    if not already_preprocessed:
        image = preprocess_for_ocr(image).image
    height, width = image.shape[:2]
    gutter_x, confidence = _detect_column_gutter(image)
    if gutter_x is None:
        columns: tuple[BBoxTuple, ...] = ((0.0, 0.0, float(width), float(height)),)
    else:
        columns = (
            (0.0, 0.0, gutter_x, float(height)),
            (gutter_x, 0.0, float(width), float(height)),
        )
    regions = tuple(detect_ruled_regions_array(image, already_preprocessed=True))
    return PageLayout(
        width=width,
        height=height,
        column_boxes=columns,
        regions=regions,
        gutter_x=gutter_x,
        column_confidence=confidence,
    )


def _detect_column_gutter(image_rgb: NDArray[np.uint8]) -> tuple[float | None, float]:
    height, width = image_rgb.shape[:2]
    if width < 200 or height < 200:
        return None, 0.0
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < 200).astype(np.uint8)
    vertical_lines = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 8))),
    )
    horizontal_lines = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, width // 12), 1)),
    )
    ruled_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
    text_ink = cv2.bitwise_and(ink, cv2.bitwise_not(ruled_lines))
    joined = cv2.dilate(
        text_ink,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 80), 1)),
    )
    y0, y1 = int(height * 0.08), int(height * 0.92)
    projection = joined[y0:y1].mean(axis=0)
    window = max(7, width // 50)
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    smoothed = np.convolve(projection, kernel, mode="same")
    search_start, search_end = int(width * 0.43), int(width * 0.57)
    if search_end <= search_start:
        return None, 0.0
    central = smoothed[search_start:search_end]
    minimum = float(central.min())
    near_minimum = np.flatnonzero(
        central <= minimum + max(1e-6, float(np.quantile(central, 0.25)) * 0.1)
    )
    page_center = width / 2
    candidate = search_start + min(
        (int(index) for index in near_minimum),
        key=lambda index: abs(search_start + index - page_center),
    )
    vertical_strength = vertical_lines[y0:y1].mean(axis=0)
    divider_x = search_start + int(np.argmax(vertical_strength[search_start:search_end]))
    divider_strength = float(vertical_strength[divider_x])
    gutter_density = float(smoothed[candidate])
    reference_values = np.concatenate(
        (smoothed[int(width * 0.1) : int(width * 0.4)], smoothed[int(width * 0.6) : int(width * 0.9)])
    )
    reference_density = float(np.quantile(reference_values, 0.7))
    if reference_density <= 0.002:
        return None, 0.0
    left_ink = float(text_ink[y0:y1, int(width * 0.05) : candidate].sum())
    right_ink = float(text_ink[y0:y1, candidate : int(width * 0.95)].sum())
    balance = min(left_ink, right_ink) / max(left_ink, right_ink, 1.0)
    density_ratio = gutter_density / reference_density
    whitespace_confidence = max(
        0.0, min(1.0, (1.0 - density_ratio) * min(1.0, balance * 2.0))
    )
    divider_supported = (
        divider_strength >= 0.15
        and density_ratio <= 0.85
        and balance >= 0.15
        and abs(divider_x - width / 2) <= width * 0.07
    )
    if divider_supported:
        divider_confidence = min(1.0, 0.55 + (divider_strength - 0.15) * 1.5)
        return float(divider_x), max(whitespace_confidence, divider_confidence)
    if density_ratio > 0.45 or balance < 0.15 or whitespace_confidence < 0.55:
        return None, whitespace_confidence
    return float(candidate), whitespace_confidence


def _intersection_ratio(first: BBoxTuple, second: BBoxTuple) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(1.0, first[2] - first[0]) * max(1.0, first[3] - first[1])
    return intersection / first_area
