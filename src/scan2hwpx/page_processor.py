from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from scan2hwpx.images import write_image
from scan2hwpx.ir.models import Page
from scan2hwpx.preprocess import PreprocessResult
from scan2hwpx.vision.formulas import FormulaProcessor


@dataclass
class PageProcessor:
    """Finish page-local work while the OCR raster is still in memory."""

    output_dir: Path
    source_name: str
    formula_processor: FormulaProcessor | None = None
    progress: Callable[[str], None] | None = None
    write_diagnostics: bool = True
    write_page_images: bool = True
    processed_indexes: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.original_dir = self.output_dir / "debug" / "original"
        self.preprocessed_dir = self.output_dir / "debug" / "preprocessed"
        self.overlay_dir = self.output_dir / "debug" / "ocr_overlay"
        self.crop_dir = self.output_dir / "review_crops"
        if self.write_page_images or self.write_diagnostics:
            self.original_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_dirs = (self.preprocessed_dir, self.overlay_dir, self.crop_dir)
        for directory in diagnostic_dirs if self.write_diagnostics else ():
            directory.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        page_index: int,
        total_pages: int,
        page: Page,
        image_rgb: NDArray[np.uint8],
        preprocessed: PreprocessResult,
    ) -> None:
        if page_index in self.processed_indexes:
            raise ValueError(f"page {page_index + 1} was processed more than once")
        if self.formula_processor is not None:
            if self.progress is not None:
                self.progress(
                    f"{self.source_name} - page {page_index + 1}/{total_pages}: 수식 인식"
                )
            page.formulas = self.formula_processor.process_page(
                image_rgb,
                page.page_no,
                self.output_dir / "formula_crops",
            )
        self._write_debug_artifacts(page, image_rgb, preprocessed)
        self.processed_indexes.add(page_index)

    def handled_all(self, pages: list[Page]) -> bool:
        return self.processed_indexes == set(range(len(pages)))

    def _write_debug_artifacts(
        self,
        page: Page,
        image_rgb: NDArray[np.uint8],
        preprocessed: PreprocessResult,
    ) -> None:
        if not self.write_page_images and not self.write_diagnostics:
            return
        height, width = image_rgb.shape[:2]
        original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        number = page.page_no
        if self.write_page_images or self.write_diagnostics:
            write_image(self.original_dir / f"page-{number}.png", original_bgr)
        if not self.write_diagnostics:
            return
        clean_bgr = cv2.cvtColor(preprocessed.image, cv2.COLOR_RGB2BGR)
        overlay = clean_bgr.copy()
        for block in page.blocks:
            x0, y0, x1, y1 = (int(value) for value in block.bbox.pixel)
            needs_review = block.confidence < 0.75 or bool(block.style.get("review_reasons"))
            color = (0, 0, 255) if needs_review else (0, 160, 0)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
            if needs_review:
                crop = original_bgr[
                    max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)
                ]
                if crop.size:
                    write_image(self.crop_dir / f"{block.id}.png", crop)
        write_image(self.preprocessed_dir / f"page-{number}.png", clean_bgr)
        write_image(self.overlay_dir / f"page-{number}.png", overlay)
