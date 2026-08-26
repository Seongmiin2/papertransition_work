from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from scan2hwpx.ir.models import Document
from scan2hwpx.vision.formulas import FormulaProcessor


def extract_formulas(
    input_path: Path,
    document: Document,
    output_dir: Path,
    dpi: int,
    processor: FormulaProcessor,
    progress: Callable[[str], None] | None = None,
    image_cache: dict[int, tuple[Any, Any]] | None = None,
) -> None:
    """Populate page formula IR while keeping the normal OCR path optional.

    ``image_cache`` holds the same-dpi page rasters the OCR provider already produced
    (page_index -> (raw_image, preprocessed)); reusing them skips a second PDF render pass.
    """
    pdf = pymupdf.open(input_path)  # type: ignore[no-untyped-call]
    try:
        total = min(pdf.page_count, len(document.pages))
        for page_index in range(total):
            if progress is not None:
                progress(f"{input_path.name} - page {page_index + 1}/{total}: ?? ??")
            cached = image_cache.get(page_index) if image_cache is not None else None
            if cached is not None:
                image = cached[0]
            else:
                pix = pdf[page_index].get_pixmap(
                    matrix=pymupdf.Matrix(dpi / 72, dpi / 72),  # type: ignore[no-untyped-call]
                    alpha=False,
                )
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
            document.pages[page_index].formulas = processor.process_page(
                image,
                page_index + 1,
                output_dir / "formula_crops",
            )
    finally:
        pdf.close()  # type: ignore[no-untyped-call]
