from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
) -> None:
    """Populate page formula IR while keeping the normal OCR path optional."""
    pdf = pymupdf.open(input_path)  # type: ignore[no-untyped-call]
    try:
        total = min(pdf.page_count, len(document.pages))
        for page_index in range(total):
            if progress is not None:
                progress(f"{input_path.name} - page {page_index + 1}/{total}: ?? ??")
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
