from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from scan2hwpx.clean_layout import build_clean_items
from scan2hwpx.hwpx.hancom import render_clean_with_hancom, verify_with_hancom
from scan2hwpx.ir.models import Document
from scan2hwpx.ocr.providers import PaddlePdfOcrProvider
from scan2hwpx.preprocess import preprocess_for_ocr
from scan2hwpx.review import write_review


def convert_pdf(
    input_path: Path,
    output_path: Path,
    dpi: int = 300,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logger = logging.getLogger("scan2hwpx.pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("start input=%s dpi=%s", input_path, dpi)

    def report(page: int, total: int, stage: str) -> None:
        message = f"{input_path.name} · {page}/{total}페이지 {stage}"
        logger.info("page=%s/%s stage=%s", page, total, stage)
        if progress is not None:
            progress(message)

    document = PaddlePdfOcrProvider(dpi=dpi).convert(
        input_path,
        report,
    )
    ir_path = output_dir / "document_ir.json"
    ir_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    _write_debug_images(input_path, document, output_dir / "debug", dpi)
    write_review(document, output_dir)
    if progress is not None:
        progress(f"{input_path.name} · 편집 가능한 한글 문서 생성 중")
    render_clean_with_hancom(document, output_path)
    first_text = next(item.text[:24] for item in build_clean_items(document) if len(item.text) >= 12)
    reopened, text_length = verify_with_hancom(output_path, first_text)
    logger.info(
        "done pages=%s questions=%s reopened=%s text_length=%s",
        len(document.pages),
        len(document.questions),
        reopened,
        text_length,
    )
    if not reopened:
        raise RuntimeError("생성된 HWPX를 한글에서 다시 열거나 OCR 텍스트를 찾지 못했습니다.")
    blocks = [block for page in document.pages for block in page.blocks]
    return {
        "pages": len(document.pages),
        "questions": len(document.questions),
        "blocks": len(blocks),
        "average_confidence": sum(block.confidence for block in blocks) / max(1, len(blocks)),
        "review_required": sum(block.confidence < 0.75 for block in blocks),
        "hancom_reopened": reopened,
        "text_length": text_length,
    }


def inspect_pdf(path: Path) -> dict[str, object]:
    pdf = pymupdf.open(path)  # type: ignore[no-untyped-call]
    try:
        return {
            "path": str(path.resolve()),
            "pages": pdf.page_count,
            "rotations": [pdf[index].rotation for index in range(pdf.page_count)],
            "sizes": [
                [pdf[index].rect.width, pdf[index].rect.height] for index in range(pdf.page_count)
            ],
            "text_characters": [
                len(pdf[index].get_text())  # type: ignore[no-untyped-call]
                for index in range(pdf.page_count)
            ],
        }
    finally:
        pdf.close()  # type: ignore[no-untyped-call]


def _write_debug_images(input_path: Path, document: Document, debug_dir: Path, dpi: int) -> None:
    original_dir = debug_dir / "original"
    preprocessed_dir = debug_dir / "preprocessed"
    overlay_dir = debug_dir / "ocr_overlay"
    crop_dir = debug_dir.parent / "review_crops"
    for directory in (original_dir, preprocessed_dir, overlay_dir, crop_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(input_path)  # type: ignore[no-untyped-call]
    try:
        for page_index, ir_page in enumerate(document.pages):
            pix = pdf[page_index].get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)  # type: ignore[no-untyped-call]
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            preprocessed = preprocess_for_ocr(image)
            original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            clean_bgr = cv2.cvtColor(preprocessed.image, cv2.COLOR_RGB2BGR)
            overlay = clean_bgr.copy()
            for block in ir_page.blocks:
                x0, y0, x1, y1 = (int(value) for value in block.bbox.pixel)
                color = (0, 0, 255) if block.confidence < 0.75 else (0, 160, 0)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
                if block.confidence < 0.75:
                    crop = original_bgr[
                        max(0, y0) : min(pix.height, y1), max(0, x0) : min(pix.width, x1)
                    ]
                    if crop.size:
                        cv2.imwrite(str(crop_dir / f"{block.id}.png"), crop)
            number = page_index + 1
            cv2.imwrite(str(original_dir / f"page-{number}.png"), original_bgr)
            cv2.imwrite(str(preprocessed_dir / f"page-{number}.png"), clean_bgr)
            cv2.imwrite(str(overlay_dir / f"page-{number}.png"), overlay)
    finally:
        pdf.close()  # type: ignore[no-untyped-call]
