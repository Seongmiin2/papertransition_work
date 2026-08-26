from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf

from scan2hwpx.clean_layout import build_clean_items
from scan2hwpx.hwpx import (
    render_clean_hwpx,
    render_fidelity_hwpx,
    render_semantic_hwpx,
    validate_hwpx,
)
from scan2hwpx.hwpx.hancom import render_clean_with_hancom
from scan2hwpx.images import write_image
from scan2hwpx.ir.models import Document
from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.preprocess import preprocess_for_ocr
from scan2hwpx.review import write_review
from scan2hwpx.vision.dataset import write_formula_manifest
from scan2hwpx.vision.formulas import FormulaProcessor
from scan2hwpx.vision.layout_dataset import build_layout_seed_dataset
from scan2hwpx.vision.pdf import extract_formulas as _extract_formulas


def convert_pdf(
    input_path: Path,
    output_path: Path,
    dpi: int = 240,
    progress: Callable[[str], None] | None = None,
    formula_processor: FormulaProcessor | None = None,
    ocr_provider: PaddlePdfOcrProvider | None = None,
    renderer: str = "fidelity",
) -> dict[str, object]:
    if renderer not in {"fidelity", "semantic", "portable", "hancom"}:
        raise ValueError("renderer must be fidelity, semantic, portable, or hancom")
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
        message = f"{input_path.name} - page {page}/{total}: {stage}"
        logger.info("page=%s/%s stage=%s", page, total, stage)
        if progress is not None:
            progress(message)

    provider = ocr_provider or PaddlePdfOcrProvider(dpi=dpi)
    page_image_cache: dict[int, tuple[Any, Any]] = {}
    document = provider.convert(input_path, report, image_cache=page_image_cache)
    if formula_processor is not None:
        _extract_formulas(
            input_path, document, output_dir, dpi, formula_processor, progress, page_image_cache
        )
        write_formula_manifest(document, output_dir / "formula_training.jsonl")
    blocks = [block for page in document.pages for block in page.blocks]
    clean_items = build_clean_items(document)
    if not blocks or not any(block.text.strip() for block in blocks):
        raise RuntimeError("OCR did not return any readable text. Check the PDF image quality.")
    if not clean_items:
        raise RuntimeError("OCR text was found, but no printable document items could be built.")

    ir_path = output_dir / "document_ir.json"
    ir_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    _write_debug_images(input_path, document, output_dir / "debug", dpi, page_image_cache)
    write_review(document, output_dir)

    if progress is not None:
        progress(f"{input_path.name} - writing editable HWPX")

    candidate = output_path.with_name(f".{output_path.stem}.candidate{output_path.suffix}")
    candidate.unlink(missing_ok=True)
    render_engine = "hancom-template-hwpx"
    reopened = False
    text_length = 0
    picture_controls = 0
    table_controls = 0
    editable_text_boxes = 0
    editable_text_characters = 0
    page_images = [
        output_dir / "debug" / "original" / f"page-{page.page_no}.png" for page in document.pages
    ]
    if renderer == "semantic":
        if progress is not None:
            progress(f"{input_path.name} - detecting tables, boxes, and pictures")
        layout_dir = output_dir / "layout_training"
        build_layout_seed_dataset(
            input_path,
            layout_dir,
            device=provider.device,
            dpi=min(dpi, 200),
        )
        semantic_stats = render_semantic_hwpx(
            page_images,
            document,
            layout_dir / "layout_seed.json",
            candidate,
        )
        result = validate_hwpx(candidate)
        if not result.valid:
            raise RuntimeError("Semantic HWPX validation failed: " + "; ".join(result.errors))
        render_engine = "semantic-layout-hwpx"
        picture_controls = semantic_stats.background_pictures + semantic_stats.extracted_pictures
        table_controls = semantic_stats.editable_tables
        editable_text_boxes = semantic_stats.editable_text_boxes
        editable_text_characters = semantic_stats.editable_text_characters
    elif renderer == "fidelity":
        fidelity_stats = render_fidelity_hwpx(page_images, candidate)
        result = validate_hwpx(candidate)
        if not result.valid:
            raise RuntimeError("Fidelity HWPX validation failed: " + "; ".join(result.errors))
        render_engine = "fidelity-page-image-hwpx"
        picture_controls = fidelity_stats.pictures
    elif renderer == "portable":
        render_clean_hwpx(document, candidate)
        result = validate_hwpx(candidate)
        if not result.valid:
            raise RuntimeError("HWPX package validation failed: " + "; ".join(result.errors))
        text_length = sum(len(item.text) for item in clean_items)
    else:
        render_engine = "hancom"
        try:
            render_clean_with_hancom(document, candidate)
            result = validate_hwpx(candidate)
            if not result.valid:
                raise RuntimeError(
                    "Hancom output failed package validation: " + "; ".join(result.errors)
                )
            text_length = sum(len(item.text) for item in clean_items)
        except Exception as exc:
            logger.exception("hancom render failed; falling back to package renderer")
            candidate.unlink(missing_ok=True)
            render_engine = "hancom-template-hwpx"
            if progress is not None:
                progress(f"{input_path.name} - Hancom automation failed; writing fallback HWPX")
            render_clean_hwpx(document, candidate)
            result = validate_hwpx(candidate)
            if not result.valid:
                raise RuntimeError(
                    "Fallback HWPX validation failed: " + "; ".join(result.errors)
                ) from exc
            text_length = sum(len(item.text) for item in clean_items)

    candidate.replace(output_path)
    logger.info(
        "done pages=%s questions=%s engine=%s reopened=%s text_length=%s",
        len(document.pages),
        len(document.questions),
        render_engine,
        reopened,
        text_length,
    )
    return {
        "pages": len(document.pages),
        "questions": len(document.questions),
        "blocks": len(blocks),
        "average_confidence": sum(block.confidence for block in blocks) / max(1, len(blocks)),
        "review_required": len(document.qa.low_confidence_blocks),
        "hancom_reopened": reopened,
        "render_engine": render_engine,
        "text_length": text_length,
        "picture_controls": picture_controls,
        "table_controls": table_controls,
        "editable_text_boxes": editable_text_boxes,
        "editable_text_characters": editable_text_characters,
        "visual_fidelity": renderer in {"fidelity", "semantic"},
        "editable_layout": renderer != "fidelity",
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


def _write_debug_images(
    input_path: Path,
    document: Document,
    debug_dir: Path,
    dpi: int,
    image_cache: dict[int, tuple[Any, Any]] | None = None,
) -> None:
    original_dir = debug_dir / "original"
    preprocessed_dir = debug_dir / "preprocessed"
    overlay_dir = debug_dir / "ocr_overlay"
    crop_dir = debug_dir.parent / "review_crops"
    for directory in (original_dir, preprocessed_dir, overlay_dir, crop_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(input_path)  # type: ignore[no-untyped-call]
    try:
        for page_index, ir_page in enumerate(document.pages):
            cached = image_cache.get(page_index) if image_cache is not None else None
            if cached is not None:
                image, preprocessed = cached
                pix_height, pix_width = image.shape[0], image.shape[1]
            else:
                matrix = pymupdf.Matrix(dpi / 72, dpi / 72)  # type: ignore[no-untyped-call]
                pix = pdf[page_index].get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                preprocessed = preprocess_for_ocr(image)
                pix_height, pix_width = pix.height, pix.width
            original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            clean_bgr = cv2.cvtColor(preprocessed.image, cv2.COLOR_RGB2BGR)
            overlay = clean_bgr.copy()
            review_ids = set(document.qa.low_confidence_blocks)
            for block in ir_page.blocks:
                x0, y0, x1, y1 = (int(value) for value in block.bbox.pixel)
                needs_review = block.id in review_ids
                color = (0, 0, 255) if needs_review else (0, 160, 0)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
                if needs_review:
                    crop = original_bgr[
                        max(0, y0) : min(pix_height, y1), max(0, x0) : min(pix_width, x1)
                    ]
                    if crop.size:
                        write_image(crop_dir / f"{block.id}.png", crop)
            number = page_index + 1
            write_image(original_dir / f"page-{number}.png", original_bgr)
            write_image(preprocessed_dir / f"page-{number}.png", clean_bgr)
            write_image(overlay_dir / f"page-{number}.png", overlay)
    finally:
        pdf.close()  # type: ignore[no-untyped-call]
