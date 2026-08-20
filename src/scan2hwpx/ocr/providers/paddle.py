from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scan2hwpx.exam_parser import parse_questions
from scan2hwpx.ir.models import (
    AnnotationState,
    BBox,
    Block,
    BlockKind,
    Column,
    Document,
    Page,
    PageQuality,
    QAReport,
)

ProgressCallback = Callable[[int, int, str], None]


class PaddlePdfOcrProvider:
    """Offline Korean OCR provider for image-only PDFs."""

    name = "paddleocr"

    def __init__(self, dpi: int = 150) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self.dpi = dpi
        self._ocr: Any | None = None

    def _engine(self) -> Any:
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]

            self._ocr = PaddleOCR(
                lang="korean",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        return self._ocr

    def convert(self, path: Path, progress: ProgressCallback | None = None) -> Document:
        import numpy as np
        import pymupdf

        from scan2hwpx.preprocess import preprocess_for_ocr

        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            pdf = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise ValueError(f"PDF를 열 수 없습니다: {exc}") from exc
        if pdf.needs_pass:
            pdf.close()  # type: ignore[no-untyped-call]
            raise ValueError("암호화된 PDF는 아직 지원하지 않습니다.")
        if pdf.page_count == 0:
            pdf.close()  # type: ignore[no-untyped-call]
            raise ValueError("페이지가 없는 PDF입니다.")
        pages: list[Page] = []
        engines: list[str] = []
        mask_ratios: list[float] = []
        mask_risks: list[float] = []
        total = pdf.page_count
        try:
            for page_index in range(total):
                if progress:
                    progress(page_index + 1, total, f"{page_index + 1}/{total} 페이지 OCR")
                page = pdf[page_index]
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(  # type: ignore[no-untyped-call]
                        self.dpi / 72, self.dpi / 72
                    ),
                    alpha=False,
                )
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                preprocessed = preprocess_for_ocr(image)
                mask_ratios.append(preprocessed.annotation_ratio)
                mask_risks.append(preprocessed.text_overlap_risk)
                results = list(self._engine().predict(preprocessed.image))
                payload = results[0].json if results else {}
                data = payload.get("res", payload) if isinstance(payload, dict) else {}
                if os.name == "nt":
                    try:
                        from scan2hwpx.ocr.providers.windows import recognize_windows_image

                        windows_lines = recognize_windows_image(preprocessed.image)
                    except (ImportError, OSError, RuntimeError, ValueError):
                        windows_lines = []
                    if windows_lines:
                        data = self._fuse_recognition(data, windows_lines)
                        engines.append("paddle+windows-ocr-ko")
                    else:
                        engines.append("paddleocr")
                else:
                    engines.append("paddleocr")
                pages.append(
                    self._page_from_result(
                        page_index + 1, pix.width, pix.height, page.rotation, data, path.name
                    )
                )
        finally:
            pdf.close()  # type: ignore[no-untyped-call]
        low = [block.id for page in pages for block in page.blocks if block.confidence < 0.75]
        warnings = [] if pages else ["OCR 결과가 없습니다."]
        if any(risk > 0.2 for risk in mask_risks):
            warnings.append("색상 마스크가 인쇄 글자와 겹칠 가능성이 있습니다.")
        return Document(
            id=f"pdf-{raw_hash[:16]}",
            source_hash=raw_hash,
            page_size=(pages[0].width, pages[0].height),
            metadata={
                "source": path.name,
                "provider": self.name,
                "recognition_engines": engines,
                "dpi": self.dpi,
                "annotation_mask_ratios": mask_ratios,
                "annotation_overlap_risks": mask_risks,
            },
            pages=pages,
            questions=parse_questions(pages),
            qa=QAReport(warnings=warnings, low_confidence_blocks=low),
        )

    def _page_from_result(
        self,
        page_no: int,
        width: int,
        height: int,
        rotation: int,
        data: dict[str, Any],
        source_name: str,
    ) -> Page:
        texts = list(data.get("rec_texts", []))
        scores = list(data.get("rec_scores", []))
        polygons = list(data.get("rec_polys", data.get("dt_polys", [])))
        raw: list[tuple[tuple[float, float, float, float], str, float]] = []
        for text, score, polygon in zip(texts, scores, polygons, strict=False):
            if not str(text).strip():
                continue
            x0, y0, x1, y1 = _polygon_bounds(polygon)
            raw.append(((x0, y0, x1, y1), str(text).strip(), float(score)))
        ordered = sorted(raw, key=lambda item: self._reading_key(item[0], width, height))
        blocks: list[Block] = []
        for index, (coords, text, score) in enumerate(ordered):
            x0, y0, x1, y1 = coords
            blocks.append(
                Block(
                    id=f"p{page_no}-b{index + 1}",
                    kind=self._kind(text, y0, y1, height),
                    bbox=BBox(
                        pixel=coords,
                        normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
                    ),
                    reading_order=index,
                    text=text,
                    confidence=max(0.0, min(1.0, score)),
                    source_provider=self.name,
                    source_payload_ref=f"{source_name}#/pages/{page_no}/lines/{index}",
                    annotation_state=AnnotationState.PRINTED,
                )
            )
        left = BBox(pixel=(0, 0, width / 2, height), normalized=(0, 0, 0.5, 1))
        right = BBox(pixel=(width / 2, 0, width, height), normalized=(0.5, 0, 1, 1))
        quality = sum(block.confidence for block in blocks) / max(1, len(blocks))
        return Page(
            page_no=page_no,
            width=width,
            height=height,
            rotation=rotation,
            columns=[Column(index=0, bbox=left), Column(index=1, bbox=right)],
            blocks=blocks,
            quality=PageQuality(score=quality, warnings=[] if blocks else ["텍스트 없음"]),
        )

    @staticmethod
    def _fuse_recognition(
        paddle: dict[str, Any],
        windows: list[tuple[tuple[float, float, float, float], str, float]],
    ) -> dict[str, Any]:
        texts = list(paddle.get("rec_texts", []))
        polygons = list(paddle.get("rec_polys", paddle.get("dt_polys", [])))
        fused = texts.copy()
        for index, (paddle_text, polygon) in enumerate(zip(texts, polygons, strict=False)):
            px0, py0, px1, py1 = _polygon_bounds(polygon)
            best: tuple[float, str] | None = None
            for (wx0, wy0, wx1, wy1), windows_text, _ in windows:
                vertical = max(0.0, min(py1, wy1) - max(py0, wy0)) / max(1.0, py1 - py0)
                horizontal = max(0.0, min(px1, wx1) - max(px0, wx0)) / max(1.0, px1 - px0)
                score = vertical * 0.7 + horizontal * 0.3
                if vertical >= 0.45 and (best is None or score > best[0]):
                    best = (score, windows_text)
            if best is None:
                continue
            windows_text = best[1]
            prefix = re.match(r"^(\d+[.)]|[①②③④⑤])\s*", str(paddle_text))
            if prefix:
                windows_text = re.sub(r"^(?:\d+[.)]|[①②③④⑤⑥⑦⑧⑨⑩]|[•?])\s*", "", windows_text)
                windows_text = f"{prefix.group(1)} {windows_text}"
            if len(windows_text.strip()) >= 2:
                fused[index] = windows_text.strip()
        return {**paddle, "rec_texts": fused}

    @staticmethod
    def _reading_key(
        bbox: tuple[float, float, float, float], width: int, height: int
    ) -> tuple[int, float, float]:
        x0, y0, x1, _ = bbox
        spans_columns = x1 - x0 > width * 0.55
        if spans_columns and y0 < height * 0.2:
            column = 0
        elif spans_columns and y0 > height * 0.8:
            column = 3
        else:
            column = 1 if (x0 + x1) / 2 < width / 2 else 2
        return column, y0, x0

    @staticmethod
    def _kind(text: str, y0: float, y1: float, height: int) -> BlockKind:
        if y0 > height * 0.92:
            return BlockKind.FOOTER
        if y1 < height * 0.08:
            return BlockKind.HEADER
        if re.match(r"^\d+[.)]\s*", text):
            return BlockKind.QUESTION
        if re.match(r"^[①②③④⑤]\s*", text):
            return BlockKind.CHOICE
        if "<보기>" in text or "<자료>" in text:
            return BlockKind.BOX
        return BlockKind.UNKNOWN


def _polygon_bounds(polygon: Any) -> tuple[float, float, float, float]:
    points = [(float(point[0]), float(point[1])) for point in polygon]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)
