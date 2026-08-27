from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from scan2hwpx.ocr.quality import KoreanQualityRouter, needs_secondary_recognition

if TYPE_CHECKING:
    from scan2hwpx.vision.page_layout import PageLayout

ProgressCallback = Callable[[int, int, str], None]
PageCallback = Callable[[int, int, Page, Any, Any], None]
_LAYOUT_KINDS = {
    "table": BlockKind.TABLE,
    "passage_box": BlockKind.PASSAGE_BOX,
}


class PaddlePdfOcrProvider:
    """Offline Korean OCR provider for image-only PDFs."""

    name = "paddleocr"

    def __init__(
        self,
        dpi: int = 240,
        fusion_mode: str = "fast",
        lexicon_path: Path | None = None,
        enable_mkldnn: bool | None = None,
        device: str = "auto",
        recognition_model_dir: Path | None = None,
        page_anomaly_model: Path | None = None,
    ) -> None:
        if fusion_mode not in {"fast", "balanced", "accurate"}:
            raise ValueError("fusion_mode must be fast, balanced, or accurate")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self.dpi = dpi
        self.fusion_mode = fusion_mode
        self.quality_router = KoreanQualityRouter(lexicon_path)
        self.device = _resolve_device(device)
        self.recognition_model_dir = (
            recognition_model_dir.resolve() if recognition_model_dir else None
        )
        self.page_anomaly_model_path = page_anomaly_model.resolve() if page_anomaly_model else None
        if self.page_anomaly_model_path and not self.page_anomaly_model_path.is_file():
            raise FileNotFoundError(f"page anomaly model not found: {self.page_anomaly_model_path}")
        self._page_anomaly_model: Any | None = None
        self.enable_mkldnn = (
            os.name != "nt" and self.device == "cpu"
            if enable_mkldnn is None
            else enable_mkldnn and self.device == "cpu"
        )
        self.cpu_threads = max(1, min(6, os.cpu_count() or 1))
        self._ocr: Any | None = None

    def _engine(self) -> Any:
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]

            recognition_options: dict[str, Any] = (
                {
                    "text_recognition_model_name": "korean_PP-OCRv5_mobile_rec",
                    "text_recognition_model_dir": str(self.recognition_model_dir),
                }
                if self.recognition_model_dir
                else {"text_recognition_model_name": "korean_PP-OCRv5_mobile_rec"}
            )
            self._ocr = PaddleOCR(
                lang="korean",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_batch_size=6,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=self.enable_mkldnn,
                cpu_threads=self.cpu_threads,
                device=self.device,
                **recognition_options,
            )
        return self._ocr

    def _predict(self, image: Any) -> list[Any]:
        try:
            return list(self._engine().predict(image))
        except RuntimeError:
            if not self.enable_mkldnn:
                raise
            self.enable_mkldnn = False
            self._ocr = None
            return list(self._engine().predict(image))

    def convert(
        self,
        path: Path,
        progress: ProgressCallback | None = None,
        page_callback: PageCallback | None = None,
    ) -> Document:
        import numpy as np
        import pymupdf

        from scan2hwpx.preprocess import preprocess_for_ocr
        from scan2hwpx.vision.page_layout import analyze_page_layout

        raw_hash = _sha256(path)
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
        anomaly_scores: list[float | None] = []
        anomaly_flags: list[bool] = []
        layout_summaries: list[dict[str, Any]] = []
        total = pdf.page_count
        try:
            for page_index in range(total):
                if progress:
                    progress(page_index + 1, total, f"{page_index + 1}/{total} 페이지 OCR")
                page = pdf[page_index]
                anomaly_score, anomaly_flag = self._score_page_anomaly(page)
                anomaly_scores.append(anomaly_score)
                anomaly_flags.append(anomaly_flag)
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
                layout = analyze_page_layout(preprocessed.image, already_preprocessed=True)
                layout_summaries.append(layout.summary())
                mask_ratios.append(preprocessed.annotation_ratio)
                mask_risks.append(preprocessed.text_overlap_risk)
                text_layer_data = _text_layer_page_data(page, self.dpi)
                if text_layer_data is not None:
                    data = text_layer_data
                    engine_name = "pdf-text-layer"
                else:
                    results = self._predict(preprocessed.image)
                    payload = results[0].json if results else {}
                    raw_data = payload.get("res", payload) if isinstance(payload, dict) else {}
                    data = raw_data if isinstance(raw_data, dict) else {}
                    engine_name = "paddleocr"
                    if os.name == "nt" and self.fusion_mode != "fast":
                        try:
                            from scan2hwpx.ocr.providers.windows import (
                                recognize_windows_image,
                                recognize_windows_regions,
                            )

                            if self.fusion_mode == "accurate":
                                windows_lines = recognize_windows_image(preprocessed.image)
                            else:
                                windows_lines = recognize_windows_regions(
                                    preprocessed.image, self._secondary_regions(data)
                                )
                        except (ImportError, OSError, RuntimeError, ValueError):
                            windows_lines = []
                        if windows_lines:
                            data = self._fuse_recognition(
                                data, windows_lines, quality_router=self.quality_router
                            )
                            engine_name = f"paddle+windows-ocr-ko:{self.fusion_mode}"
                engines.append(engine_name)
                parsed_page = self._page_from_result(
                    page_index + 1,
                    pix.width,
                    pix.height,
                    page.rotation,
                    data,
                    path.name,
                    layout,
                )
                if anomaly_flag and anomaly_score is not None:
                    parsed_page.quality.warnings.append(
                        f"페이지 분포 이탈 감지: reconstruction_mse={anomaly_score:.8f}"
                    )
                if page_callback is not None:
                    page_callback(page_index, total, parsed_page, image, preprocessed)
                pages.append(parsed_page)
        finally:
            pdf.close()  # type: ignore[no-untyped-call]
        review_blocks = [
            block.id
            for page in pages
            for block in page.blocks
            if block.confidence < 0.75 or block.style.get("review_reasons")
        ]
        warnings = [] if pages else ["OCR 결과가 없습니다."]
        if any(risk > 0.2 for risk in mask_risks):
            warnings.append("색상 마스크가 인쇄 글자와 겹칠 가능성이 있습니다.")
        if any(anomaly_flags):
            warnings.append(
                f"학습 분포와 다른 페이지 {sum(anomaly_flags)}개를 정밀 검토 대상으로 표시했습니다."
            )
        return Document(
            id=f"pdf-{raw_hash[:16]}",
            source_hash=raw_hash,
            page_size=(pages[0].width, pages[0].height),
            metadata={
                "source": path.name,
                "provider": self.name,
                "recognition_engines": engines,
                "fusion_mode": self.fusion_mode,
                "mkldnn": self.enable_mkldnn,
                "cpu_threads": self.cpu_threads,
                "device": self.device,
                "dpi": self.dpi,
                "recognition_model_dir": (
                    str(self.recognition_model_dir) if self.recognition_model_dir else None
                ),
                "page_anomaly_model": (
                    str(self.page_anomaly_model_path) if self.page_anomaly_model_path else None
                ),
                "page_anomaly_scores": anomaly_scores,
                "page_anomaly_flags": anomaly_flags,
                "annotation_mask_ratios": mask_ratios,
                "annotation_overlap_risks": mask_risks,
                "page_layouts": layout_summaries,
            },
            pages=pages,
            questions=parse_questions(pages),
            qa=QAReport(warnings=warnings, low_confidence_blocks=review_blocks),
        )

    def _score_page_anomaly(self, page: Any) -> tuple[float | None, bool]:
        if self.page_anomaly_model_path is None:
            return None, False
        if self._page_anomaly_model is None:
            from scan2hwpx.training.page_anomaly import load_page_anomaly_model

            self._page_anomaly_model = load_page_anomaly_model(self.page_anomaly_model_path)
        from scan2hwpx.training.page_anomaly import page_thumbnail

        sample = page_thumbnail(page, self._page_anomaly_model.thumbnail_size)[None, :]
        score = float(self._page_anomaly_model.score(sample)[0])
        return score, score > self._page_anomaly_model.threshold

    def _page_from_result(
        self,
        page_no: int,
        width: int,
        height: int,
        rotation: int,
        data: dict[str, Any],
        source_name: str,
        layout: PageLayout,
    ) -> Page:
        texts = list(data.get("rec_texts", []))
        scores = list(data.get("rec_scores", []))
        polygons = list(data.get("rec_polys", data.get("dt_polys", [])))
        candidates = list(data.get("rec_candidates", []))
        reasons = list(data.get("rec_review_reasons", []))
        raw: list[
            tuple[
                tuple[float, float, float, float],
                str,
                float,
                list[dict[str, Any]],
                list[str],
            ]
        ] = []
        for index, (text, score, polygon) in enumerate(zip(texts, scores, polygons, strict=False)):
            if not str(text).strip():
                continue
            x0, y0, x1, y1 = _polygon_bounds(polygon)
            item_candidates = candidates[index] if index < len(candidates) else []
            item_reasons = reasons[index] if index < len(reasons) else []
            raw.append(
                (
                    (x0, y0, x1, y1),
                    _normalize_exam_text(str(text).strip()),
                    float(score),
                    list(item_candidates),
                    list(item_reasons),
                )
            )
        ordered = (
            sorted(raw, key=lambda item: layout.reading_key(item[0]))
            if len(layout.column_boxes) > 1
            else raw
        )
        blocks: list[Block] = []
        for index, (coords, text, score, item_candidates, item_reasons) in enumerate(ordered):
            x0, y0, x1, y1 = coords
            style = {
                "ocr_candidates": item_candidates,
                "review_reasons": item_reasons,
                **layout.block_style(coords),
            }
            kind = self._kind(text, y0, y1, height)
            if kind == BlockKind.UNKNOWN:
                kind = _LAYOUT_KINDS.get(str(style["layout_region"]), kind)
            blocks.append(
                Block(
                    id=f"p{page_no}-b{index + 1}",
                    kind=kind,
                    bbox=BBox(
                        pixel=coords,
                        normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
                    ),
                    reading_order=index,
                    text=text,
                    style=style,
                    confidence=max(0.0, min(1.0, score)),
                    source_provider=self.name,
                    source_payload_ref=f"{source_name}#/pages/{page_no}/lines/{index}",
                    annotation_state=AnnotationState.PRINTED,
                )
            )
        columns = [
            Column(
                index=index,
                bbox=BBox(
                    pixel=box,
                    normalized=(
                        box[0] / width,
                        box[1] / height,
                        box[2] / width,
                        box[3] / height,
                    ),
                ),
            )
            for index, box in enumerate(layout.column_boxes)
        ]
        quality = sum(block.confidence for block in blocks) / max(1, len(blocks))
        review_count = sum(bool(block.style.get("review_reasons")) for block in blocks)
        warnings = [f"OCR 엔진 불일치 {review_count}개"] if review_count else []
        if not blocks:
            warnings.append("텍스트 없음")
        return Page(
            page_no=page_no,
            width=width,
            height=height,
            rotation=rotation,
            columns=columns,
            blocks=blocks,
            quality=PageQuality(score=quality, warnings=warnings),
        )

    @staticmethod
    def _fuse_recognition(
        paddle: dict[str, Any],
        windows: list[tuple[tuple[float, float, float, float], str, float]],
        quality_router: KoreanQualityRouter | None = None,
    ) -> dict[str, Any]:
        texts = list(paddle.get("rec_texts", []))
        scores = [float(score) for score in paddle.get("rec_scores", [])]
        polygons = list(paddle.get("rec_polys", paddle.get("dt_polys", [])))
        fused = texts.copy()
        fused_scores = scores.copy()
        candidates: list[list[dict[str, Any]]] = [[] for _ in texts]
        review_reasons: list[list[str]] = [[] for _ in texts]
        router = quality_router or KoreanQualityRouter()
        for index, (paddle_text, polygon) in enumerate(zip(texts, polygons, strict=False)):
            px0, py0, px1, py1 = _polygon_bounds(polygon)
            best: tuple[float, str, float] | None = None
            for (wx0, wy0, wx1, wy1), windows_text, windows_confidence in windows:
                vertical = max(0.0, min(py1, wy1) - max(py0, wy0)) / max(1.0, py1 - py0)
                horizontal = max(0.0, min(px1, wx1) - max(px0, wx0)) / max(1.0, px1 - px0)
                overlap_score = vertical * 0.7 + horizontal * 0.3
                if (
                    vertical >= 0.45
                    and horizontal >= 0.25
                    and (best is None or overlap_score > best[0])
                ):
                    best = (overlap_score, windows_text, windows_confidence)
            paddle_confidence = scores[index] if index < len(scores) else 0.0
            decision = router.decide(
                str(paddle_text),
                paddle_confidence,
                best[1] if best else None,
                best[2] if best else 0.88,
            )
            fused[index] = decision.selected.text
            if index < len(fused_scores):
                fused_scores[index] = decision.selected.confidence
            candidates[index] = [
                {
                    "engine": candidate.engine,
                    "text": candidate.text,
                    "confidence": candidate.confidence,
                    "quality": candidate.quality,
                    "selected": candidate.engine == decision.selected.engine,
                }
                for candidate in decision.candidates
            ]
            review_reasons[index] = list(decision.review_reasons)
        return {
            **paddle,
            "rec_texts": fused,
            "rec_scores": fused_scores,
            "rec_candidates": candidates,
            "rec_review_reasons": review_reasons,
        }

    @staticmethod
    def _secondary_regions(data: dict[str, Any]) -> list[tuple[float, float, float, float]]:
        texts = list(data.get("rec_texts", []))
        scores = list(data.get("rec_scores", []))
        polygons = list(data.get("rec_polys", data.get("dt_polys", [])))
        return [
            _polygon_bounds(polygon)
            for text, score, polygon in zip(texts, scores, polygons, strict=False)
            if needs_secondary_recognition(str(text), float(score))
        ]

    @staticmethod
    def _kind(text: str, y0: float, y1: float, height: int) -> BlockKind:
        if y0 > height * 0.92:
            return BlockKind.FOOTER
        if y1 < height * 0.08:
            return BlockKind.HEADER
        if re.match(r"^\d+[.)]\s*", text):
            return BlockKind.QUESTION
        if re.match(r"^[①-⑳㉠-㉻ⓐ-ⓩ]\s*", text):
            return BlockKind.CHOICE
        if "<보기>" in text or "<자료>" in text:
            return BlockKind.BOX
        return BlockKind.UNKNOWN


def _polygon_bounds(polygon: Any) -> tuple[float, float, float, float]:
    points = [(float(point[0]), float(point[1])) for point in polygon]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _text_layer_page_data(page: Any, dpi: int, min_characters: int = 20) -> dict[str, Any] | None:
    """Build a PaddleOCR-shaped result dict straight from a PDF's own text layer.

    Returns None when the page has no real (born-digital) text layer worth trusting,
    so the caller falls back to rasterize+OCR. Coordinates are scaled by dpi/72 to
    match the pixel space of the page image rasterized at the same dpi.
    """
    if len(page.get_text().strip()) < min_characters:
        return None
    scale = dpi / 72
    rec_texts: list[str] = []
    rec_scores: list[float] = []
    rec_polys: list[list[tuple[float, float]]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if not line_text.strip():
                continue
            x0, y0, x1, y1 = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            rec_texts.append(line_text)
            rec_scores.append(1.0)
            rec_polys.append([(x0 * scale, y0 * scale), (x1 * scale, y1 * scale)])
    if not rec_texts:
        return None
    return {"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_polys": rec_polys}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_exam_text(text: str) -> str:
    """Fix only high-confidence, score-adjacent Korean exam OCR slips."""
    text = re.sub(r"것[혼흔훈]\?(?=\s*(?:\[|$))", "것은?", text)
    text = re.sub(r"(?<![가-힣])것\?(?=\s*(?:\[|$))", "것은?", text)
    return re.sub(r"(?<!<)보기>", "<보기>", text)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import paddle

        return "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    except (ImportError, RuntimeError):
        return "cpu"
