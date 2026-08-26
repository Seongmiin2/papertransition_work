from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageOps

from scan2hwpx.preprocess import preprocess_for_ocr

LAYOUT_CLASSES = (
    "title",
    "instruction",
    "text",
    "question",
    "choice",
    "passage_box",
    "page_frame",
    "table",
    "image",
    "caption",
    "formula",
    "header",
    "footer",
    "url_footnote",
    "page_number",
)

PADDLE_LABEL_MAP = {
    "text": "text",
    "title": "title",
    "image": "image",
    "table": "table",
    "formula": "formula",
    "header": "header",
    "footer": "footer",
    "footnote": "url_footnote",
    "number": "page_number",
    "reference": "url_footnote",
}


@dataclass(frozen=True)
class LayoutAnnotation:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str
    source_label: str
    review_required: bool = True


@dataclass(frozen=True)
class TableGrid:
    bbox: tuple[float, float, float, float]
    x_lines: tuple[float, ...]
    y_lines: tuple[float, ...]

    @property
    def rows(self) -> int:
        return max(0, len(self.y_lines) - 1)

    @property
    def columns(self) -> int:
        return max(0, len(self.x_lines) - 1)


def build_layout_seed_dataset(
    input_pdf: Path,
    output_dir: Path,
    *,
    model_name: str = "PP-DocLayout-S",
    device: str = "auto",
    dpi: int = 200,
    threshold: float = 0.3,
    page_images: list[Path] | None = None,
) -> dict[str, Any]:
    """Create reviewable exam-layout seed labels; these are not verified training truth."""
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if page_images is None:
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        page_images = _rasterize_pdf(input_pdf, image_dir, dpi)

    from paddleocr import LayoutDetection  # type: ignore[import-untyped]

    resolved_device = _resolve_device(device)
    model: Any = LayoutDetection(
        model_name=model_name,
        device=resolved_device,
        threshold=threshold,
    )
    pages: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1
    category_ids = {name: index for index, name in enumerate(LAYOUT_CLASSES, start=1)}
    for page_no, image_path in enumerate(page_images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        predictions = list(model.predict(str(image_path), batch_size=1))
        paddle_annotations = _paddle_annotations(predictions, width, height)
        cv_annotations = detect_ruled_regions(image_path)
        annotations = _resolve_cross_class(
            _deduplicate([*paddle_annotations, *cv_annotations])
        )
        page = {
            "page_no": page_no,
            "image": str(image_path.resolve()),
            "width": width,
            "height": height,
            "annotations": [asdict(item) for item in annotations],
        }
        pages.append(page)
        coco_images.append(
            {"id": page_no, "file_name": image_path.name, "width": width, "height": height}
        )
        for item in annotations:
            x0, y0, x1, y1 = item.bbox
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": page_no,
                    "category_id": category_ids[item.label],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                    "confidence": item.confidence,
                    "source": item.source,
                    "review_required": item.review_required,
                }
            )
            annotation_id += 1
        _write_overlay(image_path, overlay_dir / image_path.name, annotations)

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "source_pdf": str(input_pdf.resolve()),
        "model": model_name,
        "device": resolved_device,
        "classes": list(LAYOUT_CLASSES),
        "label_status": "pseudo_label_needs_human_review",
        "pages": pages,
    }
    coco = {
        "info": {
            "description": "Exam layout seed labels; verify before model fine-tuning",
            "label_status": "pseudo_label_needs_human_review",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "layout"}
            for name in LAYOUT_CLASSES
        ],
    }
    _write_json(output_dir / "layout_seed.json", payload)
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)
    _write_json(annotations_dir / "instances_seed.json", coco)
    return {
        "pages": len(pages),
        "annotations": len(coco_annotations),
        "model": model_name,
        "device": resolved_device,
        "output": str(output_dir.resolve()),
        "label_status": "pseudo_label_needs_human_review",
    }


def detect_ruled_regions(path: Path) -> list[LayoutAnnotation]:
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    return detect_ruled_regions_array(rgb)


def detect_ruled_regions_array(
    rgb: np.ndarray[Any, Any], *, already_preprocessed: bool = False
) -> list[LayoutAnnotation]:
    """Detect ruled layout regions directly from an in-memory RGB page."""
    horizontal, vertical = _ruled_masks(rgb, already_preprocessed=already_preprocessed)
    height, width = horizontal.shape
    ruled = cv2.morphologyEx(
        cv2.bitwise_or(horizontal, vertical),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours = cv2.findContours(ruled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    regions: list[LayoutAnnotation] = []
    page_area = width * height
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_ratio = box_width * box_height / page_area
        if (
            box_width < width * 0.12
            or box_height < height * 0.025
            or area_ratio < 0.003
            or area_ratio > 0.45
        ):
            continue
        h_lines = _long_line_count(horizontal[y : y + box_height, x : x + box_width], True)
        v_lines = _long_line_count(vertical[y : y + box_height, x : x + box_width], False)
        label = "table" if h_lines >= 3 and v_lines >= 2 else "passage_box"
        regions.append(
            LayoutAnnotation(
                label=label,
                bbox=(float(x), float(y), float(x + box_width), float(y + box_height)),
                confidence=0.55 if label == "table" else 0.5,
                source="opencv_ruled_region",
                source_label=label,
            )
        )
    return _deduplicate(regions, iou_threshold=0.75)


def detect_table_grids(path: Path) -> list[TableGrid]:
    """Return complete ruled grids; merged-cell topology still requires human verification."""
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    horizontal, vertical = _ruled_masks(rgb)
    height, width = horizontal.shape
    ruled = cv2.morphologyEx(
        cv2.bitwise_or(horizontal, vertical),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours = cv2.findContours(ruled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    grids: list[TableGrid] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_ratio = box_width * box_height / max(1, width * height)
        if (
            box_width < width * 0.12
            or box_height < height * 0.025
            or area_ratio < 0.003
            or area_ratio > 0.45
        ):
            continue
        h_positions = _line_positions(
            horizontal[y : y + box_height, x : x + box_width], True, float(y)
        )
        v_positions = _line_positions(
            vertical[y : y + box_height, x : x + box_width], False, float(x)
        )
        if len(h_positions) < 3 or len(v_positions) < 2:
            continue
        h_positions = _merge_positions(
            [*h_positions, *_hough_positions(rgb, (x, y, box_width, box_height), True)]
        )
        v_positions = _merge_positions(
            [*v_positions, *_hough_positions(rgb, (x, y, box_width, box_height), False)]
        )
        grids.append(
            TableGrid(
                bbox=(float(x), float(y), float(x + box_width), float(y + box_height)),
                x_lines=tuple(v_positions),
                y_lines=tuple(h_positions),
            )
        )
    return grids


def _ruled_masks(
    rgb: np.ndarray[Any, Any], *, already_preprocessed: bool = False
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if not already_preprocessed:
        rgb = preprocess_for_ocr(rgb).image
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = binary.shape
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 100), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 120))),
    )
    return horizontal, vertical


def _rasterize_pdf(input_pdf: Path, image_dir: Path, dpi: int) -> list[Path]:
    pdf = pymupdf.open(input_pdf)  # type: ignore[no-untyped-call]
    paths: list[Path] = []
    try:
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)  # type: ignore[no-untyped-call]
        for page_index in range(pdf.page_count):
            pix = pdf[page_index].get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            path = image_dir / f"page-{page_index + 1}.jpg"
            image.save(path, "JPEG", quality=92, optimize=True, dpi=(dpi, dpi))
            paths.append(path)
    finally:
        pdf.close()  # type: ignore[no-untyped-call]
    return paths


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import paddle

        is_cuda = paddle.is_compiled_with_cuda  # type: ignore[attr-defined]
        if is_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception as exc:  # noqa: BLE001 - optional accelerator discovery
        logging.getLogger(__name__).debug("GPU discovery failed: %s", exc)
    return "cpu"


def _paddle_annotations(
    predictions: list[Any], width: int, height: int
) -> list[LayoutAnnotation]:
    if not predictions:
        return []
    payload = predictions[0].json.get("res", {})
    annotations: list[LayoutAnnotation] = []
    for box in payload.get("boxes", []):
        source_label = str(box.get("label", "unknown"))
        label = PADDLE_LABEL_MAP.get(source_label, "text")
        coordinates = box.get("coordinate", [])
        if len(coordinates) != 4:
            continue
        bbox = tuple(float(value) for value in coordinates)
        x0, y0, x1, y1 = bbox
        area_ratio = max(0.0, x1 - x0) * max(0.0, y1 - y0) / max(1, width * height)
        width_ratio = max(0.0, x1 - x0) / max(1, width)
        height_ratio = max(0.0, y1 - y0) / max(1, height)
        if label == "table" and (area_ratio > 0.25 or (width_ratio > 0.35 and height_ratio > 0.6)):
            label = "page_frame"
        score = float(box.get("score", 0.0))
        annotations.append(
            LayoutAnnotation(
                label=label,
                bbox=(x0, y0, x1, y1),
                confidence=score,
                source="paddle_pp_doclayout",
                source_label=source_label,
                review_required=True,
            )
        )
    return annotations


def _long_line_count(mask: np.ndarray[Any, Any], horizontal: bool) -> int:
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    height, width = mask.shape
    return sum(
        1
        for contour in contours
        for _x, _y, box_width, box_height in [cv2.boundingRect(contour)]
        if (box_width >= width * 0.55 if horizontal else box_height >= height * 0.55)
    )


def _line_positions(mask: np.ndarray[Any, Any], horizontal: bool, offset: float) -> list[float]:
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    height, width = mask.shape
    major_length = width if horizontal else height
    segments = [
        (
            offset + (y + box_height / 2 if horizontal else x + box_width / 2),
            float(box_width if horizontal else box_height),
        )
        for contour in contours
        for x, y, box_width, box_height in [cv2.boundingRect(contour)]
        if (box_width if horizontal else box_height) >= major_length * 0.05
    ]
    groups: list[list[tuple[float, float]]] = []
    for segment in sorted(segments):
        if groups and segment[0] - groups[-1][-1][0] <= 12:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return [
        sum(position * length for position, length in group)
        / max(1.0, sum(length for _position, length in group))
        for group in groups
        if sum(length for _position, length in group) >= major_length * 0.5
    ]


def _hough_positions(
    rgb: np.ndarray[Any, Any], bbox: tuple[int, int, int, int], horizontal: bool
) -> list[float]:
    x, y, width, height = bbox
    cleaned = preprocess_for_ocr(np.asarray(rgb, dtype=np.uint8)).image
    gray = cv2.cvtColor(cleaned[y : y + height, x : x + width], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    major = width if horizontal else height
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(25, major // 12),
        minLineLength=max(30, int(major * 0.25)),
        maxLineGap=max(12, major // 60),
    )
    if lines is None:
        return []
    positions: list[float] = []
    # OpenCV 4 commonly returns (N, 1, 4), while OpenCV 5 may return (N, 4).
    # Flatten both representations so training dependencies cannot break table detection.
    for line in np.asarray(lines).reshape(-1, 4):
        x0, y0, x1, y1 = (int(value) for value in line)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if horizontal and dx >= major * 0.25 and dy <= max(6, dx * 0.03):
            positions.append(y + (y0 + y1) / 2)
        if not horizontal and dy >= major * 0.25 and dx <= max(6, dy * 0.03):
            positions.append(x + (x0 + x1) / 2)
    return _merge_positions(positions, tolerance=15)


def _merge_positions(positions: list[float], tolerance: float = 12) -> list[float]:
    groups: list[list[float]] = []
    for position in sorted(positions):
        if groups and position - groups[-1][-1] <= tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [sum(group) / len(group) for group in groups]


def _resolve_cross_class(annotations: list[LayoutAnnotation]) -> list[LayoutAnnotation]:
    images = [item for item in annotations if item.label == "image"]
    resolved: list[LayoutAnnotation] = []
    for item in annotations:
        if item.label == "table" and item.source == "opencv_ruled_region":
            overlap = max((_intersection_ratio(item.bbox, image.bbox) for image in images), default=0)
            if overlap >= 0.5:
                item = LayoutAnnotation(
                    label="passage_box",
                    bbox=item.bbox,
                    confidence=item.confidence,
                    source=item.source,
                    source_label="table_inside_image",
                )
        resolved.append(item)
    return resolved


def _deduplicate(
    annotations: list[LayoutAnnotation], iou_threshold: float = 0.65
) -> list[LayoutAnnotation]:
    kept: list[LayoutAnnotation] = []
    for candidate in sorted(annotations, key=lambda item: item.confidence, reverse=True):
        if any(
            candidate.label == existing.label
            and _iou(candidate.bbox, existing.bbox) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.bbox[1], item.bbox[0], item.label))


def _iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1.0)


def _intersection_ratio(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(1.0, first[2] - first[0]) * max(1.0, first[3] - first[1])
    return intersection / first_area


def _write_overlay(
    source: Path, output: Path, annotations: list[LayoutAnnotation]
) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in annotations:
        color = "#e5484d" if item.review_required else "#30a46c"
        draw.rectangle(item.bbox, outline=color, width=3)
        draw.text((item.bbox[0] + 3, item.bbox[1] + 3), item.label, fill=color)
    image.save(output, "JPEG", quality=88, optimize=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    candidate = path.with_suffix(path.suffix + ".partial")
    candidate.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate.replace(path)
