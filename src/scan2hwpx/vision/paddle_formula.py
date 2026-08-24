from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scan2hwpx.images import write_image
from scan2hwpx.ir.models import BBox, Formula, FormulaStatus
from scan2hwpx.vision.formulas import (
    FormulaProcessor,
    FormulaRecognition,
    validate_formula_expression,
)


class PaddleFormulaProcessor(FormulaProcessor):
    """Official PaddleOCR layout + PP-FormulaNet adapter.

    PaddleOCR does not expose formula token probabilities, so detections remain
    review-required until render similarity or human ground truth is available.
    """

    def __init__(
        self,
        *,
        formula_model: str = "PP-FormulaNet_plus-S",
        layout_model: str = "PP-DocLayout-S",
        layout_threshold: float = 0.3,
        enable_mkldnn: bool = False,
    ) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self.formula_model = formula_model
        self.layout_model = layout_model
        self.layout_threshold = layout_threshold
        self.enable_mkldnn = enable_mkldnn
        self._pipeline: Any | None = None

    def _engine(self) -> Any:
        if self._pipeline is None:
            from paddleocr import FormulaRecognitionPipeline  # type: ignore[import-untyped]

            self._pipeline = FormulaRecognitionPipeline(
                formula_recognition_model_name=self.formula_model,
                layout_detection_model_name=self.layout_model,
                layout_threshold=self.layout_threshold,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=True,
                enable_mkldnn=self.enable_mkldnn,
            )
        return self._pipeline

    def process_page(
        self, image: np.ndarray, page_no: int, asset_dir: Path
    ) -> list[Formula]:
        asset_dir.mkdir(parents=True, exist_ok=True)
        results = list(self._engine().predict(image))
        if not results:
            return []
        payload = results[0].json
        data = payload.get("res", payload)
        height, width = image.shape[:2]
        layout_boxes = data.get("layout_det_res", {}).get("boxes", [])
        formulas: list[Formula] = []
        for index, item in enumerate(data.get("formula_res_list", []), start=1):
            coordinates = _coordinates(item.get("dt_polys"))
            if coordinates is None:
                continue
            x0, y0, x1, y1 = _clip(coordinates, width, height)
            if x1 <= x0 or y1 <= y0:
                continue
            expression = str(item.get("rec_formula", "")).strip()
            formula_id = f"p{page_no}-f{index}"
            crop_path = asset_dir / f"{formula_id}.png"
            crop = image[y0:y1, x0:x1]
            write_image(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            validation = validate_formula_expression(FormulaRecognition(expression, 0.5))
            detection_score = _detection_score(coordinates, layout_boxes)
            status = FormulaStatus.NEEDS_REVIEW
            if not expression:
                status = FormulaStatus.IMAGE_FALLBACK
            formulas.append(
                Formula(
                    id=formula_id,
                    bbox=BBox(
                        pixel=(float(x0), float(y0), float(x1), float(y1)),
                        normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
                    ),
                    source_image=crop_path.as_posix(),
                    expression=expression,
                    confidence=min(0.5, detection_score),
                    source_provider=f"{self.layout_model}+{self.formula_model}",
                    status=status,
                    validation=validation,
                )
            )
        return formulas


def _coordinates(value: Any) -> tuple[float, float, float, float] | None:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(float(part) for part in value)  # type: ignore[return-value]


def _clip(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))


def _detection_score(
    coordinates: tuple[float, float, float, float], layout_boxes: list[dict[str, Any]]
) -> float:
    for box in layout_boxes:
        if str(box.get("label", "")).lower() != "formula":
            continue
        candidate = _coordinates(box.get("coordinate"))
        if candidate is not None and max(abs(a - b) for a, b in zip(candidate, coordinates)) < 2:
            return max(0.0, min(1.0, float(box.get("score", 0.0))))
    return 0.0
