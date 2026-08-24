from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from scan2hwpx.images import write_image
from scan2hwpx.ir.models import (
    BBox,
    Formula,
    FormulaFormat,
    FormulaStatus,
    FormulaValidation,
)


@dataclass(frozen=True)
class FormulaCandidate:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class FormulaRecognition:
    expression: str
    confidence: float
    format: FormulaFormat = FormulaFormat.LATEX


class FormulaDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[FormulaCandidate]: ...


class FormulaRecognizer(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> FormulaRecognition: ...


class FormulaProcessor:
    """Model-agnostic formula detection and recognition orchestration."""

    def __init__(
        self,
        detector: FormulaDetector,
        recognizer: FormulaRecognizer,
        *,
        accept_confidence: float = 0.85,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.accept_confidence = accept_confidence

    def process_page(
        self, image: np.ndarray, page_no: int, asset_dir: Path
    ) -> list[Formula]:
        height, width = image.shape[:2]
        asset_dir.mkdir(parents=True, exist_ok=True)
        formulas: list[Formula] = []
        for index, candidate in enumerate(self.detector.detect(image), start=1):
            x0, y0, x1, y1 = _clip_bbox(candidate.bbox, width, height)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image[y0:y1, x0:x1]
            formula_id = f"p{page_no}-f{index}"
            crop_path = asset_dir / f"{formula_id}.png"
            write_image(crop_path, _to_bgr(crop))

            recognition = self.recognizer.recognize(crop)
            validation = validate_formula_expression(recognition)
            accepted = (
                candidate.confidence >= self.accept_confidence
                and recognition.confidence >= self.accept_confidence
                and validation.syntax_valid
            )
            status = FormulaStatus.RECOGNIZED if accepted else FormulaStatus.NEEDS_REVIEW
            if not recognition.expression.strip():
                status = FormulaStatus.IMAGE_FALLBACK
            formulas.append(
                Formula(
                    id=formula_id,
                    bbox=BBox(
                        pixel=(float(x0), float(y0), float(x1), float(y1)),
                        normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
                    ),
                    source_image=crop_path.as_posix(),
                    expression=recognition.expression.strip(),
                    format=recognition.format,
                    confidence=min(candidate.confidence, recognition.confidence),
                    source_provider=f"{self.detector.name}+{self.recognizer.name}",
                    status=status,
                    validation=validation,
                )
            )
        return formulas


def validate_formula_expression(recognition: FormulaRecognition) -> FormulaValidation:
    expression = recognition.expression.strip()
    warnings: list[str] = []
    if not expression:
        return FormulaValidation(warnings=["인식된 수식이 없습니다."])
    if recognition.format == FormulaFormat.MATHML:
        valid = expression.startswith("<math") and expression.endswith("</math>")
        if not valid:
            warnings.append("MathML 문법이 완전하지 않습니다.")
        return FormulaValidation(syntax_valid=valid, warnings=warnings)

    braces = 0
    valid = True
    for character in expression:
        if character == "{":
            braces += 1
        elif character == "}":
            braces -= 1
            if braces < 0:
                valid = False
                break
    if braces != 0:
        valid = False
    if not valid:
        warnings.append("LaTeX 중괄호의 짝이 맞지 않습니다.")
    if re.search(r"\\(?:frac|sqrt)\s*$", expression):
        valid = False
        warnings.append("LaTeX 명령에 필요한 인수가 없습니다.")
    return FormulaValidation(syntax_valid=valid, warnings=warnings)


def _clip_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
