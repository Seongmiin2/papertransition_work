from pathlib import Path

import numpy as np

from scan2hwpx.ir.models import FormulaStatus
from scan2hwpx.vision.formulas import (
    FormulaCandidate,
    FormulaProcessor,
    FormulaRecognition,
    validate_formula_expression,
)


class StubDetector:
    name = "stub-detector"

    def detect(self, image: np.ndarray) -> list[FormulaCandidate]:
        return [FormulaCandidate((10, 20, 90, 60), 0.98)]


class StubRecognizer:
    name = "stub-recognizer"

    def recognize(self, image: np.ndarray) -> FormulaRecognition:
        return FormulaRecognition(r"\frac{x^2+1}{x-1}", 0.96)


def test_formula_processor_writes_crop_and_structured_result(tmp_path: Path) -> None:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    formulas = FormulaProcessor(StubDetector(), StubRecognizer()).process_page(
        image, 1, tmp_path
    )

    assert len(formulas) == 1
    assert formulas[0].status == FormulaStatus.RECOGNIZED
    assert formulas[0].validation.syntax_valid
    assert formulas[0].bbox.normalized == (10 / 120, 20 / 100, 90 / 120, 60 / 100)
    assert (tmp_path / "p1-f1.png").exists()


def test_unbalanced_latex_requires_review(tmp_path: Path) -> None:
    class BrokenRecognizer:
        name = "broken"

        def recognize(self, image: np.ndarray) -> FormulaRecognition:
            return FormulaRecognition(r"\frac{x}{y", 0.99)

    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    formula = FormulaProcessor(StubDetector(), BrokenRecognizer()).process_page(
        image, 1, tmp_path
    )[0]

    assert formula.status == FormulaStatus.NEEDS_REVIEW
    assert not formula.validation.syntax_valid


def test_empty_formula_uses_image_fallback(tmp_path: Path) -> None:
    class EmptyRecognizer:
        name = "empty"

        def recognize(self, image: np.ndarray) -> FormulaRecognition:
            return FormulaRecognition("", 0.0)

    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    formula = FormulaProcessor(StubDetector(), EmptyRecognizer()).process_page(
        image, 1, tmp_path
    )[0]

    assert formula.status == FormulaStatus.IMAGE_FALLBACK
    assert not validate_formula_expression(FormulaRecognition("", 0.0)).syntax_valid
