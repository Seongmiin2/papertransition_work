import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from scan2hwpx.ir.models import FormulaStatus
from scan2hwpx.vision.paddle_formula import PaddleFormulaProcessor
from scan2hwpx.vision.training_data import (
    load_verified_formula_samples,
    normalize_latex,
    split_by_document,
)


class FakeResult:
    json: ClassVar[dict[str, object]] = {
        "res": {
            "layout_det_res": {
                "boxes": [
                    {
                        "label": "formula",
                        "score": 0.92,
                        "coordinate": [10, 20, 90, 60],
                    }
                ]
            },
            "formula_res_list": [
                {
                    "dt_polys": [10, 20, 90, 60],
                    "rec_formula": r"\frac{x}{2}",
                }
            ],
        }
    }


class FakePipeline:
    def predict(self, image: np.ndarray) -> list[FakeResult]:
        return [FakeResult()]


class StubPaddleFormulaProcessor(PaddleFormulaProcessor):
    def _engine(self) -> FakePipeline:
        return FakePipeline()


def test_paddle_adapter_maps_official_result_to_ir(tmp_path: Path) -> None:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    formula = StubPaddleFormulaProcessor().process_page(image, 1, tmp_path)[0]

    assert formula.expression == r"\frac{x}{2}"
    assert formula.status == FormulaStatus.NEEDS_REVIEW
    assert formula.confidence == 0.5
    assert (tmp_path / "p1-f1.png").exists()


def test_training_manifest_requires_verified_targets_and_deduplicates(tmp_path: Path) -> None:
    image = tmp_path / "formula.png"
    assert cv2.imwrite(str(image), np.full((20, 40), 255, dtype=np.uint8))
    manifest = tmp_path / "formula_training.jsonl"
    rows = [
        {
            "document_id": "doc-1",
            "formula_id": "f-1",
            "image": image.name,
            "target": "  x   +  y  ",
            "format": "latex",
        },
        {
            "document_id": "doc-1",
            "formula_id": "f-2",
            "image": image.name,
            "target": None,
            "format": "latex",
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )

    samples, audit = load_verified_formula_samples(manifest)

    assert len(samples) == 1
    assert samples[0].target == "x + y"
    assert audit.missing_targets == 1
    assert not audit.trainable
    assert sum(len(values) for values in split_by_document(samples).values()) == 1
    assert normalize_latex("x  +\n y") == "x + y"
