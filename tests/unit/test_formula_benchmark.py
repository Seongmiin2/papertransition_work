from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from scan2hwpx.vision.benchmark import (
    load_formula_benchmark_samples,
    normalized_edit_similarity,
    run_formula_benchmark,
)


class FakeResult:
    def __init__(self, expression: str) -> None:
        self.json = {"res": {"rec_formula": expression}}


class FakePredictor:
    expressions: ClassVar[dict[str, str]] = {
        "model-a": r"\frac{x}{2}",
        "model-b": r"\frac{x}{3}",
    }

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, input: str, *, batch_size: int = 1) -> list[FakeResult]:
        assert Path(input).is_file()
        assert batch_size == 1
        return [FakeResult(self.expressions[self.model_name])]


def _factory(model_name: str, device: str) -> FakePredictor:
    assert device == "cpu"
    return FakePredictor(model_name)


def test_formula_benchmark_writes_metrics_and_review_html(tmp_path: Path) -> None:
    image = tmp_path / "formula.png"
    assert cv2.imwrite(str(image), np.full((20, 40), 255, dtype=np.uint8))
    manifest = tmp_path / "formula_training.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "document_id": "exam-1",
                "formula_id": "f-1",
                "image": image.name,
                "target": r"\frac{x}{2}",
                "format": "latex",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report"

    report = run_formula_benchmark(
        manifest,
        output,
        ["model-a", "model-b"],
        predictor_factory=_factory,
    )

    assert report["sample_count"] == 1
    assert report["rows"][0]["disagreement"]
    assert report["summary"]["model-a"]["exact_match_rate"] == 1.0
    assert report["summary"]["model-b"]["exact_match_rate"] == 0.0
    assert (output / "formula_benchmark.json").is_file()
    assert (output / "formula_benchmark.jsonl").is_file()
    review = (output / "formula_benchmark.html").read_text(encoding="utf-8")
    assert "확정한 정답 JSONL 저장" in review
    assert "data:image/png;base64," in review


def test_benchmark_loader_accepts_an_image_directory(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "b.png"), np.full((10, 10), 255, dtype=np.uint8))
    assert cv2.imwrite(str(tmp_path / "a.jpg"), np.full((10, 10), 255, dtype=np.uint8))
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")

    samples = load_formula_benchmark_samples(tmp_path)

    assert [sample.formula_id for sample in samples] == ["a", "b"]


def test_normalized_edit_similarity_handles_exact_and_different_latex() -> None:
    assert normalized_edit_similarity("x  + y", "x + y") == 1.0
    assert normalized_edit_similarity("x", "y") == 0.0
