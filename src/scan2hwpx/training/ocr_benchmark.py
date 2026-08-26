from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf

from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider


def run_ocr_benchmark(
    source_pdf: Path,
    transcript_pdf: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    dpi: int = 200,
    fusion_mode: str = "fast",
    recognition_model_dir: Path | None = None,
    page_anomaly_model: Path | None = None,
) -> dict[str, Any]:
    """Measure one OCR model against a held-out native-text transcript."""
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    provider = PaddlePdfOcrProvider(
        dpi=dpi,
        fusion_mode=fusion_mode,
        device=device,
        recognition_model_dir=recognition_model_dir,
        page_anomaly_model=page_anomaly_model,
    )
    document = provider.convert(source_pdf)
    elapsed = time.perf_counter() - started
    prediction = "\n".join(
        block.text for page in document.pages for block in page.blocks if block.text.strip()
    )
    target = _pdf_text(transcript_pdf)
    normalized_prediction = normalize_benchmark_text(prediction)
    normalized_target = normalize_benchmark_text(target)
    distance = levenshtein_distance(normalized_target, normalized_prediction)
    denominator = max(1, len(normalized_target))
    character_error_rate = distance / denominator
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_pdf": str(source_pdf.resolve()),
        "source_sha256": _sha256(source_pdf),
        "transcript_pdf": str(transcript_pdf.resolve()),
        "transcript_sha256": _sha256(transcript_pdf),
        "recognition_model_dir": (
            str(recognition_model_dir.resolve()) if recognition_model_dir else None
        ),
        "device": document.metadata.get("device"),
        "dpi": dpi,
        "fusion_mode": fusion_mode,
        "page_anomaly_model": str(page_anomaly_model.resolve()) if page_anomaly_model else None,
        "page_anomaly_flags": document.metadata.get("page_anomaly_flags", []),
        "elapsed_seconds": round(elapsed, 3),
        "source_pages": len(document.pages),
        "target_pages": _pdf_pages(transcript_pdf),
        "ocr_blocks": sum(len(page.blocks) for page in document.pages),
        "predicted_characters": len(normalized_prediction),
        "target_characters": len(normalized_target),
        "edit_distance": distance,
        "character_error_rate": round(character_error_rate, 6),
        "character_accuracy": round(max(0.0, 1.0 - character_error_rate), 6),
        "mean_ocr_confidence": round(
            sum(block.confidence for page in document.pages for block in page.blocks)
            / max(1, sum(len(page.blocks) for page in document.pages)),
            6,
        ),
        "promotion_rule": {
            "primary": "candidate CER must be lower than baseline CER",
            "guardrails": [
                "page count must equal source page count",
                "candidate must not reduce recognized character coverage by more than 2%",
                "gold source is never included in training or threshold fitting",
            ],
        },
    }
    (destination / "prediction.txt").write_text(prediction, encoding="utf-8")
    (destination / "target.txt").write_text(target, encoding="utf-8")
    (destination / "document_ir.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    (destination / "ocr_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def compare_ocr_benchmarks(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_cer = float(baseline["character_error_rate"])
    candidate_cer = float(candidate["character_error_rate"])
    baseline_characters = max(1, int(baseline["predicted_characters"]))
    candidate_characters = int(candidate["predicted_characters"])
    coverage_ratio = candidate_characters / baseline_characters
    pages_match = int(candidate["source_pages"]) == int(baseline["source_pages"])
    promoted = candidate_cer < baseline_cer and coverage_ratio >= 0.98 and pages_match
    return {
        "baseline_cer": baseline_cer,
        "candidate_cer": candidate_cer,
        "relative_cer_change": round((candidate_cer - baseline_cer) / max(baseline_cer, 1e-9), 6),
        "character_coverage_ratio": round(coverage_ratio, 6),
        "pages_match": pages_match,
        "promoted": promoted,
        "decision": "promote" if promoted else "reject",
    }


def write_ocr_promotion_decision(
    baseline_report: Path, candidate_report: Path, output_path: Path
) -> dict[str, Any]:
    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_report.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise TypeError("benchmark reports must contain JSON objects")
    decision = {
        "schema_version": "1.0",
        "baseline_report": str(baseline_report.resolve()),
        "candidate_report": str(candidate_report.resolve()),
        **compare_ocr_benchmarks(baseline, candidate),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def normalize_benchmark_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if not character.isspace())


def levenshtein_distance(left: str, right: str) -> int:
    """Myers bit-vector edit distance; exact and fast for document-sized Unicode text."""
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    masks: dict[str, int] = {}
    for index, character in enumerate(left):
        masks[character] = masks.get(character, 0) | (1 << index)
    positive = (1 << len(left)) - 1
    negative = 0
    score = len(left)
    last = 1 << (len(left) - 1)
    limit = (1 << len(left)) - 1
    for character in right:
        equal = masks.get(character, 0)
        combined = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & last:
            score += 1
        elif negative_horizontal & last:
            score -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & limit
        negative_horizontal = (negative_horizontal << 1) & limit
        positive = (negative_horizontal | ~(combined | positive_horizontal)) & limit
        negative = positive_horizontal & combined
    return score


def _pdf_text(path: Path) -> str:
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        return "".join(page.get_text() for page in document)


def _pdf_pages(path: Path) -> int:
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        return int(document.page_count)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
