"""Flatten one existing benchmark JSON report into ai/results/performance_log.csv.

Reuses the report shapes already produced by `ocr-benchmark`, `formula-benchmark`,
`train-page-anomaly`, and `batch-convert` (see src/scan2hwpx/training/ocr_benchmark.py,
src/scan2hwpx/vision/benchmark.py, and src/scan2hwpx/batch.py).
Does not invent a new report format; only appends flattened rows to a CSV so results
over time are visible in one place (openable directly in Excel).

Usage:
    python ai/results/record_metrics.py --report path/to/ocr_benchmark.json \
        --component ocr --model-config-id korean-exam-ppocrv5-v2 [--notes "..."]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "date",
    "component",
    "model_config_id",
    "dataset_split",
    "metric_name",
    "metric_value",
    "notes",
    "source_report",
]

Row = dict[str, str]


def _rows_from_ocr(data: dict[str, Any], model_config_id: str) -> list[Row]:
    metrics = (
        "character_error_rate",
        "character_accuracy",
        "mean_ocr_confidence",
        "elapsed_seconds",
        "source_pages",
        "ocr_blocks",
    )
    return [
        {
            "model_config_id": model_config_id,
            "dataset_split": "",
            "metric_name": name,
            "metric_value": str(data[name]),
        }
        for name in metrics
        if name in data
    ]


def _rows_from_anomaly(data: dict[str, Any], model_config_id: str) -> list[Row]:
    model_config_id = model_config_id or str(data.get("model", ""))
    rows: list[Row] = []
    for split_name, split in data.get("splits", {}).items():
        for metric_name in ("pages", "flagged", "mean_mse"):
            if metric_name in split:
                rows.append(
                    {
                        "model_config_id": model_config_id,
                        "dataset_split": split_name,
                        "metric_name": metric_name,
                        "metric_value": str(split[metric_name]),
                    }
                )
    return rows


def _rows_from_formula(data: dict[str, Any], model_config_id: str) -> list[Row]:
    metrics = (
        "samples",
        "successful",
        "errors",
        "model_load_seconds",
        "average_inference_seconds",
        "exact_match_rate",
        "normalized_edit_similarity",
    )
    rows: list[Row] = []
    for model_name, summary in data.get("summary", {}).items():
        row_model_id = model_config_id or model_name
        for metric_name in metrics:
            if metric_name in summary and summary[metric_name] is not None:
                rows.append(
                    {
                        "model_config_id": row_model_id,
                        "dataset_split": model_name,
                        "metric_name": metric_name,
                        "metric_value": str(summary[metric_name]),
                    }
                )
    return rows


def _rows_from_batch(data: dict[str, Any], model_config_id: str) -> list[Row]:
    summary = data.get("summary", {})
    metrics = (
        "total_files",
        "processed",
        "completed",
        "skipped",
        "failed",
        "elapsed_seconds",
        "throughput_files_per_hour",
        "pages",
    )
    rows = [
        {
            "model_config_id": model_config_id,
            "dataset_split": "",
            "metric_name": metric_name,
            "metric_value": str(summary[metric_name]),
        }
        for metric_name in metrics
        if metric_name in summary
    ]
    if "workers" in data:
        rows.append(
            {
                "model_config_id": model_config_id,
                "dataset_split": "",
                "metric_name": "workers",
                "metric_value": str(data["workers"]),
            }
        )
    return rows


_EXTRACTORS = {
    "ocr": _rows_from_ocr,
    "anomaly": _rows_from_anomaly,
    "formula": _rows_from_formula,
    "batch": _rows_from_batch,
}


def extract_rows(
    report_path: Path,
    component: str,
    model_config_id: str,
    notes: str,
    run_date: str,
) -> list[Row]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    extractor = _EXTRACTORS[component]
    rows = extractor(data, model_config_id)
    for row in rows:
        row["date"] = run_date
        row["component"] = component
        row["notes"] = notes
        row["source_report"] = str(report_path)
    return rows


def append_rows(csv_path: Path, rows: list[Row]) -> int:
    existing_keys: set[tuple[str, ...]] = set()
    file_exists = csv_path.exists()
    if file_exists:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                existing_keys.add(
                    (
                        existing["date"],
                        existing["component"],
                        existing["model_config_id"],
                        existing["metric_name"],
                        existing["dataset_split"],
                    )
                )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = [
        row
        for row in rows
        if (
            row["date"],
            row["component"],
            row["model_config_id"],
            row["metric_name"],
            row["dataset_split"],
        )
        not in existing_keys
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    return len(new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="benchmark JSON report path")
    parser.add_argument("--component", choices=sorted(_EXTRACTORS), required=True)
    parser.add_argument("--model-config-id", default="", help="e.g. korean-exam-ppocrv5-v2")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--date", default=date.today().isoformat(), help="defaults to today"  # noqa: DTZ011
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().with_name("performance_log.csv"),
    )
    args = parser.parse_args()

    rows = extract_rows(args.report, args.component, args.model_config_id, args.notes, args.date)
    added = append_rows(args.csv, rows)
    print(f"{added}/{len(rows)} new row(s) appended to {args.csv}")


if __name__ == "__main__":
    main()
