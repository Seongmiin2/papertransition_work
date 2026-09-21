from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scan2hwpx.evaluation.safe_artifact_io import read_bounded_regular_file
from scan2hwpx.evaluation.strict_json import require_strict_json_bytes

REPORT_SCHEMA_VERSION = "model-a-table-topology-proxy-comparison/1.0"
SUMMARY_SCHEMA_VERSION = "model-a-table-topology-holdout-summary/1.0"
MAX_REPORTS = 256
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_PAGES_PER_REPORT = 4096
MAX_TABLES_PER_PAGE = 4096
MAX_UNMATCHED_PREDICTIONS = 16_384


@dataclass(frozen=True, slots=True)
class _UnmatchedPrediction:
    report_file: str
    page_id: str
    table_grid_observation_id: str


@dataclass(slots=True)
class _Totals:
    report_files: list[str] = field(default_factory=list)
    document_count: int = 0
    page_count: int = 0
    predicted_table_count: int = 0
    reference_table_count: int = 0
    grounded_reference_table_count: int = 0
    grounded_matched_table_count: int = 0
    grounded_rows_columns_exact_count: int = 0
    grounded_grid_span_exact_count: int = 0
    strict_cell_exact_count: int = 0
    strict_cell_sample_count: int = 0
    assignment_true_positive_count: int = 0
    assignment_predicted_count: int = 0
    assignment_reference_count: int = 0
    zero_ref_reference_table_count: int = 0
    unmatched_predictions: list[_UnmatchedPrediction] = field(default_factory=list)

    def add(self, other: _Totals) -> None:
        self.report_files.extend(other.report_files)
        self.document_count += other.document_count
        self.page_count += other.page_count
        self.predicted_table_count += other.predicted_table_count
        self.reference_table_count += other.reference_table_count
        self.grounded_reference_table_count += other.grounded_reference_table_count
        self.grounded_matched_table_count += other.grounded_matched_table_count
        self.grounded_rows_columns_exact_count += other.grounded_rows_columns_exact_count
        self.grounded_grid_span_exact_count += other.grounded_grid_span_exact_count
        self.strict_cell_exact_count += other.strict_cell_exact_count
        self.strict_cell_sample_count += other.strict_cell_sample_count
        self.assignment_true_positive_count += other.assignment_true_positive_count
        self.assignment_predicted_count += other.assignment_predicted_count
        self.assignment_reference_count += other.assignment_reference_count
        self.zero_ref_reference_table_count += other.zero_ref_reference_table_count
        self.unmatched_predictions.extend(other.unmatched_predictions)
        if len(self.unmatched_predictions) > MAX_UNMATCHED_PREDICTIONS:
            raise ValueError("unmatched predicted table count exceeds the supported limit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Model A table-topology holdout proxy reports while separating "
            "text-grounded references from zero-reference projection tables."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(true_positive: int, predicted: int, reference: int) -> float | None:
    precision = _rate(true_positive, predicted)
    recall = _rate(true_positive, reference)
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _require_object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_list(value: object, *, label: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds the supported item limit")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _load_report(path: Path) -> tuple[dict[str, Any], str]:
    raw = read_bounded_regular_file(
        path,
        max_bytes=MAX_REPORT_BYTES,
        label=f"table-topology report {path.name}",
    )
    require_strict_json_bytes(
        raw,
        max_depth=32,
        max_nodes=500_000,
        label=f"table-topology report {path.name}",
    )
    decoded: object = json.loads(raw)
    report = _require_object(decoded, label=f"report {path.name}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"report {path.name} has an unsupported schema_version")
    if report.get("metric_status") != "diagnostic_only_not_accuracy_not_release_evidence":
        raise ValueError(f"report {path.name} is not marked diagnostic-only")
    for field_name in ("golden_eligible", "training_eligible", "release_eligible"):
        if _require_bool(report.get(field_name), label=f"{path.name}.{field_name}"):
            raise ValueError(f"report {path.name} must have {field_name}=false")
    return report, hashlib.sha256(raw).hexdigest()


def _summarize_report(path: Path) -> tuple[_Totals, str]:
    report, report_sha256 = _load_report(path)
    pages = _require_list(
        report.get("pages"), label=f"{path.name}.pages", maximum=MAX_PAGES_PER_REPORT
    )
    selected_page_ids = _require_list(
        report.get("selected_page_ids"),
        label=f"{path.name}.selected_page_ids",
        maximum=MAX_PAGES_PER_REPORT,
    )
    selected_ids = [
        _require_string(value, label=f"{path.name}.selected_page_ids[]")
        for value in selected_page_ids
    ]

    totals = _Totals(report_files=[path.name], document_count=1, page_count=len(pages))
    observed_page_ids: list[str] = []
    observed_prediction_ids: set[str] = set()
    raw_matched_count = 0
    for page_index, raw_page in enumerate(pages):
        page_label = f"{path.name}.pages[{page_index}]"
        page = _require_object(raw_page, label=page_label)
        page_id = _require_string(page.get("page_id"), label=f"{page_label}.page_id")
        observed_page_ids.append(page_id)
        tables = _require_list(
            page.get("tables"),
            label=f"{page_label}.tables",
            maximum=MAX_TABLES_PER_PAGE,
        )
        reference_count = _require_int(
            page.get("reference_table_count"), label=f"{page_label}.reference_table_count"
        )
        if reference_count != len(tables):
            raise ValueError(f"{page_label}.reference_table_count does not match tables")
        unmatched_ids = [
            _require_string(value, label=f"{page_label}.unmatched_predicted_table_grid_ids[]")
            for value in _require_list(
                page.get("unmatched_predicted_table_grid_observation_ids"),
                label=f"{page_label}.unmatched_predicted_table_grid_observation_ids",
                maximum=MAX_TABLES_PER_PAGE,
            )
        ]
        if len(unmatched_ids) != len(set(unmatched_ids)):
            raise ValueError(f"{page_label} contains duplicate unmatched prediction ids")

        page_matched_count = 0
        for table_index, raw_table in enumerate(tables):
            table_label = f"{page_label}.tables[{table_index}]"
            table = _require_object(raw_table, label=table_label)
            predicted_id_value = table.get("predicted_table_grid_observation_id")
            if predicted_id_value is None:
                predicted_id = None
            else:
                predicted_id = _require_string(
                    predicted_id_value,
                    label=f"{table_label}.predicted_table_grid_observation_id",
                )
                page_matched_count += 1
                if predicted_id in observed_prediction_ids:
                    raise ValueError(f"report {path.name} contains a duplicate prediction id")
                observed_prediction_ids.add(predicted_id)

            reference_pairs = _require_int(
                table.get("reference_assignment_pair_count"),
                label=f"{table_label}.reference_assignment_pair_count",
            )
            if reference_pairs == 0:
                totals.zero_ref_reference_table_count += 1
                continue

            predicted_pairs = _require_int(
                table.get("predicted_assignment_pair_count"),
                label=f"{table_label}.predicted_assignment_pair_count",
            )
            true_positives = _require_int(
                table.get("assignment_pair_true_positive_count"),
                label=f"{table_label}.assignment_pair_true_positive_count",
            )
            strict_exact = _require_int(
                table.get("strict_cell_assignment_exact_count"),
                label=f"{table_label}.strict_cell_assignment_exact_count",
            )
            strict_sample = _require_int(
                table.get("strict_cell_assignment_sample_count"),
                label=f"{table_label}.strict_cell_assignment_sample_count",
            )
            if true_positives > predicted_pairs or true_positives > reference_pairs:
                raise ValueError(f"{table_label} has impossible assignment-pair counts")
            if strict_exact > strict_sample:
                raise ValueError(f"{table_label} has impossible strict-cell counts")

            totals.grounded_reference_table_count += 1
            totals.grounded_matched_table_count += predicted_id is not None
            totals.grounded_rows_columns_exact_count += _require_bool(
                table.get("rows_columns_exact"), label=f"{table_label}.rows_columns_exact"
            )
            totals.grounded_grid_span_exact_count += _require_bool(
                table.get("grid_span_exact"), label=f"{table_label}.grid_span_exact"
            )
            totals.strict_cell_exact_count += strict_exact
            totals.strict_cell_sample_count += strict_sample
            totals.assignment_true_positive_count += true_positives
            totals.assignment_predicted_count += predicted_pairs
            totals.assignment_reference_count += reference_pairs

        predicted_count = _require_int(
            page.get("predicted_table_count"), label=f"{page_label}.predicted_table_count"
        )
        matched_count = _require_int(
            page.get("matched_table_count"), label=f"{page_label}.matched_table_count"
        )
        if matched_count != page_matched_count:
            raise ValueError(f"{page_label}.matched_table_count does not match table details")
        if predicted_count != page_matched_count + len(unmatched_ids):
            raise ValueError(f"{page_label}.predicted_table_count is inconsistent")
        raw_matched_count += page_matched_count
        totals.predicted_table_count += predicted_count
        totals.reference_table_count += reference_count
        for prediction_id in unmatched_ids:
            if prediction_id in observed_prediction_ids:
                raise ValueError(f"report {path.name} contains a duplicate prediction id")
            observed_prediction_ids.add(prediction_id)
            totals.unmatched_predictions.append(
                _UnmatchedPrediction(path.name, page_id, prediction_id)
            )
            if len(totals.unmatched_predictions) > MAX_UNMATCHED_PREDICTIONS:
                raise ValueError("unmatched predicted table count exceeds the supported limit")

    if len(observed_page_ids) != len(set(observed_page_ids)):
        raise ValueError(f"report {path.name} contains duplicate page ids")
    if observed_page_ids != selected_ids:
        raise ValueError(f"report {path.name} selected_page_ids do not match pages")

    metrics = _require_object(report.get("metrics"), label=f"{path.name}.metrics")
    expected_metrics = {
        "predicted_table_count": totals.predicted_table_count,
        "reference_table_count": totals.reference_table_count,
        "matched_table_count": raw_matched_count,
    }
    for metric_name, expected in expected_metrics.items():
        actual = _require_int(metrics.get(metric_name), label=f"{path.name}.metrics.{metric_name}")
        if actual != expected:
            raise ValueError(f"{path.name}.metrics.{metric_name} is inconsistent")
    return totals, report_sha256


def _summary_dict(totals: _Totals) -> dict[str, object]:
    grounded = totals.grounded_reference_table_count
    unmatched = sorted(
        totals.unmatched_predictions,
        key=lambda item: (item.report_file, item.page_id, item.table_grid_observation_id),
    )
    return {
        "report_files": sorted(totals.report_files),
        "document_count": totals.document_count,
        "page_count": totals.page_count,
        "predicted_table_count": totals.predicted_table_count,
        "reference_table_count": totals.reference_table_count,
        "grounded_reference_table_count": grounded,
        "grounded_matched_table_count": totals.grounded_matched_table_count,
        "grounded_table_recall": _rate(totals.grounded_matched_table_count, grounded),
        "grounded_rows_columns_exact_count": totals.grounded_rows_columns_exact_count,
        "grounded_rows_columns_exact_rate": _rate(
            totals.grounded_rows_columns_exact_count, grounded
        ),
        "grounded_grid_span_exact_count": totals.grounded_grid_span_exact_count,
        "grounded_grid_span_exact_rate": _rate(totals.grounded_grid_span_exact_count, grounded),
        "grounded_strict_cell_assignment": {
            "exact_count": totals.strict_cell_exact_count,
            "sample_count": totals.strict_cell_sample_count,
            "exact_rate": _rate(totals.strict_cell_exact_count, totals.strict_cell_sample_count),
        },
        "grounded_assignment_pairs": {
            "true_positive_count": totals.assignment_true_positive_count,
            "predicted_count": totals.assignment_predicted_count,
            "reference_count": totals.assignment_reference_count,
            "precision": _rate(
                totals.assignment_true_positive_count, totals.assignment_predicted_count
            ),
            "recall": _rate(
                totals.assignment_true_positive_count, totals.assignment_reference_count
            ),
            "f1": _f1(
                totals.assignment_true_positive_count,
                totals.assignment_predicted_count,
                totals.assignment_reference_count,
            ),
        },
        "zero_ref_reference_table_count": totals.zero_ref_reference_table_count,
        "unmatched_predicted_table_count": len(unmatched),
        "unmatched_predicted_tables": [
            {
                "report_file": item.report_file,
                "page_id": item.page_id,
                "table_grid_observation_id": item.table_grid_observation_id,
            }
            for item in unmatched
        ],
    }


def summarize_report_directory(report_dir: Path) -> dict[str, object]:
    resolved = report_dir.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"report directory does not exist: {resolved}")
    split_paths = {
        "validation": sorted(resolved.glob("validation-*.json"), key=lambda path: path.name),
        "test": sorted(resolved.glob("test-*.json"), key=lambda path: path.name),
    }
    for split_name, paths in split_paths.items():
        if not paths:
            raise ValueError(f"no {split_name}-*.json reports found in {resolved}")
        if len(paths) > MAX_REPORTS:
            raise ValueError(f"{split_name} report count exceeds the supported limit")

    split_totals: dict[str, _Totals] = {}
    all_totals = _Totals()
    source_reports: list[dict[str, object]] = []
    for split_name in ("validation", "test"):
        totals = _Totals()
        for path in split_paths[split_name]:
            report_totals, report_sha256 = _summarize_report(path)
            totals.add(report_totals)
            all_totals.add(report_totals)
            source_reports.append(
                {
                    "split": split_name,
                    "file": path.name,
                    "sha256": report_sha256,
                }
            )
        split_totals[split_name] = totals

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "metric_status": "diagnostic_only_not_accuracy",
        "reference_status": "unverified_machine_projection_not_human_gold",
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "report_selection": {
            "report_dir": str(resolved),
            "patterns": ["validation-*.json", "test-*.json"],
            "source_reports": source_reports,
        },
        "splits": {
            "validation": _summary_dict(split_totals["validation"]),
            "test": _summary_dict(split_totals["test"]),
            "all": _summary_dict(all_totals),
        },
        "limitations": [
            "These proxy comparisons are diagnostic only and are not accuracy estimates.",
            "Text-grounded reference tables require reference_assignment_pair_count > 0.",
            "Zero-reference projection tables are counted separately and excluded from grounded metrics.",
            "This summary does not classify projection tables as semantic tables or layout scaffolding.",
            "Table matches depend on shared EvidenceIR text observations.",
        ],
    }


def main() -> int:
    args = _parse_args()
    summary = summarize_report_directory(args.report_dir)
    payload = _canonical_json_bytes(summary, pretty=True)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as stream:
                stream.write(payload)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {output}") from exc
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
