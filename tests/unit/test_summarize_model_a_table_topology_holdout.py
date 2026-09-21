from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ai.modeling import summarize_model_a_table_topology_holdout as summary
from scan2hwpx.evaluation.strict_json import StrictJSONError


def _table(
    reference_pairs: int,
    *,
    predicted_id: str | None = None,
    predicted_pairs: int = 0,
    true_positives: int = 0,
    strict_exact: int = 0,
    strict_sample: int = 1,
    rows_columns_exact: bool = False,
    grid_span_exact: bool = False,
) -> dict[str, object]:
    return {
        "reference_assignment_pair_count": reference_pairs,
        "predicted_assignment_pair_count": predicted_pairs,
        "assignment_pair_true_positive_count": true_positives,
        "strict_cell_assignment_exact_count": strict_exact,
        "strict_cell_assignment_sample_count": strict_sample,
        "rows_columns_exact": rows_columns_exact,
        "grid_span_exact": grid_span_exact,
        "predicted_table_grid_observation_id": predicted_id,
    }


def _report(pages: list[dict[str, Any]]) -> dict[str, object]:
    predicted = sum(int(page["predicted_table_count"]) for page in pages)
    reference = sum(int(page["reference_table_count"]) for page in pages)
    matched = sum(int(page["matched_table_count"]) for page in pages)
    return {
        "schema_version": "model-a-table-topology-proxy-comparison/1.0",
        "metric_status": "diagnostic_only_not_accuracy_not_release_evidence",
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "selected_page_ids": [page["page_id"] for page in pages],
        "metrics": {
            "predicted_table_count": predicted,
            "reference_table_count": reference,
            "matched_table_count": matched,
        },
        "pages": pages,
    }


def _page(
    page_id: str,
    tables: list[dict[str, object]],
    *,
    unmatched_ids: list[str] | None = None,
) -> dict[str, object]:
    unmatched = [] if unmatched_ids is None else unmatched_ids
    matched = sum(table["predicted_table_grid_observation_id"] is not None for table in tables)
    return {
        "page_id": page_id,
        "page_no": 1,
        "predicted_table_count": matched + len(unmatched),
        "reference_table_count": len(tables),
        "matched_table_count": matched,
        "tables": tables,
        "unmatched_predicted_table_grid_observation_ids": unmatched,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_summarize_separates_grounded_and_zero_ref_by_split(tmp_path: Path) -> None:
    validation = _report(
        [
            _page(
                "page-v",
                [
                    _table(
                        2,
                        predicted_id="grid-v-match",
                        predicted_pairs=3,
                        true_positives=2,
                        strict_exact=4,
                        strict_sample=5,
                        rows_columns_exact=True,
                        grid_span_exact=True,
                    ),
                    _table(0, strict_sample=4),
                ],
                unmatched_ids=["grid-v-unmatched"],
            )
        ]
    )
    test = _report(
        [
            _page(
                "page-t",
                [_table(3, strict_exact=0, strict_sample=6)],
            )
        ]
    )
    _write_report(tmp_path / "validation-z.json", validation)
    _write_report(tmp_path / "test-a.json", test)

    result = summary.summarize_report_directory(tmp_path)

    splits = result["splits"]
    assert isinstance(splits, dict)
    validation_result = splits["validation"]
    assert isinstance(validation_result, dict)
    assert validation_result["document_count"] == 1
    assert validation_result["predicted_table_count"] == 2
    assert validation_result["reference_table_count"] == 2
    assert validation_result["grounded_reference_table_count"] == 1
    assert validation_result["grounded_matched_table_count"] == 1
    assert validation_result["zero_ref_reference_table_count"] == 1
    assert validation_result["unmatched_predicted_table_count"] == 1

    all_result = splits["all"]
    assert isinstance(all_result, dict)
    assert all_result["document_count"] == 2
    assert all_result["page_count"] == 2
    assert all_result["grounded_reference_table_count"] == 2
    assert all_result["grounded_matched_table_count"] == 1
    assert all_result["grounded_rows_columns_exact_count"] == 1
    assert all_result["grounded_grid_span_exact_count"] == 1
    assert all_result["grounded_strict_cell_assignment"] == {
        "exact_count": 4,
        "sample_count": 11,
        "exact_rate": 4 / 11,
    }
    assert all_result["grounded_assignment_pairs"] == {
        "true_positive_count": 2,
        "predicted_count": 3,
        "reference_count": 5,
        "precision": 2 / 3,
        "recall": 2 / 5,
        "f1": 0.5,
    }
    assert all_result["zero_ref_reference_table_count"] == 1
    assert all_result["unmatched_predicted_tables"] == [
        {
            "report_file": "validation-z.json",
            "page_id": "page-v",
            "table_grid_observation_id": "grid-v-unmatched",
        }
    ]
    assert result["metric_status"] == "diagnostic_only_not_accuracy"
    assert result["golden_eligible"] is False
    assert result["training_eligible"] is False
    assert result["release_eligible"] is False


def test_main_prints_deterministic_json_and_creates_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _report([_page("page-1", [_table(1, strict_sample=1)])])
    _write_report(tmp_path / "validation-b.json", report)
    _write_report(tmp_path / "test-a.json", report)
    output = tmp_path / "result" / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize", "--report-dir", str(tmp_path), "--output", str(output)],
    )

    assert summary.main() == 0
    stdout = capsys.readouterr().out.encode("utf-8")
    assert output.read_bytes() == stdout
    assert json.loads(stdout)["splits"]["all"]["document_count"] == 2

    with pytest.raises(FileExistsError, match="output already exists"):
        summary.main()


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = b'{"schema_version":"a","schema_version":"b"}'
    (tmp_path / "validation-a.json").write_bytes(duplicate)
    (tmp_path / "test-a.json").write_bytes(duplicate)

    with pytest.raises(StrictJSONError, match="duplicate object key"):
        summary.summarize_report_directory(tmp_path)


def test_rejects_inconsistent_page_counts(tmp_path: Path) -> None:
    report = _report([_page("page-1", [_table(1, strict_sample=1)])])
    pages = report["pages"]
    assert isinstance(pages, list)
    page = pages[0]
    page["predicted_table_count"] = 1
    _write_report(tmp_path / "validation-a.json", report)
    _write_report(tmp_path / "test-a.json", _report([_page("page-1", [])]))

    with pytest.raises(ValueError, match="predicted_table_count is inconsistent"):
        summary.summarize_report_directory(tmp_path)
