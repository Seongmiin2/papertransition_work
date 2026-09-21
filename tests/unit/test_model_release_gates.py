from __future__ import annotations

import copy
import json
import operator
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

import scan2hwpx.evaluation.gates as gates_module
from scan2hwpx.evaluation import (
    DatasetEvidence,
    ModelReleaseGates,
    evaluate_model_release,
    load_model_release_gates,
)
from scan2hwpx.evaluation.gates import model_release_gates_sha256

POLICY_PATH = Path("ai/evaluation/model_release_gates.json")
MODEL_FIRST_V2_POLICY_PATH = Path("ai/evaluation/model_release_gates.model-first-v2.json")


def _policy_payload() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _write_policy(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _passing_measurements() -> dict[str, int | float]:
    gates = load_model_release_gates(POLICY_PATH)
    measurements: dict[str, int | float] = {metric.id: metric.threshold for metric in gates.metrics}
    measurements["model_b.unauthorized_content_changes"] = 0
    return measurements


def _passing_dataset() -> DatasetEvidence:
    return DatasetEvidence(
        document_level_split=True,
        template_family_holdout=True,
        real_scan_test_documents=50,
        real_scan_test_pages=500,
    )


def test_initial_v1_loads_with_fixed_dataset_and_metric_gates() -> None:
    gates = load_model_release_gates(POLICY_PATH)
    metrics = {metric.id: (metric.comparator, metric.threshold) for metric in gates.metrics}

    assert gates.gate_set_id == "initial_v1"
    assert gates.require_all_measurements is True
    assert gates.dataset_gates.minimum_real_scan_test_documents == 50
    assert gates.dataset_gates.minimum_real_scan_test_pages == 500
    assert metrics["model_a.overall_cer"] == ("<=", 0.005)
    assert metrics["model_a.reading_order_kendall_tau"] == (">=", 0.995)
    assert metrics["model_b.unauthorized_content_changes"] == ("==", 0.0)
    assert metrics["end_to_end.hancom_round_trip_pass_rate"] == ("==", 1.0)


def test_model_first_v2_preserves_initial_metrics_and_adds_two_model_a_gates() -> None:
    initial = load_model_release_gates(POLICY_PATH)
    model_first = load_model_release_gates(MODEL_FIRST_V2_POLICY_PATH)
    initial_metrics = tuple(
        (metric.id, metric.comparator, metric.threshold) for metric in initial.metrics
    )
    model_first_metrics = tuple(
        (metric.id, metric.comparator, metric.threshold) for metric in model_first.metrics
    )

    assert initial.gate_set_id == "initial_v1"
    assert model_first.gate_set_id == "model-first-v2"
    assert model_first.schema_version == initial.schema_version == "1.0"
    assert model_first.dataset_gates == initial.dataset_gates
    assert len(initial_metrics) == 15
    assert len(model_first_metrics) == 17
    assert model_first_metrics[:15] == initial_metrics
    assert model_first_metrics[15:] == (
        ("model_a.high_risk_review_recall", ">=", 0.99),
        ("model_a.confidence_ece", "<=", 0.05),
    )


def test_model_first_v2_does_not_treat_old_measurements_as_promotion_complete() -> None:
    gates = load_model_release_gates(MODEL_FIRST_V2_POLICY_PATH)

    decision = evaluate_model_release(gates, _passing_measurements(), _passing_dataset())

    assert decision.promotable is False
    assert decision.failed_gates == (
        "model_a.high_risk_review_recall.unmeasured",
        "model_a.confidence_ece.unmeasured",
    )


def test_gate_semantic_digest_is_deterministic_across_json_formatting(
    tmp_path: Path,
) -> None:
    gates = load_model_release_gates(MODEL_FIRST_V2_POLICY_PATH)
    payload = json.loads(MODEL_FIRST_V2_POLICY_PATH.read_text(encoding="utf-8"))
    reformatted = tmp_path / "model-first-v2.reformatted.json"
    reformatted.write_text(
        json.dumps(payload, ensure_ascii=False, indent=7, sort_keys=True),
        encoding="utf-8",
    )
    reloaded = load_model_release_gates(reformatted)

    assert reloaded == gates
    assert model_release_gates_sha256(reloaded) == model_release_gates_sha256(gates)
    assert model_release_gates_sha256(gates) == (
        "b4af1cdcd3bc8191bd8c70926f45a0711f7fa2bc279b56c92304f746532aaab4"
    )


def test_canonical_gate_criteria_are_runtime_immutable() -> None:
    canonical_gate_sets: Any = gates_module._METRICS_BY_GATE_SET
    set_item: Any = operator.setitem

    assert isinstance(canonical_gate_sets, MappingProxyType)
    assert isinstance(canonical_gate_sets["initial_v1"], tuple)
    with pytest.raises(TypeError):
        set_item(canonical_gate_sets, "initial_v1", ())
    with pytest.raises(TypeError):
        set_item(canonical_gate_sets["initial_v1"], 0, ("forged", "==", 0.0))


def test_loader_does_not_use_unbounded_path_read_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_unbounded_read(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes must not be used by the gate loader")

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    assert load_model_release_gates(POLICY_PATH).gate_set_id == "initial_v1"


def test_loader_rejects_oversized_gate_file_before_parsing(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized-gates.json"
    oversized.write_bytes(b" " * (gates_module.MAX_MODEL_RELEASE_GATES_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds the size limit"):
        load_model_release_gates(oversized)


def test_loader_rejects_ambiguous_or_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        b'{"schema_version":"1.0","schema_version":"1.0","gate_set_id":"initial_v1"}'
    )
    with pytest.raises(ValueError, match="duplicate object key"):
        load_model_release_gates(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"threshold":NaN}')
    with pytest.raises(ValueError, match="non-finite number"):
        load_model_release_gates(nonfinite)


def test_model_first_v2_rejects_missing_changed_or_reordered_metrics(
    tmp_path: Path,
) -> None:
    payload = json.loads(MODEL_FIRST_V2_POLICY_PATH.read_text(encoding="utf-8"))

    missing = copy.deepcopy(payload)
    missing["metrics"].pop()
    with pytest.raises(ValidationError, match="missing metrics"):
        load_model_release_gates(_write_policy(tmp_path, missing))

    changed = copy.deepcopy(payload)
    changed["metrics"][-1]["threshold"] = 0.051
    with pytest.raises(ValidationError, match="comparator or threshold changed"):
        load_model_release_gates(_write_policy(tmp_path, changed))

    reordered = copy.deepcopy(payload)
    reordered["metrics"][-2:] = reversed(reordered["metrics"][-2:])
    with pytest.raises(ValidationError, match="canonical order"):
        load_model_release_gates(_write_policy(tmp_path, reordered))


def test_unmeasured_release_cannot_be_promoted() -> None:
    gates = load_model_release_gates(POLICY_PATH)

    decision = evaluate_model_release(gates, {}, None)

    assert decision.promotable is False
    assert "dataset_evidence.missing" in decision.failed_gates
    assert "model_a.overall_cer.unmeasured" in decision.failed_gates


def test_release_is_promotable_only_when_every_gate_passes() -> None:
    gates = load_model_release_gates(POLICY_PATH)
    measurements = _passing_measurements()

    assert evaluate_model_release(gates, measurements, _passing_dataset()).promotable is True

    measurements["model_a.role_macro_f1"] = 0.969
    decision = evaluate_model_release(gates, measurements, _passing_dataset())
    assert decision.promotable is False
    assert decision.failed_gates == ("model_a.role_macro_f1",)


def test_evaluator_rejects_runtime_mutated_gate_object() -> None:
    gates = load_model_release_gates(POLICY_PATH)
    object.__setattr__(gates.metrics[0], "threshold", 1.0)

    with pytest.raises(ValueError, match="valid immutable gate set"):
        evaluate_model_release(gates, _passing_measurements(), _passing_dataset())


def test_evaluator_requires_exact_gate_model_type() -> None:
    class ForgedGateSet(ModelReleaseGates):
        pass

    gates = load_model_release_gates(POLICY_PATH)
    forged = ForgedGateSet.model_validate(gates.model_dump(mode="python"), strict=True)

    with pytest.raises(TypeError, match="exact ModelReleaseGates"):
        evaluate_model_release(forged, _passing_measurements(), _passing_dataset())


def test_evaluator_strictly_revalidates_dataset_evidence() -> None:
    gates = load_model_release_gates(POLICY_PATH)
    forged = DatasetEvidence.model_construct(
        document_level_split=1,  # type: ignore[arg-type]
        template_family_holdout=True,
        real_scan_test_documents=50,
        real_scan_test_pages=500,
    )

    with pytest.raises(ValueError, match="valid dataset evidence"):
        evaluate_model_release(gates, _passing_measurements(), forged)


def test_dataset_minimums_are_enforced() -> None:
    gates = load_model_release_gates(POLICY_PATH)
    dataset = DatasetEvidence(
        document_level_split=True,
        template_family_holdout=False,
        real_scan_test_documents=49,
        real_scan_test_pages=499,
    )

    decision = evaluate_model_release(gates, _passing_measurements(), dataset)

    assert decision.promotable is False
    assert decision.failed_gates == (
        "dataset.template_family_holdout",
        "dataset.real_scan_test_documents",
        "dataset.real_scan_test_pages",
    )


@pytest.mark.parametrize("threshold", [-0.001, 1.001])
def test_loader_rejects_thresholds_outside_metric_range(tmp_path: Path, threshold: float) -> None:
    payload = _policy_payload()
    payload["metrics"][0]["threshold"] = threshold

    with pytest.raises(ValidationError):
        load_model_release_gates(_write_policy(tmp_path, payload))


def test_loader_rejects_duplicate_and_missing_metrics(tmp_path: Path) -> None:
    duplicate = _policy_payload()
    duplicate["metrics"].append(copy.deepcopy(duplicate["metrics"][0]))
    with pytest.raises(ValidationError, match="duplicate metrics"):
        load_model_release_gates(_write_policy(tmp_path, duplicate))

    missing = _policy_payload()
    missing["metrics"].pop()
    with pytest.raises(ValidationError, match="missing metrics"):
        load_model_release_gates(_write_policy(tmp_path, missing))


def test_loader_rejects_invalid_or_changed_comparator(tmp_path: Path) -> None:
    invalid = _policy_payload()
    invalid["metrics"][0]["comparator"] = "<"
    with pytest.raises(ValidationError):
        load_model_release_gates(_write_policy(tmp_path, invalid))

    changed = _policy_payload()
    changed["metrics"][0]["comparator"] = ">="
    with pytest.raises(ValidationError, match="comparator or threshold changed"):
        load_model_release_gates(_write_policy(tmp_path, changed))


@pytest.mark.parametrize(
    ("metric_id", "value"),
    [
        ("model_a.overall_cer", -0.1),
        ("model_a.role_macro_f1", 1.1),
        ("model_b.unauthorized_content_changes", 0.5),
    ],
)
def test_evaluator_rejects_measurements_outside_their_value_range(
    metric_id: str, value: float
) -> None:
    gates = load_model_release_gates(POLICY_PATH)
    measurements = _passing_measurements()
    measurements[metric_id] = value

    with pytest.raises(ValueError, match=metric_id):
        evaluate_model_release(gates, measurements, _passing_dataset())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_evaluator_rejects_nonfinite_measurements(value: float) -> None:
    gates = load_model_release_gates(POLICY_PATH)
    measurements = _passing_measurements()
    measurements["model_a.overall_cer"] = value

    with pytest.raises(ValueError, match="must be finite"):
        evaluate_model_release(gates, measurements, _passing_dataset())
