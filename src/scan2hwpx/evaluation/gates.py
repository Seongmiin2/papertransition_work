from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .safe_artifact_io import read_bounded_regular_file
from .strict_json import require_strict_json_bytes

Comparator = Literal["<=", ">=", "=="]
GateSetId = Literal["initial_v1", "model-first-v2"]
MAX_MODEL_RELEASE_GATES_BYTES = 64 * 1024
MAX_MODEL_RELEASE_GATES_JSON_DEPTH = 16
MAX_MODEL_RELEASE_GATES_JSON_NODES = 1_000

MetricCriterion = tuple[str, Comparator, float]
_INITIAL_V1_METRICS: tuple[MetricCriterion, ...] = (
    ("model_a.overall_cer", "<=", 0.005),
    ("model_a.critical_token_error_rate", "<=", 0.001),
    ("model_a.role_macro_f1", ">=", 0.97),
    ("model_a.reading_order_exact_match", ">=", 0.99),
    ("model_a.reading_order_kendall_tau", ">=", 0.995),
    ("model_a.table_topology_exact_match", ">=", 0.95),
    ("model_a.formula_detection_recall", ">=", 0.98),
    ("model_a.normalized_formula_similarity", ">=", 0.97),
    ("model_b.schema_valid_rate", "==", 1.0),
    ("model_b.content_reference_coverage", "==", 1.0),
    ("model_b.unauthorized_content_changes", "==", 0.0),
    ("end_to_end.hwpx_package_valid_rate", "==", 1.0),
    ("end_to_end.dvc_validation_pass_rate", "==", 1.0),
    ("end_to_end.hancom_round_trip_pass_rate", "==", 1.0),
    ("end_to_end.auto_pass_acceptance_rate", ">=", 0.995),
)
_MODEL_FIRST_V2_METRICS: tuple[MetricCriterion, ...] = (
    *_INITIAL_V1_METRICS,
    ("model_a.high_risk_review_recall", ">=", 0.99),
    ("model_a.confidence_ece", "<=", 0.05),
)
_METRICS_BY_GATE_SET: Mapping[GateSetId, tuple[MetricCriterion, ...]] = MappingProxyType(
    {
        "initial_v1": _INITIAL_V1_METRICS,
        "model-first-v2": _MODEL_FIRST_V2_METRICS,
    }
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MetricGate(_StrictModel):
    id: str = Field(min_length=1)
    comparator: Comparator
    threshold: float = Field(ge=0, le=1)


class DatasetGates(_StrictModel):
    document_level_split: Literal[True]
    template_family_holdout: Literal[True]
    minimum_real_scan_test_documents: Literal[50]
    minimum_real_scan_test_pages: Literal[500]


class ModelReleaseGates(_StrictModel):
    schema_version: Literal["1.0"]
    gate_set_id: GateSetId
    require_all_measurements: Literal[True]
    dataset_gates: DatasetGates
    metrics: tuple[MetricGate, ...]

    @model_validator(mode="after")
    def validate_immutable_gate_set(self) -> ModelReleaseGates:
        expected_items = _METRICS_BY_GATE_SET[self.gate_set_id]
        expected = {
            metric_id: (comparator, threshold)
            for metric_id, comparator, threshold in expected_items
        }
        metric_ids = [metric.id for metric in self.metrics]
        duplicates = sorted(
            {metric_id for metric_id in metric_ids if metric_ids.count(metric_id) > 1}
        )
        if duplicates:
            raise ValueError("duplicate metrics: " + ", ".join(duplicates))

        actual = {metric.id: (metric.comparator, metric.threshold) for metric in self.metrics}
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append("missing metrics: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected metrics: " + ", ".join(unexpected))
            raise ValueError("; ".join(details))

        changed = sorted(
            metric_id
            for metric_id, expected_value in expected.items()
            if actual[metric_id] != expected_value
        )
        if changed:
            raise ValueError(
                f"{self.gate_set_id} comparator or threshold changed: " + ", ".join(changed)
            )
        if tuple(metric_ids) != tuple(item[0] for item in expected_items):
            raise ValueError(f"{self.gate_set_id} metrics are not in canonical order")
        return self


class DatasetEvidence(_StrictModel):
    document_level_split: bool
    template_family_holdout: bool
    real_scan_test_documents: int = Field(ge=0)
    real_scan_test_pages: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    gate_set_id: str
    promotable: bool
    failed_gates: tuple[str, ...]


def load_model_release_gates(path: Path | str) -> ModelReleaseGates:
    """Load one immutable gate set from bounded, unambiguous UTF-8 JSON."""
    payload = read_bounded_regular_file(
        Path(path),
        max_bytes=MAX_MODEL_RELEASE_GATES_BYTES,
        label="model release gates",
    )
    require_strict_json_bytes(
        payload,
        max_depth=MAX_MODEL_RELEASE_GATES_JSON_DEPTH,
        max_nodes=MAX_MODEL_RELEASE_GATES_JSON_NODES,
        label="model release gates",
    )
    return ModelReleaseGates.model_validate_json(payload, strict=True)


def model_release_gates_sha256(gates: ModelReleaseGates) -> str:
    """Return a deterministic semantic digest for one validated gate set."""
    restored = _revalidate_gates(gates)
    canonical = json.dumps(
        restored.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluate_model_release(
    gates: ModelReleaseGates,
    measurements: Mapping[str, float],
    dataset: DatasetEvidence | None,
) -> ReleaseDecision:
    """Return a non-promotable decision until every dataset and metric gate passes."""
    gates = _revalidate_gates(gates)
    if dataset is not None:
        dataset = _revalidate_dataset_evidence(dataset)
    measurement_snapshot = dict(measurements)

    expected_ids = {metric.id for metric in gates.metrics}
    unexpected = sorted(set(measurement_snapshot) - expected_ids)
    if unexpected:
        raise ValueError("unexpected measurements: " + ", ".join(unexpected))

    failed: list[str] = []
    if dataset is None:
        failed.append("dataset_evidence.missing")
    else:
        requirements = gates.dataset_gates
        if not dataset.document_level_split:
            failed.append("dataset.document_level_split")
        if not dataset.template_family_holdout:
            failed.append("dataset.template_family_holdout")
        if dataset.real_scan_test_documents < requirements.minimum_real_scan_test_documents:
            failed.append("dataset.real_scan_test_documents")
        if dataset.real_scan_test_pages < requirements.minimum_real_scan_test_pages:
            failed.append("dataset.real_scan_test_pages")

    for metric in gates.metrics:
        if metric.id not in measurement_snapshot:
            failed.append(metric.id + ".unmeasured")
            continue
        value = measurement_snapshot[metric.id]
        _validate_measurement(metric.id, value)
        if not _compare(float(value), metric.comparator, metric.threshold):
            failed.append(metric.id)

    return ReleaseDecision(
        gate_set_id=gates.gate_set_id,
        promotable=not failed,
        failed_gates=tuple(failed),
    )


def _validate_measurement(metric_id: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"measurement {metric_id} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"measurement {metric_id} must be finite")
    if metric_id == "model_b.unauthorized_content_changes":
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"measurement {metric_id} must be a non-negative integer")
        return
    if not 0 <= value <= 1:
        raise ValueError(f"measurement {metric_id} must be between 0 and 1")


def _revalidate_gates(gates: ModelReleaseGates) -> ModelReleaseGates:
    if type(gates) is not ModelReleaseGates:
        raise TypeError("gates must be an exact ModelReleaseGates instance")
    try:
        payload = gates.model_dump(mode="python", warnings="error")
        restored = ModelReleaseGates.model_validate(payload, strict=True)
        restored_payload = restored.model_dump(mode="python", warnings="error")
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("gates are not a valid immutable gate set") from exc
    if not _exact_value_equal(payload, restored_payload):
        raise ValueError("gates are not a valid immutable gate set")
    return restored


def _revalidate_dataset_evidence(dataset: DatasetEvidence) -> DatasetEvidence:
    if type(dataset) is not DatasetEvidence:
        raise TypeError("dataset must be an exact DatasetEvidence instance")
    try:
        payload = dataset.model_dump(mode="python", warnings="error")
        restored = DatasetEvidence.model_validate(payload, strict=True)
        restored_payload = restored.model_dump(mode="python", warnings="error")
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("dataset is not valid dataset evidence") from exc
    if not _exact_value_equal(payload, restored_payload):
        raise ValueError("dataset is not valid dataset evidence")
    return restored


def _exact_value_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _exact_value_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _exact_value_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def _compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == "<=":
        return value <= threshold
    if comparator == ">=":
        return value >= threshold
    return value == threshold
