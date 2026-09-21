from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scan2hwpx.evaluation.gates import load_model_release_gates
from scan2hwpx.evaluation.model_a import MODEL_A_METRIC_DEFINITION_IDS
from scan2hwpx.evaluation.release import (
    AGGREGATE_ONLY_GATE_ID,
    RELEASE_METRIC_SET_ID,
    check_model_release,
    main,
    write_release_audit,
)

GATES_PATH = Path("ai/evaluation/model_release_gates.json")
EXAMPLE_BUNDLE_PATH = Path("ai/evaluation/model_bundle.example.json")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dataset_payload() -> dict[str, Any]:
    documents = [
        {
            "id": f"test-{index:03d}",
            "lineage_id": f"source-{index:03d}",
            "source_sha256": f"{index:064x}",
            "template_family": f"held-out-family-{index:03d}",
            "split": "test",
            "capture": "real_scan",
            "page_count": 10,
            "annotation_artifact_ref": (
                f"artifact://golden-v1/annotations/test-{index:03d}.json"
            ),
            "annotation_sha256": f"{index + 1000:064x}",
            "verified": True,
            "usage_rights_verified": True,
            "verification_attestation": {
                "reviewer_id": "reviewer-001",
                "verified_at": "2026-09-09T09:00:00+09:00",
                "verification_artifact_ref": (
                    f"artifact://golden-v1/verifications/test-{index:03d}.json"
                ),
                "verification_artifact_sha256": f"{index + 2000:064x}",
            },
        }
        for index in range(1, 51)
    ]
    return {"schema_version": "1.0", "example_only": False, "documents": documents}


def _bundle_payload(dataset_path: Path) -> dict[str, Any]:
    payload = json.loads(EXAMPLE_BUNDLE_PATH.read_text(encoding="utf-8"))
    payload["bundle_id"] = "candidate-bundle-1"
    payload["release_gate_set_id"] = load_model_release_gates(GATES_PATH).gate_set_id
    payload["dataset_manifest_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    payload["model_a"] = {
        "status": "candidate",
        "model_id": "document-understanding-model-a-v1",
        "revision": "candidate-v1",
        "artifact_ref": "artifact://models/model-a/document-understanding-v1",
        "artifact_sha256": "1" * 64,
    }
    payload["model_b"] = {
        "status": "candidate",
        "model_id": "document-author-model-b-v1",
        "revision": "candidate-v1",
        "artifact_ref": "artifact://models/model-b/document-author-v1",
        "artifact_sha256": "7" * 64,
    }
    return payload


def _metric_definition_ids() -> dict[str, str]:
    gates = load_model_release_gates(GATES_PATH)
    definitions = {
        metric.id: f"metric-definition/{metric.id}/v1" for metric in gates.metrics
    }
    definitions.update(MODEL_A_METRIC_DEFINITION_IDS)
    return definitions


def _measurement_payload(dataset_path: Path, bundle_path: Path) -> dict[str, Any]:
    gates = load_model_release_gates(GATES_PATH)
    measurements = {metric.id: metric.threshold for metric in gates.metrics}
    measurements["model_b.unauthorized_content_changes"] = 0
    document_ids = [document["id"] for document in _dataset_payload()["documents"]]
    return {
        "schema_version": "1.0",
        "evidence_level": "aggregate_only",
        "model_bundle_id": "candidate-bundle-1",
        "model_bundle_manifest_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "gate_set_id": gates.gate_set_id,
        "metric_set_id": RELEASE_METRIC_SET_ID,
        "metric_definition_ids": _metric_definition_ids(),
        "evaluator_id": "scan2hwpx-release-evaluator",
        "evaluator_version": "0.0.1",
        "evaluator_artifact_ref": "repo://src/scan2hwpx/evaluation/release.py",
        "evaluator_artifact_sha256": "e" * 64,
        "evaluated_document_ids": document_ids,
        "metric_sample_counts": {
            metric_id: len(document_ids) for metric_id in measurements
        },
        "measurements": measurements,
    }


def _valid_release_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_path = _write_json(tmp_path / "dataset.json", _dataset_payload())
    bundle_path = _write_json(tmp_path / "model-bundle.json", _bundle_payload(dataset_path))
    measurements_path = _write_json(
        tmp_path / "measurements.json",
        _measurement_payload(dataset_path, bundle_path),
    )
    return dataset_path, bundle_path, measurements_path


def _check(
    dataset_path: Path,
    bundle_path: Path,
    measurements_path: Path,
):
    return check_model_release(
        GATES_PATH,
        dataset_path,
        bundle_path,
        measurements_path,
    )


def test_aggregate_measurements_are_auditable_but_never_promotable(
    tmp_path: Path,
) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)

    audit = _check(dataset_path, bundle_path, measurements_path)

    assert audit.promotable is False
    assert audit.failed_gates == (AGGREGATE_ONLY_GATE_ID,)
    assert audit.evidence_level == "aggregate_only"
    assert audit.evaluated_document_count == 50
    assert audit.dataset_evidence.real_scan_test_documents == 50
    assert audit.dataset_evidence.real_scan_test_pages == 500
    assert audit.model_bundle_manifest_sha256 == hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()


def test_release_check_reports_unmeasured_gate_before_aggregate_block(
    tmp_path: Path,
) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload["measurements"].pop("model_a.overall_cer")
    payload["metric_sample_counts"].pop("model_a.overall_cer")
    _write_json(measurements_path, payload)

    audit = _check(dataset_path, bundle_path, measurements_path)

    assert audit.promotable is False
    assert audit.failed_gates == (
        "model_a.overall_cer.unmeasured",
        AGGREGATE_ONLY_GATE_ID,
    )


def test_release_audit_is_written_atomically(tmp_path: Path) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    audit = _check(dataset_path, bundle_path, measurements_path)
    output = tmp_path / "release-audit.json"

    write_release_audit(output, audit)

    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["model_bundle_id"] == "candidate-bundle-1"
    assert restored["promotable"] is False
    assert restored["failed_gates"] == [AGGREGATE_ONLY_GATE_ID]
    assert not output.with_suffix(".json.partial").exists()


@pytest.mark.parametrize(
    ("bundle_field", "invalid_value", "message"),
    [
        (
            "bundle_id",
            "different-bundle",
            "model_bundle_id does not match model bundle",
        ),
        (
            "release_gate_set_id",
            "different-gates",
            "release_gate_set_id does not match release gates",
        ),
        (
            "dataset_manifest_sha256",
            "f" * 64,
            "dataset_manifest_sha256 does not match dataset manifest",
        ),
    ],
)
def test_release_check_rejects_bundle_provenance_mismatch(
    tmp_path: Path,
    bundle_field: str,
    invalid_value: str,
    message: str,
) -> None:
    dataset_path = _write_json(tmp_path / "dataset.json", _dataset_payload())
    bundle_payload = _bundle_payload(dataset_path)
    bundle_payload[bundle_field] = invalid_value
    bundle_path = _write_json(tmp_path / "model-bundle.json", bundle_payload)
    measurements_path = _write_json(
        tmp_path / "measurements.json",
        _measurement_payload(dataset_path, bundle_path),
    )

    with pytest.raises(ValueError, match=message):
        _check(dataset_path, bundle_path, measurements_path)


@pytest.mark.parametrize(
    ("report_field", "invalid_value", "message"),
    [
        (
            "model_bundle_manifest_sha256",
            "a" * 64,
            "model_bundle_manifest_sha256 does not match model bundle",
        ),
        (
            "dataset_manifest_sha256",
            "b" * 64,
            "dataset_manifest_sha256 does not match dataset manifest",
        ),
        (
            "gate_set_id",
            "different-gates",
            "gate_set_id does not match release gates",
        ),
    ],
)
def test_release_check_rejects_report_provenance_mismatch(
    tmp_path: Path,
    report_field: str,
    invalid_value: str,
    message: str,
) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload[report_field] = invalid_value
    _write_json(measurements_path, payload)

    with pytest.raises(ValueError, match=message):
        _check(dataset_path, bundle_path, measurements_path)


def test_release_check_rejects_placeholder_component(tmp_path: Path) -> None:
    dataset_path = _write_json(tmp_path / "dataset.json", _dataset_payload())
    bundle_payload = _bundle_payload(dataset_path)
    bundle_payload["model_a"].update(
        status="placeholder",
        revision="not-trained",
        artifact_ref=None,
        artifact_sha256=None,
    )
    bundle_path = _write_json(tmp_path / "model-bundle.json", bundle_payload)
    measurements_path = _write_json(
        tmp_path / "measurements.json",
        _measurement_payload(dataset_path, bundle_path),
    )

    with pytest.raises(ValueError, match="model bundle contains placeholder components"):
        _check(dataset_path, bundle_path, measurements_path)


def test_release_check_rejects_incomplete_document_coverage(tmp_path: Path) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload["evaluated_document_ids"].pop()
    _write_json(measurements_path, payload)

    with pytest.raises(ValueError, match="exactly match dataset test ids"):
        _check(dataset_path, bundle_path, measurements_path)


def test_release_check_rejects_incomplete_metric_coverage(tmp_path: Path) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload["metric_sample_counts"]["model_a.overall_cer"] = 49
    _write_json(measurements_path, payload)

    with pytest.raises(ValueError, match="cover every test document"):
        _check(dataset_path, bundle_path, measurements_path)


def test_release_check_rejects_changed_canonical_metric_definition(
    tmp_path: Path,
) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload["metric_definition_ids"]["model_a.overall_cer"] = "different-definition/v1"
    _write_json(measurements_path, payload)

    with pytest.raises(ValueError, match="canonical definitions"):
        _check(dataset_path, bundle_path, measurements_path)


def test_release_check_uses_strict_gate_loader(tmp_path: Path) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    gates_path = tmp_path / "duplicate-key-gates.json"
    gates_path.write_text(
        GATES_PATH.read_text(encoding="utf-8").replace(
            "{", '{"schema_version":"1.0",', 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate object key"):
        check_model_release(
            gates_path,
            dataset_path,
            bundle_path,
            measurements_path,
        )


def test_release_check_rejects_nonfinite_measurement(tmp_path: Path) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload["measurements"]["model_a.overall_cer"] = float("nan")
    _write_json(measurements_path, payload)

    with pytest.raises(ValueError, match="must be finite"):
        _check(dataset_path, bundle_path, measurements_path)


@pytest.mark.parametrize("duplicate_field", ["annotation_artifact_ref", "annotation_sha256"])
def test_release_check_rejects_duplicate_annotation_evidence(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    dataset_payload = _dataset_payload()
    dataset_payload["documents"][1][duplicate_field] = dataset_payload["documents"][0][
        duplicate_field
    ]
    dataset_path = _write_json(tmp_path / "dataset.json", dataset_payload)
    bundle_path = _write_json(tmp_path / "model-bundle.json", _bundle_payload(dataset_path))
    measurements_path = _write_json(
        tmp_path / "measurements.json",
        _measurement_payload(dataset_path, bundle_path),
    )

    with pytest.raises(ValueError, match="annotation"):
        _check(dataset_path, bundle_path, measurements_path)


def test_measurement_report_cannot_claim_unimplemented_verified_evidence(
    tmp_path: Path,
) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    payload = _measurement_payload(dataset_path, bundle_path)
    payload["evidence_level"] = "verified_records"
    _write_json(measurements_path, payload)

    with pytest.raises(ValidationError, match="aggregate_only"):
        _check(dataset_path, bundle_path, measurements_path)


def test_cli_accepts_bundle_manifest_and_returns_non_promotable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path, bundle_path, measurements_path = _valid_release_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check-model-release",
            str(measurements_path),
            str(dataset_path),
            str(bundle_path),
            "--gates",
            str(GATES_PATH),
        ],
    )

    assert main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["promotable"] is False
    assert output["failed_gates"] == [AGGREGATE_ONLY_GATE_ID]
