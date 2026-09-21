from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset import GoldenDatasetManifest, validate_golden_dataset_manifest
from .gates import DatasetEvidence, evaluate_model_release, load_model_release_gates
from .model_a import MODEL_A_METRIC_DEFINITION_IDS
from .model_bundle import (
    Identifier,
    LogicalArtifactRef,
    ModelBundleManifest,
    Sha256,
)

RELEASE_METRIC_SET_ID = "release-metrics/initial-v1"
AGGREGATE_ONLY_GATE_ID = "release.evidence.aggregate_only"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MeasurementReport(_StrictModel):
    schema_version: Literal["1.0"]
    evidence_level: Literal["aggregate_only"]
    model_bundle_id: str = Field(min_length=1)
    model_bundle_manifest_sha256: Sha256
    dataset_manifest_sha256: Sha256
    gate_set_id: str = Field(min_length=1)
    metric_set_id: Literal["release-metrics/initial-v1"]
    metric_definition_ids: dict[str, str]
    evaluator_id: Identifier
    evaluator_version: Identifier
    evaluator_artifact_ref: LogicalArtifactRef
    evaluator_artifact_sha256: Sha256
    evaluated_document_ids: tuple[str, ...] = Field(min_length=1)
    metric_sample_counts: dict[str, int]
    measurements: dict[str, int | float]

    @model_validator(mode="after")
    def validate_evidence_metadata(self) -> Self:
        for field_name, values in (
            ("measurement", self.measurements),
            ("metric definition", self.metric_definition_ids),
            ("metric sample count", self.metric_sample_counts),
        ):
            invalid = sorted(value for value in values if not value.strip())
            if invalid:
                raise ValueError(f"{field_name} ids must be non-empty")

        empty_definitions = sorted(
            metric_id
            for metric_id, definition_id in self.metric_definition_ids.items()
            if not definition_id.strip()
        )
        if empty_definitions:
            raise ValueError(
                "metric definition values must be non-empty: " + ", ".join(empty_definitions)
            )

        invalid_counts = sorted(
            metric_id for metric_id, count in self.metric_sample_counts.items() if count <= 0
        )
        if invalid_counts:
            raise ValueError("metric sample counts must be positive: " + ", ".join(invalid_counts))
        if set(self.metric_sample_counts) != set(self.measurements):
            raise ValueError("metric sample count ids must match measured metric ids")

        normalized_document_ids = [
            document_id.casefold() for document_id in self.evaluated_document_ids
        ]
        if len(set(normalized_document_ids)) != len(normalized_document_ids):
            raise ValueError("evaluated document ids must be unique")
        if any(not document_id.strip() for document_id in self.evaluated_document_ids):
            raise ValueError("evaluated document ids must be non-empty")

        artifact_path = self.evaluator_artifact_ref.split("://", maxsplit=1)[1]
        if any(part in {"", ".", ".."} for part in artifact_path.split("/")):
            raise ValueError(
                "evaluator artifact ref must not contain empty or traversal path segments"
            )
        return self


class ReleaseAudit(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_level: Literal["aggregate_only"]
    model_bundle_id: str = Field(min_length=1)
    gate_set_id: str = Field(min_length=1)
    metric_set_id: str = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    evaluator_artifact_ref: str = Field(min_length=1)
    evaluator_artifact_sha256: Sha256
    dataset_manifest_sha256: Sha256
    model_bundle_manifest_sha256: Sha256
    measurement_report_sha256: Sha256
    evaluated_document_count: int = Field(gt=0)
    promotable: Literal[False]
    failed_gates: tuple[str, ...]
    measured_metrics: tuple[str, ...]
    metric_definition_ids: dict[str, str]
    metric_sample_counts: dict[str, int]
    dataset_evidence: DatasetEvidence


def check_model_release(
    gates_path: Path | str,
    dataset_manifest_path: Path | str,
    model_bundle_manifest_path: Path | str,
    measurement_report_path: Path | str,
) -> ReleaseAudit:
    """Audit aggregate measurements without treating them as promotion evidence."""
    gates = load_model_release_gates(gates_path)
    dataset_manifest_bytes = Path(dataset_manifest_path).read_bytes()
    manifest = GoldenDatasetManifest.model_validate_json(dataset_manifest_bytes)
    dataset_evidence = validate_golden_dataset_manifest(manifest)
    model_bundle_manifest_bytes = Path(model_bundle_manifest_path).read_bytes()
    bundle = ModelBundleManifest.model_validate_json(model_bundle_manifest_bytes)
    measurement_report_bytes = Path(measurement_report_path).read_bytes()
    report = MeasurementReport.model_validate_json(measurement_report_bytes)

    dataset_manifest_sha256 = hashlib.sha256(dataset_manifest_bytes).hexdigest()
    model_bundle_manifest_sha256 = hashlib.sha256(model_bundle_manifest_bytes).hexdigest()
    measurement_report_sha256 = hashlib.sha256(measurement_report_bytes).hexdigest()

    if not bundle.components_complete:
        raise ValueError("model bundle contains placeholder components")
    if report.model_bundle_id != bundle.bundle_id:
        raise ValueError("measurement report model_bundle_id does not match model bundle")
    if report.model_bundle_manifest_sha256 != model_bundle_manifest_sha256:
        raise ValueError(
            "measurement report model_bundle_manifest_sha256 does not match model bundle"
        )
    if bundle.release_gate_set_id != gates.gate_set_id:
        raise ValueError("model bundle release_gate_set_id does not match release gates")
    if report.gate_set_id != gates.gate_set_id:
        raise ValueError("measurement report gate_set_id does not match release gates")
    if bundle.dataset_manifest_sha256 != dataset_manifest_sha256:
        raise ValueError("model bundle dataset_manifest_sha256 does not match dataset manifest")
    if report.dataset_manifest_sha256 != dataset_manifest_sha256:
        raise ValueError(
            "measurement report dataset_manifest_sha256 does not match dataset manifest"
        )

    expected_metric_ids = {metric.id for metric in gates.metrics}
    if set(report.metric_definition_ids) != expected_metric_ids:
        raise ValueError("metric definition ids must match release gate metric ids")
    changed_model_a_definitions = sorted(
        metric_id
        for metric_id, definition_id in MODEL_A_METRIC_DEFINITION_IDS.items()
        if report.metric_definition_ids[metric_id] != definition_id
    )
    if changed_model_a_definitions:
        raise ValueError(
            "model A metric definition ids do not match canonical definitions: "
            + ", ".join(changed_model_a_definitions)
        )

    test_document_ids = tuple(
        document.id for document in manifest.documents if document.split == "test"
    )
    if set(report.evaluated_document_ids) != set(test_document_ids):
        raise ValueError("evaluated document ids must exactly match dataset test ids")
    incomplete_metrics = sorted(
        metric_id
        for metric_id, sample_count in report.metric_sample_counts.items()
        if sample_count != len(test_document_ids)
    )
    if incomplete_metrics:
        raise ValueError(
            "metric sample counts must cover every test document: " + ", ".join(incomplete_metrics)
        )

    annotation_refs = [document.annotation_artifact_ref for document in manifest.documents]
    if len({value.casefold() for value in annotation_refs}) != len(annotation_refs):
        raise ValueError("dataset annotation artifact refs must be unique")
    annotation_hashes = [document.annotation_sha256 for document in manifest.documents]
    if len(set(annotation_hashes)) != len(annotation_hashes):
        raise ValueError("dataset annotation sha256 values must be unique")

    decision = evaluate_model_release(gates, report.measurements, dataset_evidence)
    return ReleaseAudit(
        evidence_level=report.evidence_level,
        model_bundle_id=bundle.bundle_id,
        gate_set_id=decision.gate_set_id,
        metric_set_id=report.metric_set_id,
        evaluator_id=report.evaluator_id,
        evaluator_version=report.evaluator_version,
        evaluator_artifact_ref=report.evaluator_artifact_ref,
        evaluator_artifact_sha256=report.evaluator_artifact_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        model_bundle_manifest_sha256=model_bundle_manifest_sha256,
        measurement_report_sha256=measurement_report_sha256,
        evaluated_document_count=len(report.evaluated_document_ids),
        promotable=False,
        failed_gates=decision.failed_gates + (AGGREGATE_ONLY_GATE_ID,),
        measured_metrics=tuple(sorted(report.measurements)),
        metric_definition_ids=report.metric_definition_ids,
        metric_sample_counts=report.metric_sample_counts,
        dataset_evidence=dataset_evidence,
    )


def write_release_audit(path: Path, audit: ReleaseAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit aggregate model measurements against release gates"
    )
    parser.add_argument("measurements", type=Path)
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("model_bundle_manifest", type=Path)
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("ai/evaluation/model_release_gates.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = check_model_release(
        args.gates,
        args.dataset_manifest,
        args.model_bundle_manifest,
        args.measurements,
    )
    if args.output is not None:
        write_release_audit(args.output, audit)
    print(audit.model_dump_json(indent=2))
    return 0 if audit.promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
