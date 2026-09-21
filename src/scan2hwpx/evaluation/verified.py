from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scan2hwpx.contracts.models import (
    ContentIR,
    EvidenceIR,
    TextContentNode,
    contract_sha256,
)

from .artifacts import VerificationArtifactPayload
from .dataset import (
    GoldenDatasetDocument,
    GoldenDatasetManifest,
    validate_golden_dataset_manifest,
)
from .model_a import (
    MODEL_A_METRIC_DEFINITION_IDS,
    MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID,
    evaluate_model_a_text_structure,
)
from .model_bundle import (
    Identifier,
    ModelBundleManifest,
    Sha256,
)

VERIFIED_MODEL_A_EVALUATOR_ID: Final = "scan2hwpx-model-a-verified-record-evaluator"
VERIFIED_MODEL_A_EVALUATOR_VERSION: Final = "1.0.0"
VERIFIED_MODEL_A_EVALUATOR_ARTIFACT_REF: Final = "repo://src/scan2hwpx/evaluation/verified.py"

NonEmpty = Annotated[str, Field(min_length=1)]
SafeLogicalArtifactRef = Annotated[
    str,
    Field(
        pattern=(
            r"^(?:artifact|repo)://[A-Za-z0-9][A-Za-z0-9._@+-]*"
            r"(?:/[A-Za-z0-9][A-Za-z0-9._@+-]*)*$"
        )
    ),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ArtifactPayload(_StrictModel):
    """Immutable bytes paired with the digest calculated by the resolver."""

    artifact_ref: SafeLogicalArtifactRef
    sha256: Sha256
    payload: bytes = Field(min_length=1, repr=False)

    @model_validator(mode="after")
    def verify_digest(self) -> Self:
        actual = hashlib.sha256(self.payload).hexdigest()
        if self.sha256 != actual:
            raise ValueError(f"artifact payload digest mismatch: {self.artifact_ref}")
        return self


class ModelAPredictionBinding(_StrictModel):
    document_id: Identifier
    evidence_artifact_ref: SafeLogicalArtifactRef
    evidence_artifact_sha256: Sha256
    prediction_artifact_ref: SafeLogicalArtifactRef
    prediction_artifact_sha256: Sha256

    @model_validator(mode="after")
    def require_distinct_artifacts(self) -> Self:
        if self.evidence_artifact_ref.casefold() == self.prediction_artifact_ref.casefold():
            raise ValueError("evidence and prediction artifact refs must be distinct")
        return self


class VerifiedModelAPredictionManifest(_StrictModel):
    """Bindings for raw predictions; aggregate measurements are intentionally absent."""

    schema_version: Literal["model-a-predictions/1.0"]
    model_bundle_id: Identifier
    model_bundle_manifest_ref: SafeLogicalArtifactRef
    model_bundle_manifest_sha256: Sha256
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    metric_set_id: Literal["model-a-text-structure/v1"]
    metric_definition_ids: dict[str, str]
    documents: tuple[ModelAPredictionBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.metric_definition_ids != MODEL_A_METRIC_DEFINITION_IDS:
            raise ValueError("metric definition ids do not match canonical Model A definitions")

        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "prediction document ids",
        )
        refs = (
            ref.casefold()
            for document in self.documents
            for ref in (
                document.evidence_artifact_ref,
                document.prediction_artifact_ref,
            )
        )
        _reject_duplicates(refs, "prediction manifest artifact refs")
        return self


class VerifiedModelADocumentArtifacts(_StrictModel):
    document_id: Identifier
    evidence: ArtifactPayload
    annotation: ArtifactPayload
    prediction: ArtifactPayload
    verification_attestation: ArtifactPayload

    @model_validator(mode="after")
    def require_distinct_artifacts(self) -> Self:
        refs = (
            self.evidence.artifact_ref.casefold(),
            self.annotation.artifact_ref.casefold(),
            self.prediction.artifact_ref.casefold(),
            self.verification_attestation.artifact_ref.casefold(),
        )
        _reject_duplicates(refs, f"artifact refs for document {self.document_id}")
        return self


class VerifiedModelAEvaluationInputs(_StrictModel):
    """Already-resolved immutable bytes consumed by the deterministic evaluator."""

    dataset_manifest: ArtifactPayload
    model_bundle_manifest: ArtifactPayload
    model_a_artifact: ArtifactPayload
    prediction_manifest: ArtifactPayload
    evaluator_artifact: ArtifactPayload
    documents: tuple[VerifiedModelADocumentArtifacts, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if self.evaluator_artifact.artifact_ref != VERIFIED_MODEL_A_EVALUATOR_ARTIFACT_REF:
            raise ValueError("unexpected verified Model A evaluator artifact ref")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "evaluation input document ids",
        )
        _reject_duplicates(
            (
                artifact.artifact_ref.casefold()
                for document in self.documents
                for artifact in (
                    document.evidence,
                    document.annotation,
                    document.prediction,
                    document.verification_attestation,
                )
            ),
            "evaluation input artifact refs",
        )
        return self


class VerifiedModelADocumentEvidence(_StrictModel):
    document_id: Identifier
    source_document_sha256: Sha256
    evidence_artifact_ref: SafeLogicalArtifactRef
    page_count: int = Field(gt=0)
    evidence_artifact_sha256: Sha256
    evidence_ir_id: NonEmpty
    evidence_ir_sha256: Sha256
    annotation_artifact_ref: SafeLogicalArtifactRef
    annotation_artifact_sha256: Sha256
    annotation_content_ir_id: NonEmpty
    annotation_content_ir_revision: int = Field(ge=1)
    annotation_content_ir_sha256: Sha256
    verification_artifact_ref: SafeLogicalArtifactRef
    verification_artifact_sha256: Sha256
    reviewer_id: Identifier
    verified_at: NonEmpty
    prediction_artifact_ref: SafeLogicalArtifactRef
    prediction_artifact_sha256: Sha256
    prediction_content_ir_id: NonEmpty
    prediction_content_ir_revision: int = Field(ge=1)
    prediction_content_ir_sha256: Sha256


class VerifiedModelAMeasurementReport(_StrictModel):
    schema_version: Literal["verified-model-a-measurements/1.0"] = (
        "verified-model-a-measurements/1.0"
    )
    evidence_level: Literal["verified_records"] = "verified_records"
    review_identity_assurance: Literal["unsigned_manifest_consistency_only"]
    model_bundle_id: Identifier
    model_bundle_manifest_ref: SafeLogicalArtifactRef
    model_bundle_manifest_sha256: Sha256
    model_a_artifact_ref: SafeLogicalArtifactRef
    model_a_artifact_sha256: Sha256
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    prediction_manifest_ref: SafeLogicalArtifactRef
    prediction_manifest_sha256: Sha256
    metric_set_id: Literal["model-a-text-structure/v1"]
    metric_definition_ids: dict[str, str]
    evaluator_id: Literal["scan2hwpx-model-a-verified-record-evaluator"]
    evaluator_version: Literal["1.0.0"]
    evaluator_artifact_ref: Literal["repo://src/scan2hwpx/evaluation/verified.py"]
    evaluator_artifact_sha256: Sha256
    evaluated_document_ids: tuple[Identifier, ...] = Field(min_length=1)
    metric_document_counts: dict[str, int]
    measurements: dict[str, float]
    documents: tuple[VerifiedModelADocumentEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        expected_metric_ids = set(MODEL_A_METRIC_DEFINITION_IDS)
        if self.metric_definition_ids != MODEL_A_METRIC_DEFINITION_IDS:
            raise ValueError("report metric definitions are not canonical")
        if set(self.measurements) != expected_metric_ids:
            raise ValueError("report measurements do not match canonical metric ids")
        if set(self.metric_document_counts) != expected_metric_ids:
            raise ValueError("report metric counts do not match canonical metric ids")
        document_count = len(self.evaluated_document_ids)
        if any(count != document_count for count in self.metric_document_counts.values()):
            raise ValueError("every metric must cover every evaluated document")
        if tuple(document.document_id for document in self.documents) != (
            self.evaluated_document_ids
        ):
            raise ValueError("report document evidence order does not match evaluated ids")
        return self


def evaluate_verified_model_a_records(
    inputs: VerifiedModelAEvaluationInputs,
) -> VerifiedModelAMeasurementReport:
    """Recompute canonical Model A metrics from digest-bound raw IR artifacts.

    ``release.py`` does not yet accept this report schema, so producing this report
    alone cannot promote a model bundle.
    """
    if not isinstance(inputs, VerifiedModelAEvaluationInputs):
        raise TypeError("inputs must be VerifiedModelAEvaluationInputs")

    evaluator_artifact_sha256 = _verify_running_evaluator(inputs.evaluator_artifact)
    dataset = GoldenDatasetManifest.model_validate_json(
        inputs.dataset_manifest.payload,
        strict=True,
    )
    validate_golden_dataset_manifest(dataset)
    bundle = ModelBundleManifest.model_validate_json(
        inputs.model_bundle_manifest.payload,
        strict=True,
    )
    predictions = VerifiedModelAPredictionManifest.model_validate_json(
        inputs.prediction_manifest.payload,
        strict=True,
    )

    _validate_top_level_lineage(inputs, bundle, predictions)

    test_documents = tuple(document for document in dataset.documents if document.split == "test")
    test_ids = tuple(document.id for document in test_documents)
    prediction_by_id = {document.document_id: document for document in predictions.documents}
    artifacts_by_id = {document.document_id: document for document in inputs.documents}
    _require_exact_coverage(test_ids, tuple(prediction_by_id), "prediction manifest")
    _require_exact_coverage(test_ids, tuple(artifacts_by_id), "evaluation inputs")

    reference_texts: list[str] = []
    predicted_texts: list[str] = []
    reference_roles: list[str] = []
    predicted_roles: list[str] = []
    reference_orders: list[tuple[str, ...]] = []
    predicted_orders: list[tuple[str, ...]] = []
    document_evidence: list[VerifiedModelADocumentEvidence] = []

    for dataset_document in test_documents:
        document_id = dataset_document.id
        binding = prediction_by_id[document_id]
        artifacts = artifacts_by_id[document_id]
        _validate_document_artifact_bindings(dataset_document, binding, artifacts)

        evidence = EvidenceIR.model_validate_json(artifacts.evidence.payload, strict=True)
        annotation = ContentIR.model_validate_json(artifacts.annotation.payload, strict=True)
        prediction = ContentIR.model_validate_json(artifacts.prediction.payload, strict=True)
        attestation = _validate_attestation(dataset_document, artifacts.verification_attestation)

        if len(evidence.pages) != dataset_document.page_count:
            raise ValueError(
                f"document {document_id} EvidenceIR page count does not match dataset"
            )
        if evidence.source_document_sha256 != dataset_document.source_sha256:
            raise ValueError(
                f"document {document_id} EvidenceIR source digest does not match dataset"
            )
        annotation.assert_evidence_integrity(evidence)
        prediction.assert_evidence_integrity(evidence)
        _append_document_metrics(
            document_id,
            annotation,
            prediction,
            reference_texts,
            predicted_texts,
            reference_roles,
            predicted_roles,
            reference_orders,
            predicted_orders,
        )
        document_evidence.append(
            VerifiedModelADocumentEvidence(
                document_id=document_id,
                source_document_sha256=dataset_document.source_sha256,
                page_count=len(evidence.pages),
                evidence_artifact_ref=artifacts.evidence.artifact_ref,
                evidence_artifact_sha256=artifacts.evidence.sha256,
                evidence_ir_id=evidence.id,
                evidence_ir_sha256=contract_sha256(evidence),
                annotation_artifact_ref=artifacts.annotation.artifact_ref,
                verification_artifact_ref=artifacts.verification_attestation.artifact_ref,
                verification_artifact_sha256=artifacts.verification_attestation.sha256,
                reviewer_id=attestation.reviewer_id,
                verified_at=attestation.verified_at,
                annotation_artifact_sha256=artifacts.annotation.sha256,
                annotation_content_ir_id=annotation.id,
                annotation_content_ir_revision=annotation.revision,
                annotation_content_ir_sha256=contract_sha256(annotation),
                prediction_artifact_ref=artifacts.prediction.artifact_ref,
                prediction_artifact_sha256=artifacts.prediction.sha256,
                prediction_content_ir_id=prediction.id,
                prediction_content_ir_revision=prediction.revision,
                prediction_content_ir_sha256=contract_sha256(prediction),
            )
        )

    measurements = evaluate_model_a_text_structure(
        reference_texts,
        predicted_texts,
        reference_roles,
        predicted_roles,
        reference_orders,
        predicted_orders,
    )
    metric_document_counts = {
        metric_id: len(test_documents) for metric_id in MODEL_A_METRIC_DEFINITION_IDS
    }
    return VerifiedModelAMeasurementReport(
        model_bundle_id=bundle.bundle_id,
        review_identity_assurance="unsigned_manifest_consistency_only",
        model_bundle_manifest_ref=inputs.model_bundle_manifest.artifact_ref,
        model_bundle_manifest_sha256=inputs.model_bundle_manifest.sha256,
        model_a_artifact_ref=inputs.model_a_artifact.artifact_ref,
        model_a_artifact_sha256=inputs.model_a_artifact.sha256,
        dataset_manifest_ref=inputs.dataset_manifest.artifact_ref,
        dataset_manifest_sha256=inputs.dataset_manifest.sha256,
        prediction_manifest_ref=inputs.prediction_manifest.artifact_ref,
        prediction_manifest_sha256=inputs.prediction_manifest.sha256,
        metric_set_id=MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID,
        metric_definition_ids=dict(MODEL_A_METRIC_DEFINITION_IDS),
        evaluator_id=VERIFIED_MODEL_A_EVALUATOR_ID,
        evaluator_version=VERIFIED_MODEL_A_EVALUATOR_VERSION,
        evaluator_artifact_ref=VERIFIED_MODEL_A_EVALUATOR_ARTIFACT_REF,
        evaluator_artifact_sha256=evaluator_artifact_sha256,
        evaluated_document_ids=test_ids,
        metric_document_counts=metric_document_counts,
        measurements=measurements,
        documents=tuple(document_evidence),
    )


def _validate_top_level_lineage(
    inputs: VerifiedModelAEvaluationInputs,
    bundle: ModelBundleManifest,
    predictions: VerifiedModelAPredictionManifest,
) -> None:
    if bundle.model_a.status == "placeholder":
        raise ValueError("model bundle Model A component is a placeholder")
    if bundle.model_a.artifact_ref is None or bundle.model_a.artifact_sha256 is None:
        raise ValueError("model bundle Model A component has no artifact")
    _require_artifact_binding(
        inputs.model_a_artifact,
        bundle.model_a.artifact_ref,
        bundle.model_a.artifact_sha256,
        "model A",
    )
    if bundle.dataset_manifest_ref != inputs.dataset_manifest.artifact_ref:
        raise ValueError("model bundle dataset manifest ref mismatch")
    if bundle.dataset_manifest_sha256 != inputs.dataset_manifest.sha256:
        raise ValueError("model bundle dataset manifest digest mismatch")
    if predictions.dataset_manifest_ref != inputs.dataset_manifest.artifact_ref:
        raise ValueError("prediction manifest dataset manifest ref mismatch")
    if predictions.dataset_manifest_sha256 != inputs.dataset_manifest.sha256:
        raise ValueError("prediction manifest dataset manifest digest mismatch")
    if predictions.model_bundle_id != bundle.bundle_id:
        raise ValueError("prediction manifest model bundle id mismatch")
    if predictions.model_bundle_manifest_ref != inputs.model_bundle_manifest.artifact_ref:
        raise ValueError("prediction manifest model bundle ref mismatch")
    if predictions.model_bundle_manifest_sha256 != inputs.model_bundle_manifest.sha256:
        raise ValueError("prediction manifest model bundle digest mismatch")


def _validate_document_artifact_bindings(
    dataset_document: GoldenDatasetDocument,
    prediction_binding: ModelAPredictionBinding,
    artifacts: VerifiedModelADocumentArtifacts,
) -> None:
    _require_artifact_binding(
        artifacts.annotation,
        dataset_document.annotation_artifact_ref,
        dataset_document.annotation_sha256,
        "annotation",
    )
    _require_artifact_binding(
        artifacts.verification_attestation,
        dataset_document.verification_attestation.verification_artifact_ref,
        dataset_document.verification_attestation.verification_artifact_sha256,
        "verification attestation",
    )
    _require_artifact_binding(
        artifacts.evidence,
        prediction_binding.evidence_artifact_ref,
        prediction_binding.evidence_artifact_sha256,
        "evidence",
    )
    _require_artifact_binding(
        artifacts.prediction,
        prediction_binding.prediction_artifact_ref,
        prediction_binding.prediction_artifact_sha256,
        "prediction",
    )


def _require_artifact_binding(
    artifact: ArtifactPayload,
    expected_ref: str,
    expected_sha256: str,
    label: str,
) -> None:
    if artifact.artifact_ref != expected_ref:
        raise ValueError(f"{label} artifact ref does not match manifest binding")
    if artifact.sha256 != expected_sha256:
        raise ValueError(f"{label} artifact digest does not match manifest binding")


def _append_document_metrics(
    document_id: str,
    annotation: ContentIR,
    prediction: ContentIR,
    reference_texts: list[str],
    predicted_texts: list[str],
    reference_roles: list[str],
    predicted_roles: list[str],
    reference_orders: list[tuple[str, ...]],
    predicted_orders: list[tuple[str, ...]],
) -> None:
    annotation_by_id = {node.id: node for node in annotation.nodes}
    prediction_by_id = {node.id: node for node in prediction.nodes}
    if set(annotation_by_id) != set(prediction_by_id):
        raise ValueError(
            f"document {document_id} prediction node coverage does not match annotation"
        )

    kind_mismatches = sorted(
        node_id
        for node_id in annotation_by_id
        if annotation_by_id[node_id].kind != prediction_by_id[node_id].kind
    )
    if kind_mismatches:
        raise ValueError(
            f"document {document_id} prediction node kinds do not match annotation: "
            + ", ".join(kind_mismatches)
        )

    for node_id in annotation.reading_order:
        expected = annotation_by_id[node_id]
        actual = prediction_by_id[node_id]
        reference_roles.append(expected.role.value)
        predicted_roles.append(actual.role.value)
        if isinstance(expected, TextContentNode):
            if not isinstance(actual, TextContentNode):
                raise TypeError(
                    f"document {document_id} text node kind mismatch: {node_id}"
                )
            reference_texts.append(expected.text)
            predicted_texts.append(actual.text)

    reference_orders.append(annotation.reading_order)
    predicted_orders.append(prediction.reading_order)


def _require_exact_coverage(
    expected_ids: tuple[str, ...],
    actual_ids: tuple[str, ...],
    label: str,
) -> None:
    expected = set(expected_ids)
    actual = set(actual_ids)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} document coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _reject_duplicates(values: Iterable[str], label: str) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: " + ", ".join(duplicates))


def _verify_running_evaluator(artifact: ArtifactPayload) -> str:
    running_payload = Path(__file__).read_bytes()
    running_sha256 = hashlib.sha256(running_payload).hexdigest()
    if artifact.payload != running_payload or artifact.sha256 != running_sha256:
        raise ValueError("evaluator artifact does not match running evaluator bytes")
    return running_sha256


def _validate_attestation(
    dataset_document: GoldenDatasetDocument,
    artifact: ArtifactPayload,
) -> VerificationArtifactPayload:
    record = VerificationArtifactPayload.model_validate_json(
        artifact.payload,
        strict=True,
    )
    expected = {
        "document_id": dataset_document.id,
        "source_sha256": dataset_document.source_sha256,
        "annotation_sha256": dataset_document.annotation_sha256,
        "reviewer_id": dataset_document.verification_attestation.reviewer_id,
        "verified_at": dataset_document.verification_attestation.verified_at,
    }
    mismatches = sorted(
        field_name
        for field_name, expected_value in expected.items()
        if getattr(record, field_name) != expected_value
    )
    if mismatches:
        raise ValueError(
            f"verification attestation for {dataset_document.id} disagrees with manifest: "
            + ", ".join(mismatches)
        )
    return record
