from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final, Literal, Never, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from scan2hwpx.contracts.models import (
    ContentIR,
    ContentPlanItem,
    HwpDocumentPlan,
    contract_sha256,
)

from .artifacts import VerificationArtifactPayload
from .dataset import GoldenDatasetDocument, GoldenDatasetManifest, validate_golden_dataset_manifest
from .model_a import levenshtein_distance
from .model_bundle import Identifier, ModelBundleManifest, Sha256
from .verified import ArtifactPayload, SafeLogicalArtifactRef

MODEL_B_PLAN_METRIC_SET_ID: Final = "model-b-plan-integrity/v1"
MODEL_B_METRIC_DEFINITION_IDS: Final[dict[str, str]] = {
    "model_b.schema_valid_rate": "hwp-document-plan/strict-schema-document-valid-rate/v1",
    "model_b.content_reference_coverage": (
        "hwp-document-plan/content-reference-integrity-document-exact-match-rate/v1"
    ),
    "model_b.unauthorized_content_changes": (
        "hwp-document-plan/approved-content-projection-and-lineage-edit-count/v1"
    ),
}
VERIFIED_MODEL_B_EVALUATOR_ID: Final = "scan2hwpx-model-b-verified-record-evaluator"
VERIFIED_MODEL_B_EVALUATOR_VERSION: Final = "1.0.0"
VERIFIED_MODEL_B_EVALUATOR_ARTIFACT_REF: Final = "repo://src/scan2hwpx/evaluation/model_b.py"

NonEmpty = Annotated[str, Field(min_length=1)]
ContractT = TypeVar("ContractT", bound=BaseModel)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ModelBPlanVerificationBinding(_StrictModel):
    reviewer_id: Identifier
    verified_at: NonEmpty
    verification_artifact_ref: SafeLogicalArtifactRef
    verification_artifact_sha256: Sha256

    @field_validator("verified_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("verified_at must be an ISO-8601 datetime") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("verified_at must include a timezone")
        return value


class ModelBGoldenDocumentBinding(_StrictModel):
    document_id: Identifier
    lineage_id: Identifier
    source_document_sha256: Sha256
    content_ir_artifact_ref: SafeLogicalArtifactRef
    content_ir_artifact_sha256: Sha256
    content_ir_contract_sha256: Sha256
    content_verification_artifact_ref: SafeLogicalArtifactRef
    content_verification_artifact_sha256: Sha256
    gold_plan_artifact_ref: SafeLogicalArtifactRef
    gold_plan_artifact_sha256: Sha256
    gold_plan_contract_sha256: Sha256
    plan_verification: ModelBPlanVerificationBinding

    @model_validator(mode="after")
    def require_distinct_artifacts(self) -> Self:
        _reject_duplicates(
            (
                self.content_ir_artifact_ref.casefold(),
                self.content_verification_artifact_ref.casefold(),
                self.gold_plan_artifact_ref.casefold(),
                self.plan_verification.verification_artifact_ref.casefold(),
            ),
            f"golden artifact refs for document {self.document_id}",
        )
        return self


class VerifiedModelBGoldenManifest(_StrictModel):
    """Digest bindings for held-out ContentIR and human-approved plan records."""

    schema_version: Literal["model-b-golden/1.0"]
    example_only: bool
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    documents: tuple[ModelBGoldenDocumentBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.example_only:
            raise ValueError("example-only Model B golden manifest cannot be evaluated")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "Model B golden document ids",
        )
        _reject_duplicates(
            (
                artifact_ref.casefold()
                for document in self.documents
                for artifact_ref in _golden_binding_artifact_refs(document)
            ),
            "Model B golden artifact refs",
        )
        return self


class ModelBPredictionBinding(_StrictModel):
    document_id: Identifier
    prediction_artifact_ref: SafeLogicalArtifactRef
    prediction_artifact_sha256: Sha256


class VerifiedModelBPredictionManifest(_StrictModel):
    """Raw prediction bindings; callers cannot supply aggregate measurements."""

    schema_version: Literal["model-b-predictions/1.0"]
    model_bundle_id: Identifier
    model_bundle_manifest_ref: SafeLogicalArtifactRef
    model_bundle_manifest_sha256: Sha256
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    golden_manifest_ref: SafeLogicalArtifactRef
    golden_manifest_sha256: Sha256
    metric_set_id: Literal["model-b-plan-integrity/v1"]
    metric_definition_ids: dict[str, str]
    documents: tuple[ModelBPredictionBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.metric_definition_ids != MODEL_B_METRIC_DEFINITION_IDS:
            raise ValueError("metric definition ids do not match canonical Model B definitions")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "Model B prediction document ids",
        )
        _reject_duplicates(
            (document.prediction_artifact_ref.casefold() for document in self.documents),
            "Model B prediction artifact refs",
        )
        return self


class ModelBPlanVerificationArtifact(_StrictModel):
    """Unsigned human-workflow record binding one approved plan to its ContentIR."""

    schema_version: Literal["model-b-plan-verification/1.0"]
    document_id: Identifier
    lineage_id: Identifier
    source_document_sha256: Sha256
    dataset_manifest_sha256: Sha256
    content_ir_artifact_ref: SafeLogicalArtifactRef
    content_ir_artifact_sha256: Sha256
    content_ir_contract_sha256: Sha256
    gold_plan_artifact_ref: SafeLogicalArtifactRef
    gold_plan_artifact_sha256: Sha256
    gold_plan_contract_sha256: Sha256
    reviewer_id: Identifier
    verified_at: NonEmpty
    decision: Literal["approved"]

    @field_validator("verified_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        return ModelBPlanVerificationBinding.require_timezone(value)


class VerifiedModelBDocumentArtifacts(_StrictModel):
    document_id: Identifier
    content_ir: ArtifactPayload
    content_verification_attestation: ArtifactPayload
    gold_plan: ArtifactPayload
    gold_plan_verification_attestation: ArtifactPayload
    prediction: ArtifactPayload

    @model_validator(mode="after")
    def require_distinct_artifacts(self) -> Self:
        _reject_duplicates(
            (
                self.content_ir.artifact_ref.casefold(),
                self.content_verification_attestation.artifact_ref.casefold(),
                self.gold_plan.artifact_ref.casefold(),
                self.gold_plan_verification_attestation.artifact_ref.casefold(),
                self.prediction.artifact_ref.casefold(),
            ),
            f"evaluation artifact refs for document {self.document_id}",
        )
        return self


class VerifiedModelBEvaluationInputs(_StrictModel):
    """Already-resolved immutable bytes consumed by the Model B evaluator."""

    dataset_manifest: ArtifactPayload
    model_bundle_manifest: ArtifactPayload
    model_b_artifact: ArtifactPayload
    golden_manifest: ArtifactPayload
    prediction_manifest: ArtifactPayload
    evaluator_artifact: ArtifactPayload
    documents: tuple[VerifiedModelBDocumentArtifacts, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if self.evaluator_artifact.artifact_ref != VERIFIED_MODEL_B_EVALUATOR_ARTIFACT_REF:
            raise ValueError("unexpected verified Model B evaluator artifact ref")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "Model B evaluation input document ids",
        )
        _reject_duplicates(
            (artifact.artifact_ref.casefold() for artifact in _all_input_artifacts(self)),
            "Model B evaluation input artifact refs",
        )
        return self


class VerifiedModelBDocumentEvidence(_StrictModel):
    document_id: Identifier
    lineage_id: Identifier
    source_document_sha256: Sha256
    content_ir_artifact_ref: SafeLogicalArtifactRef
    content_ir_artifact_sha256: Sha256
    content_ir_contract_sha256: Sha256
    content_verification_artifact_ref: SafeLogicalArtifactRef
    content_verification_artifact_sha256: Sha256
    content_reviewer_id: Identifier
    content_verified_at: NonEmpty
    gold_plan_artifact_ref: SafeLogicalArtifactRef
    gold_plan_artifact_sha256: Sha256
    gold_plan_contract_sha256: Sha256
    plan_verification_artifact_ref: SafeLogicalArtifactRef
    plan_verification_artifact_sha256: Sha256
    plan_reviewer_id: Identifier
    plan_verified_at: NonEmpty
    prediction_artifact_ref: SafeLogicalArtifactRef
    prediction_artifact_sha256: Sha256
    prediction_schema_valid: bool
    prediction_plan_contract_sha256: Sha256 | None
    content_reference_integrity_exact: bool
    unauthorized_content_changes: int = Field(ge=0)


class VerifiedModelBMeasurementReport(_StrictModel):
    schema_version: Literal["verified-model-b-measurements/1.0"] = (
        "verified-model-b-measurements/1.0"
    )
    evidence_level: Literal["verified_records"] = "verified_records"
    review_identity_assurance: Literal["unsigned_manifest_consistency_only"]
    promotable: Literal[False]
    model_bundle_id: Identifier
    model_bundle_manifest_ref: SafeLogicalArtifactRef
    model_bundle_manifest_sha256: Sha256
    model_b_artifact_ref: SafeLogicalArtifactRef
    model_b_artifact_sha256: Sha256
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    golden_manifest_ref: SafeLogicalArtifactRef
    golden_manifest_sha256: Sha256
    prediction_manifest_ref: SafeLogicalArtifactRef
    prediction_manifest_sha256: Sha256
    metric_set_id: Literal["model-b-plan-integrity/v1"]
    metric_definition_ids: dict[str, str]
    evaluator_id: Literal["scan2hwpx-model-b-verified-record-evaluator"]
    evaluator_version: Literal["1.0.0"]
    evaluator_artifact_ref: Literal["repo://src/scan2hwpx/evaluation/model_b.py"]
    evaluator_artifact_sha256: Sha256
    evaluated_document_ids: tuple[Identifier, ...] = Field(min_length=1)
    metric_document_counts: dict[str, int]
    measurements: dict[str, int | float]
    documents: tuple[VerifiedModelBDocumentEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        metric_ids = set(MODEL_B_METRIC_DEFINITION_IDS)
        if self.metric_definition_ids != MODEL_B_METRIC_DEFINITION_IDS:
            raise ValueError("report metric definitions are not canonical")
        if set(self.measurements) != metric_ids:
            raise ValueError("report measurements do not match canonical metric ids")
        if set(self.metric_document_counts) != metric_ids:
            raise ValueError("report metric counts do not match canonical metric ids")
        document_count = len(self.evaluated_document_ids)
        if any(count != document_count for count in self.metric_document_counts.values()):
            raise ValueError("every metric must cover every evaluated document")
        if tuple(document.document_id for document in self.documents) != (
            self.evaluated_document_ids
        ):
            raise ValueError("report document evidence order does not match evaluated ids")
        expected = {
            "model_b.schema_valid_rate": (
                sum(document.prediction_schema_valid for document in self.documents)
                / document_count
            ),
            "model_b.content_reference_coverage": (
                sum(document.content_reference_integrity_exact for document in self.documents)
                / document_count
            ),
            "model_b.unauthorized_content_changes": sum(
                document.unauthorized_content_changes for document in self.documents
            ),
        }
        if self.measurements != expected:
            raise ValueError("report measurements do not match document evidence")
        return self


def evaluate_verified_model_b_records(
    inputs: VerifiedModelBEvaluationInputs,
) -> VerifiedModelBMeasurementReport:
    """Recompute Model B integrity metrics from digest-bound raw plan bytes.

    Invalid prediction JSON is a measured document failure. Invalid manifests,
    gold artifacts, attestations, digests, or lineage abort the entire evaluation.
    This report cannot promote a model because reviewer identity is not authenticated.
    """
    if not isinstance(inputs, VerifiedModelBEvaluationInputs):
        raise TypeError("inputs must be VerifiedModelBEvaluationInputs")

    evaluator_sha256 = _verify_running_evaluator(inputs.evaluator_artifact)
    dataset = _parse_trusted_contract(
        inputs.dataset_manifest.payload,
        GoldenDatasetManifest,
        "dataset manifest",
    )
    validate_golden_dataset_manifest(dataset)
    bundle = _parse_trusted_contract(
        inputs.model_bundle_manifest.payload,
        ModelBundleManifest,
        "model bundle manifest",
    )
    golden = _parse_trusted_contract(
        inputs.golden_manifest.payload,
        VerifiedModelBGoldenManifest,
        "Model B golden manifest",
    )
    predictions = _parse_trusted_contract(
        inputs.prediction_manifest.payload,
        VerifiedModelBPredictionManifest,
        "Model B prediction manifest",
    )
    _validate_top_level_lineage(inputs, bundle, golden, predictions)

    test_documents = tuple(document for document in dataset.documents if document.split == "test")
    if not test_documents:
        raise ValueError("dataset must contain at least one test document")
    test_ids = tuple(document.id for document in test_documents)
    golden_by_id = {document.document_id: document for document in golden.documents}
    prediction_by_id = {document.document_id: document for document in predictions.documents}
    artifacts_by_id = {document.document_id: document for document in inputs.documents}
    _require_exact_coverage(test_ids, tuple(golden_by_id), "Model B golden manifest")
    _require_exact_coverage(test_ids, tuple(prediction_by_id), "Model B prediction manifest")
    _require_exact_coverage(test_ids, tuple(artifacts_by_id), "Model B evaluation inputs")

    evidence: list[VerifiedModelBDocumentEvidence] = []
    for dataset_document in test_documents:
        document_id = dataset_document.id
        golden_binding = golden_by_id[document_id]
        prediction_binding = prediction_by_id[document_id]
        artifacts = artifacts_by_id[document_id]
        _validate_document_artifact_bindings(
            dataset_document,
            golden_binding,
            prediction_binding,
            artifacts,
        )
        content = _parse_trusted_contract(
            artifacts.content_ir.payload,
            ContentIR,
            f"gold ContentIR for {document_id}",
        )
        gold_plan = _parse_trusted_contract(
            artifacts.gold_plan.payload,
            HwpDocumentPlan,
            f"gold plan for {document_id}",
        )
        content_contract_sha256 = contract_sha256(content)
        gold_plan_contract_sha256 = contract_sha256(gold_plan)
        _require_contract_digest(
            content_contract_sha256,
            golden_binding.content_ir_contract_sha256,
            f"gold ContentIR for {document_id}",
        )
        _require_contract_digest(
            gold_plan_contract_sha256,
            golden_binding.gold_plan_contract_sha256,
            f"gold plan for {document_id}",
        )
        gold_plan.assert_content_integrity(content)
        content_attestation = _validate_content_attestation(
            dataset_document,
            artifacts.content_verification_attestation,
        )
        plan_attestation = _validate_plan_attestation(
            dataset_document,
            golden_binding,
            artifacts.gold_plan_verification_attestation,
            inputs.dataset_manifest.sha256,
            content_contract_sha256,
            gold_plan_contract_sha256,
        )

        prediction = _parse_prediction(artifacts.prediction.payload)
        schema_valid = prediction is not None
        reference_integrity = False
        prediction_contract_sha256: str | None = None
        if prediction is None:
            unauthorized_changes = len(_content_projection(gold_plan))
        else:
            prediction_contract_sha256 = contract_sha256(prediction)
            try:
                prediction.assert_content_integrity(content)
            except ValueError:
                pass
            else:
                reference_integrity = True
            unauthorized_changes = _unauthorized_content_changes(
                content,
                gold_plan,
                prediction,
            )

        evidence.append(
            VerifiedModelBDocumentEvidence(
                document_id=document_id,
                lineage_id=dataset_document.lineage_id,
                source_document_sha256=dataset_document.source_sha256,
                content_ir_artifact_ref=artifacts.content_ir.artifact_ref,
                content_ir_artifact_sha256=artifacts.content_ir.sha256,
                content_ir_contract_sha256=content_contract_sha256,
                content_verification_artifact_ref=(
                    artifacts.content_verification_attestation.artifact_ref
                ),
                content_verification_artifact_sha256=(
                    artifacts.content_verification_attestation.sha256
                ),
                content_reviewer_id=content_attestation.reviewer_id,
                content_verified_at=content_attestation.verified_at,
                gold_plan_artifact_ref=artifacts.gold_plan.artifact_ref,
                gold_plan_artifact_sha256=artifacts.gold_plan.sha256,
                gold_plan_contract_sha256=gold_plan_contract_sha256,
                plan_verification_artifact_ref=(
                    artifacts.gold_plan_verification_attestation.artifact_ref
                ),
                plan_verification_artifact_sha256=(
                    artifacts.gold_plan_verification_attestation.sha256
                ),
                plan_reviewer_id=plan_attestation.reviewer_id,
                plan_verified_at=plan_attestation.verified_at,
                prediction_artifact_ref=artifacts.prediction.artifact_ref,
                prediction_artifact_sha256=artifacts.prediction.sha256,
                prediction_schema_valid=schema_valid,
                prediction_plan_contract_sha256=prediction_contract_sha256,
                content_reference_integrity_exact=reference_integrity,
                unauthorized_content_changes=unauthorized_changes,
            )
        )

    measurements: dict[str, int | float] = {
        "model_b.schema_valid_rate": (
            sum(document.prediction_schema_valid for document in evidence) / len(evidence)
        ),
        "model_b.content_reference_coverage": (
            sum(document.content_reference_integrity_exact for document in evidence) / len(evidence)
        ),
        "model_b.unauthorized_content_changes": sum(
            document.unauthorized_content_changes for document in evidence
        ),
    }
    return VerifiedModelBMeasurementReport(
        review_identity_assurance="unsigned_manifest_consistency_only",
        promotable=False,
        model_bundle_id=bundle.bundle_id,
        model_bundle_manifest_ref=inputs.model_bundle_manifest.artifact_ref,
        model_bundle_manifest_sha256=inputs.model_bundle_manifest.sha256,
        model_b_artifact_ref=inputs.model_b_artifact.artifact_ref,
        model_b_artifact_sha256=inputs.model_b_artifact.sha256,
        dataset_manifest_ref=inputs.dataset_manifest.artifact_ref,
        dataset_manifest_sha256=inputs.dataset_manifest.sha256,
        golden_manifest_ref=inputs.golden_manifest.artifact_ref,
        golden_manifest_sha256=inputs.golden_manifest.sha256,
        prediction_manifest_ref=inputs.prediction_manifest.artifact_ref,
        prediction_manifest_sha256=inputs.prediction_manifest.sha256,
        metric_set_id=MODEL_B_PLAN_METRIC_SET_ID,
        metric_definition_ids=dict(MODEL_B_METRIC_DEFINITION_IDS),
        evaluator_id=VERIFIED_MODEL_B_EVALUATOR_ID,
        evaluator_version=VERIFIED_MODEL_B_EVALUATOR_VERSION,
        evaluator_artifact_ref=VERIFIED_MODEL_B_EVALUATOR_ARTIFACT_REF,
        evaluator_artifact_sha256=evaluator_sha256,
        evaluated_document_ids=test_ids,
        metric_document_counts={
            metric_id: len(evidence) for metric_id in MODEL_B_METRIC_DEFINITION_IDS
        },
        measurements=measurements,
        documents=tuple(evidence),
    )


def _validate_top_level_lineage(
    inputs: VerifiedModelBEvaluationInputs,
    bundle: ModelBundleManifest,
    golden: VerifiedModelBGoldenManifest,
    predictions: VerifiedModelBPredictionManifest,
) -> None:
    if bundle.model_b.status == "placeholder":
        raise ValueError("model bundle Model B component is a placeholder")
    if bundle.model_b.artifact_ref is None or bundle.model_b.artifact_sha256 is None:
        raise ValueError("model bundle Model B component has no artifact")
    _require_artifact_binding(
        inputs.model_b_artifact,
        bundle.model_b.artifact_ref,
        bundle.model_b.artifact_sha256,
        "model B",
    )
    for owner, artifact_ref, artifact_sha256 in (
        ("model bundle", bundle.dataset_manifest_ref, bundle.dataset_manifest_sha256),
        ("golden manifest", golden.dataset_manifest_ref, golden.dataset_manifest_sha256),
        (
            "prediction manifest",
            predictions.dataset_manifest_ref,
            predictions.dataset_manifest_sha256,
        ),
    ):
        _require_artifact_binding(
            inputs.dataset_manifest,
            artifact_ref,
            artifact_sha256,
            f"{owner} dataset manifest",
        )
    if predictions.model_bundle_id != bundle.bundle_id:
        raise ValueError("prediction manifest model bundle id mismatch")
    _require_artifact_binding(
        inputs.model_bundle_manifest,
        predictions.model_bundle_manifest_ref,
        predictions.model_bundle_manifest_sha256,
        "prediction manifest model bundle",
    )
    _require_artifact_binding(
        inputs.golden_manifest,
        predictions.golden_manifest_ref,
        predictions.golden_manifest_sha256,
        "prediction manifest golden manifest",
    )


def _validate_document_artifact_bindings(
    dataset_document: GoldenDatasetDocument,
    golden: ModelBGoldenDocumentBinding,
    prediction: ModelBPredictionBinding,
    artifacts: VerifiedModelBDocumentArtifacts,
) -> None:
    expected_scalar_values = {
        "golden lineage id": (golden.lineage_id, dataset_document.lineage_id),
        "golden source digest": (
            golden.source_document_sha256,
            dataset_document.source_sha256,
        ),
        "golden content ref": (
            golden.content_ir_artifact_ref,
            dataset_document.annotation_artifact_ref,
        ),
        "golden content digest": (
            golden.content_ir_artifact_sha256,
            dataset_document.annotation_sha256,
        ),
        "golden content verification ref": (
            golden.content_verification_artifact_ref,
            dataset_document.verification_attestation.verification_artifact_ref,
        ),
        "golden content verification digest": (
            golden.content_verification_artifact_sha256,
            dataset_document.verification_attestation.verification_artifact_sha256,
        ),
    }
    mismatches = sorted(
        label for label, (actual, expected) in expected_scalar_values.items() if actual != expected
    )
    if mismatches:
        raise ValueError(
            f"document {dataset_document.id} lineage mismatch: " + ", ".join(mismatches)
        )
    if (
        artifacts.document_id != dataset_document.id
        or prediction.document_id != dataset_document.id
    ):
        raise ValueError(f"document {dataset_document.id} input document id mismatch")
    for artifact, expected_ref, expected_sha256, label in (
        (
            artifacts.content_ir,
            golden.content_ir_artifact_ref,
            golden.content_ir_artifact_sha256,
            "gold ContentIR",
        ),
        (
            artifacts.content_verification_attestation,
            golden.content_verification_artifact_ref,
            golden.content_verification_artifact_sha256,
            "content verification attestation",
        ),
        (
            artifacts.gold_plan,
            golden.gold_plan_artifact_ref,
            golden.gold_plan_artifact_sha256,
            "gold plan",
        ),
        (
            artifacts.gold_plan_verification_attestation,
            golden.plan_verification.verification_artifact_ref,
            golden.plan_verification.verification_artifact_sha256,
            "gold plan verification attestation",
        ),
        (
            artifacts.prediction,
            prediction.prediction_artifact_ref,
            prediction.prediction_artifact_sha256,
            "prediction",
        ),
    ):
        _require_artifact_binding(artifact, expected_ref, expected_sha256, label)


def _validate_content_attestation(
    dataset_document: GoldenDatasetDocument,
    artifact: ArtifactPayload,
) -> VerificationArtifactPayload:
    record = _parse_trusted_contract(
        artifact.payload,
        VerificationArtifactPayload,
        f"content verification attestation for {dataset_document.id}",
    )
    expected = {
        "document_id": dataset_document.id,
        "source_sha256": dataset_document.source_sha256,
        "annotation_sha256": dataset_document.annotation_sha256,
        "reviewer_id": dataset_document.verification_attestation.reviewer_id,
        "verified_at": dataset_document.verification_attestation.verified_at,
    }
    _require_record_values(record, expected, f"content attestation for {dataset_document.id}")
    return record


def _validate_plan_attestation(
    dataset_document: GoldenDatasetDocument,
    golden: ModelBGoldenDocumentBinding,
    artifact: ArtifactPayload,
    dataset_manifest_sha256: str,
    content_contract_sha256: str,
    gold_plan_contract_sha256: str,
) -> ModelBPlanVerificationArtifact:
    record = _parse_trusted_contract(
        artifact.payload,
        ModelBPlanVerificationArtifact,
        f"gold plan verification attestation for {dataset_document.id}",
    )
    expected = {
        "document_id": dataset_document.id,
        "lineage_id": dataset_document.lineage_id,
        "source_document_sha256": dataset_document.source_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "content_ir_artifact_ref": golden.content_ir_artifact_ref,
        "content_ir_artifact_sha256": golden.content_ir_artifact_sha256,
        "content_ir_contract_sha256": content_contract_sha256,
        "gold_plan_artifact_ref": golden.gold_plan_artifact_ref,
        "gold_plan_artifact_sha256": golden.gold_plan_artifact_sha256,
        "gold_plan_contract_sha256": gold_plan_contract_sha256,
        "reviewer_id": golden.plan_verification.reviewer_id,
        "verified_at": golden.plan_verification.verified_at,
    }
    _require_record_values(record, expected, f"plan attestation for {dataset_document.id}")
    return record


def _unauthorized_content_changes(
    content: ContentIR,
    gold_plan: HwpDocumentPlan,
    prediction: HwpDocumentPlan,
) -> int:
    """Count changes to the approved content projection and exact ContentIR lineage.

    Layout, styles, official references, and page/column breaks are design choices and
    are intentionally excluded. Invalid plans are handled by the caller as omission of
    every approved content item.
    """
    lineage_changes = sum(
        (
            prediction.content_ir_id != content.id,
            prediction.content_ir_revision != content.revision,
            prediction.content_ir_sha256 != contract_sha256(content),
        )
    )
    return lineage_changes + levenshtein_distance(
        _content_projection(gold_plan),
        _content_projection(prediction),
    )


def _content_projection(plan: HwpDocumentPlan) -> tuple[tuple[str, str], ...]:
    return tuple(
        (item.content_ref, item.render_as)
        for item in plan.flow
        if isinstance(item, ContentPlanItem)
    )


def _parse_prediction(payload: bytes) -> HwpDocumentPlan | None:
    try:
        return _parse_trusted_contract(payload, HwpDocumentPlan, "Model B prediction")
    except (StrictJsonError, ValidationError):
        return None


class StrictJsonError(ValueError):
    """Raised when JSON bytes are not unambiguous finite UTF-8 JSON."""


def _parse_trusted_contract(
    payload: bytes,
    contract_type: type[ContractT],
    label: str,
) -> ContractT:
    _validate_strict_json(payload, label)
    return contract_type.model_validate_json(payload, strict=True)


def _validate_strict_json(payload: bytes, label: str) -> None:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJsonError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Never:
        raise StrictJsonError(f"{label} contains non-finite JSON number: {value}")

    try:
        text = payload.decode("utf-8")
        json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except StrictJsonError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise StrictJsonError(f"{label} is not valid strict UTF-8 JSON") from exc


def _require_record_values(
    record: BaseModel,
    expected: Mapping[str, object],
    label: str,
) -> None:
    mismatches = sorted(
        field_name
        for field_name, expected_value in expected.items()
        if getattr(record, field_name) != expected_value
    )
    if mismatches:
        raise ValueError(f"{label} disagrees with manifest: " + ", ".join(mismatches))


def _require_contract_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} contract digest mismatch")


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


def _require_exact_coverage(
    expected_ids: Sequence[str],
    actual_ids: Sequence[str],
    label: str,
) -> None:
    expected = set(expected_ids)
    actual = set(actual_ids)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} document coverage mismatch: missing={missing}, unexpected={unexpected}"
        )


def _verify_running_evaluator(artifact: ArtifactPayload) -> str:
    running_payload = Path(__file__).read_bytes()
    running_sha256 = hashlib.sha256(running_payload).hexdigest()
    if artifact.payload != running_payload or artifact.sha256 != running_sha256:
        raise ValueError("evaluator artifact does not match running evaluator bytes")
    return running_sha256


def _all_input_artifacts(
    inputs: VerifiedModelBEvaluationInputs,
) -> Iterable[ArtifactPayload]:
    yield inputs.dataset_manifest
    yield inputs.model_bundle_manifest
    yield inputs.model_b_artifact
    yield inputs.golden_manifest
    yield inputs.prediction_manifest
    yield inputs.evaluator_artifact
    for document in inputs.documents:
        yield document.content_ir
        yield document.content_verification_attestation
        yield document.gold_plan
        yield document.gold_plan_verification_attestation
        yield document.prediction


def _golden_binding_artifact_refs(
    document: ModelBGoldenDocumentBinding,
) -> tuple[str, str, str, str]:
    return (
        document.content_ir_artifact_ref,
        document.content_verification_artifact_ref,
        document.gold_plan_artifact_ref,
        document.plan_verification.verification_artifact_ref,
    )


def _reject_duplicates(values: Iterable[str], label: str) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: " + ", ".join(duplicates))
