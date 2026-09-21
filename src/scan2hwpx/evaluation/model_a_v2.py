from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Final, Literal, Never, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scan2hwpx.contracts.models import (
    ContentIR,
    ContentNode,
    ContentRole,
    EvidenceIR,
    FormulaContentNode,
    ImageContentNode,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)

from .artifacts import VerificationArtifactPayload
from .dataset import GoldenDatasetDocument, GoldenDatasetManifest, validate_golden_dataset_manifest
from .model_a import (
    extract_critical_tokens,
    levenshtein_distance,
    reading_order_kendall_tau,
    role_macro_f1,
)
from .model_bundle import Identifier, ModelBundleManifest, Sha256
from .verified import ArtifactPayload, SafeLogicalArtifactRef

MODEL_A_FULL_RECORD_METRIC_SET_ID: Final = "model-a-full-record/v2"
MODEL_A_V2_METRIC_DEFINITION_IDS: Final[dict[str, str]] = {
    "model_a.overall_cer": (
        "cer/nfc-whitespace-sensitive-node-id-and-table-cell-topology-aligned/v2"
    ),
    "model_a.critical_token_error_rate": (
        "critical-token-edit-rate/node-id-and-table-cell-topology-aligned/v2"
    ),
    "model_a.role_macro_f1": (
        "role-macro-f1/gold-id-aligned-missing-kind-extra-sentinels/v2"
    ),
    "model_a.reading_order_exact_match": (
        "reading-order/document-exact-match-invalid-or-id-mismatch-zero/v2"
    ),
    "model_a.reading_order_kendall_tau": (
        "reading-order/normalized-kendall-mean-invalid-or-id-mismatch-zero/v2"
    ),
    "model_a.table_topology_exact_match": (
        "table-topology/gold-table-grid-span-exact-match-missing-or-wrong-kind-zero/v1"
    ),
    "model_a.formula_detection_recall": (
        "formula-detection/gold-formula-id-kind-recall-invalid-or-missing-zero/v1"
    ),
    "model_a.normalized_formula_similarity": (
        "formula-expression/same-format-nfc-strip-whitespace-edit-similarity-gold-mean/v1"
    ),
}
MODEL_A_V2_METRIC_SAMPLE_UNITS: Final[dict[str, str]] = {
    "model_a.overall_cer": "character_slots",
    "model_a.critical_token_error_rate": "critical_token_slots",
    "model_a.role_macro_f1": "content_node_alignment_slots",
    "model_a.reading_order_exact_match": "documents",
    "model_a.reading_order_kendall_tau": "documents",
    "model_a.table_topology_exact_match": "gold_table_nodes",
    "model_a.formula_detection_recall": "gold_formula_nodes",
    "model_a.normalized_formula_similarity": "gold_formula_nodes",
}
VERIFIED_MODEL_A_V2_EVALUATOR_ID: Final = (
    "scan2hwpx-model-a-full-verified-record-evaluator"
)
VERIFIED_MODEL_A_V2_EVALUATOR_VERSION: Final = "2.0.0"
VERIFIED_MODEL_A_V2_EVALUATOR_ARTIFACT_REF: Final = (
    "repo://src/scan2hwpx/evaluation/model_a_v2.py"
)
MAX_MODEL_A_PREDICTION_BYTES: Final = 8 * 1024 * 1024
MAX_TRUSTED_JSON_BYTES: Final = 64 * 1024 * 1024
MAX_JSON_DEPTH: Final = 64
MAX_MODEL_A_PREDICTION_JSON_NODES: Final = 2_000_000
MAX_TRUSTED_JSON_NODES: Final = 2_000_000
MAX_CONTENT_NODES: Final = 100_000
MAX_TABLE_CELLS_PER_DOCUMENT: Final = 100_000
MAX_TABLE_GRID_POSITIONS: Final = 100_000
MAX_MODEL_A_COMPARISON_WORK_UNITS: Final = 10_000_000
_MIN_JSON_INTEGER: Final = -(2**63)
_MAX_JSON_INTEGER: Final = 2**63 - 1

NonEmpty = Annotated[str, Field(min_length=1)]
ContractT = TypeVar("ContractT", bound=BaseModel)
UnimplementedRequiredGateIds = tuple[
    Literal["model_a.high_risk_review_recall"],
    Literal["model_a.confidence_ece"],
]
MODEL_A_V2_UNIMPLEMENTED_REQUIRED_GATE_IDS: Final[UnimplementedRequiredGateIds] = (
    "model_a.high_risk_review_recall",
    "model_a.confidence_ece",
)
PredictionFailure = Literal[
    "none",
    "size_limit",
    "invalid_json",
    "collection_limit",
    "schema_invalid",
    "evidence_integrity_invalid",
    "comparison_limit",
]

_MISSING_PREDICTION_ROLE: Final = "__missing_prediction__"
_PREDICTION_ONLY_ROLE: Final = "__prediction_only__"
_KIND_MISMATCH_ROLE: Final = "__kind_mismatch__"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ModelAV2PredictionBinding(_StrictModel):
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


class VerifiedModelAV2PredictionManifest(_StrictModel):
    """Digest bindings for raw predictions; aggregate measurements are forbidden."""

    schema_version: Literal["model-a-predictions/2.0"]
    model_bundle_id: Identifier
    model_bundle_manifest_ref: SafeLogicalArtifactRef
    model_bundle_manifest_sha256: Sha256
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    metric_set_id: Literal["model-a-full-record/v2"]
    metric_definition_ids: dict[str, str]
    documents: tuple[ModelAV2PredictionBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.metric_definition_ids != MODEL_A_V2_METRIC_DEFINITION_IDS:
            raise ValueError("metric definition ids do not match canonical Model A v2 definitions")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "Model A v2 prediction document ids",
        )
        _reject_duplicates(
            (
                ref.casefold()
                for document in self.documents
                for ref in (
                    document.evidence_artifact_ref,
                    document.prediction_artifact_ref,
                )
            ),
            "Model A v2 prediction artifact refs",
        )
        return self


class VerifiedModelAV2DocumentArtifacts(_StrictModel):
    document_id: Identifier
    evidence: ArtifactPayload
    annotation: ArtifactPayload
    prediction: ArtifactPayload
    verification_attestation: ArtifactPayload

    @model_validator(mode="after")
    def require_distinct_artifacts(self) -> Self:
        _reject_duplicates(
            (
                self.evidence.artifact_ref.casefold(),
                self.annotation.artifact_ref.casefold(),
                self.prediction.artifact_ref.casefold(),
                self.verification_attestation.artifact_ref.casefold(),
            ),
            f"Model A v2 document artifact refs for {self.document_id}",
        )
        return self


class VerifiedModelAV2EvaluationInputs(_StrictModel):
    """Already-resolved immutable bytes consumed by the v2 evaluator."""

    dataset_manifest: ArtifactPayload
    model_bundle_manifest: ArtifactPayload
    model_a_artifact: ArtifactPayload
    prediction_manifest: ArtifactPayload
    evaluator_artifact: ArtifactPayload
    documents: tuple[VerifiedModelAV2DocumentArtifacts, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if self.evaluator_artifact.artifact_ref != VERIFIED_MODEL_A_V2_EVALUATOR_ARTIFACT_REF:
            raise ValueError("unexpected verified Model A v2 evaluator artifact ref")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "Model A v2 evaluation input document ids",
        )
        _reject_duplicates(
            (artifact.artifact_ref.casefold() for artifact in _all_input_artifacts(self)),
            "Model A v2 evaluation input artifact refs",
        )
        return self


class VerifiedModelAV2DocumentEvidence(_StrictModel):
    document_id: Identifier
    source_document_sha256: Sha256
    page_count: int = Field(gt=0)
    evidence_artifact_ref: SafeLogicalArtifactRef
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
    prediction_record_valid: bool
    prediction_failure: PredictionFailure
    prediction_content_ir_id: NonEmpty | None
    prediction_content_ir_revision: int | None = Field(default=None, ge=1)
    prediction_content_ir_sha256: Sha256 | None
    gold_node_count: int = Field(ge=1)
    predicted_node_count: int | None = Field(default=None, ge=1)
    missing_node_count: int = Field(ge=0)
    extra_node_count: int | None = Field(default=None, ge=0)
    kind_mismatch_count: int | None = Field(default=None, ge=0)
    extra_formula_count: int | None = Field(default=None, ge=0)
    evidence_grounding_mismatch_count: int = Field(ge=0)
    image_asset_mismatch_count: int = Field(ge=0)
    text_edit_count: int = Field(ge=0)
    text_character_slot_count: int = Field(ge=0)
    critical_token_edit_count: int = Field(ge=0)
    critical_token_slot_count: int = Field(ge=0)
    reference_roles: tuple[NonEmpty, ...] = Field(min_length=1)
    predicted_roles: tuple[NonEmpty, ...] = Field(min_length=1)
    reading_order_exact: bool
    reading_order_kendall_tau: float = Field(ge=0, le=1)
    table_topology_exact_count: int = Field(ge=0)
    table_topology_sample_count: int = Field(ge=0)
    formula_detection_true_positive_count: int = Field(ge=0)
    formula_sample_count: int = Field(ge=0)
    formula_similarity_sum: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_contributions(self) -> Self:
        if len(self.reference_roles) != len(self.predicted_roles):
            raise ValueError("document role contributions must be aligned")
        content_roles = {role.value for role in ContentRole}
        if any(
            role not in content_roles | {_PREDICTION_ONLY_ROLE}
            for role in self.reference_roles
        ) or any(
            role not in content_roles
            | {_MISSING_PREDICTION_ROLE, _KIND_MISMATCH_ROLE}
            for role in self.predicted_roles
        ):
            raise ValueError("role alignment contains a noncanonical role")
        if self.text_edit_count > self.text_character_slot_count:
            raise ValueError("text edit count exceeds character slots")
        if self.critical_token_edit_count > self.critical_token_slot_count:
            raise ValueError("critical-token edit count exceeds token slots")
        if self.table_topology_exact_count > self.table_topology_sample_count:
            raise ValueError("table exact count exceeds table samples")
        if self.formula_detection_true_positive_count > self.formula_sample_count:
            raise ValueError("formula true positives exceed formula samples")
        if self.formula_similarity_sum > self.formula_sample_count:
            raise ValueError("formula similarity sum exceeds formula samples")
        if self.missing_node_count > self.gold_node_count:
            raise ValueError("missing node count exceeds gold node count")
        if (
            self.table_topology_sample_count + self.formula_sample_count
            > self.gold_node_count
        ):
            raise ValueError("structured sample counts exceed gold node count")
        if self.prediction_record_valid != (self.prediction_failure == "none"):
            raise ValueError("prediction validity and failure reason disagree")
        if self.prediction_record_valid:
            required = (
                self.prediction_content_ir_id,
                self.prediction_content_ir_revision,
                self.prediction_content_ir_sha256,
                self.predicted_node_count,
                self.extra_node_count,
                self.kind_mismatch_count,
                self.extra_formula_count,
            )
            if any(value is None for value in required):
                raise ValueError("valid prediction diagnostics must be complete")
            assert self.predicted_node_count is not None
            assert self.extra_node_count is not None
            assert self.kind_mismatch_count is not None
            assert self.extra_formula_count is not None
            if self.predicted_node_count != (
                self.gold_node_count - self.missing_node_count + self.extra_node_count
            ):
                raise ValueError("valid prediction node counts are inconsistent")
            matched_node_count = self.gold_node_count - self.missing_node_count
            if self.kind_mismatch_count > matched_node_count:
                raise ValueError("kind mismatch count exceeds matched node count")
            if self.extra_formula_count > self.extra_node_count:
                raise ValueError("extra formula count exceeds extra node count")
            if len(self.reference_roles) != self.gold_node_count + self.extra_node_count:
                raise ValueError("valid prediction role slots are inconsistent")
            if (
                self.reference_roles.count(_PREDICTION_ONLY_ROLE)
                != self.extra_node_count
                or self.predicted_roles.count(_MISSING_PREDICTION_ROLE)
                != self.missing_node_count
                or self.predicted_roles.count(_KIND_MISMATCH_ROLE)
                != self.kind_mismatch_count
            ):
                raise ValueError("role alignment diagnostics are inconsistent")
            if (self.missing_node_count or self.extra_node_count) and (
                self.reading_order_exact or self.reading_order_kendall_tau != 0
            ):
                raise ValueError("node-set mismatch cannot receive reading-order credit")
        elif any(
            value is not None
            for value in (
                self.prediction_content_ir_id,
                self.prediction_content_ir_revision,
                self.prediction_content_ir_sha256,
                self.predicted_node_count,
                self.extra_node_count,
                self.kind_mismatch_count,
                self.extra_formula_count,
            )
        ):
            raise ValueError("invalid prediction diagnostics must remain unknown")
        elif (
            self.missing_node_count != self.gold_node_count
            or len(self.reference_roles) != self.gold_node_count
            or any(role != _MISSING_PREDICTION_ROLE for role in self.predicted_roles)
            or self.text_edit_count != self.text_character_slot_count
            or self.critical_token_edit_count != self.critical_token_slot_count
            or self.reading_order_exact
            or self.reading_order_kendall_tau != 0
            or self.table_topology_exact_count != 0
            or self.formula_detection_true_positive_count != 0
            or self.formula_similarity_sum != 0
        ):
            raise ValueError("invalid prediction contributions must be conservative failures")
        return self


class VerifiedModelAV2MeasurementReport(_StrictModel):
    """Eight gate measurements plus explicit blockers outside those definitions."""

    schema_version: Literal["verified-model-a-measurements/2.0"] = (
        "verified-model-a-measurements/2.0"
    )
    evidence_level: Literal["verified_records"] = "verified_records"
    review_identity_assurance: Literal["unsigned_manifest_consistency_only"]
    promotable: Literal[False]
    model_bundle_id: Identifier
    model_bundle_manifest_ref: SafeLogicalArtifactRef
    model_bundle_manifest_sha256: Sha256
    model_a_artifact_ref: SafeLogicalArtifactRef
    model_a_artifact_sha256: Sha256
    dataset_manifest_ref: SafeLogicalArtifactRef
    dataset_manifest_sha256: Sha256
    prediction_manifest_ref: SafeLogicalArtifactRef
    prediction_manifest_sha256: Sha256
    metric_set_id: Literal["model-a-full-record/v2"]
    metric_definition_ids: dict[str, str]
    metric_sample_units: dict[str, str]
    metric_sample_counts: dict[str, int]
    measurements: dict[str, float | None]
    unmeasured_metric_ids: tuple[str, ...]
    structural_hard_failure_document_ids: tuple[Identifier, ...]
    unimplemented_required_gate_ids: UnimplementedRequiredGateIds
    evaluator_id: Literal["scan2hwpx-model-a-full-verified-record-evaluator"]
    evaluator_version: Literal["2.0.0"]
    evaluator_artifact_ref: Literal["repo://src/scan2hwpx/evaluation/model_a_v2.py"]
    evaluator_artifact_sha256: Sha256
    evaluated_document_ids: tuple[Identifier, ...] = Field(min_length=1)
    documents: tuple[VerifiedModelAV2DocumentEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "report document ids",
        )
        expected_structural_failures = tuple(
            document.document_id
            for document in self.documents
            if document.evidence_grounding_mismatch_count > 0
            or document.image_asset_mismatch_count > 0
        )
        if self.structural_hard_failure_document_ids != expected_structural_failures:
            raise ValueError(
                "structural hard-failure ids do not match document diagnostics"
            )
        if (
            self.unimplemented_required_gate_ids
            != MODEL_A_V2_UNIMPLEMENTED_REQUIRED_GATE_IDS
        ):
            raise ValueError("unimplemented required gate ids are not canonical")
        metric_ids = set(MODEL_A_V2_METRIC_DEFINITION_IDS)
        if self.metric_definition_ids != MODEL_A_V2_METRIC_DEFINITION_IDS:
            raise ValueError("report metric definitions are not canonical")
        if self.metric_sample_units != MODEL_A_V2_METRIC_SAMPLE_UNITS:
            raise ValueError("report metric sample units are not canonical")
        if set(self.metric_sample_counts) != metric_ids:
            raise ValueError("report metric sample counts do not match canonical metric ids")
        if set(self.measurements) != metric_ids:
            raise ValueError("report measurements do not match canonical metric ids")
        expected_unmeasured = tuple(
            metric_id
            for metric_id in MODEL_A_V2_METRIC_DEFINITION_IDS
            if self.metric_sample_counts[metric_id] == 0
        )
        if self.unmeasured_metric_ids != expected_unmeasured:
            raise ValueError("unmeasured metric ids do not match zero sample counts")
        for metric_id, sample_count in self.metric_sample_counts.items():
            measurement = self.measurements[metric_id]
            if sample_count == 0:
                if measurement is not None:
                    raise ValueError(f"unmeasured metric must have null value: {metric_id}")
            elif measurement is None or not 0 <= measurement <= 1:
                raise ValueError(f"measured metric must have a 0..1 value: {metric_id}")
        if tuple(document.document_id for document in self.documents) != (
            self.evaluated_document_ids
        ):
            raise ValueError("report document evidence order does not match evaluated ids")
        expected_counts, expected_measurements = _aggregate_document_metrics(self.documents)
        if self.metric_sample_counts != expected_counts:
            raise ValueError("report metric sample counts do not match document evidence")
        if self.measurements != expected_measurements:
            raise ValueError("report measurements do not match document evidence")
        return self


def evaluate_verified_model_a_v2_records(
    inputs: VerifiedModelAV2EvaluationInputs,
) -> VerifiedModelAV2MeasurementReport:
    """Recompute all eight Model A gates from digest-bound raw records.

    Invalid prediction bytes are measured as conservative failures. Trusted dataset,
    bundle, EvidenceIR, annotation, attestation, digest, or evaluator mismatches abort.
    Grounding and image-asset mismatches remain outside the eight metric definitions
    and are exposed as structural hard failures. Missing review-recall and calibration
    gates are named explicitly. The result is always nonpromotable and is not accepted
    by release.py.
    """
    if type(inputs) is not VerifiedModelAV2EvaluationInputs:
        raise TypeError("inputs must be VerifiedModelAV2EvaluationInputs")

    inputs = _revalidate_evaluation_inputs(inputs)
    _validate_runtime_inputs(inputs)
    evaluator_sha256 = _verify_running_evaluator(inputs.evaluator_artifact)
    dataset = _parse_trusted_contract(
        inputs.dataset_manifest.payload,
        GoldenDatasetManifest,
        "Golden dataset manifest",
    )
    validate_golden_dataset_manifest(dataset)
    bundle = _parse_trusted_contract(
        inputs.model_bundle_manifest.payload,
        ModelBundleManifest,
        "model bundle manifest",
    )
    predictions = _parse_trusted_contract(
        inputs.prediction_manifest.payload,
        VerifiedModelAV2PredictionManifest,
        "Model A v2 prediction manifest",
    )
    _validate_top_level_lineage(inputs, bundle, predictions)

    test_documents = tuple(document for document in dataset.documents if document.split == "test")
    if not test_documents:
        raise ValueError("Golden dataset must contain at least one test document")
    test_ids = tuple(document.id for document in test_documents)
    prediction_by_id = {document.document_id: document for document in predictions.documents}
    artifacts_by_id = {document.document_id: document for document in inputs.documents}
    _require_exact_coverage(test_ids, tuple(prediction_by_id), "prediction manifest")
    _require_exact_coverage(test_ids, tuple(artifacts_by_id), "evaluation inputs")

    document_evidence: list[VerifiedModelAV2DocumentEvidence] = []
    for dataset_document in test_documents:
        binding = prediction_by_id[dataset_document.id]
        artifacts = artifacts_by_id[dataset_document.id]
        _validate_document_artifact_bindings(dataset_document, binding, artifacts)

        evidence = _parse_trusted_contract(
            artifacts.evidence.payload,
            EvidenceIR,
            f"EvidenceIR for {dataset_document.id}",
        )
        annotation = _parse_trusted_contract(
            artifacts.annotation.payload,
            ContentIR,
            f"Golden ContentIR for {dataset_document.id}",
        )
        attestation = _validate_attestation(
            dataset_document,
            artifacts.verification_attestation,
        )
        _validate_trusted_document(dataset_document, evidence, annotation)
        prediction, failure = _parse_prediction(artifacts.prediction.payload, evidence)
        document_evidence.append(
            _evaluate_document(
                dataset_document,
                artifacts,
                evidence,
                annotation,
                attestation,
                prediction,
                failure,
            )
        )

    sample_counts, measurements = _aggregate_document_metrics(document_evidence)
    return VerifiedModelAV2MeasurementReport(
        review_identity_assurance="unsigned_manifest_consistency_only",
        promotable=False,
        model_bundle_id=bundle.bundle_id,
        model_bundle_manifest_ref=inputs.model_bundle_manifest.artifact_ref,
        model_bundle_manifest_sha256=inputs.model_bundle_manifest.sha256,
        model_a_artifact_ref=inputs.model_a_artifact.artifact_ref,
        model_a_artifact_sha256=inputs.model_a_artifact.sha256,
        dataset_manifest_ref=inputs.dataset_manifest.artifact_ref,
        dataset_manifest_sha256=inputs.dataset_manifest.sha256,
        prediction_manifest_ref=inputs.prediction_manifest.artifact_ref,
        prediction_manifest_sha256=inputs.prediction_manifest.sha256,
        metric_set_id=MODEL_A_FULL_RECORD_METRIC_SET_ID,
        metric_definition_ids=dict(MODEL_A_V2_METRIC_DEFINITION_IDS),
        metric_sample_units=dict(MODEL_A_V2_METRIC_SAMPLE_UNITS),
        metric_sample_counts=sample_counts,
        measurements=measurements,
        unmeasured_metric_ids=tuple(
            metric_id
            for metric_id in MODEL_A_V2_METRIC_DEFINITION_IDS
            if sample_counts[metric_id] == 0
        ),
        structural_hard_failure_document_ids=tuple(
            document.document_id
            for document in document_evidence
            if document.evidence_grounding_mismatch_count > 0
            or document.image_asset_mismatch_count > 0
        ),
        unimplemented_required_gate_ids=MODEL_A_V2_UNIMPLEMENTED_REQUIRED_GATE_IDS,
        evaluator_id=VERIFIED_MODEL_A_V2_EVALUATOR_ID,
        evaluator_version=VERIFIED_MODEL_A_V2_EVALUATOR_VERSION,
        evaluator_artifact_ref=VERIFIED_MODEL_A_V2_EVALUATOR_ARTIFACT_REF,
        evaluator_artifact_sha256=evaluator_sha256,
        evaluated_document_ids=test_ids,
        documents=tuple(document_evidence),
    )


def normalize_formula_expression(expression: str) -> str:
    """Return the v1 formula comparison form: NFC with Unicode whitespace removed."""
    return "".join(
        character
        for character in unicodedata.normalize("NFC", expression)
        if not character.isspace()
    )


def _validate_top_level_lineage(
    inputs: VerifiedModelAV2EvaluationInputs,
    bundle: ModelBundleManifest,
    predictions: VerifiedModelAV2PredictionManifest,
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
    for owner, artifact_ref, artifact_sha256 in (
        ("model bundle", bundle.dataset_manifest_ref, bundle.dataset_manifest_sha256),
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


def _validate_document_artifact_bindings(
    dataset_document: GoldenDatasetDocument,
    prediction_binding: ModelAV2PredictionBinding,
    artifacts: VerifiedModelAV2DocumentArtifacts,
) -> None:
    if artifacts.document_id != dataset_document.id:
        raise ValueError(f"document {dataset_document.id} input document id mismatch")
    for artifact, expected_ref, expected_sha256, label in (
        (
            artifacts.annotation,
            dataset_document.annotation_artifact_ref,
            dataset_document.annotation_sha256,
            "annotation",
        ),
        (
            artifacts.verification_attestation,
            dataset_document.verification_attestation.verification_artifact_ref,
            dataset_document.verification_attestation.verification_artifact_sha256,
            "verification attestation",
        ),
        (
            artifacts.evidence,
            prediction_binding.evidence_artifact_ref,
            prediction_binding.evidence_artifact_sha256,
            "evidence",
        ),
        (
            artifacts.prediction,
            prediction_binding.prediction_artifact_ref,
            prediction_binding.prediction_artifact_sha256,
            "prediction",
        ),
    ):
        _require_artifact_binding(artifact, expected_ref, expected_sha256, label)


def _validate_trusted_document(
    dataset_document: GoldenDatasetDocument,
    evidence: EvidenceIR,
    annotation: ContentIR,
) -> None:
    if len(evidence.pages) != dataset_document.page_count:
        raise ValueError(
            f"document {dataset_document.id} EvidenceIR page count does not match dataset"
        )
    if evidence.source_document_sha256 != dataset_document.source_sha256:
        raise ValueError(
            f"document {dataset_document.id} EvidenceIR source digest does not match dataset"
        )
    annotation.assert_evidence_integrity(evidence)
    for node in annotation.nodes:
        if isinstance(node, FormulaContentNode) and not normalize_formula_expression(
            node.expression
        ):
            raise ValueError(
                f"document {dataset_document.id} Golden formula normalizes to empty: {node.id}"
            )


def _validate_attestation(
    dataset_document: GoldenDatasetDocument,
    artifact: ArtifactPayload,
) -> VerificationArtifactPayload:
    record = _parse_trusted_contract(
        artifact.payload,
        VerificationArtifactPayload,
        f"verification attestation for {dataset_document.id}",
    )
    expected = {
        "document_id": dataset_document.id,
        "source_sha256": dataset_document.source_sha256,
        "annotation_sha256": dataset_document.annotation_sha256,
        "reviewer_id": dataset_document.verification_attestation.reviewer_id,
        "verified_at": dataset_document.verification_attestation.verified_at,
    }
    mismatches = sorted(
        name for name, expected_value in expected.items() if getattr(record, name) != expected_value
    )
    if mismatches:
        raise ValueError(
            f"verification attestation for {dataset_document.id} disagrees with manifest: "
            + ", ".join(mismatches)
        )
    return record


def _parse_prediction(
    payload: bytes,
    evidence: EvidenceIR,
) -> tuple[ContentIR | None, PredictionFailure]:
    if len(payload) > MAX_MODEL_A_PREDICTION_BYTES:
        return None, "size_limit"
    try:
        raw = _decode_strict_json(
            payload,
            "Model A prediction",
            max_bytes=MAX_MODEL_A_PREDICTION_BYTES,
            max_nodes=MAX_MODEL_A_PREDICTION_JSON_NODES,
        )
    except StrictJsonError:
        return None, "invalid_json"
    try:
        _require_raw_content_collection_bounds(raw, "Model A prediction")
    except ContentCollectionLimitError:
        return None, "collection_limit"
    try:
        prediction = ContentIR.model_validate_json(payload, strict=True)
    except (ValidationError, ValueError, TypeError, RecursionError):
        return None, "schema_invalid"
    try:
        prediction.assert_evidence_integrity(evidence)
    except (ValueError, TypeError):
        return None, "evidence_integrity_invalid"
    return prediction, "none"


def _evaluate_document(
    dataset_document: GoldenDatasetDocument,
    artifacts: VerifiedModelAV2DocumentArtifacts,
    evidence: EvidenceIR,
    annotation: ContentIR,
    attestation: VerificationArtifactPayload,
    prediction: ContentIR | None,
    failure: PredictionFailure,
) -> VerifiedModelAV2DocumentEvidence:
    annotation_by_id = {node.id: node for node in annotation.nodes}
    prediction_by_id = {} if prediction is None else {node.id: node for node in prediction.nodes}
    evidence_grounding_mismatch_count = _evidence_grounding_mismatch_count(
        annotation,
        prediction,
    )
    image_asset_mismatch_count = _image_asset_mismatch_count(annotation, prediction)
    if prediction is not None and _comparison_work_exceeds_limit(
        annotation,
        prediction,
        annotation_by_id,
        prediction_by_id,
    ):
        prediction = None
        prediction_by_id = {}
        failure = "comparison_limit"
    missing_ids = set(annotation_by_id) - set(prediction_by_id)
    extra_ids = set(prediction_by_id) - set(annotation_by_id)
    kind_mismatches = {
        node_id
        for node_id in set(annotation_by_id) & set(prediction_by_id)
        if annotation_by_id[node_id].kind != prediction_by_id[node_id].kind
    }

    reference_texts, predicted_texts = _align_texts(
        annotation,
        prediction,
        annotation_by_id,
        prediction_by_id,
    )
    reference_roles, predicted_roles = _align_roles(
        annotation,
        prediction,
        annotation_by_id,
        prediction_by_id,
        kind_mismatches,
    )
    text_edit_count, text_character_slots = _text_edit_totals(
        reference_texts,
        predicted_texts,
    )
    critical_edit_count, critical_slots = _critical_token_edit_totals(
        reference_texts,
        predicted_texts,
    )
    reading_exact, reading_tau = _reading_order_scores(annotation, prediction)
    table_exact, table_samples = _table_topology_counts(annotation, prediction_by_id)
    formula_true_positives, formula_samples, formula_similarity_sum = _formula_counts(
        annotation,
        prediction_by_id,
    )
    extra_formula_count = (
        None
        if prediction is None
        else sum(
            isinstance(prediction_by_id[node_id], FormulaContentNode)
            for node_id in extra_ids
        )
    )

    return VerifiedModelAV2DocumentEvidence(
        document_id=dataset_document.id,
        source_document_sha256=dataset_document.source_sha256,
        page_count=len(evidence.pages),
        evidence_artifact_ref=artifacts.evidence.artifact_ref,
        evidence_artifact_sha256=artifacts.evidence.sha256,
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        annotation_artifact_ref=artifacts.annotation.artifact_ref,
        annotation_artifact_sha256=artifacts.annotation.sha256,
        annotation_content_ir_id=annotation.id,
        annotation_content_ir_revision=annotation.revision,
        annotation_content_ir_sha256=contract_sha256(annotation),
        verification_artifact_ref=artifacts.verification_attestation.artifact_ref,
        verification_artifact_sha256=artifacts.verification_attestation.sha256,
        reviewer_id=attestation.reviewer_id,
        verified_at=attestation.verified_at,
        prediction_artifact_ref=artifacts.prediction.artifact_ref,
        prediction_artifact_sha256=artifacts.prediction.sha256,
        prediction_record_valid=prediction is not None,
        prediction_failure=failure,
        prediction_content_ir_id=None if prediction is None else prediction.id,
        prediction_content_ir_revision=None if prediction is None else prediction.revision,
        prediction_content_ir_sha256=(
            None if prediction is None else contract_sha256(prediction)
        ),
        gold_node_count=len(annotation.nodes),
        predicted_node_count=None if prediction is None else len(prediction.nodes),
        missing_node_count=len(missing_ids),
        extra_node_count=None if prediction is None else len(extra_ids),
        kind_mismatch_count=None if prediction is None else len(kind_mismatches),
        extra_formula_count=extra_formula_count,
        evidence_grounding_mismatch_count=evidence_grounding_mismatch_count,
        image_asset_mismatch_count=image_asset_mismatch_count,
        text_edit_count=text_edit_count,
        text_character_slot_count=text_character_slots,
        critical_token_edit_count=critical_edit_count,
        critical_token_slot_count=critical_slots,
        reference_roles=reference_roles,
        predicted_roles=predicted_roles,
        reading_order_exact=reading_exact,
        reading_order_kendall_tau=reading_tau,
        table_topology_exact_count=table_exact,
        table_topology_sample_count=table_samples,
        formula_detection_true_positive_count=formula_true_positives,
        formula_sample_count=formula_samples,
        formula_similarity_sum=formula_similarity_sum,
    )


def _align_texts(
    annotation: ContentIR,
    prediction: ContentIR | None,
    annotation_by_id: Mapping[str, ContentNode],
    prediction_by_id: Mapping[str, ContentNode],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reference: list[str] = []
    predicted: list[str] = []
    for node_id in annotation.reading_order:
        expected = annotation_by_id[node_id]
        actual = prediction_by_id.get(node_id)
        if isinstance(expected, TextContentNode):
            reference.append(expected.text)
            predicted.append(actual.text if isinstance(actual, TextContentNode) else "")
        elif isinstance(expected, TableContentNode):
            _append_table_cell_text_alignment(
                expected,
                actual if isinstance(actual, TableContentNode) else None,
                reference,
                predicted,
            )
    if prediction is not None:
        for node_id in prediction.reading_order:
            actual = prediction_by_id[node_id]
            gold_node = annotation_by_id.get(node_id)
            if isinstance(actual, TextContentNode) and not isinstance(
                gold_node, TextContentNode
            ):
                reference.append("")
                predicted.append(actual.text)
            elif isinstance(actual, TableContentNode) and not isinstance(
                gold_node, TableContentNode
            ):
                for cell in sorted(actual.cells, key=_table_cell_key):
                    reference.append("")
                    predicted.append(cell.text)
    return tuple(reference), tuple(predicted)


def _append_table_cell_text_alignment(
    expected: TableContentNode,
    actual: TableContentNode | None,
    reference: list[str],
    predicted: list[str],
) -> None:
    expected_by_key = {_table_cell_key(cell): cell for cell in expected.cells}
    actual_by_key = (
        {} if actual is None else {_table_cell_key(cell): cell for cell in actual.cells}
    )
    for key in sorted(expected_by_key):
        reference.append(expected_by_key[key].text)
        actual_cell = actual_by_key.get(key)
        predicted.append("" if actual_cell is None else actual_cell.text)
    for key in sorted(set(actual_by_key) - set(expected_by_key)):
        reference.append("")
        predicted.append(actual_by_key[key].text)


def _table_cell_key(cell: TableCell) -> tuple[int, int, int, int]:
    return cell.row, cell.column, cell.row_span, cell.column_span


def _evidence_grounding_mismatch_count(
    annotation: ContentIR,
    prediction: ContentIR | None,
) -> int:
    expected = _evidence_grounding_map(annotation)
    actual = {} if prediction is None else _evidence_grounding_map(prediction)
    return sum(
        expected.get(key) != actual.get(key)
        for key in set(expected) | set(actual)
    )


def _evidence_grounding_map(
    content: ContentIR,
) -> dict[tuple[object, ...], tuple[str, ...]]:
    grounding: dict[tuple[object, ...], tuple[str, ...]] = {}
    for node in content.nodes:
        grounding[("node", node.id)] = node.evidence_refs
        if isinstance(node, TableContentNode):
            for cell in node.cells:
                grounding[("table_cell", node.id, *_table_cell_key(cell))] = (
                    cell.evidence_refs
                )
    return grounding


def _image_asset_mismatch_count(
    annotation: ContentIR,
    prediction: ContentIR | None,
) -> int:
    expected = {
        node.id: node.asset_ref
        for node in annotation.nodes
        if isinstance(node, ImageContentNode)
    }
    actual = (
        {}
        if prediction is None
        else {
            node.id: node.asset_ref
            for node in prediction.nodes
            if isinstance(node, ImageContentNode)
        }
    )
    return sum(
        expected.get(node_id) != actual.get(node_id)
        for node_id in set(expected) | set(actual)
    )


def _comparison_work_exceeds_limit(
    annotation: ContentIR,
    prediction: ContentIR,
    annotation_by_id: Mapping[str, ContentNode],
    prediction_by_id: Mapping[str, ContentNode],
) -> bool:
    reference_texts, predicted_texts = _align_texts(
        annotation,
        prediction,
        annotation_by_id,
        prediction_by_id,
    )
    work = 0
    for expected, actual in zip(reference_texts, predicted_texts, strict=True):
        normalized_expected = unicodedata.normalize("NFC", expected)
        normalized_actual = unicodedata.normalize("NFC", actual)
        work += _edit_work_units(normalized_expected, normalized_actual)
        expected_tokens = extract_critical_tokens(normalized_expected)
        actual_tokens = extract_critical_tokens(normalized_actual)
        work += _edit_work_units(expected_tokens, actual_tokens)
        if work > MAX_MODEL_A_COMPARISON_WORK_UNITS:
            return True

    for expected_node in annotation.nodes:
        if not isinstance(expected_node, FormulaContentNode):
            continue
        actual_node = prediction_by_id.get(expected_node.id)
        if (
            not isinstance(actual_node, FormulaContentNode)
            or actual_node.format != expected_node.format
        ):
            continue
        normalized_expected = normalize_formula_expression(expected_node.expression)
        normalized_actual = normalize_formula_expression(actual_node.expression)
        work += _edit_work_units(normalized_expected, normalized_actual)
        if work > MAX_MODEL_A_COMPARISON_WORK_UNITS:
            return True

    if (
        annotation.reading_order != prediction.reading_order
        and set(annotation.reading_order) == set(prediction.reading_order)
    ):
        item_count = len(annotation.reading_order)
        work += item_count * (item_count - 1) // 2
    return work > MAX_MODEL_A_COMPARISON_WORK_UNITS


def _edit_work_units(left: Sequence[object], right: Sequence[object]) -> int:
    return len(left) * len(right) + max(len(left), len(right))


def _align_roles(
    annotation: ContentIR,
    prediction: ContentIR | None,
    annotation_by_id: Mapping[str, ContentNode],
    prediction_by_id: Mapping[str, ContentNode],
    kind_mismatches: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reference: list[str] = []
    predicted: list[str] = []
    for node_id in annotation.reading_order:
        expected = annotation_by_id[node_id]
        actual = prediction_by_id.get(node_id)
        reference.append(expected.role.value)
        if actual is None:
            predicted.append(_MISSING_PREDICTION_ROLE)
        elif node_id in kind_mismatches:
            predicted.append(_KIND_MISMATCH_ROLE)
        else:
            predicted.append(actual.role.value)
    if prediction is not None:
        for node_id in prediction.reading_order:
            if node_id not in annotation_by_id:
                reference.append(_PREDICTION_ONLY_ROLE)
                predicted.append(prediction_by_id[node_id].role.value)
    return tuple(reference), tuple(predicted)


def _text_edit_totals(
    reference: Sequence[str], prediction: Sequence[str]
) -> tuple[int, int]:
    edits = 0
    slots = 0
    for expected, actual in zip(reference, prediction, strict=True):
        normalized_expected = unicodedata.normalize("NFC", expected)
        normalized_actual = unicodedata.normalize("NFC", actual)
        edits += levenshtein_distance(normalized_expected, normalized_actual)
        slots += max(len(normalized_expected), len(normalized_actual))
    return edits, slots


def _critical_token_edit_totals(
    reference: Sequence[str], prediction: Sequence[str]
) -> tuple[int, int]:
    edits = 0
    slots = 0
    for expected, actual in zip(reference, prediction, strict=True):
        expected_tokens = extract_critical_tokens(expected)
        actual_tokens = extract_critical_tokens(actual)
        edits += levenshtein_distance(expected_tokens, actual_tokens)
        slots += max(len(expected_tokens), len(actual_tokens))
    return edits, slots


def _reading_order_scores(
    annotation: ContentIR,
    prediction: ContentIR | None,
) -> tuple[bool, float]:
    if prediction is None:
        return False, 0.0
    exact = annotation.reading_order == prediction.reading_order
    if exact:
        return True, 1.0
    if set(annotation.reading_order) != set(prediction.reading_order):
        return exact, 0.0
    return exact, reading_order_kendall_tau(
        annotation.reading_order,
        prediction.reading_order,
    )


def _table_topology_counts(
    annotation: ContentIR,
    prediction_by_id: Mapping[str, ContentNode],
) -> tuple[int, int]:
    gold_tables = [node for node in annotation.nodes if isinstance(node, TableContentNode)]
    exact = 0
    for expected in gold_tables:
        actual = prediction_by_id.get(expected.id)
        if isinstance(actual, TableContentNode) and _table_topology(
            expected
        ) == _table_topology(actual):
            exact += 1
    return exact, len(gold_tables)


def _table_topology(table: TableContentNode) -> tuple[object, ...]:
    return (
        table.rows,
        table.columns,
        tuple(
            sorted(
                (cell.row, cell.column, cell.row_span, cell.column_span)
                for cell in table.cells
            )
        ),
    )


def _formula_counts(
    annotation: ContentIR,
    prediction_by_id: Mapping[str, ContentNode],
) -> tuple[int, int, float]:
    gold_formulas = [node for node in annotation.nodes if isinstance(node, FormulaContentNode)]
    detected = 0
    similarity_sum = 0.0
    for expected in gold_formulas:
        actual = prediction_by_id.get(expected.id)
        if not isinstance(actual, FormulaContentNode):
            continue
        detected += 1
        if actual.format != expected.format:
            continue
        normalized_expected = normalize_formula_expression(expected.expression)
        normalized_actual = normalize_formula_expression(actual.expression)
        if not normalized_actual:
            continue
        distance = levenshtein_distance(normalized_expected, normalized_actual)
        similarity_sum += 1.0 - distance / max(
            len(normalized_expected),
            len(normalized_actual),
        )
    return detected, len(gold_formulas), similarity_sum


def _aggregate_document_metrics(
    documents: Sequence[VerifiedModelAV2DocumentEvidence],
) -> tuple[dict[str, int], dict[str, float | None]]:
    text_slots = sum(document.text_character_slot_count for document in documents)
    critical_slots = sum(document.critical_token_slot_count for document in documents)
    reference_roles = tuple(role for document in documents for role in document.reference_roles)
    predicted_roles = tuple(role for document in documents for role in document.predicted_roles)
    table_samples = sum(document.table_topology_sample_count for document in documents)
    formula_samples = sum(document.formula_sample_count for document in documents)
    document_count = len(documents)
    sample_counts = {
        "model_a.overall_cer": text_slots,
        "model_a.critical_token_error_rate": critical_slots,
        "model_a.role_macro_f1": len(reference_roles),
        "model_a.reading_order_exact_match": document_count,
        "model_a.reading_order_kendall_tau": document_count,
        "model_a.table_topology_exact_match": table_samples,
        "model_a.formula_detection_recall": formula_samples,
        "model_a.normalized_formula_similarity": formula_samples,
    }
    measurements: dict[str, float | None] = {
        "model_a.overall_cer": (
            None
            if text_slots == 0
            else sum(document.text_edit_count for document in documents) / text_slots
        ),
        "model_a.critical_token_error_rate": (
            None
            if critical_slots == 0
            else sum(document.critical_token_edit_count for document in documents)
            / critical_slots
        ),
        "model_a.role_macro_f1": (
            None if not reference_roles else role_macro_f1(reference_roles, predicted_roles)
        ),
        "model_a.reading_order_exact_match": (
            sum(document.reading_order_exact for document in documents) / document_count
        ),
        "model_a.reading_order_kendall_tau": (
            sum(document.reading_order_kendall_tau for document in documents)
            / document_count
        ),
        "model_a.table_topology_exact_match": (
            None
            if table_samples == 0
            else sum(document.table_topology_exact_count for document in documents)
            / table_samples
        ),
        "model_a.formula_detection_recall": (
            None
            if formula_samples == 0
            else sum(
                document.formula_detection_true_positive_count for document in documents
            )
            / formula_samples
        ),
        "model_a.normalized_formula_similarity": (
            None
            if formula_samples == 0
            else sum(document.formula_similarity_sum for document in documents)
            / formula_samples
        ),
    }
    return sample_counts, measurements


class StrictJsonError(ValueError):
    """Raised when JSON bytes are ambiguous, nonfinite, oversized, or too deep."""


def _parse_trusted_contract(
    payload: bytes,
    contract_type: type[ContractT],
    label: str,
) -> ContractT:
    raw = _decode_strict_json(
        payload,
        label,
        max_bytes=MAX_TRUSTED_JSON_BYTES,
        max_nodes=MAX_TRUSTED_JSON_NODES,
    )
    if contract_type is ContentIR:
        _require_raw_content_collection_bounds(raw, label)
    return contract_type.model_validate_json(payload, strict=True)


def _decode_strict_json(
    payload: bytes,
    label: str,
    *,
    max_bytes: int,
    max_nodes: int,
) -> object:
    if len(payload) > max_bytes:
        raise StrictJsonError(f"{label} exceeds maximum byte size")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJsonError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Never:
        raise StrictJsonError(f"{label} contains non-finite JSON number: {value}")

    def parse_integer(value: str) -> int:
        digits = value.removeprefix("-")
        if len(digits) > 19:
            raise StrictJsonError(f"{label} contains an integer with more than 19 digits")
        parsed = int(value)
        if parsed < _MIN_JSON_INTEGER or parsed > _MAX_JSON_INTEGER:
            raise StrictJsonError(f"{label} contains an integer outside signed 64-bit range")
        return parsed

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise StrictJsonError(f"{label} contains a non-finite JSON float")
        return parsed

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
            parse_int=parse_integer,
            parse_float=parse_float,
        )
    except StrictJsonError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise StrictJsonError(f"{label} is not valid strict UTF-8 JSON") from exc
    _require_bounded_json_structure(decoded, label, max_nodes=max_nodes)
    return decoded


def _require_bounded_json_structure(
    value: object,
    label: str,
    *,
    max_nodes: int,
) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > max_nodes:
            raise StrictJsonError(f"{label} exceeds maximum JSON node count")
        if depth > MAX_JSON_DEPTH:
            raise StrictJsonError(f"{label} exceeds maximum JSON depth")
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)


class ContentCollectionLimitError(ValueError):
    """Raised before ContentIR validation can allocate unbounded collections."""


def _require_raw_content_collection_bounds(value: object, label: str) -> None:
    if not isinstance(value, dict):
        return
    raw_nodes = value.get("nodes")
    raw_order = value.get("reading_order")
    if isinstance(raw_nodes, list) and len(raw_nodes) > MAX_CONTENT_NODES:
        raise ContentCollectionLimitError(f"{label} content node count exceeds limit")
    if isinstance(raw_order, list) and len(raw_order) > MAX_CONTENT_NODES:
        raise ContentCollectionLimitError(f"{label} reading-order count exceeds limit")
    if not isinstance(raw_nodes, list):
        return

    total_cells = 0
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue
        raw_cells = raw_node.get("cells")
        if isinstance(raw_cells, list):
            total_cells += len(raw_cells)
            if total_cells > MAX_TABLE_CELLS_PER_DOCUMENT:
                raise ContentCollectionLimitError(
                    f"{label} total table cell count exceeds limit"
                )
        rows = raw_node.get("rows")
        columns = raw_node.get("columns")
        if (
            isinstance(rows, int)
            and not isinstance(rows, bool)
            and rows > 0
            and isinstance(columns, int)
            and not isinstance(columns, bool)
            and columns > 0
            and rows * columns > MAX_TABLE_GRID_POSITIONS
        ):
            raise ContentCollectionLimitError(
                f"{label} table grid position count exceeds limit"
            )
        if not isinstance(raw_cells, list):
            continue
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict):
                continue
            row_span = raw_cell.get("row_span", 1)
            column_span = raw_cell.get("column_span", 1)
            if (
                isinstance(row_span, int)
                and not isinstance(row_span, bool)
                and row_span > 0
                and isinstance(column_span, int)
                and not isinstance(column_span, bool)
                and column_span > 0
                and row_span * column_span > MAX_TABLE_GRID_POSITIONS
            ):
                raise ContentCollectionLimitError(
                    f"{label} table cell span position count exceeds limit"
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


def _validate_runtime_inputs(inputs: VerifiedModelAV2EvaluationInputs) -> None:
    if inputs.evaluator_artifact.artifact_ref != VERIFIED_MODEL_A_V2_EVALUATOR_ARTIFACT_REF:
        raise ValueError("unexpected verified Model A v2 evaluator artifact ref")
    _reject_duplicates(
        (document.document_id.casefold() for document in inputs.documents),
        "Model A v2 evaluation input document ids",
    )
    artifacts = tuple(_all_input_artifacts(inputs))
    _reject_duplicates(
        (artifact.artifact_ref.casefold() for artifact in artifacts),
        "Model A v2 evaluation input artifact refs",
    )
    for artifact in artifacts:
        actual_sha256 = hashlib.sha256(artifact.payload).hexdigest()
        if actual_sha256 != artifact.sha256:
            raise ValueError(f"artifact payload digest mismatch: {artifact.artifact_ref}")


def _revalidate_evaluation_inputs(
    inputs: VerifiedModelAV2EvaluationInputs,
) -> VerifiedModelAV2EvaluationInputs:
    try:
        restored = VerifiedModelAV2EvaluationInputs.model_validate(
            inputs.model_dump(mode="python", warnings="error"),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"inputs are not valid strict evaluation inputs: {exc}") from exc
    if restored != inputs:
        raise ValueError("inputs are not canonical strict evaluation inputs")
    return restored


def _all_input_artifacts(
    inputs: VerifiedModelAV2EvaluationInputs,
) -> Iterable[ArtifactPayload]:
    yield inputs.dataset_manifest
    yield inputs.model_bundle_manifest
    yield inputs.model_a_artifact
    yield inputs.prediction_manifest
    yield inputs.evaluator_artifact
    for document in inputs.documents:
        yield document.evidence
        yield document.annotation
        yield document.prediction
        yield document.verification_attestation


def _reject_duplicates(values: Iterable[str], label: str) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: " + ", ".join(duplicates))
