from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self, TypeGuard, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scan2hwpx.contracts.models import (
    ContentIR,
    ContentNode,
    EvidenceIR,
    FormulaContentNode,
    ImageContentNode,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.evaluation.strict_json import require_strict_json_bytes
from scan2hwpx.model_a.inference import (
    DEFAULT_MAX_RESPONSE_BYTES,
    MAX_REQUEST_BYTES,
    BuiltModelAInferenceResult,
    ModelAInferenceRequest,
    ModelAModelArtifact,
    ModelAPageImage,
    assemble_model_a_document,
    build_model_a_inference_request,
    validate_model_a_inference_response,
)

from .model_a_v2 import (
    MAX_CONTENT_NODES,
    MAX_JSON_DEPTH,
    MAX_MODEL_A_COMPARISON_WORK_UNITS,
    MAX_TABLE_CELLS_PER_DOCUMENT,
    MAX_TABLE_GRID_POSITIONS,
    MAX_TRUSTED_JSON_BYTES,
    MAX_TRUSTED_JSON_NODES,
    MODEL_A_FULL_RECORD_METRIC_SET_ID,
    MODEL_A_V2_METRIC_DEFINITION_IDS,
    MODEL_A_V2_METRIC_SAMPLE_UNITS,
    VerifiedModelAV2DocumentArtifacts,
    VerifiedModelAV2DocumentEvidence,
    VerifiedModelAV2EvaluationInputs,
    VerifiedModelAV2MeasurementReport,
    evaluate_verified_model_a_v2_records,
)
from .model_bundle import Identifier, ModelBundleManifest, Sha256
from .verified import ArtifactPayload, SafeLogicalArtifactRef

MODEL_A_V3_METRIC_SET_ID: Final = "model-a-full-record/v3"
MODEL_A_V3_METRIC_DEFINITION_IDS: Final[dict[str, str]] = {
    **MODEL_A_V2_METRIC_DEFINITION_IDS,
    "model_a.high_risk_review_recall": (
        "high-risk-gold-surface-mismatch-validator-block-or-exact-node-review-recall/v1"
    ),
    "model_a.confidence_ece": (
        "top-level-node-confidence-gold-id-exact-correctness-10-equal-width-bins/v1"
    ),
}
MODEL_A_V3_METRIC_SAMPLE_UNITS: Final[dict[str, str]] = {
    **MODEL_A_V2_METRIC_SAMPLE_UNITS,
    "model_a.high_risk_review_recall": "high_risk_mismatched_surfaces",
    "model_a.confidence_ece": "schema_valid_predicted_top_level_nodes",
}
_CANONICAL_MODEL_A_V2_METRIC_DEFINITION_ITEMS: Final = tuple(
    MODEL_A_V2_METRIC_DEFINITION_IDS.items()
)
_CANONICAL_MODEL_A_V2_METRIC_SAMPLE_UNIT_ITEMS: Final = tuple(
    MODEL_A_V2_METRIC_SAMPLE_UNITS.items()
)
_CANONICAL_MODEL_A_V3_METRIC_DEFINITION_ITEMS: Final = tuple(
    MODEL_A_V3_METRIC_DEFINITION_IDS.items()
)
_CANONICAL_MODEL_A_V3_METRIC_SAMPLE_UNIT_ITEMS: Final = tuple(
    MODEL_A_V3_METRIC_SAMPLE_UNITS.items()
)
MODEL_A_RISK_SURFACE_DEFINITION_ID: Final = "model-a-gold-risk-surfaces/1.0"
VERIFIED_MODEL_A_V3_EVALUATOR_ID: Final = "scan2hwpx-model-a-full-verified-execution-evaluator"
VERIFIED_MODEL_A_V3_EVALUATOR_VERSION: Final = "3.0.0"
VERIFIED_MODEL_A_V3_EVALUATOR_ARTIFACT_REF: Final = "repo://src/scan2hwpx/evaluation/model_a_v3.py"

MAX_MODEL_A_V3_DOCUMENTS: Final = 10_000
MAX_MODEL_A_V3_PAGES: Final = 5_000
MAX_MODEL_A_V3_RISK_SURFACES: Final = 500_000
# Measured v4 requests for 485 pages are about 491 MiB and the complete typical
# corpus is about 1 GiB. This in-memory evaluator is capped at 1 GiB for an 8 GiB
# host; larger corpora require a future streaming evaluator rather than a larger cap.
MAX_MODEL_A_V3_TOTAL_ARTIFACT_BYTES: Final = 1024 * 1024 * 1024
MAX_MODEL_A_V3_RESULT_BYTES: Final = 64 * 1024 * 1024
MAX_MODEL_A_V3_ASSEMBLY_BYTES: Final = 64 * 1024 * 1024
MAX_MODEL_A_V3_CAPTURED_RESPONSE_BYTES: Final = DEFAULT_MAX_RESPONSE_BYTES + 1
_MEASURED_V4_PAGE_IMAGE_BYTES: Final = 368_458_051
_MEASURED_V4_IMAGE_BASE64_BYTES: Final = 4 * ((_MEASURED_V4_PAGE_IMAGE_BYTES + 2) // 3)

NonEmpty = Annotated[str, Field(min_length=1, max_length=4_096)]
ContractT = TypeVar("ContractT", bound=BaseModel)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ModelARiskSurfaceKind(StrEnum):
    PAGE_OUTPUT_VALID = "page_output_valid"
    NODE_PRESENCE = "node_presence"
    NODE_KIND = "node_kind"
    NODE_ROLE = "node_role"
    NODE_EVIDENCE_REFS = "node_evidence_refs"
    TEXT_VALUE = "text_value"
    TABLE_TOPOLOGY = "table_topology"
    TABLE_CELL_TEXT = "table_cell_text"
    TABLE_CELL_EVIDENCE_REFS = "table_cell_evidence_refs"
    FORMULA_EXPRESSION = "formula_expression"
    FORMULA_FORMAT = "formula_format"
    IMAGE_ASSET_REF = "image_asset_ref"
    READING_ORDER = "reading_order"
    EXTRA_NODES = "extra_nodes"


_NODE_SURFACE_KINDS: Final = frozenset(
    {
        ModelARiskSurfaceKind.NODE_PRESENCE,
        ModelARiskSurfaceKind.NODE_KIND,
        ModelARiskSurfaceKind.NODE_ROLE,
        ModelARiskSurfaceKind.NODE_EVIDENCE_REFS,
        ModelARiskSurfaceKind.TEXT_VALUE,
        ModelARiskSurfaceKind.TABLE_TOPOLOGY,
        ModelARiskSurfaceKind.TABLE_CELL_TEXT,
        ModelARiskSurfaceKind.TABLE_CELL_EVIDENCE_REFS,
        ModelARiskSurfaceKind.FORMULA_EXPRESSION,
        ModelARiskSurfaceKind.FORMULA_FORMAT,
        ModelARiskSurfaceKind.IMAGE_ASSET_REF,
    }
)
_CELL_SURFACE_KINDS: Final = frozenset(
    {
        ModelARiskSurfaceKind.TABLE_CELL_TEXT,
        ModelARiskSurfaceKind.TABLE_CELL_EVIDENCE_REFS,
    }
)


class ModelARiskSurface(_StrictModel):
    """A deterministic evaluation unit; it contains no inferred risk label."""

    surface_id: Identifier
    kind: ModelARiskSurfaceKind
    page_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    node_id: NonEmpty | None = None
    table_cell: tuple[int, int, int, int] | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if len(self.page_ids) != len(set(self.page_ids)):
            raise ValueError("risk surface page ids must be unique")
        if (self.kind in _NODE_SURFACE_KINDS) != (self.node_id is not None):
            raise ValueError("risk surface node binding does not match its kind")
        if (self.kind in _CELL_SURFACE_KINDS) != (self.table_cell is not None):
            raise ValueError("risk surface table-cell binding does not match its kind")
        if self.table_cell is not None and any(value < 0 for value in self.table_cell):
            raise ValueError("risk surface table-cell coordinates must be non-negative")
        expected = _risk_surface_id(
            self.kind,
            self.page_ids,
            self.node_id,
            self.table_cell,
        )
        if self.surface_id != expected:
            raise ValueError("risk surface id is not canonical")
        return self


class ModelAHighRiskSurfaceLabel(_StrictModel):
    surface: ModelARiskSurface
    high_risk: bool


class ModelAHighRiskAnnotation(_StrictModel):
    """Externally supplied labels bound to every mechanically generated surface."""

    schema_version: Literal["model-a-high-risk-annotations/1.0"] = (
        "model-a-high-risk-annotations/1.0"
    )
    document_id: Identifier
    evidence_ir_id: NonEmpty
    evidence_ir_sha256: Sha256
    gold_content_ir_id: NonEmpty
    gold_content_ir_revision: int = Field(ge=1)
    gold_content_ir_sha256: Sha256
    surface_definition_id: Literal["model-a-gold-risk-surfaces/1.0"] = (
        MODEL_A_RISK_SURFACE_DEFINITION_ID
    )
    label_assurance: Literal["externally_supplied_not_authenticated"] = (
        "externally_supplied_not_authenticated"
    )
    labels: tuple[ModelAHighRiskSurfaceLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_surfaces(self) -> Self:
        _reject_duplicates(
            (label.surface.surface_id for label in self.labels),
            "high-risk annotation surface ids",
        )
        return self


class ModelARawResponsePayload(_StrictModel):
    """A digest-bound model response; empty bytes are a legitimate failed response."""

    artifact_ref: SafeLogicalArtifactRef
    sha256: Sha256
    payload: bytes = Field(repr=False)

    @model_validator(mode="after")
    def verify_digest(self) -> Self:
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError(f"artifact payload digest mismatch: {self.artifact_ref}")
        return self


class ModelAPageExecutionBinding(_StrictModel):
    page_id: NonEmpty
    page_no: int = Field(ge=1)
    response_byte_limit: int = Field(ge=1, le=DEFAULT_MAX_RESPONSE_BYTES)
    request_artifact_ref: SafeLogicalArtifactRef
    request_artifact_sha256: Sha256
    raw_response_artifact_ref: SafeLogicalArtifactRef
    raw_response_artifact_sha256: Sha256
    validation_result_artifact_ref: SafeLogicalArtifactRef
    validation_result_artifact_sha256: Sha256

    @model_validator(mode="after")
    def require_distinct_refs(self) -> Self:
        _reject_duplicates(
            (
                self.request_artifact_ref.casefold(),
                self.raw_response_artifact_ref.casefold(),
                self.validation_result_artifact_ref.casefold(),
            ),
            f"page execution artifact refs for {self.page_id}",
        )
        return self


class ModelADocumentExecutionManifest(_StrictModel):
    schema_version: Literal["model-a-page-executions/1.0"] = "model-a-page-executions/1.0"
    execution_status: Literal["complete"] = "complete"
    document_id: Identifier
    evidence_artifact_ref: SafeLogicalArtifactRef
    evidence_artifact_sha256: Sha256
    evidence_ir_id: NonEmpty
    evidence_ir_sha256: Sha256
    gold_annotation_artifact_ref: SafeLogicalArtifactRef
    gold_annotation_artifact_sha256: Sha256
    prediction_artifact_ref: SafeLogicalArtifactRef
    prediction_artifact_sha256: Sha256
    high_risk_annotation_artifact_ref: SafeLogicalArtifactRef
    high_risk_annotation_artifact_sha256: Sha256
    model: ModelAModelArtifact
    pages: tuple[ModelAPageExecutionBinding, ...] = Field(min_length=1)
    assembly_artifact_ref: SafeLogicalArtifactRef
    assembly_artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_canonical_manifest(self) -> Self:
        keys = tuple((page.page_no, page.page_id) for page in self.pages)
        if keys != tuple(sorted(keys)):
            raise ValueError("execution manifest pages must be in canonical page order")
        _reject_duplicates((page.page_id for page in self.pages), "execution page ids")
        _reject_duplicates((str(page.page_no) for page in self.pages), "execution page numbers")
        refs = (
            self.evidence_artifact_ref,
            self.gold_annotation_artifact_ref,
            self.prediction_artifact_ref,
            self.high_risk_annotation_artifact_ref,
            self.assembly_artifact_ref,
            *(
                ref
                for page in self.pages
                for ref in (
                    page.request_artifact_ref,
                    page.raw_response_artifact_ref,
                    page.validation_result_artifact_ref,
                )
            ),
        )
        _reject_duplicates((ref.casefold() for ref in refs), "execution manifest refs")
        return self


class ModelAPageExecutionArtifacts(_StrictModel):
    page_id: NonEmpty
    page_no: int = Field(ge=1)
    request: ArtifactPayload
    raw_response: ModelARawResponsePayload
    validation_result: ArtifactPayload

    @model_validator(mode="after")
    def require_distinct_refs(self) -> Self:
        _reject_duplicates(
            (
                self.request.artifact_ref.casefold(),
                self.raw_response.artifact_ref.casefold(),
                self.validation_result.artifact_ref.casefold(),
            ),
            f"resolved page execution refs for {self.page_id}",
        )
        return self


class VerifiedModelAV3DocumentArtifacts(_StrictModel):
    document_id: Identifier
    execution_manifest: ArtifactPayload
    high_risk_annotation: ArtifactPayload
    assembly: ArtifactPayload
    pages: tuple[ModelAPageExecutionArtifacts, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_resolved_artifacts(self) -> Self:
        _reject_duplicates(
            (page.page_id for page in self.pages),
            f"resolved execution page ids for {self.document_id}",
        )
        _reject_duplicates(
            (str(page.page_no) for page in self.pages),
            f"resolved execution page numbers for {self.document_id}",
        )
        refs = (
            self.execution_manifest.artifact_ref,
            self.high_risk_annotation.artifact_ref,
            self.assembly.artifact_ref,
            *(
                ref
                for page in self.pages
                for ref in (
                    page.request.artifact_ref,
                    page.raw_response.artifact_ref,
                    page.validation_result.artifact_ref,
                )
            ),
        )
        _reject_duplicates(
            (ref.casefold() for ref in refs),
            f"resolved v3 artifact refs for {self.document_id}",
        )
        return self


class VerifiedModelAV3EvaluationInputs(_StrictModel):
    """v2 raw records plus complete page-level Model A execution evidence."""

    schema_version: Literal["verified-model-a-evaluation-inputs/3.0"] = (
        "verified-model-a-evaluation-inputs/3.0"
    )
    predecessor_metric_set_id: Literal["model-a-full-record/v2"] = MODEL_A_FULL_RECORD_METRIC_SET_ID
    v2_inputs: VerifiedModelAV2EvaluationInputs
    evaluator_artifact: ArtifactPayload
    documents: tuple[VerifiedModelAV3DocumentArtifacts, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if self.evaluator_artifact.artifact_ref != VERIFIED_MODEL_A_V3_EVALUATOR_ARTIFACT_REF:
            raise ValueError("unexpected verified Model A v3 evaluator artifact ref")
        if len(self.documents) > MAX_MODEL_A_V3_DOCUMENTS:
            raise ValueError("v3 evaluation document count exceeds limit")
        _reject_duplicates(
            (document.document_id.casefold() for document in self.documents),
            "Model A v3 evaluation document ids",
        )
        return self


class ModelACalibrationBin(_StrictModel):
    index: int = Field(ge=0, le=9)
    lower_bound: float = Field(ge=0, le=1)
    upper_bound: float = Field(ge=0, le=1)
    upper_inclusive: bool
    sample_count: int = Field(ge=0)
    confidence_sum: float = Field(ge=0)
    correct_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_bin(self) -> Self:
        if self.lower_bound != self.index / 10:
            raise ValueError("calibration bin lower bound is not canonical")
        if self.upper_bound != (self.index + 1) / 10:
            raise ValueError("calibration bin upper bound is not canonical")
        if self.upper_inclusive != (self.index == 9):
            raise ValueError("only the final calibration bin includes its upper bound")
        if self.correct_count > self.sample_count:
            raise ValueError("calibration correct count exceeds sample count")
        if self.confidence_sum > self.sample_count:
            raise ValueError("calibration confidence sum exceeds sample count")
        return self


class ModelAHighRiskMismatchEvidence(_StrictModel):
    surface_id: Identifier
    kind: ModelARiskSurfaceKind
    page_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    node_id: NonEmpty | None
    table_cell: tuple[int, int, int, int] | None = None
    route: Literal["validator_block", "node_needs_review"] | None

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if len(self.page_ids) != len(set(self.page_ids)):
            raise ValueError("mismatch evidence page ids must be unique")
        if (self.kind in _NODE_SURFACE_KINDS) != (self.node_id is not None):
            raise ValueError("mismatch evidence node binding does not match its kind")
        if (self.kind in _CELL_SURFACE_KINDS) != (self.table_cell is not None):
            raise ValueError("mismatch evidence table-cell binding does not match its kind")
        if self.table_cell is not None and any(value < 0 for value in self.table_cell):
            raise ValueError("mismatch evidence table-cell coordinates must be non-negative")
        if self.surface_id != _risk_surface_id(
            self.kind,
            self.page_ids,
            self.node_id,
            self.table_cell,
        ):
            raise ValueError("mismatch evidence surface id is not canonical")
        return self


class VerifiedModelAV3DocumentEvidence(_StrictModel):
    document_id: Identifier
    execution_manifest_artifact_ref: SafeLogicalArtifactRef
    execution_manifest_artifact_sha256: Sha256
    high_risk_annotation_artifact_ref: SafeLogicalArtifactRef
    high_risk_annotation_artifact_sha256: Sha256
    model: ModelAModelArtifact
    page_count: int = Field(ge=1)
    blocked_page_ids: tuple[NonEmpty, ...]
    assembly_outcome: Literal["blocked", "valid_unverified"]
    risk_surface_count: int = Field(ge=1)
    high_risk_surface_count: int = Field(ge=0)
    high_risk_mismatch_count: int = Field(ge=0)
    high_risk_routed_count: int = Field(ge=0)
    high_risk_mismatches: tuple[ModelAHighRiskMismatchEvidence, ...]
    missing_gold_node_count: int = Field(ge=0)
    extra_predicted_node_count: int = Field(ge=0)
    calibration_bins: tuple[ModelACalibrationBin, ...]

    @model_validator(mode="after")
    def validate_contributions(self) -> Self:
        if len(self.blocked_page_ids) != len(set(self.blocked_page_ids)):
            raise ValueError("blocked page ids must be unique")
        if len(self.calibration_bins) != 10:
            raise ValueError("document calibration must contain exactly ten bins")
        if tuple(item.index for item in self.calibration_bins) != tuple(range(10)):
            raise ValueError("document calibration bins are not canonical")
        if self.high_risk_surface_count > self.risk_surface_count:
            raise ValueError("high-risk surface count exceeds all surfaces")
        if self.high_risk_mismatch_count > self.high_risk_surface_count:
            raise ValueError("high-risk mismatch count exceeds high-risk surfaces")
        if self.high_risk_routed_count > self.high_risk_mismatch_count:
            raise ValueError("routed count exceeds high-risk mismatch count")
        if len(self.high_risk_mismatches) != self.high_risk_mismatch_count:
            raise ValueError("high-risk mismatch evidence count is inconsistent")
        if (
            sum(item.route is not None for item in self.high_risk_mismatches)
            != self.high_risk_routed_count
        ):
            raise ValueError("high-risk routed count is inconsistent")
        return self


class VerifiedModelAV3MeasurementReport(_StrictModel):
    """Ten raw-recomputed metrics; this migration report is never promotable."""

    schema_version: Literal["verified-model-a-measurements/3.0"] = (
        "verified-model-a-measurements/3.0"
    )
    evidence_level: Literal["verified_records_plus_replayed_unverified_execution"] = (
        "verified_records_plus_replayed_unverified_execution"
    )
    promotable: Literal[False]
    migration_policy: Literal["preserve_v2_metrics_add_raw_review_recall_and_ece"] = (
        "preserve_v2_metrics_add_raw_review_recall_and_ece"
    )
    predecessor_metric_set_id: Literal["model-a-full-record/v2"]
    predecessor_report_sha256: Sha256
    base_v2_report: VerifiedModelAV2MeasurementReport
    metric_set_id: Literal["model-a-full-record/v3"]
    metric_definition_ids: dict[str, str]
    metric_sample_units: dict[str, str]
    metric_sample_counts: dict[str, int]
    measurements: dict[str, float | None]
    unmeasured_metric_ids: tuple[str, ...]
    calibration_bins: tuple[ModelACalibrationBin, ...]
    evaluator_id: Literal["scan2hwpx-model-a-full-verified-execution-evaluator"]
    evaluator_version: Literal["3.0.0"]
    evaluator_artifact_ref: Literal["repo://src/scan2hwpx/evaluation/model_a_v3.py"]
    evaluator_artifact_sha256: Sha256
    evaluated_document_ids: tuple[Identifier, ...] = Field(min_length=1)
    documents: tuple[VerifiedModelAV3DocumentEvidence, ...] = Field(min_length=1)
    remaining_promotion_blockers: tuple[
        Literal[
            "risk_labels_not_authenticated",
            "review_identity_unsigned",
            "release_gate_not_integrated",
            "typed_cross_page_relations_unmeasured",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.promotable is not False:
            raise ValueError("Model A v3 evaluation report must remain nonpromotable")
        if self.predecessor_metric_set_id != MODEL_A_FULL_RECORD_METRIC_SET_ID:
            raise ValueError("v3 predecessor metric set is not canonical")
        if self.predecessor_report_sha256 != _sha256(
            _canonical_json_bytes(self.base_v2_report.model_dump(mode="json"))
        ):
            raise ValueError("v3 predecessor report digest mismatch")
        if self.metric_definition_ids != MODEL_A_V3_METRIC_DEFINITION_IDS:
            raise ValueError("v3 metric definitions are not canonical")
        if self.metric_sample_units != MODEL_A_V3_METRIC_SAMPLE_UNITS:
            raise ValueError("v3 metric sample units are not canonical")
        metric_ids = set(MODEL_A_V3_METRIC_DEFINITION_IDS)
        if set(self.metric_sample_counts) != metric_ids:
            raise ValueError("v3 metric sample counts do not match canonical metrics")
        if set(self.measurements) != metric_ids:
            raise ValueError("v3 measurements do not match canonical metrics")
        for metric_id in MODEL_A_V2_METRIC_DEFINITION_IDS:
            if (
                self.metric_sample_counts[metric_id]
                != self.base_v2_report.metric_sample_counts[metric_id]
                or self.measurements[metric_id] != self.base_v2_report.measurements[metric_id]
            ):
                raise ValueError("v3 report changed a preserved v2 metric")
        expected_counts, expected_measurements, expected_bins = _aggregate_v3_metrics(
            self.base_v2_report,
            self.documents,
        )
        if self.metric_sample_counts != expected_counts:
            raise ValueError("v3 metric sample counts do not match document evidence")
        if self.measurements != expected_measurements:
            raise ValueError("v3 measurements do not match document evidence")
        if self.calibration_bins != expected_bins:
            raise ValueError("v3 calibration bins do not match document evidence")
        expected_unmeasured = tuple(
            metric_id
            for metric_id in MODEL_A_V3_METRIC_DEFINITION_IDS
            if expected_counts[metric_id] == 0
        )
        if self.unmeasured_metric_ids != expected_unmeasured:
            raise ValueError("v3 unmeasured metrics do not match zero denominators")
        if tuple(document.document_id for document in self.documents) != (
            self.evaluated_document_ids
        ):
            raise ValueError("v3 document order does not match evaluated document ids")
        if self.remaining_promotion_blockers != (
            "risk_labels_not_authenticated",
            "review_identity_unsigned",
            "release_gate_not_integrated",
            "typed_cross_page_relations_unmeasured",
        ):
            raise ValueError("v3 promotion blockers are not canonical")
        return self


def canonical_model_a_risk_surfaces(
    evidence: EvidenceIR,
    gold: ContentIR,
) -> tuple[ModelARiskSurface, ...]:
    """Enumerate the exact label surface without assigning any risk values."""

    evidence = _revalidate_exact_model(evidence, EvidenceIR, "risk EvidenceIR")
    gold = _revalidate_exact_model(gold, ContentIR, "risk gold ContentIR")
    gold.assert_evidence_integrity(evidence)
    return _canonical_risk_surfaces(evidence, gold)


def evaluate_verified_model_a_v3_records(
    inputs: VerifiedModelAV3EvaluationInputs,
) -> VerifiedModelAV3MeasurementReport:
    """Replay raw page executions and add review-recall and calibration to v2.

    The supplied high-risk booleans are never derived or repaired. Trusted artifact,
    execution, lineage, or coverage disagreement aborts evaluation. A failed model
    output is treated as routed only when replaying the real inference validator
    produces a blocked page result. The report remains nonpromotable because label and
    reviewer identity are not authenticated here and release policy is not integrated.
    """

    _require_unmutated_runtime_metric_tables()
    _preflight_total_artifact_bytes(inputs)
    inputs = _revalidate_v3_inputs(inputs)
    _validate_runtime_inputs(inputs)
    evaluator_sha256 = _verify_running_evaluator(inputs.evaluator_artifact)
    base_report = evaluate_verified_model_a_v2_records(inputs.v2_inputs)
    bundle = _parse_contract(
        inputs.v2_inputs.model_bundle_manifest.payload,
        ModelBundleManifest,
        "Model A v3 model bundle",
    )
    document_ids = base_report.evaluated_document_ids
    v2_artifacts_by_id = {document.document_id: document for document in inputs.v2_inputs.documents}
    base_evidence_by_id = {document.document_id: document for document in base_report.documents}
    v3_artifacts_by_id = {document.document_id: document for document in inputs.documents}
    _require_exact_coverage(document_ids, tuple(v3_artifacts_by_id), "v3 execution inputs")

    document_evidence: list[VerifiedModelAV3DocumentEvidence] = []
    work_units = 0
    for document_id in document_ids:
        v2_artifacts = v2_artifacts_by_id[document_id]
        base_evidence = base_evidence_by_id[document_id]
        evidence = _parse_contract(
            v2_artifacts.evidence.payload,
            EvidenceIR,
            f"v3 EvidenceIR for {document_id}",
        )
        gold = _parse_contract(
            v2_artifacts.annotation.payload,
            ContentIR,
            f"v3 gold ContentIR for {document_id}",
        )
        gold.assert_evidence_integrity(evidence)
        contribution, consumed = _evaluate_v3_document(
            v2_artifacts,
            base_evidence,
            v3_artifacts_by_id[document_id],
            evidence,
            gold,
            bundle,
        )
        work_units += consumed
        if work_units > MAX_MODEL_A_COMPARISON_WORK_UNITS:
            raise ValueError("Model A v3 comparison work exceeds limit")
        document_evidence.append(contribution)

    counts, measurements, bins = _aggregate_v3_metrics(
        base_report,
        tuple(document_evidence),
    )
    return VerifiedModelAV3MeasurementReport(
        promotable=False,
        predecessor_metric_set_id=MODEL_A_FULL_RECORD_METRIC_SET_ID,
        predecessor_report_sha256=_sha256(
            _canonical_json_bytes(base_report.model_dump(mode="json"))
        ),
        base_v2_report=base_report,
        metric_set_id=MODEL_A_V3_METRIC_SET_ID,
        metric_definition_ids=dict(MODEL_A_V3_METRIC_DEFINITION_IDS),
        metric_sample_units=dict(MODEL_A_V3_METRIC_SAMPLE_UNITS),
        metric_sample_counts=counts,
        measurements=measurements,
        unmeasured_metric_ids=tuple(
            metric_id for metric_id in MODEL_A_V3_METRIC_DEFINITION_IDS if counts[metric_id] == 0
        ),
        calibration_bins=bins,
        evaluator_id=VERIFIED_MODEL_A_V3_EVALUATOR_ID,
        evaluator_version=VERIFIED_MODEL_A_V3_EVALUATOR_VERSION,
        evaluator_artifact_ref=VERIFIED_MODEL_A_V3_EVALUATOR_ARTIFACT_REF,
        evaluator_artifact_sha256=evaluator_sha256,
        evaluated_document_ids=document_ids,
        documents=tuple(document_evidence),
        remaining_promotion_blockers=(
            "risk_labels_not_authenticated",
            "review_identity_unsigned",
            "release_gate_not_integrated",
            "typed_cross_page_relations_unmeasured",
        ),
    )


def canonical_model_a_v3_report_bytes(
    report: VerifiedModelAV3MeasurementReport,
) -> bytes:
    """Return deterministic report bytes after strict anti-forgery revalidation."""

    report = _revalidate_exact_model(
        report,
        VerifiedModelAV3MeasurementReport,
        "Model A v3 report",
    )
    return _canonical_json_bytes(report.model_dump(mode="json"))


def _evaluate_v3_document(
    v2_artifacts: VerifiedModelAV2DocumentArtifacts,
    base_evidence: VerifiedModelAV2DocumentEvidence,
    artifacts: VerifiedModelAV3DocumentArtifacts,
    evidence: EvidenceIR,
    gold: ContentIR,
    bundle: ModelBundleManifest,
) -> tuple[VerifiedModelAV3DocumentEvidence, int]:
    document_id = base_evidence.document_id
    if artifacts.document_id != document_id:
        raise ValueError(f"v3 execution document id mismatch for {document_id}")
    manifest = _parse_contract(
        artifacts.execution_manifest.payload,
        ModelADocumentExecutionManifest,
        f"execution manifest for {document_id}",
    )
    annotation = _parse_contract(
        artifacts.high_risk_annotation.payload,
        ModelAHighRiskAnnotation,
        f"high-risk annotation for {document_id}",
    )
    _validate_execution_bindings(
        manifest,
        artifacts,
        v2_artifacts,
        evidence,
        gold,
        annotation,
        bundle,
    )
    expected_surfaces = _canonical_risk_surfaces(evidence, gold)
    if len(expected_surfaces) > MAX_MODEL_A_V3_RISK_SURFACES:
        raise ValueError(f"risk surface count exceeds limit for {document_id}")
    supplied_surfaces = tuple(label.surface for label in annotation.labels)
    if supplied_surfaces != expected_surfaces:
        raise ValueError(
            f"high-risk annotation surface coverage or canonical order mismatch for {document_id}"
        )

    binding_by_page = {page.page_id: page for page in manifest.pages}
    resolved_by_page = {page.page_id: page for page in artifacts.pages}
    page_results: list[BuiltModelAInferenceResult] = []
    for page_index, page in enumerate(
        sorted(evidence.pages, key=lambda item: (item.page_no, item.id)),
        start=1,
    ):
        if page_index > MAX_MODEL_A_V3_PAGES:
            raise ValueError("Model A v3 page count exceeds limit")
        binding = binding_by_page[page.id]
        resolved = resolved_by_page[page.id]
        request_artifact = _parse_contract(
            resolved.request.payload,
            ModelAInferenceRequest,
            f"canonical Model A request for {document_id}/{page.id}",
            max_bytes=MAX_REQUEST_BYTES,
        )
        raw_image = base64.b64decode(request_artifact.page_image.data_base64, validate=True)
        page_image = ModelAPageImage(
            page_id=request_artifact.page_image.page_id,
            page_no=request_artifact.page_image.page_no,
            image_source_ref=request_artifact.page_image.image_source_ref,
            media_type=request_artifact.page_image.media_type,
            raw_bytes=raw_image,
            sha256=request_artifact.page_image.sha256,
        )
        rebuilt_request = build_model_a_inference_request(
            evidence,
            page_image=page_image,
            model=manifest.model,
        )
        if (
            rebuilt_request.artifact != request_artifact
            or rebuilt_request.raw_request_bytes != resolved.request.payload
            or rebuilt_request.raw_request_sha256 != resolved.request.sha256
        ):
            raise ValueError(f"trusted request cannot be reproduced for {document_id}/{page.id}")
        rebuilt_result = validate_model_a_inference_response(
            rebuilt_request,
            resolved.raw_response.payload,
            max_response_bytes=binding.response_byte_limit,
        )
        if (
            rebuilt_result.result_bytes != resolved.validation_result.payload
            or rebuilt_result.result_sha256 != resolved.validation_result.sha256
        ):
            raise ValueError(
                f"trusted validation result cannot be reproduced for {document_id}/{page.id}"
            )
        page_results.append(rebuilt_result)

    assembly = assemble_model_a_document(evidence, page_results)
    if (
        assembly.result_bytes != artifacts.assembly.payload
        or assembly.result_sha256 != artifacts.assembly.sha256
    ):
        raise ValueError(f"trusted assembly cannot be reproduced for {document_id}")
    if assembly.artifact.outcome == "valid_unverified":
        if not base_evidence.prediction_record_valid:
            raise ValueError(
                f"valid assembly disagrees with invalid v2 prediction for {document_id}"
            )
        prediction = _parse_contract(
            v2_artifacts.prediction.payload,
            ContentIR,
            f"v3 prediction ContentIR for {document_id}",
        )
        if prediction != assembly.artifact.content_ir:
            raise ValueError(
                f"v2 prediction is stale relative to replayed assembly for {document_id}"
            )
    elif base_evidence.prediction_record_valid:
        raise ValueError(f"blocked assembly disagrees with valid v2 prediction for {document_id}")

    blocked_page_ids = tuple(
        result.artifact.source.target_page_id
        for result in page_results
        if result.artifact.outcome == "blocked"
    )
    blocked_page_id_set = frozenset(blocked_page_ids)
    predicted_occurrences = tuple(
        (node, result.artifact.source.target_page_id)
        for result in page_results
        if result.artifact.page_analysis is not None
        for node in result.artifact.page_analysis.nodes
    )
    predicted_reading_order = tuple(
        node_id
        for result in page_results
        if result.artifact.page_analysis is not None
        for node_id in result.artifact.page_analysis.reading_order
    )
    prediction_counts = Counter(node.id for node, _ in predicted_occurrences)
    prediction_by_id = {
        node.id: node for node, _ in predicted_occurrences if prediction_counts[node.id] == 1
    }
    gold_by_id = {node.id: node for node in gold.nodes}
    gold_cells_by_node = {
        node.id: {_table_cell_key(cell): cell for cell in node.cells}
        for node in gold.nodes
        if isinstance(node, TableContentNode)
    }
    prediction_cells_by_node = {
        node.id: {_table_cell_key(cell): cell for cell in node.cells}
        for node in prediction_by_id.values()
        if isinstance(node, TableContentNode)
    }
    missing_gold_node_count = sum(node_id not in prediction_counts for node_id in gold_by_id)
    extra_predicted_node_count = sum(
        count - (1 if node_id in gold_by_id else 0)
        for node_id, count in prediction_counts.items()
        if count > (1 if node_id in gold_by_id else 0)
    )
    extra_nodes = tuple(
        node
        for node, _ in predicted_occurrences
        if node.id not in gold_by_id or prediction_counts[node.id] > 1
    )

    mismatches: list[ModelAHighRiskMismatchEvidence] = []
    high_risk_surface_count = 0
    for label in annotation.labels:
        if not label.high_risk:
            continue
        high_risk_surface_count += 1
        surface = label.surface
        if not _surface_mismatches(
            surface,
            gold,
            gold_by_id,
            prediction_by_id,
            gold_cells_by_node,
            prediction_cells_by_node,
            prediction_counts,
            predicted_occurrences,
            predicted_reading_order,
            blocked_page_id_set,
        ):
            continue
        route = _surface_route(
            surface,
            prediction_by_id,
            extra_nodes,
            blocked_page_id_set,
        )
        mismatches.append(
            ModelAHighRiskMismatchEvidence(
                surface_id=surface.surface_id,
                kind=surface.kind,
                page_ids=surface.page_ids,
                node_id=surface.node_id,
                table_cell=surface.table_cell,
                route=route,
            )
        )

    calibration_bins = _calibration_bins(predicted_occurrences, gold_by_id, prediction_counts)
    contribution = VerifiedModelAV3DocumentEvidence(
        document_id=document_id,
        execution_manifest_artifact_ref=artifacts.execution_manifest.artifact_ref,
        execution_manifest_artifact_sha256=artifacts.execution_manifest.sha256,
        high_risk_annotation_artifact_ref=artifacts.high_risk_annotation.artifact_ref,
        high_risk_annotation_artifact_sha256=artifacts.high_risk_annotation.sha256,
        model=manifest.model,
        page_count=len(evidence.pages),
        blocked_page_ids=blocked_page_ids,
        assembly_outcome=assembly.artifact.outcome,
        risk_surface_count=len(expected_surfaces),
        high_risk_surface_count=high_risk_surface_count,
        high_risk_mismatch_count=len(mismatches),
        high_risk_routed_count=sum(item.route is not None for item in mismatches),
        high_risk_mismatches=tuple(mismatches),
        missing_gold_node_count=missing_gold_node_count,
        extra_predicted_node_count=extra_predicted_node_count,
        calibration_bins=calibration_bins,
    )
    work_units = (
        len(expected_surfaces)
        + len(predicted_occurrences)
        + len(gold.nodes)
        + sum(len(cells) for cells in gold_cells_by_node.values())
        + sum(len(cells) for cells in prediction_cells_by_node.values())
    )
    return contribution, work_units


def _validate_execution_bindings(
    manifest: ModelADocumentExecutionManifest,
    artifacts: VerifiedModelAV3DocumentArtifacts,
    v2_artifacts: VerifiedModelAV2DocumentArtifacts,
    evidence: EvidenceIR,
    gold: ContentIR,
    annotation: ModelAHighRiskAnnotation,
    bundle: ModelBundleManifest,
) -> None:
    document_id = artifacts.document_id
    if manifest.document_id != document_id:
        raise ValueError(f"execution manifest document id mismatch for {document_id}")
    if bundle.model_a.artifact_sha256 is None:
        raise ValueError("model bundle Model A artifact digest is missing")
    expected_model = ModelAModelArtifact(
        model_id=bundle.model_a.model_id,
        model_revision=bundle.model_a.revision,
        artifact_sha256=bundle.model_a.artifact_sha256,
    )
    if manifest.model != expected_model:
        raise ValueError(f"execution model identity mismatch for {document_id}")
    expected_bindings = (
        (
            v2_artifacts.evidence,
            manifest.evidence_artifact_ref,
            manifest.evidence_artifact_sha256,
            "execution EvidenceIR",
        ),
        (
            v2_artifacts.annotation,
            manifest.gold_annotation_artifact_ref,
            manifest.gold_annotation_artifact_sha256,
            "execution gold annotation",
        ),
        (
            v2_artifacts.prediction,
            manifest.prediction_artifact_ref,
            manifest.prediction_artifact_sha256,
            "execution prediction",
        ),
        (
            artifacts.high_risk_annotation,
            manifest.high_risk_annotation_artifact_ref,
            manifest.high_risk_annotation_artifact_sha256,
            "execution high-risk annotation",
        ),
        (
            artifacts.assembly,
            manifest.assembly_artifact_ref,
            manifest.assembly_artifact_sha256,
            "execution assembly",
        ),
    )
    for artifact, expected_ref, expected_sha256, label in expected_bindings:
        _require_artifact_binding(artifact, expected_ref, expected_sha256, label)
    if manifest.evidence_ir_id != evidence.id or manifest.evidence_ir_sha256 != contract_sha256(
        evidence
    ):
        raise ValueError(f"execution EvidenceIR lineage mismatch for {document_id}")
    if (
        annotation.document_id != document_id
        or annotation.evidence_ir_id != evidence.id
        or annotation.evidence_ir_sha256 != contract_sha256(evidence)
        or annotation.gold_content_ir_id != gold.id
        or annotation.gold_content_ir_revision != gold.revision
        or annotation.gold_content_ir_sha256 != contract_sha256(gold)
    ):
        raise ValueError(f"high-risk annotation lineage mismatch for {document_id}")

    expected_pages = tuple(
        (page.id, page.page_no)
        for page in sorted(evidence.pages, key=lambda item: (item.page_no, item.id))
    )
    manifest_pages = tuple((page.page_id, page.page_no) for page in manifest.pages)
    resolved_pages = tuple((page.page_id, page.page_no) for page in artifacts.pages)
    if manifest_pages != expected_pages:
        raise ValueError(f"execution manifest page coverage mismatch for {document_id}")
    _require_exact_coverage(expected_pages, resolved_pages, f"resolved pages for {document_id}")
    resolved_by_page = {page.page_id: page for page in artifacts.pages}
    for binding in manifest.pages:
        resolved = resolved_by_page[binding.page_id]
        _require_artifact_binding(
            resolved.request,
            binding.request_artifact_ref,
            binding.request_artifact_sha256,
            "page request",
        )
        _require_artifact_binding(
            resolved.raw_response,
            binding.raw_response_artifact_ref,
            binding.raw_response_artifact_sha256,
            "raw page response",
        )
        _require_artifact_binding(
            resolved.validation_result,
            binding.validation_result_artifact_ref,
            binding.validation_result_artifact_sha256,
            "page validation result",
        )


def _canonical_risk_surfaces(
    evidence: EvidenceIR,
    gold: ContentIR,
) -> tuple[ModelARiskSurface, ...]:
    page_order = {
        page.id: index
        for index, page in enumerate(
            sorted(evidence.pages, key=lambda item: (item.page_no, item.id))
        )
    }
    page_by_observation = {
        observation.id: page.id for page in evidence.pages for observation in page.observations
    }
    all_page_ids = tuple(
        page.id for page in sorted(evidence.pages, key=lambda item: (item.page_no, item.id))
    )

    def pages_for_refs(refs: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            sorted(
                {page_by_observation[ref] for ref in refs},
                key=lambda page_id: (page_order[page_id], page_id),
            )
        )

    surfaces: list[ModelARiskSurface] = []
    for page_id in all_page_ids:
        surfaces.append(_surface(ModelARiskSurfaceKind.PAGE_OUTPUT_VALID, (page_id,)))
    for node in gold.nodes:
        node_pages = pages_for_refs(node.evidence_refs)
        for kind in (
            ModelARiskSurfaceKind.NODE_PRESENCE,
            ModelARiskSurfaceKind.NODE_KIND,
            ModelARiskSurfaceKind.NODE_ROLE,
            ModelARiskSurfaceKind.NODE_EVIDENCE_REFS,
        ):
            surfaces.append(_surface(kind, node_pages, node_id=node.id))
        if isinstance(node, TextContentNode):
            surfaces.append(
                _surface(
                    ModelARiskSurfaceKind.TEXT_VALUE,
                    node_pages,
                    node_id=node.id,
                )
            )
        elif isinstance(node, TableContentNode):
            surfaces.append(
                _surface(
                    ModelARiskSurfaceKind.TABLE_TOPOLOGY,
                    node_pages,
                    node_id=node.id,
                )
            )
            for cell in sorted(node.cells, key=_table_cell_key):
                cell_key = _table_cell_key(cell)
                cell_pages = pages_for_refs(cell.evidence_refs)
                surfaces.extend(
                    (
                        _surface(
                            ModelARiskSurfaceKind.TABLE_CELL_TEXT,
                            cell_pages,
                            node_id=node.id,
                            table_cell=cell_key,
                        ),
                        _surface(
                            ModelARiskSurfaceKind.TABLE_CELL_EVIDENCE_REFS,
                            cell_pages,
                            node_id=node.id,
                            table_cell=cell_key,
                        ),
                    )
                )
        elif isinstance(node, FormulaContentNode):
            surfaces.extend(
                (
                    _surface(
                        ModelARiskSurfaceKind.FORMULA_EXPRESSION,
                        node_pages,
                        node_id=node.id,
                    ),
                    _surface(
                        ModelARiskSurfaceKind.FORMULA_FORMAT,
                        node_pages,
                        node_id=node.id,
                    ),
                )
            )
        elif isinstance(node, ImageContentNode):
            surfaces.append(
                _surface(
                    ModelARiskSurfaceKind.IMAGE_ASSET_REF,
                    node_pages,
                    node_id=node.id,
                )
            )
    surfaces.extend(
        (
            _surface(ModelARiskSurfaceKind.READING_ORDER, all_page_ids),
            _surface(ModelARiskSurfaceKind.EXTRA_NODES, all_page_ids),
        )
    )
    if len(surfaces) > MAX_MODEL_A_V3_RISK_SURFACES:
        raise ValueError("Model A v3 risk surface count exceeds limit")
    return tuple(surfaces)


def _surface(
    kind: ModelARiskSurfaceKind,
    page_ids: tuple[str, ...],
    *,
    node_id: str | None = None,
    table_cell: tuple[int, int, int, int] | None = None,
) -> ModelARiskSurface:
    return ModelARiskSurface(
        surface_id=_risk_surface_id(kind, page_ids, node_id, table_cell),
        kind=kind,
        page_ids=page_ids,
        node_id=node_id,
        table_cell=table_cell,
    )


def _risk_surface_id(
    kind: ModelARiskSurfaceKind,
    page_ids: Sequence[str],
    node_id: str | None,
    table_cell: tuple[int, int, int, int] | None,
) -> str:
    identity = {
        "definition": MODEL_A_RISK_SURFACE_DEFINITION_ID,
        "kind": kind.value,
        "node_id": node_id,
        "page_ids": list(page_ids),
        "table_cell": list(table_cell) if table_cell is not None else None,
    }
    return "risk:" + _sha256(_canonical_json_bytes(identity))


def _surface_mismatches(
    surface: ModelARiskSurface,
    gold: ContentIR,
    gold_by_id: dict[str, ContentNode],
    prediction_by_id: dict[str, ContentNode],
    gold_cells_by_node: dict[str, dict[tuple[int, int, int, int], TableCell]],
    prediction_cells_by_node: dict[str, dict[tuple[int, int, int, int], TableCell]],
    prediction_counts: Counter[str],
    predicted_occurrences: tuple[tuple[ContentNode, str], ...],
    predicted_reading_order: tuple[str, ...],
    blocked_page_ids: frozenset[str],
) -> bool:
    kind = surface.kind
    if kind == ModelARiskSurfaceKind.PAGE_OUTPUT_VALID:
        return surface.page_ids[0] in blocked_page_ids
    if kind == ModelARiskSurfaceKind.READING_ORDER:
        if blocked_page_ids or any(count != 1 for count in prediction_counts.values()):
            return True
        return predicted_reading_order != gold.reading_order
    if kind == ModelARiskSurfaceKind.EXTRA_NODES:
        return any(
            node_id not in gold_by_id or count > 1 for node_id, count in prediction_counts.items()
        )
    assert surface.node_id is not None
    gold_node = gold_by_id[surface.node_id]
    predicted_node = prediction_by_id.get(surface.node_id)
    if kind == ModelARiskSurfaceKind.NODE_PRESENCE:
        return predicted_node is None
    if kind == ModelARiskSurfaceKind.NODE_KIND:
        return predicted_node is None or predicted_node.kind != gold_node.kind
    if kind == ModelARiskSurfaceKind.NODE_ROLE:
        return predicted_node is None or predicted_node.role != gold_node.role
    if kind == ModelARiskSurfaceKind.NODE_EVIDENCE_REFS:
        return predicted_node is None or predicted_node.evidence_refs != gold_node.evidence_refs
    if kind == ModelARiskSurfaceKind.TEXT_VALUE:
        return (
            not isinstance(gold_node, TextContentNode)
            or not isinstance(predicted_node, TextContentNode)
            or predicted_node.text != gold_node.text
        )
    if kind == ModelARiskSurfaceKind.TABLE_TOPOLOGY:
        return (
            not isinstance(gold_node, TableContentNode)
            or not isinstance(predicted_node, TableContentNode)
            or _table_topology(predicted_node) != _table_topology(gold_node)
        )
    if kind in _CELL_SURFACE_KINDS:
        if (
            not isinstance(gold_node, TableContentNode)
            or not isinstance(predicted_node, TableContentNode)
            or surface.table_cell is None
        ):
            return True
        gold_cell = gold_cells_by_node[gold_node.id].get(surface.table_cell)
        predicted_cell = prediction_cells_by_node[predicted_node.id].get(surface.table_cell)
        if gold_cell is None or predicted_cell is None:
            return True
        if kind == ModelARiskSurfaceKind.TABLE_CELL_TEXT:
            return predicted_cell.text != gold_cell.text
        return predicted_cell.evidence_refs != gold_cell.evidence_refs
    if kind == ModelARiskSurfaceKind.FORMULA_EXPRESSION:
        return (
            not isinstance(gold_node, FormulaContentNode)
            or not isinstance(predicted_node, FormulaContentNode)
            or predicted_node.expression != gold_node.expression
        )
    if kind == ModelARiskSurfaceKind.FORMULA_FORMAT:
        return (
            not isinstance(gold_node, FormulaContentNode)
            or not isinstance(predicted_node, FormulaContentNode)
            or predicted_node.format != gold_node.format
        )
    if kind == ModelARiskSurfaceKind.IMAGE_ASSET_REF:
        return (
            not isinstance(gold_node, ImageContentNode)
            or not isinstance(predicted_node, ImageContentNode)
            or predicted_node.asset_ref != gold_node.asset_ref
        )
    raise AssertionError(f"unhandled Model A risk surface kind: {kind}")


def _surface_route(
    surface: ModelARiskSurface,
    prediction_by_id: dict[str, ContentNode],
    extra_nodes: tuple[ContentNode, ...],
    blocked_page_ids: frozenset[str],
) -> Literal["validator_block", "node_needs_review"] | None:
    if any(page_id in blocked_page_ids for page_id in surface.page_ids):
        return "validator_block"
    if surface.node_id is not None:
        predicted = prediction_by_id.get(surface.node_id)
        return "node_needs_review" if predicted is not None and predicted.needs_review else None
    if (
        surface.kind == ModelARiskSurfaceKind.EXTRA_NODES
        and extra_nodes
        and all(node.needs_review for node in extra_nodes)
    ):
        return "node_needs_review"
    return None


def _table_cell_key(cell: TableCell) -> tuple[int, int, int, int]:
    return (cell.row, cell.column, cell.row_span, cell.column_span)


def _table_topology(table: TableContentNode) -> tuple[object, ...]:
    return (
        table.rows,
        table.columns,
        tuple(sorted(_table_cell_key(cell) for cell in table.cells)),
    )


def _calibration_bins(
    predicted_occurrences: tuple[tuple[ContentNode, str], ...],
    gold_by_id: dict[str, ContentNode],
    prediction_counts: Counter[str],
) -> tuple[ModelACalibrationBin, ...]:
    confidences: list[list[float]] = [[] for _ in range(10)]
    correct: list[int] = [0] * 10
    for node, _page_id in predicted_occurrences:
        index = min(int(node.confidence * 10), 9)
        confidences[index].append(node.confidence)
        gold = gold_by_id.get(node.id)
        if (
            gold is not None
            and prediction_counts[node.id] == 1
            and _semantic_node_payload(node) == _semantic_node_payload(gold)
        ):
            correct[index] += 1
    return tuple(
        ModelACalibrationBin(
            index=index,
            lower_bound=index / 10,
            upper_bound=(index + 1) / 10,
            upper_inclusive=index == 9,
            sample_count=len(confidences[index]),
            confidence_sum=math.fsum(confidences[index]),
            correct_count=correct[index],
        )
        for index in range(10)
    )


def _semantic_node_payload(node: ContentNode) -> dict[str, object]:
    payload = node.model_dump(mode="json")
    payload.pop("confidence")
    payload.pop("needs_review")
    if isinstance(node, TableContentNode):
        payload["cells"] = [
            cell.model_dump(mode="json") for cell in sorted(node.cells, key=_table_cell_key)
        ]
    return payload


def _aggregate_v3_metrics(
    base_report: VerifiedModelAV2MeasurementReport,
    documents: Sequence[VerifiedModelAV3DocumentEvidence],
) -> tuple[
    dict[str, int],
    dict[str, float | None],
    tuple[ModelACalibrationBin, ...],
]:
    counts = dict(base_report.metric_sample_counts)
    measurements = dict(base_report.measurements)
    high_risk_errors = sum(document.high_risk_mismatch_count for document in documents)
    high_risk_routed = sum(document.high_risk_routed_count for document in documents)
    counts["model_a.high_risk_review_recall"] = high_risk_errors
    measurements["model_a.high_risk_review_recall"] = (
        None if high_risk_errors == 0 else high_risk_routed / high_risk_errors
    )

    aggregate_bins: list[ModelACalibrationBin] = []
    calibration_samples = 0
    ece_terms: list[float] = []
    for index in range(10):
        sample_count = sum(document.calibration_bins[index].sample_count for document in documents)
        confidence_sum = math.fsum(
            document.calibration_bins[index].confidence_sum for document in documents
        )
        correct_count = sum(
            document.calibration_bins[index].correct_count for document in documents
        )
        aggregate_bins.append(
            ModelACalibrationBin(
                index=index,
                lower_bound=index / 10,
                upper_bound=(index + 1) / 10,
                upper_inclusive=index == 9,
                sample_count=sample_count,
                confidence_sum=confidence_sum,
                correct_count=correct_count,
            )
        )
        calibration_samples += sample_count
    if calibration_samples:
        for item in aggregate_bins:
            if item.sample_count:
                mean_confidence = item.confidence_sum / item.sample_count
                accuracy = item.correct_count / item.sample_count
                ece_terms.append(
                    item.sample_count / calibration_samples * abs(mean_confidence - accuracy)
                )
    counts["model_a.confidence_ece"] = calibration_samples
    measurements["model_a.confidence_ece"] = (
        None if calibration_samples == 0 else math.fsum(ece_terms)
    )
    return counts, measurements, tuple(aggregate_bins)


def _parse_contract(
    payload: bytes,
    contract_type: type[ContractT],
    label: str,
    *,
    max_bytes: int | None = None,
) -> ContractT:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be bytes")
    if max_bytes is None:
        max_bytes = MAX_TRUSTED_JSON_BYTES
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds maximum byte size")
    require_strict_json_bytes(
        payload,
        max_depth=MAX_JSON_DEPTH,
        max_nodes=MAX_TRUSTED_JSON_NODES,
        label=label,
    )
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if contract_type is ContentIR:
        _require_raw_content_bounds(raw, label)
    elif contract_type is EvidenceIR:
        _require_raw_evidence_bounds(raw, label)
    elif contract_type is ModelAHighRiskAnnotation:
        _require_raw_annotation_bounds(raw, label)
    try:
        return contract_type.model_validate_json(payload, strict=True)
    except (ValidationError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"{label} does not satisfy its strict contract") from exc


def _require_raw_content_bounds(value: object, label: str) -> None:
    if not isinstance(value, dict):
        return
    nodes = value.get("nodes")
    reading_order = value.get("reading_order")
    if isinstance(nodes, list) and len(nodes) > MAX_CONTENT_NODES:
        raise ValueError(f"{label} content node count exceeds limit")
    if isinstance(reading_order, list) and len(reading_order) > MAX_CONTENT_NODES:
        raise ValueError(f"{label} reading-order count exceeds limit")
    if not isinstance(nodes, list):
        return
    total_cells = 0
    total_grid = 0
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "table":
            continue
        cells = node.get("cells")
        if isinstance(cells, list):
            total_cells += len(cells)
            if total_cells > MAX_TABLE_CELLS_PER_DOCUMENT:
                raise ValueError(f"{label} total table cell count exceeds limit")
        rows = node.get("rows")
        columns = node.get("columns")
        if _is_json_int(rows) and _is_json_int(columns) and rows > 0 and columns > 0:
            grid = rows * columns
            if grid > MAX_TABLE_GRID_POSITIONS:
                raise ValueError(f"{label} table grid exceeds limit")
            total_grid += grid
            if total_grid > MAX_TABLE_GRID_POSITIONS:
                raise ValueError(f"{label} total table grid exceeds limit")
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            row_span = cell.get("row_span", 1)
            column_span = cell.get("column_span", 1)
            if (
                _is_json_int(row_span)
                and _is_json_int(column_span)
                and row_span > 0
                and column_span > 0
                and row_span * column_span > MAX_TABLE_GRID_POSITIONS
            ):
                raise ValueError(f"{label} table cell span exceeds limit")


def _require_raw_evidence_bounds(value: object, label: str) -> None:
    if not isinstance(value, dict):
        return
    sources = value.get("sources")
    pages = value.get("pages")
    if isinstance(sources, list) and len(sources) > MAX_MODEL_A_V3_RISK_SURFACES:
        raise ValueError(f"{label} source count exceeds limit")
    if isinstance(pages, list) and len(pages) > MAX_MODEL_A_V3_PAGES:
        raise ValueError(f"{label} page count exceeds limit")
    if not isinstance(pages, list):
        return
    observation_count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        observations = page.get("observations")
        if isinstance(observations, list):
            observation_count += len(observations)
            if observation_count > MAX_MODEL_A_V3_RISK_SURFACES:
                raise ValueError(f"{label} observation count exceeds limit")


def _require_raw_annotation_bounds(value: object, label: str) -> None:
    if not isinstance(value, dict):
        return
    labels = value.get("labels")
    if isinstance(labels, list) and len(labels) > MAX_MODEL_A_V3_RISK_SURFACES:
        raise ValueError(f"{label} risk label count exceeds limit")


def _is_json_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _revalidate_v3_inputs(
    inputs: VerifiedModelAV3EvaluationInputs,
) -> VerifiedModelAV3EvaluationInputs:
    if type(inputs) is not VerifiedModelAV3EvaluationInputs:
        raise TypeError("inputs must be VerifiedModelAV3EvaluationInputs")
    _require_exact_nested_input_types(inputs)
    restored = _revalidate_exact_model(
        inputs,
        VerifiedModelAV3EvaluationInputs,
        "Model A v3 inputs",
    )
    _require_exact_nested_input_types(restored)
    return restored


def _require_unmutated_runtime_metric_tables() -> None:
    if (
        tuple(MODEL_A_V2_METRIC_DEFINITION_IDS.items())
        != _CANONICAL_MODEL_A_V2_METRIC_DEFINITION_ITEMS
        or tuple(MODEL_A_V2_METRIC_SAMPLE_UNITS.items())
        != _CANONICAL_MODEL_A_V2_METRIC_SAMPLE_UNIT_ITEMS
        or tuple(MODEL_A_V3_METRIC_DEFINITION_IDS.items())
        != _CANONICAL_MODEL_A_V3_METRIC_DEFINITION_ITEMS
        or tuple(MODEL_A_V3_METRIC_SAMPLE_UNITS.items())
        != _CANONICAL_MODEL_A_V3_METRIC_SAMPLE_UNIT_ITEMS
    ):
        raise ValueError("Model A v2/v3 runtime metric tables were mutated")


def _preflight_total_artifact_bytes(inputs: VerifiedModelAV3EvaluationInputs) -> None:
    if type(inputs) is not VerifiedModelAV3EvaluationInputs:
        raise TypeError("inputs must be VerifiedModelAV3EvaluationInputs")
    _require_exact_nested_input_types(inputs)
    total_bytes = 0
    for artifact in _all_v3_artifacts(inputs):
        if type(artifact.payload) is not bytes:
            raise TypeError(f"artifact payload must be bytes: {artifact.artifact_ref}")
        total_bytes += len(artifact.payload)
        if total_bytes > MAX_MODEL_A_V3_TOTAL_ARTIFACT_BYTES:
            raise ValueError(
                "Model A v3 total resolved artifact bytes exceed the 1 GiB in-memory "
                "limit; this corpus requires a future streaming evaluator"
            )


def _revalidate_exact_model(
    value: ContractT,
    expected_type: type[ContractT],
    label: str,
) -> ContractT:
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be exact {expected_type.__name__}")
    try:
        restored = expected_type.model_validate(
            value.model_dump(mode="python", warnings="error"),
            strict=True,
        )
    except (ValidationError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"{label} is not a valid strict model") from exc
    if restored != value:
        raise ValueError(f"{label} is not canonical")
    return restored


def _require_exact_nested_input_types(inputs: VerifiedModelAV3EvaluationInputs) -> None:
    if type(inputs.v2_inputs) is not VerifiedModelAV2EvaluationInputs:
        raise TypeError("v2_inputs must be exact VerifiedModelAV2EvaluationInputs")
    if type(inputs.evaluator_artifact) is not ArtifactPayload:
        raise TypeError("v3 evaluator artifact must be exact ArtifactPayload")
    for document in inputs.documents:
        if type(document) is not VerifiedModelAV3DocumentArtifacts:
            raise TypeError("v3 documents must use exact document artifact models")
        for artifact in (
            document.execution_manifest,
            document.high_risk_annotation,
            document.assembly,
        ):
            if type(artifact) is not ArtifactPayload:
                raise TypeError("v3 document artifacts must be exact ArtifactPayload")
        for page in document.pages:
            if type(page) is not ModelAPageExecutionArtifacts:
                raise TypeError("v3 page executions must use exact artifact models")
            if (
                type(page.request) is not ArtifactPayload
                or type(page.validation_result) is not ArtifactPayload
                or type(page.raw_response) is not ModelARawResponsePayload
            ):
                raise TypeError("v3 page artifacts must use exact payload models")


def _validate_runtime_inputs(inputs: VerifiedModelAV3EvaluationInputs) -> None:
    artifacts = list(_all_v3_artifacts(inputs))
    page_count = 0
    for document in inputs.documents:
        page_count += len(document.pages)
        for page in document.pages:
            if len(page.request.payload) > MAX_REQUEST_BYTES:
                raise ValueError("v3 canonical request exceeds byte limit")
            if len(page.raw_response.payload) > MAX_MODEL_A_V3_CAPTURED_RESPONSE_BYTES:
                raise ValueError("v3 captured raw response exceeds absolute byte limit")
            if len(page.validation_result.payload) > MAX_MODEL_A_V3_RESULT_BYTES:
                raise ValueError("v3 validation result exceeds byte limit")
        if len(document.assembly.payload) > MAX_MODEL_A_V3_ASSEMBLY_BYTES:
            raise ValueError("v3 assembly artifact exceeds byte limit")
    if page_count > MAX_MODEL_A_V3_PAGES:
        raise ValueError("Model A v3 total page count exceeds limit")
    total_bytes = sum(len(artifact.payload) for artifact in artifacts)
    if total_bytes > MAX_MODEL_A_V3_TOTAL_ARTIFACT_BYTES:
        raise ValueError(
            "Model A v3 total resolved artifact bytes exceed the 1 GiB in-memory "
            "limit; this corpus requires a future streaming evaluator"
        )
    _reject_duplicates(
        (artifact.artifact_ref.casefold() for artifact in artifacts),
        "Model A v3 resolved artifact refs",
    )
    for artifact in artifacts:
        if type(artifact.payload) is not bytes:
            raise TypeError(f"artifact payload must be bytes: {artifact.artifact_ref}")
        if hashlib.sha256(artifact.payload).hexdigest() != artifact.sha256:
            raise ValueError(f"artifact payload digest mismatch: {artifact.artifact_ref}")


def _all_v3_artifacts(
    inputs: VerifiedModelAV3EvaluationInputs,
) -> Iterable[ArtifactPayload | ModelARawResponsePayload]:
    yield from _all_v2_artifacts(inputs.v2_inputs)
    yield inputs.evaluator_artifact
    for document in inputs.documents:
        yield document.execution_manifest
        yield document.high_risk_annotation
        yield document.assembly
        for page in document.pages:
            yield page.request
            yield page.raw_response
            yield page.validation_result


def _all_v2_artifacts(
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


def _verify_running_evaluator(artifact: ArtifactPayload) -> str:
    if artifact.artifact_ref != VERIFIED_MODEL_A_V3_EVALUATOR_ARTIFACT_REF:
        raise ValueError("unexpected Model A v3 evaluator artifact ref")
    running_payload = Path(__file__).read_bytes()
    running_sha256 = _sha256(running_payload)
    if artifact.payload != running_payload or artifact.sha256 != running_sha256:
        raise ValueError("v3 evaluator artifact does not match running evaluator bytes")
    return running_sha256


def _require_artifact_binding(
    artifact: ArtifactPayload | ModelARawResponsePayload,
    expected_ref: str,
    expected_sha256: str,
    label: str,
) -> None:
    if artifact.artifact_ref != expected_ref:
        raise ValueError(f"{label} artifact ref does not match manifest binding")
    if artifact.sha256 != expected_sha256:
        raise ValueError(f"{label} artifact digest does not match manifest binding")


def _require_exact_coverage(
    expected_ids: Sequence[object],
    actual_ids: Sequence[object],
    label: str,
) -> None:
    expected = set(expected_ids)
    actual = set(actual_ids)
    if expected != actual or len(expected_ids) != len(expected) or len(actual_ids) != len(actual):
        missing = sorted(repr(value) for value in expected - actual)
        unexpected = sorted(repr(value) for value in actual - expected)
        raise ValueError(
            f"{label} exact coverage mismatch: missing={missing}, unexpected={unexpected}"
        )


def _reject_duplicates(values: Iterable[str], label: str) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: " + ", ".join(duplicates))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
