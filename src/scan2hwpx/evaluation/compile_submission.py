from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from scan2hwpx.contracts import (
    REQUIRED_DOCUMENT_QUALITY_CHECKS_V1,
    CheckDecision,
    ContentIR,
    EvidenceIR,
    FormulaContentNode,
    HwpDocumentPlan,
    QualityCheck,
    QualityOutcome,
    QualityReport,
    contract_sha256,
)
from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    CandidateToModelBHandoff,
    VerifiedCandidateToModelBHandoff,
)
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBPlanGroundingEvidence,
    ModelBPlanReviewDraft,
    ModelBPlanReviewInferenceBinding,
    verify_model_b_plan_review_draft,
)
from scan2hwpx.evaluation.review_draft import (
    CandidateReviewDraft,
    VerifiedCandidateCompileInputs,
    materialize_verified_candidate_compile_inputs,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    SafeArtifactIOError,
    publish_file_create_only,
    sha256_bounded_regular_file,
)
from scan2hwpx.hwpx.assets import ImageAssetBundle
from scan2hwpx.hwpx.compiler import PlanCompileResult, compile_plan_hwpx

if TYPE_CHECKING:
    from scan2hwpx.model_b.inference import BuiltModelBInferenceResult

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECK_ORDER = (
    "schema-reference-integrity",
    "content-conservation",
    "reading-order",
    "table-topology",
    "formula-grounding",
    "visual-layout",
    "hwpx-package-valid",
    "dvc-validation",
    "editable-objects",
    "hancom-round-trip",
)


class CompileSubmissionError(RuntimeError):
    """Raised when reviewed artifacts cannot be bound and safely published."""


@dataclass(frozen=True, slots=True)
class CompileSubmissionResult:
    output_path: Path
    compile_result: PlanCompileResult
    quality_report: QualityReport


def compile_reviewed_submission(
    candidate_review: CandidateReviewDraft,
    verified_handoff: VerifiedCandidateToModelBHandoff,
    completed_model_b_review: ModelBPlanReviewDraft,
    model_b_inference_result: BuiltModelBInferenceResult,
    *,
    candidate_root: Path | str,
    output_path: Path | str,
    compiled_artifact_ref: str,
) -> CompileSubmissionResult:
    """Compile one fully reviewed candidate and publish its HWPX create-only.

    The result remains review-only: this boundary does not run visual comparison,
    Hancom DVC, or a Hancom open/save/reopen round trip.
    """

    candidate = _strict_revalidate(
        candidate_review,
        CandidateReviewDraft,
        "candidate review",
    )
    completed_model_b = _strict_revalidate(
        completed_model_b_review,
        ModelBPlanReviewDraft,
        "completed Model B review",
    )
    envelope = _revalidate_verified_handoff(verified_handoff)

    inputs = materialize_verified_candidate_compile_inputs(
        candidate,
        candidate_root=candidate_root,
    )
    candidate, evidence, image_assets = _revalidate_materialized_inputs(
        candidate,
        inputs,
    )
    _require_candidate_handoff_match(candidate, envelope)
    (
        inference_binding,
        inference_plan,
        inference_content,
        inference_grounding,
    ) = _verify_model_b_inference_result(
        model_b_inference_result,
        verified_handoff=verified_handoff,
        envelope=envelope,
        completed_model_b=completed_model_b,
    )
    completed_model_b = verify_model_b_plan_review_draft(
        completed_model_b,
        content_ir=inference_content,
        base_plan=inference_plan,
        grounding_evidence=inference_grounding,
    )
    if completed_model_b.status != "complete":
        raise CompileSubmissionError("Model B plan review must be complete before compilation")
    if completed_model_b.document_id != candidate.document_id:
        raise CompileSubmissionError("completed Model B review document does not match candidate")

    candidate_root_path = Path(candidate_root).resolve(strict=True)
    destination, parent = _resolve_new_output(
        output_path,
        candidate_root=candidate_root_path,
    )
    descriptor, staged_output_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".partial.hwpx",
    )
    staged_output = Path(staged_output_name)
    committed = False
    try:
        os.close(descriptor)
        if image_assets is None:
            compile_result = compile_plan_hwpx(
                completed_model_b.content_ir,
                completed_model_b.reviewed_hwp_document_plan,
                staged_output,
            )
        else:
            compile_result = compile_plan_hwpx(
                completed_model_b.content_ir,
                completed_model_b.reviewed_hwp_document_plan,
                staged_output,
                evidence_ir=evidence,
                image_assets=image_assets,
            )
        _require_compile_result_lineage(completed_model_b, compile_result)
        quality_report = _build_quality_report(
            candidate=candidate,
            evidence=evidence,
            completed_model_b=completed_model_b,
            compile_result=compile_result,
            compiled_artifact_ref=compiled_artifact_ref,
            inference_binding=inference_binding,
        )
        _require_staged_digest(staged_output, compile_result.artifact_sha256)
        publish_file_create_only(staged_output, destination)
        committed = True
    finally:
        if not committed:
            staged_output.unlink(missing_ok=True)

    return CompileSubmissionResult(
        output_path=destination,
        compile_result=compile_result,
        quality_report=quality_report,
    )


def _strict_revalidate(
    value: object,
    expected_type: type[_ModelT],
    label: str,
) -> _ModelT:
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be an exact {expected_type.__name__}")
    try:
        return expected_type.model_validate(
            value.model_dump(mode="python", round_trip=True, warnings=False),
            strict=True,
        )
    except ValidationError as exc:
        raise CompileSubmissionError(f"{label} is not a valid strict contract") from exc


def _revalidate_verified_handoff(
    verified: VerifiedCandidateToModelBHandoff,
) -> CandidateToModelBHandoff:
    if type(verified) is not VerifiedCandidateToModelBHandoff:
        raise TypeError("verified handoff must be an exact VerifiedCandidateToModelBHandoff")
    if not _SHA256.fullmatch(verified.artifact_sha256):
        raise CompileSubmissionError("verified handoff artifact SHA-256 is invalid")
    envelope = _strict_revalidate(
        verified.envelope,
        CandidateToModelBHandoff,
        "verified handoff envelope",
    )
    canonical_payload = (
        json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(canonical_payload).hexdigest() != verified.artifact_sha256:
        raise CompileSubmissionError("verified handoff artifact SHA-256 does not match envelope")
    return envelope


def _revalidate_materialized_inputs(
    candidate: CandidateReviewDraft,
    inputs: VerifiedCandidateCompileInputs,
) -> tuple[CandidateReviewDraft, EvidenceIR, ImageAssetBundle | None]:
    if type(inputs) is not VerifiedCandidateCompileInputs:
        raise TypeError("candidate materializer returned an invalid result type")
    materialized_candidate = _strict_revalidate(
        inputs.candidate_review,
        CandidateReviewDraft,
        "materialized candidate review",
    )
    if materialized_candidate != candidate:
        raise CompileSubmissionError("candidate materializer returned another review revision")
    if materialized_candidate.status != "complete":
        raise CompileSubmissionError("candidate review must be complete before compilation")
    evidence = _strict_revalidate(inputs.evidence_ir, EvidenceIR, "materialized EvidenceIR")
    if inputs.image_assets is not None and type(inputs.image_assets) is not ImageAssetBundle:
        raise TypeError("materialized image assets must be an exact ImageAssetBundle")
    try:
        materialized_candidate.reviewed_content_ir.assert_evidence_integrity(evidence)
    except ValueError as exc:
        raise CompileSubmissionError(
            "materialized EvidenceIR does not match reviewed content"
        ) from exc
    return materialized_candidate, evidence, inputs.image_assets


def _verify_model_b_inference_result(
    value: BuiltModelBInferenceResult,
    *,
    verified_handoff: VerifiedCandidateToModelBHandoff,
    envelope: CandidateToModelBHandoff,
    completed_model_b: ModelBPlanReviewDraft,
) -> tuple[
    ModelBPlanReviewInferenceBinding,
    HwpDocumentPlan,
    ContentIR,
    ModelBPlanGroundingEvidence,
]:
    # Local import avoids the existing evaluation <-> model_b import cycle.
    from scan2hwpx.model_b.inference import BuiltModelBInferenceResult

    if type(value) is not BuiltModelBInferenceResult:
        raise CompileSubmissionError(
            "the exact Model B inference result is required for submission"
        )
    try:
        BuiltModelBInferenceResult.__post_init__(value)
    except (TypeError, ValueError, ValidationError) as exc:
        raise CompileSubmissionError("Model B inference result is not valid") from exc
    artifact = value.artifact
    plan = artifact.hwp_document_plan
    if artifact.outcome != "valid_unverified" or plan is None:
        raise CompileSubmissionError("blocked Model B inference cannot be submitted")
    request = value.request
    if (
        request.verified_handoff.artifact_sha256 != verified_handoff.artifact_sha256
        or request.verified_handoff.envelope != envelope
        or request.artifact.source.handoff_artifact_sha256 != verified_handoff.artifact_sha256
    ):
        raise CompileSubmissionError("Model B inference belongs to another handoff")
    seed = envelope.model_b_plan_review_draft
    expected = ModelBPlanReviewInferenceBinding(
        handoff_artifact_sha256=verified_handoff.artifact_sha256,
        request_sha256=request.raw_request_sha256,
        result_sha256=value.result_sha256,
        raw_response_sha256=artifact.raw_response_sha256,
        inferred_hwp_document_plan_contract_sha256=contract_sha256(plan),
        model_id=artifact.model.model_id,
        model_revision=artifact.model.model_revision,
        model_artifact_sha256=artifact.model.artifact_sha256,
        handoff_source_candidate_contract_sha256=(seed.source_candidate_contract_sha256),
    )
    if completed_model_b.schema_version != "model-b-plan-review-draft/1.3":
        raise CompileSubmissionError("submission requires an inference-bound review draft")
    if completed_model_b.model_b_inference_binding != expected:
        raise CompileSubmissionError("Model B review inference binding does not match")
    if (
        completed_model_b.base_hwp_document_plan != plan
        or completed_model_b.base_hwp_document_plan_contract_sha256 != contract_sha256(plan)
    ):
        raise CompileSubmissionError("Model B review base plan does not match inference")
    return expected, plan, request.content_ir, request.grounding_evidence


def _require_candidate_handoff_match(
    candidate: CandidateReviewDraft,
    envelope: CandidateToModelBHandoff,
) -> None:
    binding = envelope.completed_candidate_review
    mismatches: list[str] = []
    expected_values = (
        ("document_id", envelope.document_id, candidate.document_id),
        ("lineage_id", envelope.lineage_id, candidate.lineage_id),
        ("binding.document_id", binding.document_id, candidate.document_id),
        ("binding.lineage_id", binding.lineage_id, candidate.lineage_id),
        (
            "candidate_manifest_sha256",
            binding.candidate_manifest_sha256,
            candidate.candidate_manifest_sha256,
        ),
        ("draft_revision", binding.draft_revision, candidate.draft_revision),
        ("base_content_ir", binding.base_content_ir, candidate.base_content_ir),
        (
            "base_hwp_document_plan",
            binding.base_hwp_document_plan,
            candidate.base_hwp_document_plan,
        ),
        (
            "reviewed_content_ir_contract_sha256",
            binding.reviewed_content_ir_contract_sha256,
            contract_sha256(candidate.reviewed_content_ir),
        ),
        (
            "reviewed_hwp_document_plan_contract_sha256",
            binding.reviewed_hwp_document_plan_contract_sha256,
            contract_sha256(candidate.reviewed_hwp_document_plan),
        ),
    )
    mismatches.extend(label for label, actual, expected in expected_values if actual != expected)
    initial_model_b = envelope.model_b_plan_review_draft
    if initial_model_b.content_ir != candidate.reviewed_content_ir:
        mismatches.append("Model B initial ContentIR")
    if initial_model_b.base_hwp_document_plan != candidate.reviewed_hwp_document_plan:
        mismatches.append("Model B initial base HwpDocumentPlan")
    if mismatches:
        raise CompileSubmissionError(
            "candidate review does not match verified handoff: " + ", ".join(mismatches)
        )


def _resolve_new_output(
    output_path: Path | str,
    *,
    candidate_root: Path,
) -> tuple[Path, Path]:
    requested = Path(output_path)
    if requested.name in {"", ".", ".."}:
        raise ValueError("submission output must name a file")
    if requested.suffix.lower() != ".hwpx":
        raise ValueError("submission output path must use the .hwpx suffix")
    preliminary = requested.resolve(strict=False)
    if preliminary == candidate_root or preliminary.is_relative_to(candidate_root):
        raise ValueError("submission output must be outside the candidate root")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if destination == candidate_root or destination.is_relative_to(candidate_root):
        raise ValueError("submission output must be outside the candidate root")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    return destination, parent


def _build_quality_report(
    *,
    candidate: CandidateReviewDraft,
    evidence: EvidenceIR,
    completed_model_b: ModelBPlanReviewDraft,
    compile_result: PlanCompileResult,
    compiled_artifact_ref: str,
    inference_binding: ModelBPlanReviewInferenceBinding,
) -> QualityReport:
    content = completed_model_b.content_ir
    plan = completed_model_b.reviewed_hwp_document_plan
    has_formula = any(isinstance(node, FormulaContentNode) for node in content.nodes)
    decisions = {
        "schema-reference-integrity": (
            CheckDecision.PASS,
            "Strict contracts, lineage bindings, and package references were revalidated.",
        ),
        "content-conservation": (
            CheckDecision.PASS,
            "The compiler verified the complete planned content projection.",
        ),
        "reading-order": (
            CheckDecision.PASS,
            "The compiler verified native output objects in HwpDocumentPlan flow order.",
        ),
        "table-topology": (
            CheckDecision.PASS,
            "The compiler verified every emitted native table against its declared topology.",
        ),
        "formula-grounding": (
            CheckDecision.REVIEW if has_formula else CheckDecision.PASS,
            "Formula grounding requires review."
            if has_formula
            else "No formula nodes were present in the compiled ContentIR.",
        ),
        "visual-layout": (
            CheckDecision.REVIEW,
            "No visual layout comparison was run for this submission.",
        ),
        "hwpx-package-valid": (
            CheckDecision.PASS,
            "The staged HWPX passed the bounded package validator and compiler checks.",
        ),
        "dvc-validation": (
            CheckDecision.REVIEW,
            "Hancom DVC validation was not run for this submission.",
        ),
        "editable-objects": (
            CheckDecision.PASS,
            "The compiler emitted supported content as native HWPX objects.",
        ),
        "hancom-round-trip": (
            CheckDecision.REVIEW,
            "A Hancom open/save/reopen round trip was not run for this submission.",
        ),
    }
    if set(decisions) != REQUIRED_DOCUMENT_QUALITY_CHECKS_V1:
        raise CompileSubmissionError("document quality check inventory is incomplete")
    try:
        report = QualityReport(
            id=f"quality-{compile_result.artifact_sha256}",
            document_id=candidate.document_id,
            evidence_ir_id=evidence.id,
            evidence_ir_sha256=contract_sha256(evidence),
            content_ir_id=content.id,
            content_ir_revision=content.revision,
            content_ir_sha256=contract_sha256(content),
            hwp_document_plan_id=plan.id,
            hwp_document_plan_sha256=contract_sha256(plan),
            compiled_artifact_ref=compiled_artifact_ref,
            compiled_artifact_sha256=compile_result.artifact_sha256,
            compiler_version=compile_result.compiler_version,
            model_bundle_id=(f"unbundled-model-b-{inference_binding.model_artifact_sha256}"),
            check_set_id="document-quality-v1",
            outcome=QualityOutcome.REVIEW,
            checks=tuple(
                QualityCheck(
                    id=check_id, decision=decisions[check_id][0], message=decisions[check_id][1]
                )
                for check_id in _CHECK_ORDER
            ),
        )
        report.assert_contract_lineage(evidence, content, plan)
    except (ValidationError, ValueError) as exc:
        raise CompileSubmissionError("quality report inputs or lineage are invalid") from exc
    return report


def _require_compile_result_lineage(
    completed_model_b: ModelBPlanReviewDraft,
    compile_result: PlanCompileResult,
) -> None:
    if type(compile_result) is not PlanCompileResult:
        raise TypeError("compiler must return an exact PlanCompileResult")
    expected_content_sha256 = contract_sha256(completed_model_b.content_ir)
    expected_plan_sha256 = contract_sha256(completed_model_b.reviewed_hwp_document_plan)
    if compile_result.content_ir_sha256 != expected_content_sha256:
        raise CompileSubmissionError("compiler result ContentIR lineage does not match")
    if compile_result.hwp_document_plan_sha256 != expected_plan_sha256:
        raise CompileSubmissionError("compiler result HwpDocumentPlan lineage does not match")


def _require_staged_digest(staged_output: Path, expected_sha256: str) -> None:
    try:
        size = staged_output.lstat().st_size
        actual_sha256 = sha256_bounded_regular_file(
            staged_output,
            max_bytes=size,
            label="staged HWPX",
        )
    except (OSError, SafeArtifactIOError) as exc:
        raise CompileSubmissionError("staged HWPX cannot be verified") from exc
    if actual_sha256 != expected_sha256:
        raise CompileSubmissionError("staged HWPX digest does not match compiler result")
