from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Self

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from scan2hwpx.contracts import contract_sha256
from scan2hwpx.contracts.models import StrictContractModel
from scan2hwpx.evaluation.model_b_grounding_sidecars import (
    VerifiedModelBGroundingSidecar,
    load_verified_model_b_grounding_sidecar,
)
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBPlanReviewDraft,
    start_model_b_plan_review_draft,
    verify_model_b_plan_review_draft,
)
from scan2hwpx.evaluation.review_draft import (
    CandidateContractArtifactBinding,
    CandidateReviewDraft,
    verify_candidate_review_draft,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    SafeArtifactIOError,
    publish_file_create_only,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_BoundedText = Annotated[str, Field(min_length=1, max_length=1_024)]
_MAX_CANDIDATE_REVIEW_BYTES = 64 * 1024 * 1024
_MAX_SIDECAR_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_HANDOFF_BYTES = 128 * 1024 * 1024


class CandidateToModelBHandoffError(ValueError):
    """Raised when the reviewed-content-to-Model-B boundary is not intact."""


class _StrictModel(StrictContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class _NonEligibleBoundary(_StrictModel):
    research_only: Literal[True] = True
    human_review_required: Literal[True] = True
    golden_eligible: Literal[False] = False
    training_eligible: Literal[False] = False
    release_eligible: Literal[False] = False
    rights_status: Literal["unverified"] = "unverified"
    identity_assurance: Literal["self_asserted_untrusted"] = (
        "self_asserted_untrusted"
    )


class CompletedCandidateReviewBinding(_StrictModel):
    """Exact completed CandidateReviewDraft revision consumed by the handoff."""

    schema_version: Literal["candidate-review-draft/1.0"] = (
        "candidate-review-draft/1.0"
    )
    artifact_sha256: Sha256
    status: Literal["complete"] = "complete"
    draft_revision: int = Field(ge=1)
    candidate_manifest_sha256: Sha256
    document_id: _BoundedText
    lineage_id: Sha256
    base_content_ir: CandidateContractArtifactBinding
    base_hwp_document_plan: CandidateContractArtifactBinding
    reviewed_content_ir_contract_sha256: Sha256
    reviewed_hwp_document_plan_contract_sha256: Sha256


class GroundingSidecarHandoffBinding(_StrictModel):
    """Exact verified sidecar and grounding identity consumed by the handoff."""

    schema_version: Literal["model-b-grounding-sidecars/1.0"] = (
        "model-b-grounding-sidecars/1.0"
    )
    manifest_sha256: Sha256
    candidate_manifest_sha256: Sha256
    document_id: _BoundedText
    lineage_id: Sha256
    knowledge_corpus_sha256: Sha256
    capability_profile_id: _BoundedText
    capability_profile_sha256: Sha256
    retrieval_artifact_sha256: Sha256
    grounding_evidence_contract_sha256: Sha256


class CandidateToModelBHandoff(_NonEligibleBoundary):
    """Strict audit envelope for one completed Model A review to Model B handoff."""

    schema_version: Literal["candidate-to-model-b-handoff/1.0"] = (
        "candidate-to-model-b-handoff/1.0"
    )
    artifact_role: Literal["completed_candidate_review_to_model_b_plan_review"] = (
        "completed_candidate_review_to_model_b_plan_review"
    )
    created_at: Annotated[str, Field(min_length=1, max_length=64)]
    document_id: _BoundedText
    lineage_id: Sha256
    completed_candidate_review: CompletedCandidateReviewBinding
    grounding_sidecar: GroundingSidecarHandoffBinding
    model_b_plan_review_draft: ModelBPlanReviewDraft

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 datetime") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_cross_contract_bindings(self) -> Self:
        review = self.completed_candidate_review
        sidecar = self.grounding_sidecar
        model_b = self.model_b_plan_review_draft
        if not (
            self.document_id
            == review.document_id
            == sidecar.document_id
            == model_b.document_id
        ):
            raise ValueError("handoff document identities do not match")
        if self.lineage_id != review.lineage_id or self.lineage_id != sidecar.lineage_id:
            raise ValueError("handoff lineage identities do not match")
        if review.candidate_manifest_sha256 != sidecar.candidate_manifest_sha256:
            raise ValueError("handoff candidate manifest bindings do not match")
        if (
            model_b.content_ir_contract_sha256
            != review.reviewed_content_ir_contract_sha256
        ):
            raise ValueError("Model B input is not the completed reviewed ContentIR")
        if (
            model_b.base_hwp_document_plan_contract_sha256
            != review.reviewed_hwp_document_plan_contract_sha256
        ):
            raise ValueError("Model B base plan is not the completed reviewed plan")
        if (
            model_b.grounding_evidence_contract_sha256
            != sidecar.grounding_evidence_contract_sha256
        ):
            raise ValueError("Model B grounding digest does not match the sidecar")
        evidence = model_b.grounding_evidence
        if (
            evidence.knowledge_corpus_sha256 != sidecar.knowledge_corpus_sha256
            or evidence.capability_profile_id != sidecar.capability_profile_id
            or evidence.capability_profile_sha256
            != sidecar.capability_profile_sha256
            or evidence.retrieval_artifact_sha256
            != sidecar.retrieval_artifact_sha256
        ):
            raise ValueError("Model B grounding fields do not match the sidecar")
        if model_b.base_hwp_document_plan.capability_profile_id != (
            sidecar.capability_profile_id
        ):
            raise ValueError("Model B plan capability does not match the sidecar")
        if model_b.status != "in_progress" or model_b.draft_revision != 1:
            raise ValueError("handoff must contain the initial in-progress Model B revision")
        if (
            model_b.reviewed_hwp_document_plan
            != model_b.base_hwp_document_plan
            or model_b.reviewed_hwp_document_plan_contract_sha256
            != model_b.base_hwp_document_plan_contract_sha256
        ):
            raise ValueError("handoff Model B revision must begin from the exact base plan")
        if self.created_at != model_b.updated_at:
            raise ValueError("handoff timestamp must match the initial Model B revision")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedCandidateToModelBHandoff:
    """A handoff returned only after all external sources were rebound."""

    envelope: CandidateToModelBHandoff
    artifact_sha256: str

    @property
    def model_b_plan_review_draft(self) -> ModelBPlanReviewDraft:
        return self.envelope.model_b_plan_review_draft


@dataclass(frozen=True, slots=True)
class _SourceContext:
    candidate_review: CandidateReviewDraft
    candidate_review_artifact_sha256: str
    sidecar_manifest_sha256: str
    sidecar: VerifiedModelBGroundingSidecar


def start_candidate_to_model_b_handoff(
    candidate_root: Path | str,
    completed_candidate_review_path: Path | str,
    sidecar_root: Path | str,
    knowledge_corpus_path: Path | str,
    capability_profile_path: Path | str,
    *,
    reviewer_label: str,
    output_path: Path | str,
    updated_at: datetime | None = None,
    expected_candidate_review_artifact_sha256: str | None = None,
    expected_candidate_review_revision: int | None = None,
    expected_candidate_manifest_sha256: str | None = None,
    expected_lineage_id: str | None = None,
) -> VerifiedCandidateToModelBHandoff:
    """Create one immutable handoff from completed reviewed content, never base content."""

    candidate_bundle = _required_directory(candidate_root, "candidate root")
    sidecar_bundle = _required_directory(sidecar_root, "grounding sidecar root")
    destination = _resolve_new_output(output_path)
    _require_output_outside_inputs(destination, (candidate_bundle, sidecar_bundle))

    first = _load_source_context(
        candidate_bundle,
        completed_candidate_review_path,
        sidecar_bundle,
        knowledge_corpus_path,
        capability_profile_path,
    )
    _require_expected_bindings(
        first,
        expected_candidate_review_artifact_sha256=(
            expected_candidate_review_artifact_sha256
        ),
        expected_candidate_review_revision=expected_candidate_review_revision,
        expected_candidate_manifest_sha256=expected_candidate_manifest_sha256,
        expected_lineage_id=expected_lineage_id,
    )
    timestamp = _handoff_timestamp(updated_at)
    model_b_draft = start_model_b_plan_review_draft(
        first.candidate_review.reviewed_content_ir,
        first.candidate_review.reviewed_hwp_document_plan,
        document_id=first.candidate_review.document_id,
        reviewer_label=reviewer_label,
        grounding_evidence=first.sidecar.grounding_evidence,
        updated_at=timestamp,
    )
    envelope = CandidateToModelBHandoff(
        created_at=timestamp.isoformat(),
        document_id=first.candidate_review.document_id,
        lineage_id=first.candidate_review.lineage_id,
        completed_candidate_review=_completed_review_binding(first),
        grounding_sidecar=_grounding_sidecar_binding(first),
        model_b_plan_review_draft=model_b_draft,
    )

    # Re-read and fully reverify every external boundary immediately before publish.
    second = _load_source_context(
        candidate_bundle,
        completed_candidate_review_path,
        sidecar_bundle,
        knowledge_corpus_path,
        capability_profile_path,
    )
    _require_same_source_context(first, second)
    _verify_envelope_against_context(envelope, second)
    payload = _json_bytes(envelope.model_dump(mode="json"))
    _atomic_create_handoff(destination, payload)
    return VerifiedCandidateToModelBHandoff(
        envelope=envelope,
        artifact_sha256=_sha256_bytes(payload),
    )


def reopen_candidate_to_model_b_handoff(
    handoff_path: Path | str,
    candidate_root: Path | str,
    completed_candidate_review_path: Path | str,
    sidecar_root: Path | str,
    knowledge_corpus_path: Path | str,
    capability_profile_path: Path | str,
) -> VerifiedCandidateToModelBHandoff:
    """Reopen an envelope only after rebinding its completed review and sidecar."""

    handoff_payload = _read_bounded_regular_file(
        handoff_path,
        "candidate-to-Model-B handoff",
        max_bytes=_MAX_HANDOFF_BYTES,
    )
    _assert_strict_json(handoff_payload, "candidate-to-Model-B handoff")
    try:
        envelope = CandidateToModelBHandoff.model_validate_json(
            handoff_payload,
            strict=True,
        )
    except ValidationError as exc:
        raise CandidateToModelBHandoffError(
            "candidate-to-Model-B handoff is not valid strict JSON"
        ) from exc

    context = _load_source_context(
        _required_directory(candidate_root, "candidate root"),
        completed_candidate_review_path,
        _required_directory(sidecar_root, "grounding sidecar root"),
        knowledge_corpus_path,
        capability_profile_path,
    )
    _verify_envelope_against_context(envelope, context)
    return VerifiedCandidateToModelBHandoff(
        envelope=envelope,
        artifact_sha256=_sha256_bytes(handoff_payload),
    )


def model_b_plan_review_draft_from_verified_handoff(
    verified: VerifiedCandidateToModelBHandoff,
) -> ModelBPlanReviewDraft:
    """Extract a fresh typed draft for the existing in-memory patch workflow."""

    if not isinstance(verified, VerifiedCandidateToModelBHandoff):
        raise TypeError("verified must be a VerifiedCandidateToModelBHandoff")
    try:
        return ModelBPlanReviewDraft.model_validate(
            verified.model_b_plan_review_draft.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise CandidateToModelBHandoffError(
            "verified handoff contains an invalid Model B review draft"
        ) from exc


def _load_source_context(
    candidate_root: Path,
    completed_candidate_review_path: Path | str,
    sidecar_root: Path,
    knowledge_corpus_path: Path | str,
    capability_profile_path: Path | str,
) -> _SourceContext:
    review_payload = _read_bounded_regular_file(
        completed_candidate_review_path,
        "completed candidate review",
        max_bytes=_MAX_CANDIDATE_REVIEW_BYTES,
    )
    _assert_strict_json(review_payload, "completed candidate review")
    try:
        candidate_review = CandidateReviewDraft.model_validate_json(
            review_payload,
            strict=True,
        )
    except ValidationError as exc:
        raise CandidateToModelBHandoffError(
            "completed candidate review is not valid strict JSON"
        ) from exc
    candidate_review = verify_candidate_review_draft(
        candidate_review,
        candidate_root=candidate_root,
    )
    _require_completed_candidate_review(candidate_review)

    sidecar_manifest_path = sidecar_root / "manifest.json"
    manifest_before = _read_bounded_regular_file(
        sidecar_manifest_path,
        "grounding sidecar manifest",
        max_bytes=_MAX_SIDECAR_MANIFEST_BYTES,
    )
    sidecar = load_verified_model_b_grounding_sidecar(
        candidate_root,
        sidecar_root,
        knowledge_corpus_path,
        capability_profile_path,
        lineage_id=candidate_review.lineage_id,
    )
    manifest_after = _read_bounded_regular_file(
        sidecar_manifest_path,
        "grounding sidecar manifest",
        max_bytes=_MAX_SIDECAR_MANIFEST_BYTES,
    )
    review_after = _read_bounded_regular_file(
        completed_candidate_review_path,
        "completed candidate review",
        max_bytes=_MAX_CANDIDATE_REVIEW_BYTES,
    )
    if manifest_before != manifest_after:
        raise CandidateToModelBHandoffError(
            "grounding sidecar manifest changed during handoff verification"
        )
    if review_payload != review_after:
        raise CandidateToModelBHandoffError(
            "completed candidate review changed during handoff verification"
        )
    _require_review_sidecar_match(candidate_review, sidecar)
    return _SourceContext(
        candidate_review=candidate_review,
        candidate_review_artifact_sha256=_sha256_bytes(review_payload),
        sidecar_manifest_sha256=_sha256_bytes(manifest_before),
        sidecar=sidecar,
    )


def _require_completed_candidate_review(review: CandidateReviewDraft) -> None:
    if review.status != "complete":
        raise CandidateToModelBHandoffError(
            "candidate review must be complete before Model B handoff"
        )
    pending = [
        item.issue_code
        for item in review.issue_dispositions
        if item.disposition == "pending"
    ]
    if pending:
        raise CandidateToModelBHandoffError(
            "completed candidate review still contains pending issues"
        )
    unresolved = [node.id for node in review.reviewed_content_ir.nodes if node.needs_review]
    if unresolved:
        raise CandidateToModelBHandoffError(
            "completed candidate review still contains nodes needing review"
        )


def _require_review_sidecar_match(
    review: CandidateReviewDraft,
    sidecar: VerifiedModelBGroundingSidecar,
) -> None:
    if review.document_id != sidecar.document_id:
        raise CandidateToModelBHandoffError("review and sidecar document ids do not match")
    if review.lineage_id != sidecar.lineage_id:
        raise CandidateToModelBHandoffError("review and sidecar lineage ids do not match")
    if review.candidate_manifest_sha256 != sidecar.candidate_manifest_sha256:
        raise CandidateToModelBHandoffError(
            "review and sidecar candidate manifest bindings do not match"
        )
    if contract_sha256(sidecar.content_ir) != review.base_content_ir.contract_sha256:
        raise CandidateToModelBHandoffError(
            "sidecar base ContentIR does not match the reviewed candidate binding"
        )
    if (
        contract_sha256(sidecar.hwp_document_plan)
        != review.base_hwp_document_plan.contract_sha256
    ):
        raise CandidateToModelBHandoffError(
            "sidecar base plan does not match the reviewed candidate binding"
        )
    if review.reviewed_hwp_document_plan.capability_profile_id != (
        sidecar.grounding_evidence.capability_profile_id
    ):
        raise CandidateToModelBHandoffError(
            "reviewed plan capability does not match sidecar grounding"
        )


def _completed_review_binding(context: _SourceContext) -> CompletedCandidateReviewBinding:
    review = context.candidate_review
    return CompletedCandidateReviewBinding(
        artifact_sha256=context.candidate_review_artifact_sha256,
        draft_revision=review.draft_revision,
        candidate_manifest_sha256=review.candidate_manifest_sha256,
        document_id=review.document_id,
        lineage_id=review.lineage_id,
        base_content_ir=review.base_content_ir,
        base_hwp_document_plan=review.base_hwp_document_plan,
        reviewed_content_ir_contract_sha256=(
            review.reviewed_content_ir_contract_sha256
        ),
        reviewed_hwp_document_plan_contract_sha256=(
            review.reviewed_hwp_document_plan_contract_sha256
        ),
    )


def _grounding_sidecar_binding(context: _SourceContext) -> GroundingSidecarHandoffBinding:
    sidecar = context.sidecar
    evidence = sidecar.grounding_evidence
    return GroundingSidecarHandoffBinding(
        manifest_sha256=context.sidecar_manifest_sha256,
        candidate_manifest_sha256=sidecar.candidate_manifest_sha256,
        document_id=sidecar.document_id,
        lineage_id=sidecar.lineage_id,
        knowledge_corpus_sha256=evidence.knowledge_corpus_sha256,
        capability_profile_id=evidence.capability_profile_id,
        capability_profile_sha256=evidence.capability_profile_sha256,
        retrieval_artifact_sha256=evidence.retrieval_artifact_sha256,
        grounding_evidence_contract_sha256=contract_sha256(evidence),
    )


def _verify_envelope_against_context(
    envelope: CandidateToModelBHandoff,
    context: _SourceContext,
) -> None:
    expected_review = _completed_review_binding(context)
    expected_sidecar = _grounding_sidecar_binding(context)
    if envelope.completed_candidate_review != expected_review:
        raise CandidateToModelBHandoffError(
            "handoff completed-review provenance does not match current input"
        )
    if envelope.grounding_sidecar != expected_sidecar:
        raise CandidateToModelBHandoffError(
            "handoff sidecar provenance does not match current input"
        )
    review = context.candidate_review
    try:
        verify_model_b_plan_review_draft(
            envelope.model_b_plan_review_draft,
            content_ir=review.reviewed_content_ir,
            base_plan=review.reviewed_hwp_document_plan,
            grounding_evidence=context.sidecar.grounding_evidence,
        )
    except ValueError as exc:
        raise CandidateToModelBHandoffError(
            "handoff Model B draft does not match reviewed sources"
        ) from exc


def _require_expected_bindings(
    context: _SourceContext,
    *,
    expected_candidate_review_artifact_sha256: str | None,
    expected_candidate_review_revision: int | None,
    expected_candidate_manifest_sha256: str | None,
    expected_lineage_id: str | None,
) -> None:
    review = context.candidate_review
    checks = (
        (
            expected_candidate_review_artifact_sha256,
            context.candidate_review_artifact_sha256,
            "candidate review artifact SHA-256",
        ),
        (
            expected_candidate_review_revision,
            review.draft_revision,
            "candidate review revision",
        ),
        (
            expected_candidate_manifest_sha256,
            review.candidate_manifest_sha256,
            "candidate manifest SHA-256",
        ),
        (expected_lineage_id, review.lineage_id, "candidate lineage"),
    )
    for expected, actual, label in checks:
        if expected is not None and expected != actual:
            raise CandidateToModelBHandoffError(f"expected {label} does not match")


def _require_same_source_context(first: _SourceContext, second: _SourceContext) -> None:
    if first != second:
        raise CandidateToModelBHandoffError(
            "handoff inputs changed before immutable output publication"
        )


def _handoff_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("updated_at must include a timezone")
    return timestamp


def _required_directory(path: Path | str, label: str) -> Path:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
        _require_directory_non_reparse(metadata, label)
        resolved = candidate.resolve(strict=True)
        _require_directory_non_reparse(resolved.lstat(), label)
    except CandidateToModelBHandoffError:
        raise
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise CandidateToModelBHandoffError(f"{label} is missing or unreadable") from exc
    return resolved


def _read_bounded_regular_file(
    path: Path | str,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    source = Path(path)
    descriptor: int | None = None
    try:
        path_metadata = source.lstat()
        _require_regular_non_reparse_file(path_metadata, label)
        if path_metadata.st_size > max_bytes:
            raise CandidateToModelBHandoffError(f"{label} exceeds size limit")
        flags = os.O_RDONLY
        for flag_name in (
            "O_BINARY",
            "O_CLOEXEC",
            "O_NOINHERIT",
            "O_NOFOLLOW",
            "O_NONBLOCK",
        ):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(source, flags)
        opened_metadata = os.fstat(descriptor)
        _require_regular_non_reparse_file(opened_metadata, label)
        if opened_metadata.st_size > max_bytes:
            raise CandidateToModelBHandoffError(f"{label} exceeds size limit")
        _require_same_file_snapshot(path_metadata, opened_metadata, label)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = _read_stream_bounded(stream, label, max_bytes=max_bytes)
            read_metadata = os.fstat(stream.fileno())
        _require_same_file_snapshot(opened_metadata, read_metadata, label)
        final_metadata = source.lstat()
        _require_regular_non_reparse_file(final_metadata, label)
        _require_same_file_snapshot(read_metadata, final_metadata, label)
    except CandidateToModelBHandoffError:
        raise
    except OSError as exc:
        raise CandidateToModelBHandoffError(f"{label} is missing or unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload


def _read_stream_bounded(
    stream: BinaryIO,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    payload = bytearray()
    while True:
        remaining = max_bytes + 1 - len(payload)
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise CandidateToModelBHandoffError(f"{label} exceeds size limit")
    return bytes(payload)


def _require_regular_non_reparse_file(metadata: os.stat_result, label: str) -> None:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise CandidateToModelBHandoffError(
            f"{label} must be a regular non-symlink file"
        )


def _require_directory_non_reparse(metadata: os.stat_result, label: str) -> None:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise CandidateToModelBHandoffError(
            f"{label} must be a regular non-symlink directory"
        )


def _require_same_file_snapshot(
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    if not (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    ):
        raise CandidateToModelBHandoffError(f"{label} changed while it was read")


def _assert_strict_json(payload: bytes, label: str) -> None:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite number: {value}")

    try:
        json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise CandidateToModelBHandoffError(
            f"{label} must be valid strict UTF-8 JSON"
        ) from exc


def _resolve_new_output(path: Path | str) -> Path:
    requested = Path(path)
    if requested.name in {"", ".", ".."}:
        raise ValueError("handoff output must name a file")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("handoff output already exists")
    destination = requested.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("handoff output already exists")
    return destination


def _require_output_outside_inputs(destination: Path, roots: tuple[Path, ...]) -> None:
    if any(destination == root or destination.is_relative_to(root) for root in roots):
        raise ValueError("handoff output must be outside candidate and sidecar bundles")


def _atomic_create_handoff(destination: Path, payload: bytes) -> None:
    if len(payload) > _MAX_HANDOFF_BYTES:
        raise CandidateToModelBHandoffError("handoff artifact exceeds size limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _required_directory(destination.parent, "handoff output parent")
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("handoff output already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            publish_file_create_only(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError("handoff output already exists") from exc
        except SafeArtifactIOError as exc:
            raise CandidateToModelBHandoffError(
                "handoff output changed during publication"
            ) from exc
        committed = True
    finally:
        if not committed:
            temporary.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
