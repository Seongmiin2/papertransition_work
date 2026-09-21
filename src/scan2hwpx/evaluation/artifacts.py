from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .dataset import (
    ARTIFACT_REF_PATTERN,
    GoldenDatasetManifest,
    validate_golden_dataset_manifest,
)
from .safe_artifact_io import SafeArtifactIOError
from .safe_artifact_io import read_bounded_regular_file as _read_bounded_regular_file
from .strict_json import StrictJSONError, require_strict_json_bytes

_ARTIFACT_REF = re.compile(ARTIFACT_REF_PATTERN)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_ANNOTATION_BYTES = 64 * 1024 * 1024
_MAX_VERIFICATION_ARTIFACT_BYTES = 1024 * 1024
_MAX_TOTAL_VERIFIED_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_ARTIFACT_REF_LENGTH = 1024
_MAX_ATTESTATION_JSON_DEPTH = 16
_MAX_ATTESTATION_JSON_NODES = 256
_UNSIGNED_ASSURANCE: Literal["unsigned_manifest_consistency_only"] = (
    "unsigned_manifest_consistency_only"
)


class ArtifactVerificationError(ValueError):
    """Raised when a manifest artifact cannot be trusted."""


class VerificationArtifactPayload(BaseModel):
    """Minimal unsigned approval claim; parsing does not authenticate its author."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["1.0"]
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    source_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    annotation_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    reviewer_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    verified_at: str = Field(min_length=1, max_length=64)
    decision: Literal["approved"]
    usage_rights_verified: Literal[True]

    @field_validator("source_sha256", "annotation_sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()

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


@dataclass(frozen=True, slots=True)
class VerifiedArtifactBinding:
    artifact_ref: str
    resolved_path: Path
    sha256: str
    payload: bytes = field(repr=False)

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True, slots=True)
class VerifiedDocumentArtifacts:
    document_id: str
    annotation: VerifiedArtifactBinding
    verification_attestation: VerifiedArtifactBinding
    reviewer_identity_assurance: Literal["unsigned_manifest_consistency_only"] = (
        field(default=_UNSIGNED_ASSURANCE, init=False)
    )
    usage_rights_assurance: Literal["unsigned_manifest_consistency_only"] = (
        field(default=_UNSIGNED_ASSURANCE, init=False)
    )


def resolve_workspace_artifact(
    workspace_root: Path | str,
    artifact_ref: str,
) -> Path:
    """Resolve an artifact reference without allowing it to escape the workspace.

    Every component after ``artifact://`` must be a non-reparse path beneath a
    non-reparse workspace root. URI decoding is intentionally unsupported, so encoded
    traversal and platform-specific absolute paths cannot acquire filesystem semantics.

    This returns a location snapshot, not a capability. Security-sensitive callers
    must consume verified bytes returned by :func:`verify_golden_dataset_artifacts`
    instead of reopening this path.
    """
    root = _resolve_workspace_root(workspace_root)
    return _resolve_artifact_from_root(root, artifact_ref)


def verify_golden_dataset_artifacts(
    manifest: GoldenDatasetManifest,
    *,
    workspace_root: Path | str,
) -> tuple[VerifiedDocumentArtifacts, ...]:
    """Verify all annotation and human-attestation bytes declared by a manifest.

    Returned payloads are the immutable bytes used for SHA-256 calculation, allowing
    downstream evaluators to parse the verified annotation without reopening a path.
    Annotation schema and page-count validation deliberately remain evaluator concerns.

    This proves only unsigned artifact/manifest consistency for immutable returned
    bytes. It does not authenticate reviewer identity, establish usage rights, or
    verify a digital signature, and cannot authorize a production model promotion.
    """
    if not isinstance(manifest, GoldenDatasetManifest):
        raise TypeError("manifest must be a GoldenDatasetManifest")
    try:
        manifest = GoldenDatasetManifest.model_validate(
            manifest.model_dump(mode="python"),
            strict=True,
        )
        validate_golden_dataset_manifest(manifest)
    except ValueError as exc:
        raise ArtifactVerificationError(str(exc)) from exc

    root = _resolve_workspace_root(workspace_root)
    seen_refs: dict[str, str] = {}
    seen_paths: dict[Path, str] = {}
    seen_file_ids: dict[tuple[int, int], str] = {}
    verified_documents: list[VerifiedDocumentArtifacts] = []
    total_verified_bytes = 0

    for document in manifest.documents:
        if not document.verified:
            raise ArtifactVerificationError(
                f"document {document.id!r} is not marked as human verified"
            )
        if not document.usage_rights_verified:
            raise ArtifactVerificationError(
                f"document {document.id!r} does not have verified usage rights"
            )

        annotation = _verify_binding(
            root=root,
            document_id=document.id,
            role="annotation",
            artifact_ref=document.annotation_artifact_ref,
            expected_sha256=document.annotation_sha256,
            seen_refs=seen_refs,
            seen_paths=seen_paths,
            seen_file_ids=seen_file_ids,
            max_bytes=_MAX_ANNOTATION_BYTES,
            remaining_total_bytes=(
                _MAX_TOTAL_VERIFIED_ARTIFACT_BYTES - total_verified_bytes
            ),
        )
        total_verified_bytes += annotation.size_bytes
        attestation = _verify_binding(
            root=root,
            document_id=document.id,
            role="verification_attestation",
            artifact_ref=(
                document.verification_attestation.verification_artifact_ref
            ),
            expected_sha256=(
                document.verification_attestation.verification_artifact_sha256
            ),
            seen_refs=seen_refs,
            seen_paths=seen_paths,
            seen_file_ids=seen_file_ids,
            max_bytes=_MAX_VERIFICATION_ARTIFACT_BYTES,
            remaining_total_bytes=(
                _MAX_TOTAL_VERIFIED_ARTIFACT_BYTES - total_verified_bytes
            ),
        )
        total_verified_bytes += attestation.size_bytes
        _validate_attestation_payload(
            payload=attestation.payload,
            document_id=document.id,
            source_sha256=document.source_sha256,
            annotation_sha256=document.annotation_sha256,
            reviewer_id=document.verification_attestation.reviewer_id,
            verified_at=document.verification_attestation.verified_at,
        )
        verified_documents.append(
            VerifiedDocumentArtifacts(
                document_id=document.id,
                annotation=annotation,
                verification_attestation=attestation,
            )
        )

    return tuple(verified_documents)


def _resolve_workspace_root(workspace_root: Path | str) -> Path:
    candidate = Path(workspace_root).absolute()
    try:
        before = candidate.lstat()
        _require_non_reparse_directory(before, "workspace root")
        root = candidate.resolve(strict=True)
        after = candidate.lstat()
    except (OSError, RuntimeError) as exc:
        raise ArtifactVerificationError("workspace root does not exist or cannot be resolved") from exc
    _require_same_snapshot(before, after, "workspace root")
    return root


def _resolve_artifact_from_root(root: Path, artifact_ref: str) -> Path:
    resolved_path, _ = _capture_artifact_path(root, artifact_ref)
    return resolved_path


def _verify_binding(
    *,
    root: Path,
    document_id: str,
    role: str,
    artifact_ref: str,
    expected_sha256: str,
    seen_refs: dict[str, str],
    seen_paths: dict[Path, str],
    seen_file_ids: dict[tuple[int, int], str],
    max_bytes: int,
    remaining_total_bytes: int,
) -> VerifiedArtifactBinding:
    label = f"{document_id}:{role}"
    ref_key = artifact_ref.casefold()
    previous_ref = seen_refs.get(ref_key)
    if previous_ref is not None:
        raise ArtifactVerificationError(
            f"duplicate artifact binding between {previous_ref} and {label}"
        )
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ArtifactVerificationError(f"invalid expected SHA-256 for {label}")

    if remaining_total_bytes < 0:
        raise ArtifactVerificationError("verified artifact total byte limit exceeded")
    resolved_path, payload, actual_sha256, file_id = _read_artifact(
        root,
        artifact_ref,
        max_bytes=max_bytes,
        remaining_total_bytes=remaining_total_bytes,
    )
    previous_path = seen_paths.get(resolved_path)
    if previous_path is not None:
        raise ArtifactVerificationError(
            f"duplicate artifact binding between {previous_path} and {label}"
        )

    if actual_sha256 != expected_sha256.lower():
        raise ArtifactVerificationError(f"artifact SHA-256 mismatch for {label}")
    previous_file = seen_file_ids.get(file_id)
    if previous_file is not None:
        raise ArtifactVerificationError(
            f"duplicate artifact binding between {previous_file} and {label}"
        )

    seen_refs[ref_key] = label
    seen_paths[resolved_path] = label
    seen_file_ids[file_id] = label
    return VerifiedArtifactBinding(
        artifact_ref=artifact_ref,
        resolved_path=resolved_path,
        sha256=actual_sha256,
        payload=payload,
    )


def _read_artifact(
    root: Path,
    artifact_ref: str,
    *,
    max_bytes: int,
    remaining_total_bytes: int,
) -> tuple[Path, bytes, str, tuple[int, int]]:
    resolved_path, path_snapshot = _capture_artifact_path(root, artifact_ref)
    file_metadata = path_snapshot[-1][1]
    if file_metadata.st_size > max_bytes:
        raise ArtifactVerificationError(f"artifact exceeds size limit: {artifact_ref}")
    if file_metadata.st_size > remaining_total_bytes:
        raise ArtifactVerificationError("verified artifact total byte limit exceeded")
    if file_metadata.st_ino == 0:
        raise ArtifactVerificationError(f"artifact file identity is unavailable: {artifact_ref}")
    if file_metadata.st_nlink != 1:
        raise ArtifactVerificationError(
            f"duplicate artifact binding or hard-link alias: {artifact_ref}"
        )
    try:
        payload = _read_bounded_regular_file(
            resolved_path,
            max_bytes=max_bytes,
            label=f"artifact {artifact_ref}",
        )
    except SafeArtifactIOError as exc:
        raise ArtifactVerificationError(str(exc)) from exc
    _require_unchanged_path_snapshot(path_snapshot, artifact_ref)
    resolved_after, _ = _capture_artifact_path(root, artifact_ref)
    if resolved_after != resolved_path:
        raise ArtifactVerificationError(f"artifact changed while it was read: {artifact_ref}")
    file_id = (file_metadata.st_dev, file_metadata.st_ino)
    return resolved_path, payload, hashlib.sha256(payload).hexdigest(), file_id


def _validate_attestation_payload(
    *,
    payload: bytes,
    document_id: str,
    source_sha256: str,
    annotation_sha256: str,
    reviewer_id: str,
    verified_at: str,
) -> None:
    try:
        require_strict_json_bytes(
            payload,
            max_depth=_MAX_ATTESTATION_JSON_DEPTH,
            label=f"verification artifact for {document_id!r}",
            max_nodes=_MAX_ATTESTATION_JSON_NODES,
        )
        record = VerificationArtifactPayload.model_validate_json(payload, strict=True)
    except (StrictJSONError, ValidationError):
        raise ArtifactVerificationError(
            f"verification artifact for {document_id!r} is not valid strict JSON"
        ) from None

    expected = {
        "document_id": document_id,
        "source_sha256": source_sha256,
        "annotation_sha256": annotation_sha256,
        "reviewer_id": reviewer_id,
        "verified_at": verified_at,
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(record, field_name) != expected_value
    ]
    if mismatches:
        raise ArtifactVerificationError(
            f"verification artifact for {document_id!r} disagrees with manifest: "
            + ", ".join(mismatches)
        )


def _capture_artifact_path(
    root: Path,
    artifact_ref: str,
) -> tuple[Path, tuple[tuple[Path, os.stat_result], ...]]:
    if (
        not isinstance(artifact_ref, str)
        or len(artifact_ref) > _MAX_ARTIFACT_REF_LENGTH
        or _ARTIFACT_REF.fullmatch(artifact_ref) is None
    ):
        raise ArtifactVerificationError("invalid artifact reference")
    relative_parts = artifact_ref.removeprefix("artifact://").split("/")
    candidate = root.joinpath(*relative_parts)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise ArtifactVerificationError("workspace root changed during verification") from exc
    _require_non_reparse_directory(root_metadata, "workspace root")
    snapshots: list[tuple[Path, os.stat_result]] = [(root, root_metadata)]
    current = root
    for index, part in enumerate(relative_parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise ArtifactVerificationError(f"artifact is missing: {artifact_ref}") from exc
        except OSError as exc:
            raise ArtifactVerificationError(
                f"artifact cannot be resolved: {artifact_ref}"
            ) from exc
        if _is_reparse(metadata):
            raise ArtifactVerificationError(
                "artifact escapes workspace root or uses a symlink/reparse point: "
                f"{artifact_ref}"
            )
        if index < len(relative_parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVerificationError(
                    f"artifact parent is not a directory: {artifact_ref}"
                )
        elif not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVerificationError(
                f"artifact is not a regular file: {artifact_ref}"
            )
        snapshots.append((current, metadata))
    try:
        resolved_path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtifactVerificationError(
            f"artifact cannot be resolved: {artifact_ref}"
        ) from exc
    if not resolved_path.is_relative_to(root):
        raise ArtifactVerificationError(f"artifact escapes workspace root: {artifact_ref}")
    snapshot = tuple(snapshots)
    _require_unchanged_path_snapshot(snapshot, artifact_ref)
    return resolved_path, snapshot


def _require_unchanged_path_snapshot(
    snapshots: tuple[tuple[Path, os.stat_result], ...],
    artifact_ref: str,
) -> None:
    for path, before in snapshots:
        try:
            after = path.lstat()
        except OSError as exc:
            raise ArtifactVerificationError(
                f"artifact changed while it was read: {artifact_ref}"
            ) from exc
        if _is_reparse(after) or not _same_snapshot(before, after):
            raise ArtifactVerificationError(
                f"artifact changed while it was read: {artifact_ref}"
            )


def _require_non_reparse_directory(metadata: os.stat_result, label: str) -> None:
    if _is_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactVerificationError(f"{label} must be a non-reparse directory")


def _require_same_snapshot(
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    if not _same_snapshot(before, after):
        raise ArtifactVerificationError(f"{label} changed while it was resolved")


def _same_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & reparse_attribute
    )
