from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .gates import DatasetEvidence
from .safe_artifact_io import read_bounded_regular_file
from .strict_json import require_strict_json_bytes

Split = Literal["train", "validation", "test"]
Capture = Literal["real_scan", "born_digital", "synthetic"]

ARTIFACT_REF_PATTERN = (
    r"^artifact://[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?)*$"
)
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 300_000
_MAX_DOCUMENTS = 10_000
_MAX_TOTAL_PAGES = 10_000_000


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class VerificationAttestation(_StrictModel):
    """Unsigned claims whose consistency is checked against an artifact payload."""

    reviewer_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    verified_at: str = Field(min_length=1, max_length=64)
    verification_artifact_ref: str = Field(
        max_length=1024,
        pattern=ARTIFACT_REF_PATTERN,
    )
    verification_artifact_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

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

    @field_validator("verification_artifact_sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class GoldenDatasetDocument(_StrictModel):
    """Dataset claims; boolean flags alone do not authenticate people or rights."""

    id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    lineage_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    source_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    template_family: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    split: Split
    capture: Capture
    page_count: int = Field(gt=0, le=100_000)
    annotation_artifact_ref: str = Field(
        max_length=1024,
        pattern=ARTIFACT_REF_PATTERN,
    )
    annotation_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    verified: bool
    usage_rights_verified: bool
    verification_attestation: VerificationAttestation

    @field_validator("source_sha256", "annotation_sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class GoldenDatasetManifest(_StrictModel):
    """Strict metadata manifest, not an artifact resolver or checksum verifier.

    Production callers must still resolve every artifact reference and verify its
    bytes, declared SHA-256, annotation schema, and source page count.
    """

    schema_version: Literal["1.0"]
    example_only: bool
    documents: tuple[GoldenDatasetDocument, ...] = Field(
        min_length=1,
        max_length=_MAX_DOCUMENTS,
    )

    @model_validator(mode="after")
    def validate_documents(self) -> GoldenDatasetManifest:
        if len(self.documents) > _MAX_DOCUMENTS:
            raise ValueError("golden dataset document count exceeds the limit")
        if sum(document.page_count for document in self.documents) > _MAX_TOTAL_PAGES:
            raise ValueError("golden dataset total page count exceeds the limit")

        duplicate_ids = _duplicates(document.id for document in self.documents)
        if duplicate_ids:
            raise ValueError("duplicate document ids: " + ", ".join(duplicate_ids))

        duplicate_sources = _duplicates(document.source_sha256 for document in self.documents)
        if duplicate_sources:
            raise ValueError("duplicate source sha256: " + ", ".join(duplicate_sources))

        duplicate_annotation_refs = _duplicates(
            document.annotation_artifact_ref for document in self.documents
        )
        if duplicate_annotation_refs:
            raise ValueError(
                "duplicate annotation artifact refs: "
                + ", ".join(duplicate_annotation_refs)
            )

        duplicate_annotation_hashes = _duplicates(
            document.annotation_sha256 for document in self.documents
        )
        if duplicate_annotation_hashes:
            raise ValueError(
                "duplicate annotation sha256: " + ", ".join(duplicate_annotation_hashes)
            )

        duplicate_verification_refs = _duplicates(
            document.verification_attestation.verification_artifact_ref
            for document in self.documents
        )
        if duplicate_verification_refs:
            raise ValueError(
                "duplicate verification artifact refs: "
                + ", ".join(duplicate_verification_refs)
            )

        duplicate_verification_hashes = _duplicates(
            document.verification_attestation.verification_artifact_sha256
            for document in self.documents
        )
        if duplicate_verification_hashes:
            raise ValueError(
                "duplicate verification artifact sha256: "
                + ", ".join(duplicate_verification_hashes)
            )

        unverified = sorted(document.id for document in self.documents if not document.verified)
        if unverified:
            raise ValueError("unverified documents: " + ", ".join(unverified))

        rights_unverified = sorted(
            document.id for document in self.documents if not document.usage_rights_verified
        )
        if rights_unverified:
            raise ValueError("usage rights not verified: " + ", ".join(rights_unverified))

        lineage_splits: dict[str, set[Split]] = {}
        for document in self.documents:
            lineage_splits.setdefault(document.lineage_id.casefold(), set()).add(document.split)
        leaked_lineages = sorted(
            lineage_id for lineage_id, splits in lineage_splits.items() if len(splits) > 1
        )
        if leaked_lineages:
            raise ValueError(
                "lineage leakage across splits: " + ", ".join(leaked_lineages)
            )

        test_families = {
            document.template_family.casefold()
            for document in self.documents
            if document.split == "test"
        }
        development_families = {
            document.template_family.casefold()
            for document in self.documents
            if document.split in {"train", "validation"}
        }
        leaked = sorted(test_families & development_families)
        if leaked:
            raise ValueError("template family leakage into test: " + ", ".join(leaked))
        return self


def load_golden_dataset_manifest(
    path: Path | str,
    *,
    allow_example_only: bool = False,
) -> GoldenDatasetManifest:
    """Validate manifest metadata, rejecting example fixtures by default.

    This does not resolve artifacts or verify their bytes against declared checksums;
    the production release pipeline must perform those I/O checks separately.
    """
    payload = read_bounded_regular_file(
        Path(path),
        max_bytes=_MAX_MANIFEST_BYTES,
        label="golden dataset manifest",
    )
    require_strict_json_bytes(
        payload,
        max_depth=_MAX_JSON_DEPTH,
        label="golden dataset manifest",
        max_nodes=_MAX_JSON_NODES,
    )
    manifest = GoldenDatasetManifest.model_validate_json(payload, strict=True)
    validate_golden_dataset_manifest(manifest, allow_example_only=allow_example_only)
    return manifest


def summarize_golden_dataset_manifest(manifest: GoldenDatasetManifest) -> DatasetEvidence:
    """Calculate the dataset evidence consumed by the model release gates."""
    if not isinstance(manifest, GoldenDatasetManifest):
        raise TypeError("manifest must be a GoldenDatasetManifest")
    real_scan_test = [
        document
        for document in manifest.documents
        if document.split == "test" and document.capture == "real_scan"
    ]
    lineage_splits: dict[str, set[Split]] = {}
    for document in manifest.documents:
        lineage_splits.setdefault(document.lineage_id.casefold(), set()).add(document.split)
    test_families = {
        document.template_family.casefold()
        for document in manifest.documents
        if document.split == "test"
    }
    development_families = {
        document.template_family.casefold()
        for document in manifest.documents
        if document.split in {"train", "validation"}
    }
    return DatasetEvidence(
        document_level_split=all(len(splits) == 1 for splits in lineage_splits.values()),
        template_family_holdout=not bool(test_families & development_families),
        real_scan_test_documents=len(real_scan_test),
        real_scan_test_pages=sum(document.page_count for document in real_scan_test),
    )


def validate_golden_dataset_manifest(
    manifest: GoldenDatasetManifest,
    *,
    allow_example_only: bool = False,
) -> DatasetEvidence:
    """Validate production eligibility and return release-gate evidence."""
    if not isinstance(manifest, GoldenDatasetManifest):
        raise TypeError("manifest must be a GoldenDatasetManifest")
    validated_manifest = GoldenDatasetManifest.model_validate(
        manifest.model_dump(mode="python"),
        strict=True,
    )
    if validated_manifest.example_only and not allow_example_only:
        raise ValueError("example-only manifest cannot be used as production evidence")
    return summarize_golden_dataset_manifest(validated_manifest)


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    return sorted(duplicates)
