from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]
Split = Literal["train", "validation", "test"]
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_LABEL_BYTES = 64 * 1024 * 1024
_MAX_CHARACTER_DICT_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_TRAINING_IMAGES = 2_000_000
_ATTESTOR_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_EXPECTED_ARTIFACTS = {
    "train_label": "train.txt",
    "validation_label": "validation.txt",
    "test_label": "test.txt",
    "character_dict": "korean_exam_dict.txt",
}


class OcrTrainingRightsError(ValueError):
    """Raised before training when dataset rights or lineage are not proven."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class OcrTrainingRightsAttestation(_StrictModel):
    attestor_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_ATTESTOR_ID_PATTERN,
    )
    attested_at: str = Field(min_length=1, max_length=64)
    basis: Literal[
        "owner",
        "licensed",
        "public_domain",
        "authorized_by_rights_holder",
    ]
    evidence_artifact_ref: str = Field(
        pattern=r"^artifact://[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
    )
    evidence_sha256: Sha256

    @field_validator("attested_at")
    @classmethod
    def require_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("attested_at must be an ISO-8601 datetime") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("attested_at must include a timezone")
        return value

    @field_validator("evidence_sha256")
    @classmethod
    def normalize_evidence_sha256(cls, value: str) -> str:
        return value.casefold()


class OcrTrainingRightsDocument(_StrictModel):
    source_sha256: Sha256
    split: Split
    template_family: str = Field(min_length=1, max_length=256)
    usage_scope: Literal["ocr_model_training_and_evaluation"]
    rights_verified: Literal[True]
    attestation: OcrTrainingRightsAttestation

    @field_validator("source_sha256")
    @classmethod
    def normalize_source_sha256(cls, value: str) -> str:
        return value.casefold()

    @field_validator("template_family")
    @classmethod
    def normalize_template_family(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value.strip()).casefold()
        if (
            not normalized
            or "/" in normalized
            or "\\" in normalized
            or any(
                unicodedata.category(character) in {"Cc", "Cf"}
                for character in normalized
            )
        ):
            raise ValueError("template_family must be a bounded label, not a path")
        return normalized


class OcrTrainingRightsManifest(_StrictModel):
    schema_version: Literal["ocr-training-rights/1.0"]
    example_only: bool
    documents: tuple[OcrTrainingRightsDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory_and_split_isolation(self) -> Self:
        hashes = [document.source_sha256 for document in self.documents]
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate source_sha256 in OCR training rights manifest")
        present_splits = {document.split for document in self.documents}
        missing_splits = sorted({"train", "validation", "test"} - present_splits)
        if missing_splits:
            raise ValueError(
                "OCR training rights manifest is missing splits: "
                + ", ".join(missing_splits)
            )
        family_splits: dict[str, set[Split]] = {}
        for document in self.documents:
            family_splits.setdefault(document.template_family, set()).add(document.split)
        leaked = sorted(
            family for family, splits in family_splits.items() if len(splits) > 1
        )
        if leaked:
            raise ValueError(
                "template family leakage across OCR splits: " + ", ".join(leaked)
            )
        return self


@dataclass(frozen=True)
class AuthenticatedAttestorContext:
    """Exact report trust and identities established by an authenticated caller.

    This is an external trust input, not local proof that source documents were
    reproducibly rendered and cropped into the registered images.
    """

    attestor_ids: frozenset[str]
    trusted_dataset_report_sha256: str
    derivation_attestor_id: str
    assurance: Literal["authenticated_external_context"] = "authenticated_external_context"

    def __post_init__(self) -> None:
        if not isinstance(self.attestor_ids, frozenset) or not self.attestor_ids:
            raise ValueError(
                "authenticated attestor context identities must be a non-empty frozenset"
            )
        if self.assurance != "authenticated_external_context":
            raise ValueError("authenticated attestor context has an invalid assurance")
        for attestor_id in self.attestor_ids:
            if not isinstance(attestor_id, str) or not re.fullmatch(
                _ATTESTOR_ID_PATTERN, attestor_id
            ):
                raise ValueError("authenticated attestor context contains an invalid identity")
        if (
            not isinstance(self.trusted_dataset_report_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.trusted_dataset_report_sha256) is None
        ):
            raise ValueError("authenticated context must trust one dataset report SHA-256")
        if self.derivation_attestor_id not in self.attestor_ids:
            raise ValueError("dataset derivation attestor must be authenticated by the caller")


@dataclass(frozen=True)
class OcrTrainingRightsEvidence:
    rights_manifest_sha256: str
    dataset_report_sha256: str
    document_count: int
    identity_assurance: Literal["authenticated_external_context"]
    evidence_artifacts_sha256: str
    training_images_sha256: str
    derivation_assurance: Literal["authenticated_external_report_attestation"]
    derivation_verification_scope: Literal["report_digest_and_artifact_integrity_only"]
    derivation_attestor_id: str


def load_ocr_training_rights_manifest(
    path: Path | str,
    *,
    allow_example_only: bool = False,
) -> OcrTrainingRightsManifest:
    manifest_path = Path(path)
    payload = _read_bounded_regular_file(
        manifest_path,
        "rights manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    return _parse_rights_manifest(payload, allow_example_only=allow_example_only)


def _parse_rights_manifest(
    payload: bytes,
    *,
    allow_example_only: bool,
) -> OcrTrainingRightsManifest:
    try:
        strict_json = _parse_strict_json(payload, "rights manifest")
        manifest = OcrTrainingRightsManifest.model_validate_json(
            json.dumps(strict_json, ensure_ascii=False), strict=True
        )
    except ValueError as exc:
        raise OcrTrainingRightsError("OCR training rights manifest is invalid") from exc
    if manifest.example_only and not allow_example_only:
        raise OcrTrainingRightsError(
            "example-only rights manifest cannot authorize OCR training"
        )
    return manifest


def verify_ocr_training_dataset_rights(
    dataset_root: Path | str,
    rights_manifest_path: Path | str,
    *,
    train_label: Path | str,
    validation_label: Path | str,
    character_dict: Path | str,
    evidence_root: Path | str,
    attestor_context: AuthenticatedAttestorContext | None,
) -> OcrTrainingRightsEvidence:
    if attestor_context is None:
        raise OcrTrainingRightsError(
            "OCR training requires identities from an authenticated external context"
        )
    root = _required_directory(Path(dataset_root), "dataset root")
    manifest_path = Path(rights_manifest_path)
    manifest_payload = _read_bounded_regular_file(
        manifest_path,
        "rights manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = _parse_rights_manifest(manifest_payload, allow_example_only=False)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    evidence_directory = _required_directory(Path(evidence_root), "rights evidence root")
    evidence_artifacts_sha256 = _verify_attestation_evidence(
        manifest,
        evidence_directory,
    )
    unknown_attestors = sorted(
        {
            document.attestation.attestor_id
            for document in manifest.documents
            if document.attestation.attestor_id not in attestor_context.attestor_ids
        }
    )
    if unknown_attestors:
        raise OcrTrainingRightsError(
            "OCR rights attestor was not authenticated by the caller: "
            + ", ".join(unknown_attestors)
        )
    rights_attestors = {
        document.attestation.attestor_id for document in manifest.documents
    }
    if attestor_context.derivation_attestor_id in rights_attestors:
        raise OcrTrainingRightsError(
            "OCR dataset derivation requires an independently authenticated attestor"
        )
    report_path = root / "dataset_report.json"
    report_payload = _read_bounded_regular_file(
        report_path,
        "dataset report",
        max_bytes=_MAX_REPORT_BYTES,
    )
    report = _parse_strict_json(report_payload, "dataset report")
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    if report_sha256 != attestor_context.trusted_dataset_report_sha256:
        raise OcrTrainingRightsError(
            "OCR dataset report was not trusted by the authenticated caller"
        )
    if not isinstance(report, dict) or report.get("schema_version") != "1.1":
        raise OcrTrainingRightsError("OCR dataset report is invalid")

    split_policy = report.get("split_policy")
    if (
        not isinstance(split_policy, dict)
        or split_policy.get("frozen_manifest_sha256") != manifest_sha256
        or split_policy.get("template_family_contract") != "explicit_manifest_only"
    ):
        raise OcrTrainingRightsError(
            "OCR dataset report is not bound to the supplied rights manifest"
        )

    report_documents = _report_document_inventory(report.get("documents"))
    rights_documents = {
        document.source_sha256: (document.split, document.template_family)
        for document in manifest.documents
    }
    if report_documents != rights_documents:
        raise OcrTrainingRightsError(
            "OCR dataset document inventory does not match the rights manifest"
        )

    requested_artifacts = {
        "train_label": Path(train_label),
        "validation_label": Path(validation_label),
        "test_label": root / "test.txt",
        "character_dict": Path(character_dict),
    }
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_EXPECTED_ARTIFACTS):
        raise OcrTrainingRightsError("OCR dataset artifact bindings are missing")
    artifact_payloads: dict[str, bytes] = {}
    for key, expected_name in _EXPECTED_ARTIFACTS.items():
        binding = artifacts.get(key)
        expected_path = root / expected_name
        if _normalized_path(requested_artifacts[key]) != _normalized_path(expected_path):
            raise OcrTrainingRightsError(
                f"OCR training {key} is not the dataset-report registered artifact"
            )
        max_bytes = (
            _MAX_CHARACTER_DICT_BYTES if key == "character_dict" else _MAX_LABEL_BYTES
        )
        artifact_payload = _read_bounded_regular_file(
            expected_path,
            f"dataset {key}",
            max_bytes=max_bytes,
        )
        artifact_payloads[key] = artifact_payload
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_name
            or not isinstance(binding.get("sha256"), str)
            or binding["sha256"] != hashlib.sha256(artifact_payload).hexdigest()
        ):
            raise OcrTrainingRightsError(f"OCR dataset {key} digest mismatch")

    training_image_count, training_images_sha256 = _verify_training_images(
        root,
        artifact_payloads,
        report_documents,
    )
    image_inventory = report.get("image_inventory")
    if (
        not isinstance(image_inventory, dict)
        or set(image_inventory) != {"digest_definition", "image_count", "sha256"}
        or image_inventory.get("digest_definition")
        != "split-label-ref-source-lineage-and-image-bytes/v1"
        or image_inventory.get("image_count") != training_image_count
        or image_inventory.get("sha256") != training_images_sha256
    ):
        raise OcrTrainingRightsError("OCR dataset image inventory digest mismatch")

    return OcrTrainingRightsEvidence(
        rights_manifest_sha256=manifest_sha256,
        dataset_report_sha256=report_sha256,
        document_count=len(manifest.documents),
        identity_assurance=attestor_context.assurance,
        evidence_artifacts_sha256=evidence_artifacts_sha256,
        training_images_sha256=training_images_sha256,
        derivation_assurance="authenticated_external_report_attestation",
        derivation_verification_scope="report_digest_and_artifact_integrity_only",
        derivation_attestor_id=attestor_context.derivation_attestor_id,
    )


def _report_document_inventory(value: object) -> dict[str, tuple[Split, str]]:
    if not isinstance(value, list):
        raise OcrTrainingRightsError("OCR dataset report document inventory is missing")
    result: dict[str, tuple[Split, str]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise OcrTrainingRightsError("OCR dataset report document is invalid")
        digest = item.get("sha256")
        split = item.get("split")
        family = item.get("template_family")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or split not in {"train", "validation", "test"}
            or not isinstance(family, str)
        ):
            raise OcrTrainingRightsError(
                "OCR dataset report lacks explicit source, split, or template family"
            )
        normalized_family = unicodedata.normalize("NFC", family.strip()).casefold()
        if not normalized_family or digest in result:
            raise OcrTrainingRightsError("OCR dataset report document inventory is invalid")
        result[digest] = (split, normalized_family)
    return result


def _verify_attestation_evidence(
    manifest: OcrTrainingRightsManifest,
    evidence_root: Path,
) -> str:
    verified: dict[str, str] = {}
    for document in manifest.documents:
        attestation = document.attestation
        relative_ref = attestation.evidence_artifact_ref.removeprefix("artifact://")
        parts = PurePosixPath(relative_ref).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise OcrTrainingRightsError("OCR rights evidence artifact ref is unsafe")
        evidence_path = evidence_root.joinpath(*parts)
        _require_non_reparse_directory_chain(
            evidence_root,
            parts[:-1],
            "rights evidence artifact",
        )
        payload = _read_bounded_regular_file(
            evidence_path,
            "rights evidence artifact",
            max_bytes=_MAX_EVIDENCE_BYTES,
        )
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != attestation.evidence_sha256:
            raise OcrTrainingRightsError("OCR rights evidence artifact digest mismatch")
        _require_non_reparse_directory_chain(
            evidence_root,
            parts[:-1],
            "rights evidence artifact",
        )
        existing = verified.setdefault(attestation.evidence_artifact_ref, actual_sha256)
        if existing != actual_sha256:
            raise OcrTrainingRightsError("OCR rights evidence artifact binding is inconsistent")

    digest = hashlib.sha256(b"ocr-training-rights-evidence-artifacts/v1\0")
    for artifact_ref, artifact_sha256 in sorted(verified.items()):
        digest.update(artifact_ref.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(artifact_sha256))
    return digest.hexdigest()


def _verify_training_images(
    root: Path,
    artifact_payloads: dict[str, bytes],
    report_documents: dict[str, tuple[Split, str]],
) -> tuple[int, str]:
    digest = hashlib.sha256(b"ocr-training-images/v1\0")
    seen_refs: set[str] = set()
    image_count = 0
    for split, key in (
        ("train", "train_label"),
        ("validation", "validation_label"),
        ("test", "test_label"),
    ):
        payload = artifact_payloads[key]
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OcrTrainingRightsError(f"OCR dataset {key} is not UTF-8") from exc
        raw_lines = text.split("\n")
        if raw_lines and raw_lines[-1] == "":
            raw_lines.pop()
        for line_number, raw_line in enumerate(raw_lines, 1):
            raw_line = raw_line.removesuffix("\r")
            if not raw_line or raw_line.count("\t") != 1:
                raise OcrTrainingRightsError(
                    f"OCR dataset {key} row {line_number} is malformed"
                )
            image_ref, target_text = raw_line.split("\t")
            if not target_text:
                raise OcrTrainingRightsError(
                    f"OCR dataset {key} row {line_number} has an empty target"
                )
            pure_ref = PurePosixPath(image_ref)
            parts = pure_ref.parts
            if (
                pure_ref.is_absolute()
                or pure_ref.as_posix() != image_ref
                or "\\" in image_ref
                or ":" in image_ref
                or any(part in {"", ".", ".."} for part in parts)
                or len(parts) != 4
                or parts[0] != "images"
                or parts[1] != split
                or re.fullmatch(r"[0-9a-f]{64}", parts[2]) is None
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.png", parts[3])
                is None
            ):
                raise OcrTrainingRightsError(
                    f"OCR dataset {key} row {line_number} has an unsafe lineage ref"
                )
            source_sha256 = parts[2]
            document = report_documents.get(source_sha256)
            if document is None or document[0] != split:
                raise OcrTrainingRightsError(
                    f"OCR dataset {key} row {line_number} is not bound to a source document"
                )
            ref_key = image_ref.casefold()
            if ref_key in seen_refs:
                raise OcrTrainingRightsError("OCR dataset contains a duplicate image ref")
            seen_refs.add(ref_key)
            image_path = root.joinpath(*parts)
            _require_non_reparse_directory_chain(
                root,
                parts[:-1],
                "training image",
            )
            image_payload = _read_bounded_regular_file(
                image_path,
                "training image",
                max_bytes=_MAX_IMAGE_BYTES,
            )
            _require_non_reparse_directory_chain(
                root,
                parts[:-1],
                "training image",
            )
            image_sha256 = hashlib.sha256(image_payload).hexdigest()
            digest.update(split.encode("ascii"))
            digest.update(b"\0")
            digest.update(image_ref.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(image_sha256))
            image_count += 1
            if image_count > _MAX_TRAINING_IMAGES:
                raise OcrTrainingRightsError("OCR dataset contains too many training images")
    return image_count, digest.hexdigest()


def _parse_strict_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_raise_invalid_constant(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OcrTrainingRightsError(f"OCR {label} is not strict JSON") from exc


def _read_bounded_regular_file(path: Path, label: str, *, max_bytes: int) -> bytes:
    source = Path(path)
    descriptor: int | None = None
    try:
        path_metadata = source.lstat()
        _require_regular_non_reparse(path_metadata, label)
        if path_metadata.st_size > max_bytes:
            raise OcrTrainingRightsError(f"OCR {label} exceeds the size limit")
        flags = os.O_RDONLY
        for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(source, flags)
        opened_metadata = os.fstat(descriptor)
        _require_regular_non_reparse(opened_metadata, label)
        _require_same_snapshot(path_metadata, opened_metadata, label)
        if opened_metadata.st_size > max_bytes:
            raise OcrTrainingRightsError(f"OCR {label} exceeds the size limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = _read_stream_bounded(stream, label, max_bytes=max_bytes)
            read_metadata = os.fstat(stream.fileno())
        _require_same_snapshot(opened_metadata, read_metadata, label)
        final_metadata = source.lstat()
        _require_regular_non_reparse(final_metadata, label)
        _require_same_snapshot(read_metadata, final_metadata, label)
        return payload
    except OcrTrainingRightsError:
        raise
    except OSError as exc:
        raise OcrTrainingRightsError(f"OCR {label} is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_stream_bounded(stream: BinaryIO, label: str, *, max_bytes: int) -> bytes:
    payload = bytearray()
    while True:
        remaining = max_bytes + 1 - len(payload)
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise OcrTrainingRightsError(f"OCR {label} exceeds the size limit")


def _required_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise OcrTrainingRightsError(
                f"OCR {label} must be a non-symlink directory"
            )
        return path.resolve(strict=True)
    except OcrTrainingRightsError:
        raise
    except OSError as exc:
        raise OcrTrainingRightsError(f"OCR {label} is unavailable") from exc


def _require_non_reparse_directory_chain(
    root: Path,
    relative_directories: tuple[str, ...],
    label: str,
) -> None:
    cursor = root
    try:
        _require_directory_non_reparse(cursor.lstat(), label)
        for part in relative_directories:
            cursor /= part
            _require_directory_non_reparse(cursor.lstat(), label)
    except OcrTrainingRightsError:
        raise
    except OSError as exc:
        raise OcrTrainingRightsError(f"OCR {label} directory is unavailable") from exc


def _require_regular_non_reparse(metadata: os.stat_result, label: str) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise OcrTrainingRightsError(
            f"OCR {label} must be a regular non-symlink file"
        )


def _require_directory_non_reparse(metadata: os.stat_result, label: str) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise OcrTrainingRightsError(
            f"OCR {label} must use non-symlink directories"
        )


def _require_same_snapshot(
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
        raise OcrTrainingRightsError(f"OCR {label} changed while it was read")


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _sha256(path: Path) -> str:
    payload = _read_bounded_regular_file(
        path,
        "registered artifact",
        max_bytes=_MAX_REPORT_BYTES,
    )
    return hashlib.sha256(payload).hexdigest()
