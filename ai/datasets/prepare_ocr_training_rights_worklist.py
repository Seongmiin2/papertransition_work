from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PAIR_SCHEMA = "hwp-hwpx-projection-pairs/1.0"
WORKLIST_SCHEMA = "ocr-training-rights-worklist/1.0"
SPLITS = frozenset({"train", "validation", "test"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_DOCUMENTS = 10_000
_MAX_PAGE_COUNT = 100_000
_MAX_SOURCE_REF_BYTES = 4_096
_MAX_SOURCE_ENTRIES = 100_000
_MAX_METADATA_BYTES = 1_024
_HUMAN_FIELDS = (
    "template_family",
    "final_split",
    "rights_basis",
    "rights_evidence_artifact_ref",
    "rights_evidence_sha256",
    "attestor_id",
    "attested_at",
)


@dataclass(frozen=True, slots=True)
class PairDocument:
    source_ref: str
    source_sha256: str
    candidate_split: str
    page_count: int


@dataclass(frozen=True, slots=True)
class PairManifest:
    path: Path
    sha256: str
    payload: bytes
    documents: tuple[PairDocument, ...]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-authorizing human worklist for OCR source rights, template "
            "families, and final splits"
        )
    )
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    worklist = prepare_ocr_training_rights_worklist(
        args.pair_manifest,
        args.source_root,
        args.out,
    )
    print(
        json.dumps(
            {
                "schema_version": worklist["schema_version"],
                "training_authorized": worklist["training_authorized"],
                "document_count": worklist["document_count"],
                "source_manifest_sha256": worklist["source_manifest_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def prepare_ocr_training_rights_worklist(
    pair_manifest: str | Path,
    source_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = _load_pair_manifest(Path(pair_manifest))
    root = _required_directory(Path(source_root), "source root")
    destination = Path(output_path).resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if destination == manifest.path or _is_within(destination, root):
        raise ValueError("output must not overlap source inputs")

    _reject_source_symlinks(root)
    verified_sources = _verify_listed_sources(root, manifest.documents)
    documents = [
        {
            "source_ref": document.source_ref,
            "source_sha256": document.source_sha256,
            "page_count": document.page_count,
            "candidate_split": document.candidate_split,
            "template_family": None,
            "final_split": None,
            "rights_basis": None,
            "rights_evidence_artifact_ref": None,
            "rights_evidence_sha256": None,
            "attestor_id": None,
            "attested_at": None,
            "needs_human_input": list(_HUMAN_FIELDS),
        }
        for document in manifest.documents
    ]
    worklist: dict[str, Any] = {
        "schema_version": WORKLIST_SCHEMA,
        "artifact_role": "human_rights_review_worklist_only",
        "training_authorized": False,
        "candidate_split_is_informational": True,
        "source_manifest_sha256": manifest.sha256,
        "document_count": len(documents),
        "documents": documents,
    }
    serialized = (
        json.dumps(worklist, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    current_manifest = _load_pair_manifest(manifest.path)
    if current_manifest.payload != manifest.payload:
        raise RuntimeError("pair manifest changed while preparing the rights worklist")
    _reject_source_symlinks(root)
    if _verify_listed_sources(root, manifest.documents) != verified_sources:
        raise RuntimeError("listed source files changed while preparing the rights worklist")
    _publish_create_only(destination, serialized)
    return worklist


def _load_pair_manifest(path: Path) -> PairManifest:
    manifest_path = _required_regular_file(path, "pair manifest")
    payload = _read_bounded_file(manifest_path, _MAX_MANIFEST_BYTES, "pair manifest")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_raise_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("pair manifest must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("pair manifest must be an object")
    expected_keys = {
        "schema_version",
        "artifact_role",
        "research_only",
        "rights_manifest",
        "source_dataset_report_sha256",
        "source_inventory_sha256",
        "document_count",
        "splits",
        "producer",
        "documents",
    }
    if set(value) != expected_keys:
        raise ValueError("pair manifest has an unexpected field set")
    if value["schema_version"] != PAIR_SCHEMA:
        raise ValueError(f"pair manifest schema_version must be {PAIR_SCHEMA}")
    if value["artifact_role"] != "paired_source_for_projection":
        raise ValueError("pair manifest has an unexpected artifact role")
    if value["research_only"] is not True:
        raise ValueError("pair manifest must remain research_only")
    if value["rights_manifest"] != {"status": "missing"}:
        raise ValueError("pair manifest rights status must remain explicitly missing")
    _required_sha256(value["source_dataset_report_sha256"], "source dataset report")
    _required_sha256(value["source_inventory_sha256"], "source inventory")
    _validate_producer(value["producer"])

    raw_documents = value["documents"]
    count = value["document_count"]
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("pair manifest documents must be a non-empty list")
    if len(raw_documents) > _MAX_DOCUMENTS:
        raise ValueError("pair manifest has too many documents")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(raw_documents):
        raise ValueError("pair manifest document_count mismatch")

    documents: list[PairDocument] = []
    seen_hashes: set[str] = set()
    seen_refs: set[str] = set()
    seen_artifact_refs: set[str] = set()
    for index, item in enumerate(raw_documents):
        if not isinstance(item, dict) or set(item) != {
            "source_sha256",
            "source_ref",
            "split",
            "page_count",
            "hwpx",
            "pdf",
        }:
            raise ValueError(f"pair manifest document {index} has an unexpected field set")
        source_sha256 = _required_sha256(item["source_sha256"], f"document {index} source")
        if source_sha256 in seen_hashes:
            raise ValueError("pair manifest contains duplicate source SHA-256")
        seen_hashes.add(source_sha256)
        source_ref = _required_source_ref(item["source_ref"], index)
        source_ref_key = unicodedata.normalize("NFC", source_ref).casefold()
        if source_ref_key in seen_refs:
            raise ValueError("pair manifest contains duplicate source refs")
        seen_refs.add(source_ref_key)
        split = item["split"]
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"pair manifest document {index} has an invalid split")
        page_count = item["page_count"]
        if (
            isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or not 1 <= page_count <= _MAX_PAGE_COUNT
        ):
            raise ValueError(f"pair manifest document {index} has an invalid page_count")
        for kind in ("hwpx", "pdf"):
            artifact_ref = _validate_artifact_binding(
                item[kind], source_sha256, kind, index
            )
            if artifact_ref.casefold() in seen_artifact_refs:
                raise ValueError("pair manifest contains duplicate artifact refs")
            seen_artifact_refs.add(artifact_ref.casefold())
        documents.append(
            PairDocument(
                source_ref=source_ref,
                source_sha256=source_sha256,
                candidate_split=split,
                page_count=page_count,
            )
        )

    documents.sort(key=lambda document: document.source_sha256)
    _validate_split_counts(value["splits"], documents)
    return PairManifest(
        path=manifest_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
        documents=tuple(documents),
    )


def _required_source_ref(value: object, index: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_SOURCE_REF_BYTES
        or _has_control_character(value)
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"pair manifest document {index} has an unsafe source_ref")
    ref = PurePosixPath(value)
    if (
        ref.is_absolute()
        or ref.as_posix() != value
        or any(part in {"", ".", ".."} for part in ref.parts)
        or ref.suffix.casefold() != ".hwp"
    ):
        raise ValueError(f"pair manifest document {index} has an unsafe source_ref")
    return value


def _validate_artifact_binding(
    value: object,
    source_sha256: str,
    kind: str,
    index: int,
) -> str:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"pair manifest document {index} has an invalid {kind} binding")
    expected_path = f"{source_sha256}/source.{kind}"
    if value["path"] != expected_path:
        raise ValueError(f"pair manifest document {index} has an unsafe {kind} path")
    _required_sha256(value["sha256"], f"document {index} {kind}")
    return expected_path


def _validate_producer(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"name", "version"}:
        raise ValueError("pair manifest producer has an unexpected field set")
    for field in ("name", "version"):
        item = value[field]
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item.encode("utf-8")) > _MAX_METADATA_BYTES
            or _has_control_character(item)
        ):
            raise ValueError(f"pair manifest producer {field} is invalid")


def _validate_split_counts(value: object, documents: list[PairDocument]) -> None:
    if not isinstance(value, dict) or set(value) != SPLITS:
        raise ValueError("pair manifest splits must contain train, validation, and test")
    expected = Counter(document.candidate_split for document in documents)
    for split in SPLITS:
        count = value[split]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= _MAX_DOCUMENTS
            or count != expected[split]
        ):
            raise ValueError(f"pair manifest split {split} document count mismatch")


def _verify_listed_sources(
    root: Path,
    documents: tuple[PairDocument, ...],
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for document in documents:
        path = root.joinpath(*PurePosixPath(document.source_ref).parts)
        if path.is_symlink():
            raise ValueError(f"listed source must not be a symlink: {document.source_ref}")
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(f"listed source is missing: {document.source_ref}") from exc
        if not resolved.is_file() or not _is_within(resolved, root):
            raise ValueError(f"listed source is unsafe: {document.source_ref}")
        actual_sha256 = _sha256(resolved)
        if actual_sha256 != document.source_sha256:
            raise ValueError(f"listed source SHA-256 mismatch: {document.source_ref}")
        verified[document.source_ref] = actual_sha256
    return verified


def _reject_source_symlinks(root: Path) -> None:
    entries_seen = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > _MAX_SOURCE_ENTRIES:
                        raise ValueError("source root contains too many entries")
                    if entry.is_symlink():
                        raise ValueError("source root must not contain symlinks")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
        except OSError as exc:
            raise ValueError("source root could not be safely enumerated") from exc


def _publish_create_only(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    final_path = parent / destination.name
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(f"output already exists: {final_path}")
    descriptor, staging_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, final_path)
        except FileExistsError as exc:
            raise FileExistsError(f"output appeared during build: {final_path}") from exc
    finally:
        staging.unlink(missing_ok=True)


def _read_bounded_file(path: Path, limit: int, label: str) -> bytes:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is unavailable: {path}") from exc
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds the size limit")
    return payload


def _required_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    return resolved


def _required_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return resolved


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_invalid_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"source file is unavailable: {path.name}") from exc
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
