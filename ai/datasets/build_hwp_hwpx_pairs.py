from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Protocol

import pymupdf

from scan2hwpx.hwpx.validate import validate_hwpx  # type: ignore[import-untyped]

SPLITS = frozenset({"train", "validation", "test"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
OPEN_ARGUMENTS = "forceopen:true;suspendpassword:true;versionwarning:false"


class PairConverter(Protocol):
    producer: str
    version: str

    def convert(self, source_hwp: Path, output_hwpx: Path, output_pdf: Path) -> int: ...

    def close(self) -> None: ...


AdapterFactory = Callable[[], PairConverter]
HwpFactory = Callable[[], Any]


@dataclass(frozen=True)
class ExpectedDocument:
    source_sha256: str
    split: str
    page_count: int


@dataclass(frozen=True)
class SourceDocument:
    source_sha256: str
    source_ref: str
    path: Path


@dataclass(frozen=True)
class SourceInventory:
    by_sha256: dict[str, SourceDocument]
    ref_to_sha256: dict[str, str]
    sha256: str


class PyhwpxComAdapter:
    producer = "pyhwpx"

    def __init__(
        self,
        *,
        hwp_factory: HwpFactory | None = None,
        producer_version: str | None = None,
    ) -> None:
        if hwp_factory is None:
            from pyhwpx import Hwp  # type: ignore[import-untyped]

            from scan2hwpx.hwpx import hancom  # type: ignore[import-untyped]

            hancom._ensure_hancom_security_module()
            self._hwp: Any = Hwp(new=True, visible=False)
        else:
            self._hwp = hwp_factory()
        self.version = producer_version or package_version("pyhwpx")

    def convert(self, source_hwp: Path, output_hwpx: Path, output_pdf: Path) -> int:
        if not self._hwp.open(str(source_hwp), arg=OPEN_ARGUMENTS):
            raise RuntimeError("Hancom failed to open source HWP")
        try:
            if not self._hwp.save_as(str(output_hwpx), format="HWPX"):
                raise RuntimeError("Hancom failed to save HWPX")
        finally:
            self._hwp.close(is_dirty=False)

        if not self._hwp.open(str(output_hwpx), arg=OPEN_ARGUMENTS):
            raise RuntimeError("Hancom failed to reopen generated HWPX")
        try:
            page_count = self._hwp.PageCount
            if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
                raise ValueError("Hancom returned an invalid HWPX page count")
            if not self._hwp.save_as(str(output_pdf), format="PDF"):
                raise RuntimeError("Hancom failed to save PDF from generated HWPX")
        finally:
            self._hwp.close(is_dirty=False)
        return page_count

    def close(self) -> None:
        self._hwp.quit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build SHA-mapped HWPX/PDF projection-source pairs"
    )
    parser.add_argument(
        "--dataset-report",
        type=Path,
        default=Path("output/hwp-ocr-training-v2/dataset_report.json"),
    )
    parser.add_argument("--source", type=Path, default=Path("ai/datasets/source"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_hwp_hwpx_pairs(args.dataset_report, args.source, args.out)
    print(
        json.dumps(
            {
                "document_count": manifest["document_count"],
                "splits": manifest["splits"],
                "research_only": manifest["research_only"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_hwp_hwpx_pairs(
    dataset_report: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    report_path = _required_regular_file(Path(dataset_report), "dataset report")
    source_root = _required_directory(Path(source_dir), "source")
    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if _is_within(destination, source_root) or destination == report_path:
        raise ValueError("output must not overlap source inputs")

    report_bytes = report_path.read_bytes()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    expected = _load_expected_documents(report_bytes)
    inventory = _scan_source_hwp(source_root)
    _assert_exact_source_mapping(expected, inventory)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
        )
    )
    published = False
    adapter: PairConverter | None = None
    try:
        factory = adapter_factory or PyhwpxComAdapter
        adapter = factory()
        producer = _nonempty_metadata(adapter.producer, "producer")
        producer_version = _nonempty_metadata(adapter.version, "producer version")
        documents: list[dict[str, Any]] = []
        expected_artifacts: dict[str, str] = {}

        for document in expected:
            source = inventory.by_sha256[document.source_sha256]
            _assert_source_hash(source)
            document_dir = staging / document.source_sha256
            document_dir.mkdir()
            output_hwpx = document_dir / "source.hwpx"
            output_pdf = document_dir / "source.pdf"
            page_count = adapter.convert(source.path, output_hwpx, output_pdf)
            _assert_source_hash(source)
            if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
                raise ValueError("converter page_count must be a positive integer")
            if page_count != document.page_count:
                raise RuntimeError("converted HWPX page_count differs from dataset report")

            hwpx_sha256 = _artifact_sha256(output_hwpx, "HWPX")
            if not validate_hwpx(output_hwpx).valid:
                raise RuntimeError("generated HWPX failed package validation")
            pdf_sha256 = _artifact_sha256(output_pdf, "PDF")
            hwpx_ref = f"{document.source_sha256}/source.hwpx"
            if _pdf_page_count(output_pdf) != page_count:
                raise RuntimeError("generated PDF page_count differs from reopened HWPX")
            pdf_ref = f"{document.source_sha256}/source.pdf"
            expected_artifacts[hwpx_ref] = hwpx_sha256
            expected_artifacts[pdf_ref] = pdf_sha256
            documents.append(
                {
                    "source_sha256": document.source_sha256,
                    "source_ref": source.source_ref,
                    "split": document.split,
                    "page_count": page_count,
                    "hwpx": {"path": hwpx_ref, "sha256": hwpx_sha256},
                    "pdf": {"path": pdf_ref, "sha256": pdf_sha256},
                }
            )

        adapter.close()
        adapter = None
        _assert_inputs_unchanged(report_path, report_sha256, source_root, inventory)
        _assert_artifacts_unchanged(staging, expected_artifacts)

        split_counts = Counter(document.split for document in expected)
        manifest: dict[str, Any] = {
            "schema_version": "hwp-hwpx-projection-pairs/1.0",
            "artifact_role": "paired_source_for_projection",
            "research_only": True,
            "rights_manifest": {"status": "missing"},
            "source_dataset_report_sha256": report_sha256,
            "source_inventory_sha256": inventory.sha256,
            "document_count": len(documents),
            "splits": {split: split_counts[split] for split in sorted(SPLITS)},
            "producer": {"name": producer, "version": producer_version},
            "documents": documents,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output appeared during build: {destination}")
        staging.rename(destination)
        published = True
        return manifest
    finally:
        if adapter is not None:
            adapter.close()
        if not published and staging.exists():
            shutil.rmtree(staging)


def _load_expected_documents(payload: bytes) -> tuple[ExpectedDocument, ...]:
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("dataset report must be valid UTF-8 JSON") from exc
    if not isinstance(report, dict) or not isinstance(report.get("documents"), list):
        raise TypeError("dataset report must contain a documents list")
    if report.get("schema_version") != "1.0":
        raise ValueError("dataset report schema_version must be 1.0")
    if not report["documents"]:
        raise ValueError("dataset report documents must not be empty")

    documents: list[ExpectedDocument] = []
    seen_sha256: set[str] = set()
    for index, item in enumerate(report["documents"]):
        if not isinstance(item, dict):
            raise TypeError(f"dataset report document {index} must be an object")
        source_sha256 = item.get("sha256")
        split = item.get("split")
        page_count = item.get("pages")
        if not isinstance(source_sha256, str) or SHA256_PATTERN.fullmatch(source_sha256) is None:
            raise ValueError(f"dataset report document {index} has invalid sha256")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"dataset report document {index} has invalid split")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
            raise ValueError(f"dataset report document {index} has invalid pages")
        if source_sha256 in seen_sha256:
            raise ValueError(f"dataset report contains duplicate sha256: {source_sha256}")
        seen_sha256.add(source_sha256)
        documents.append(
            ExpectedDocument(
                source_sha256=source_sha256,
                split=split,
                page_count=page_count,
            )
        )
    _validate_report_split_counts(report.get("splits"), documents)
    return tuple(sorted(documents, key=lambda document: document.source_sha256))


def _validate_report_split_counts(value: object, documents: list[ExpectedDocument]) -> None:
    if not isinstance(value, dict) or set(value) != SPLITS:
        raise TypeError("dataset report splits must contain train, validation, and test")
    expected_counts = Counter(document.split for document in documents)
    for split in SPLITS:
        split_summary = value[split]
        if not isinstance(split_summary, dict):
            raise TypeError(f"dataset report split {split} must be an object")
        count = split_summary.get("documents")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"dataset report split {split} has invalid document count")
        if count != expected_counts[split]:
            raise ValueError(f"dataset report split {split} document count mismatch")


def _scan_source_hwp(source_root: Path) -> SourceInventory:
    entries = sorted(source_root.rglob("*"), key=lambda path: path.relative_to(source_root).as_posix())
    if any(path.is_symlink() for path in entries):
        raise ValueError("source directory must not contain symlinks")

    by_sha256: dict[str, SourceDocument] = {}
    ref_to_sha256: dict[str, str] = {}
    seen_casefold_refs: set[str] = set()
    for path in entries:
        if not path.is_file() or path.suffix.casefold() != ".hwp":
            continue
        resolved = path.resolve(strict=True)
        if not _is_within(resolved, source_root):
            raise ValueError("source HWP escapes source directory")
        source_ref = resolved.relative_to(source_root).as_posix()
        if any(ord(character) < 32 for character in source_ref):
            raise ValueError("source HWP path contains control characters")
        ref_key = source_ref.casefold()
        if ref_key in seen_casefold_refs:
            raise ValueError("source directory contains duplicate HWP paths")
        seen_casefold_refs.add(ref_key)

        source_sha256 = _sha256(resolved)
        if source_sha256 in by_sha256:
            raise ValueError(f"source directory contains duplicate HWP bytes: {source_sha256}")
        ref_to_sha256[source_ref] = source_sha256
        by_sha256[source_sha256] = SourceDocument(
            source_sha256=source_sha256,
            source_ref=source_ref,
            path=resolved,
        )
    if not by_sha256:
        raise ValueError("source directory contains no HWP files")

    digest = hashlib.sha256(b"hwp-source-inventory/v1\0")
    for source_ref, source_sha256 in sorted(ref_to_sha256.items()):
        digest.update(source_ref.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(source_sha256))
    return SourceInventory(
        by_sha256=by_sha256,
        ref_to_sha256=ref_to_sha256,
        sha256=digest.hexdigest(),
    )


def _assert_exact_source_mapping(
    expected: tuple[ExpectedDocument, ...], inventory: SourceInventory
) -> None:
    expected_sha256 = {document.source_sha256 for document in expected}
    actual_sha256 = set(inventory.by_sha256)
    if expected_sha256 != actual_sha256:
        missing = len(expected_sha256 - actual_sha256)
        unexpected = len(actual_sha256 - expected_sha256)
        raise ValueError(
            f"source HWP SHA mapping mismatch: missing={missing}, unexpected={unexpected}"
        )


def _assert_source_hash(source: SourceDocument) -> None:
    if (
        not source.path.is_file()
        or source.path.is_symlink()
        or _sha256(source.path) != source.source_sha256
    ):
        raise RuntimeError(f"source HWP changed during conversion: {source.source_sha256}")


def _assert_inputs_unchanged(
    report_path: Path,
    report_sha256: str,
    source_root: Path,
    inventory: SourceInventory,
) -> None:
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or _sha256(report_path) != report_sha256
    ):
        raise RuntimeError("dataset report changed during conversion")
    current = _scan_source_hwp(source_root)
    if current.ref_to_sha256 != inventory.ref_to_sha256:
        raise RuntimeError("source HWP inventory changed during conversion")


def _assert_artifacts_unchanged(staging: Path, expected: dict[str, str]) -> None:
    entries = list(staging.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise RuntimeError("generated artifacts must not contain symlinks")
    actual_files = {
        path.relative_to(staging).as_posix(): path
        for path in entries
        if path.is_file()
    }
    if set(actual_files) != set(expected):
        raise RuntimeError("converter produced an unexpected artifact set")
    for artifact_ref, expected_sha256 in expected.items():
        if _sha256(actual_files[artifact_ref]) != expected_sha256:
            raise RuntimeError("generated artifact changed before publication")


def _artifact_sha256(path: Path, label: str) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise RuntimeError(f"converter did not create a regular non-empty {label} artifact")
    return _sha256(path)

def _pdf_page_count(path: Path) -> int:
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise RuntimeError("generated PDF has an invalid signature")
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        if document.needs_pass:
            raise RuntimeError("generated PDF must not require a password")
        page_count = int(document.page_count)
    if page_count < 1:
        raise RuntimeError("generated PDF must contain at least one page")
    return page_count


def _nonempty_metadata(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"converter {label} must be a string")
    if not value.strip() or value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"converter {label} must be non-empty metadata")
    return value


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
        raise FileNotFoundError(f"{label} directory not found: {path}") from exc
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


if __name__ == "__main__":
    raise SystemExit(main())
