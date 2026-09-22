from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pymupdf
import yaml  # type: ignore[import-untyped]
from PIL import Image
from pyhwpx import Hwp  # type: ignore[import-untyped]

_SPACE = re.compile(r"[\t\u00a0 ]+")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_NAMESPACE = "hwp-ocr-line-split/v2"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_SPLIT_MANIFEST_BYTES = 16 * 1024 * 1024
_RENDERED_PDF_PROVENANCE_SCHEMA = "hwp-rendered-pdf-provenance/1.0"

Split = Literal["train", "validation", "test"]
SplitSource = Literal[
    "frozen_document",
    "frozen_template_family",
    "template_family_hash",
    "lineage_hash",
]
_ORDERED_SPLITS: tuple[Split, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class FrozenSplitEntry:
    source_sha256: str
    template_family: str | None
    split: Split | None


@dataclass(frozen=True)
class DocumentSplit:
    source_sha256: str
    split: Split
    template_family: str | None
    source: SplitSource


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build real Korean exam OCR line data from Hancom-rendered HWP files"
    )
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output", type=Path, default=Path("output/datasets/hwp-ocr-training"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-text-length", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "Optional prior dataset_report.json or split map. Entries may pin a split and "
            "declare template_family; family isolation is guaranteed only for declared families."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path.home()
        / ".paddlex"
        / "official_models"
        / "korean_PP-OCRv5_mobile_rec",
    )
    parser.add_argument("--rebuild-pdfs", action="store_true")
    args = parser.parse_args()

    if args.dpi < 150:
        parser.error("--dpi must be at least 150")
    if args.max_text_length < 2:
        parser.error("--max-text-length must be at least 2")

    project_root = Path(__file__).resolve().parents[2]
    default_inputs = [Path("ai/datasets/source/고2")]
    input_dirs = [(project_root / path).resolve() for path in (args.inputs or default_inputs)]
    output_dir = (project_root / args.output).resolve()
    model_dir = args.base_model.expanduser().resolve()
    hwp_paths = sorted(
        {path.resolve() for input_dir in input_dirs for path in input_dir.glob("*.hwp")}
    )
    if not hwp_paths:
        parser.error("no HWP files found in: " + ", ".join(str(path) for path in input_dirs))

    split_manifest_path = (
        (project_root / args.split_manifest).resolve() if args.split_manifest else None
    )
    frozen_splits = _load_split_manifest(split_manifest_path)
    split_by_path = _split_documents(hwp_paths, args.seed, frozen_splits=frozen_splits)
    dictionary = _load_dictionary(model_dir / "inference.yml")
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = output_dir / "korean_exam_dict.txt"
    dictionary_path.write_text("\n".join(dictionary) + "\n", encoding="utf-8")

    rendered_dir = output_dir / "rendered_pdf"
    rendered_dir.mkdir(exist_ok=True)
    rows: dict[Split, list[str]] = {"train": [], "validation": [], "test": []}
    counts: Counter[str] = Counter()
    documents: list[dict[str, Any]] = []

    hwp = Hwp(new=True, visible=False)
    try:
        for index, source_path in enumerate(hwp_paths, start=1):
            assignment = split_by_path[source_path]
            split = assignment.split
            source_hash = assignment.source_sha256
            if _sha256(source_path) != source_hash:
                raise RuntimeError(f"source changed before rendering: {source_path.name}")
            pdf_path = rendered_dir / f"{source_hash}.pdf"
            provenance_path = rendered_dir / f"{source_hash}.provenance.json"
            print(f"[{index}/{len(hwp_paths)}] {source_path.name} -> {split}", flush=True)
            if args.rebuild_pdfs or not _cached_pdf_matches_source(
                pdf_path,
                provenance_path,
                source_hash,
            ):
                if not hwp.open(
                    str(source_path),
                    arg="forceopen:true;suspendpassword:true;versionwarning:false",
                ):
                    counts["open_failed"] += 1
                    continue
                page_count = hwp.PageCount
                if not hwp.save_as(str(pdf_path), "PDF"):
                    counts["export_failed"] += 1
                    hwp.close(is_dirty=False)
                    continue
                hwp.close(is_dirty=False)
                if _sha256(source_path) != source_hash:
                    raise RuntimeError(f"source changed while rendering: {source_path.name}")
                _write_pdf_provenance(provenance_path, source_hash, _sha256(pdf_path))
            else:
                with pymupdf.open(pdf_path) as document:  # type: ignore[no-untyped-call]
                    page_count = len(document)

            result = _extract_lines(
                pdf_path,
                output_dir,
                split,
                source_hash,
                set(dictionary),
                dpi=args.dpi,
                max_text_length=args.max_text_length,
            )
            rows[split].extend(result["rows"])
            if _sha256(source_path) != source_hash:
                raise RuntimeError(f"source changed while extracting lines: {source_path.name}")
            counts.update(result["counts"])
            documents.append(
                {
                    "source": str(source_path),
                    "sha256": source_hash,
                    "split": split,
                    "template_family": assignment.template_family,
                    "split_assignment_source": assignment.source,
                    "pages": page_count,
                    "accepted_lines": len(result["rows"]),
                    "rendered_pdf_sha256": _sha256(pdf_path),
                }
            )
    finally:
        hwp.quit()

    for split, split_rows in rows.items():
        (output_dir / f"{split}.txt").write_text(
            "\n".join(split_rows) + ("\n" if split_rows else ""), encoding="utf-8"
        )

    artifact_paths = {
        "train_label": output_dir / "train.txt",
        "validation_label": output_dir / "validation.txt",
        "test_label": output_dir / "test.txt",
        "character_dict": dictionary_path,
    }
    for source_path, assignment in split_by_path.items():
        if _sha256(source_path) != assignment.source_sha256:
            raise RuntimeError(f"source changed before dataset report: {source_path.name}")
    image_inventory = _build_image_inventory(output_dir, rows)
    report = {
        "schema_version": "1.1",
        "source": "Hancom-rendered HWP PDF text lines; no OCR-generated labels",
        "inputs": [str(path) for path in input_dirs],
        "base_model": str(model_dir),
        "seed": args.seed,
        "split_policy": {
            "id": _SPLIT_NAMESPACE,
            "thresholds": {"train": 80, "validation": 10, "test": 10},
            "frozen_manifest": (
                str(split_manifest_path) if split_manifest_path is not None else None
            ),
            "frozen_manifest_sha256": (
                _sha256(split_manifest_path) if split_manifest_path is not None else None
            ),
            "template_family_contract": "explicit_manifest_only",
            "template_family_documents": sum(
                assignment.template_family is not None for assignment in split_by_path.values()
            ),
        },
        "dpi": args.dpi,
        "max_text_length": args.max_text_length,
        "dictionary_characters": len(dictionary),
        "splits": {
            split: {
                "documents": sum(document["split"] == split for document in documents),
                "lines": len(split_rows),
            }
            for split, split_rows in rows.items()
        },
        "counts": dict(sorted(counts.items())),
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        "image_inventory": image_inventory,
        "documents": documents,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["splits"], ensure_ascii=False), flush=True)
    return 0 if all(rows[split] for split in rows) else 1


def _extract_lines(
    pdf_path: Path,
    output_dir: Path,
    split: str,
    document_hash: str,
    dictionary: set[str],
    *,
    dpi: int,
    max_text_length: int,
) -> dict[str, Any]:
    rows: list[str] = []
    counts: Counter[str] = Counter()
    scale = dpi / 72
    image_dir = output_dir / "images" / split / document_hash
    image_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as document:  # type: ignore[no-untyped-call]
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),  # type: ignore[no-untyped-call]
                colorspace=pymupdf.csGRAY,
                alpha=False,
            )
            page_image = Image.frombytes(
                "L", (pixmap.width, pixmap.height), pixmap.samples
            )
            payload = page.get_text("dict", sort=True)
            line_index = 0
            for block in payload.get("blocks", []):
                for line in block.get("lines", []):
                    text = _normalize_text(
                        "".join(str(span.get("text", "")) for span in line.get("spans", []))
                    )
                    reason = _rejection_reason(text, dictionary, max_text_length)
                    if reason:
                        counts[reason] += 1
                        continue
                    bbox = line.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        counts["invalid_bbox"] += 1
                        continue
                    crop_box = _scaled_crop_box(bbox, scale, page_image.size)
                    if crop_box[2] - crop_box[0] < 8 or crop_box[3] - crop_box[1] < 8:
                        counts["tiny_bbox"] += 1
                        continue
                    token = hashlib.sha256(
                        f"{document_hash}:{page_index}:{line_index}:{text}".encode()
                    ).hexdigest()[:24]
                    relative_path = Path("images") / split / document_hash / f"{token}.png"
                    page_image.crop(crop_box).save(output_dir / relative_path, format="PNG")
                    rows.append(f"{relative_path.as_posix()}\t{text}")
                    line_index += 1
                    counts["accepted"] += 1
    return {"rows": rows, "counts": counts}


def _cached_pdf_matches_source(
    pdf_path: Path,
    provenance_path: Path,
    source_sha256: str,
) -> bool:
    if pdf_path.is_symlink() or provenance_path.is_symlink():
        raise ValueError("rendered PDF cache must not contain symlinks")
    if not pdf_path.is_file() or not provenance_path.is_file():
        return False
    try:
        with provenance_path.open("rb") as stream:
            raw_payload = stream.read(4096 + 1)
        if len(raw_payload) > 4096:
            return False
        payload = json.loads(raw_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "source_sha256",
        "pdf_sha256",
    }:
        return False
    return (
        payload["schema_version"] == _RENDERED_PDF_PROVENANCE_SCHEMA
        and payload["source_sha256"] == source_sha256
        and isinstance(payload["pdf_sha256"], str)
        and _SHA256.fullmatch(payload["pdf_sha256"]) is not None
        and _sha256(pdf_path) == payload["pdf_sha256"]
    )


def _write_pdf_provenance(
    provenance_path: Path,
    source_sha256: str,
    pdf_sha256: str,
) -> None:
    if provenance_path.is_symlink():
        raise ValueError("rendered PDF provenance must not be a symlink")
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": _RENDERED_PDF_PROVENANCE_SCHEMA,
                "source_sha256": source_sha256,
                "pdf_sha256": pdf_sha256,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_image_inventory(
    output_dir: Path,
    rows: dict[Split, list[str]],
) -> dict[str, str | int]:
    digest = hashlib.sha256(b"ocr-training-images/v1\0")
    image_count = 0
    for split in _ORDERED_SPLITS:
        for row in rows[split]:
            image_ref, _ = row.split("\t", 1)
            image_sha256 = _sha256(output_dir.joinpath(*image_ref.split("/")))
            digest.update(split.encode("ascii"))
            digest.update(b"\0")
            digest.update(image_ref.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(image_sha256))
            image_count += 1
    return {
        "digest_definition": "split-label-ref-source-lineage-and-image-bytes/v1",
        "image_count": image_count,
        "sha256": digest.hexdigest(),
    }


def _load_dictionary(inference_config: Path) -> list[str]:
    if not inference_config.is_file():
        raise FileNotFoundError(f"base model inference config is missing: {inference_config}")
    payload = yaml.safe_load(inference_config.read_text(encoding="utf-8"))
    characters = payload.get("PostProcess", {}).get("character_dict", [])
    if not isinstance(characters, list) or not characters:
        raise ValueError(f"character dictionary is missing: {inference_config}")
    return list(dict.fromkeys(str(character) for character in characters))


def _load_split_manifest(path: Path | None) -> dict[str, FrozenSplitEntry]:
    if path is None:
        return {}
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"split manifest is missing: {path}")
    with path.open("rb") as stream:
        raw_payload = stream.read(_MAX_SPLIT_MANIFEST_BYTES + 1)
    if len(raw_payload) > _MAX_SPLIT_MANIFEST_BYTES:
        raise ValueError("split manifest exceeds the size limit")
    try:
        payload: Any = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("split manifest must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise TypeError("split manifest must contain a documents list")

    entries: dict[str, FrozenSplitEntry] = {}
    family_splits: dict[str, Split] = {}
    for index, value in enumerate(payload["documents"]):
        if not isinstance(value, dict):
            raise TypeError(f"split manifest document {index} must be an object")
        source_sha256 = _manifest_sha256(value, index)
        if source_sha256 in entries:
            raise ValueError(f"split manifest contains duplicate lineage: {source_sha256}")

        family_value = value.get("template_family")
        if family_value is None:
            template_family = None
        elif not isinstance(family_value, str) or not family_value.strip():
            raise ValueError(f"split manifest document {index} has invalid template_family")
        else:
            template_family = unicodedata.normalize("NFC", family_value.strip()).casefold()
            if any(
                unicodedata.category(character) in {"Cc", "Cf"}
                for character in template_family
            ):
                raise ValueError(
                    f"split manifest document {index} has invalid template_family"
                )

        split_value = value.get("split")
        if split_value is None:
            split = None
        elif not isinstance(split_value, str) or split_value not in _SPLITS:
            raise ValueError(f"split manifest document {index} has invalid split")
        else:
            split = cast(Split, split_value)

        entry = FrozenSplitEntry(
            source_sha256=source_sha256,
            template_family=template_family,
            split=split,
        )
        entries[source_sha256] = entry
        if template_family is not None and split is not None:
            existing = family_splits.setdefault(template_family, split)
            if existing != split:
                raise ValueError(
                    f"template family {template_family!r} is assigned to multiple splits"
                )
    return entries


def _manifest_sha256(value: dict[Any, Any], index: int) -> str:
    current = value.get("source_sha256")
    legacy = value.get("sha256")
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(f"split manifest document {index} has conflicting sha256 fields")
    digest = current if current is not None else legacy
    if not isinstance(digest, str) or _SHA256.fullmatch(digest.casefold()) is None:
        raise ValueError(f"split manifest document {index} has invalid sha256")
    return digest.casefold()


def _split_documents(
    paths: list[Path],
    seed: int,
    *,
    frozen_splits: dict[str, FrozenSplitEntry] | None = None,
) -> dict[Path, DocumentSplit]:
    frozen = frozen_splits or {}
    frozen_family_splits: dict[str, Split] = {}
    for frozen_entry in frozen.values():
        if frozen_entry.template_family is None or frozen_entry.split is None:
            continue
        existing = frozen_family_splits.setdefault(
            frozen_entry.template_family, frozen_entry.split
        )
        if existing != frozen_entry.split:
            raise ValueError(
                f"template family {frozen_entry.template_family!r} is assigned to multiple splits"
            )

    result: dict[Path, DocumentSplit] = {}
    for path in paths:
        source_sha256 = _sha256(path)
        entry = frozen.get(source_sha256)
        template_family = entry.template_family if entry is not None else None
        if entry is not None and entry.split is not None:
            split = entry.split
            source: SplitSource = "frozen_document"
        elif template_family is not None and template_family in frozen_family_splits:
            split = frozen_family_splits[template_family]
            source = "frozen_template_family"
        elif template_family is not None:
            split = _stable_split(f"template-family:{template_family}", seed)
            source = "template_family_hash"
        else:
            split = _stable_split(f"lineage:{source_sha256}", seed)
            source = "lineage_hash"
        result[path] = DocumentSplit(
            source_sha256=source_sha256,
            split=split,
            template_family=template_family,
            source=source,
        )

    _validate_split_isolation(result.values())
    return result


def _stable_split(key: str, seed: int) -> Split:
    digest = hashlib.sha256(f"{_SPLIT_NAMESPACE}\0{seed}\0{key}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _validate_split_isolation(assignments: Iterable[DocumentSplit]) -> None:
    lineage_splits: dict[str, set[Split]] = {}
    family_splits: dict[str, set[Split]] = {}
    for assignment in assignments:
        lineage_splits.setdefault(assignment.source_sha256, set()).add(assignment.split)
        if assignment.template_family is not None:
            family_splits.setdefault(assignment.template_family, set()).add(assignment.split)
    leaked_lineages = sorted(key for key, splits in lineage_splits.items() if len(splits) > 1)
    if leaked_lineages:
        raise ValueError("lineage leakage across splits: " + ", ".join(leaked_lineages))
    leaked_families = sorted(key for key, splits in family_splits.items() if len(splits) > 1)
    if leaked_families:
        raise ValueError("template family leakage across splits: " + ", ".join(leaked_families))


def _normalize_text(text: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _rejection_reason(text: str, dictionary: set[str], max_text_length: int) -> str | None:
    if len(text) < 2 or not any(character.isalnum() for character in text):
        return "empty_or_symbol_only"
    if len(text) > max_text_length:
        return "too_long"
    if any(character != " " and character not in dictionary for character in text):
        return "unsupported_character"
    return None


def _scaled_crop_box(
    bbox: list[float] | tuple[float, ...], scale: float, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) * scale for value in bbox)
    horizontal_padding = max(3, round(scale))
    vertical_padding = max(2, round(scale * 0.6))
    return (
        max(0, math.floor(x0) - horizontal_padding),
        max(0, math.floor(y0) - vertical_padding),
        min(size[0], math.ceil(x1) + horizontal_padding),
        min(size[1], math.ceil(y1) + vertical_padding),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
