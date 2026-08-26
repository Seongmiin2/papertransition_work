from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import yaml  # type: ignore[import-untyped]
from PIL import Image
from pyhwpx import Hwp

_SPACE = re.compile(r"[\t\u00a0 ]+")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build real Korean exam OCR line data from Hancom-rendered HWP files"
    )
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output", type=Path, default=Path("output/hwp-ocr-training"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-text-length", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260826)
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

    project_root = Path(__file__).resolve().parents[1]
    input_dirs = [(project_root / path).resolve() for path in (args.inputs or [Path("고2")])]
    output_dir = (project_root / args.output).resolve()
    model_dir = args.base_model.expanduser().resolve()
    hwp_paths = sorted(
        {path.resolve() for input_dir in input_dirs for path in input_dir.glob("*.hwp")}
    )
    if not hwp_paths:
        parser.error("no HWP files found in: " + ", ".join(str(path) for path in input_dirs))

    dictionary = _load_dictionary(model_dir / "inference.yml")
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = output_dir / "korean_exam_dict.txt"
    dictionary_path.write_text("\n".join(dictionary) + "\n", encoding="utf-8")

    split_by_path = _split_documents(hwp_paths, args.seed)
    rendered_dir = output_dir / "rendered_pdf"
    rendered_dir.mkdir(exist_ok=True)
    rows: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    counts: Counter[str] = Counter()
    documents: list[dict[str, Any]] = []

    hwp = Hwp(new=True, visible=False)
    try:
        for index, source_path in enumerate(hwp_paths, start=1):
            split = split_by_path[source_path]
            source_hash = _sha256(source_path)
            pdf_path = rendered_dir / f"{source_hash[:16]}.pdf"
            print(f"[{index}/{len(hwp_paths)}] {source_path.name} -> {split}", flush=True)
            if args.rebuild_pdfs or not pdf_path.is_file():
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
            else:
                with pymupdf.open(pdf_path) as document:
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
            counts.update(result["counts"])
            documents.append(
                {
                    "source": str(source_path),
                    "sha256": source_hash,
                    "split": split,
                    "pages": page_count,
                    "accepted_lines": len(result["rows"]),
                }
            )
    finally:
        hwp.quit()

    for split, split_rows in rows.items():
        (output_dir / f"{split}.txt").write_text(
            "\n".join(split_rows) + ("\n" if split_rows else ""), encoding="utf-8"
        )

    report = {
        "schema_version": "1.0",
        "source": "Hancom-rendered HWP PDF text lines; no OCR-generated labels",
        "inputs": [str(path) for path in input_dirs],
        "base_model": str(model_dir),
        "seed": args.seed,
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
    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
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
                    relative_path = Path("images") / split / f"{token}.png"
                    page_image.crop(crop_box).save(output_dir / relative_path, format="PNG")
                    rows.append(f"{relative_path.as_posix()}\t{text}")
                    line_index += 1
                    counts["accepted"] += 1
    return {"rows": rows, "counts": counts}


def _load_dictionary(inference_config: Path) -> list[str]:
    if not inference_config.is_file():
        raise FileNotFoundError(f"base model inference config is missing: {inference_config}")
    payload = yaml.safe_load(inference_config.read_text(encoding="utf-8"))
    characters = payload.get("PostProcess", {}).get("character_dict", [])
    if not isinstance(characters, list) or not characters:
        raise ValueError(f"character dictionary is missing: {inference_config}")
    return list(dict.fromkeys(str(character) for character in characters))


def _split_documents(paths: list[Path], seed: int) -> dict[Path, str]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    holdout = max(1, round(len(shuffled) * 0.1))
    result: dict[Path, str] = {}
    for index, path in enumerate(shuffled):
        if index < holdout:
            result[path] = "test"
        elif index < holdout * 2:
            result[path] = "validation"
        else:
            result[path] = "train"
    return result


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
