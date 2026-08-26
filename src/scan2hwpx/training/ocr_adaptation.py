from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image

from scan2hwpx.ir.models import Block
from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.preprocess import preprocess_for_ocr

ProgressCallback = Callable[[str], None]


def build_ocr_adaptation_dataset(
    corpus_manifest: Path,
    base_dataset: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    dpi: int = 200,
    minimum_confidence: float = 0.985,
    minimum_quality: float = 0.92,
    include_source_test: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create conservative silver scan crops and merge them with the existing synthetic set.

    Silver labels are teacher predictions, never gold. The held-out gold and regression PDFs are
    excluded unconditionally.
    """
    destination = output_dir.resolve()
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    base_counts = _merge_base_dataset(base_dataset, destination, split_rows)
    documents = [
        item
        for item in _load_manifest(corpus_manifest)
        if item.get("role") == "source_pdf"
        and item.get("duplicate_of") is None
        and (include_source_test or item.get("split") in {"train", "validation"})
    ]
    provider = PaddlePdfOcrProvider(
        dpi=dpi,
        fusion_mode="fast",
        lexicon_path=base_dataset / "lexicon.json",
        device=device,
    )
    accepted_by_split: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    page_count = 0
    block_count = 0
    for document_index, item in enumerate(documents, start=1):
        source = Path(str(item["source_path"]))
        split = str(item["split"])
        if progress:
            progress(f"[{document_index}/{len(documents)}] silver OCR: {source.name}")
        document = provider.convert(
            source,
            progress=(
                (lambda current, total, message: progress(f"  {message}")) if progress else None
            ),
        )
        page_count += len(document.pages)
        with pymupdf.open(source) as pdf:  # type: ignore[no-untyped-call]
            for page, ir_page in zip(pdf, document.pages, strict=True):
                rendered = page.get_pixmap(
                    matrix=pymupdf.Matrix(  # type: ignore[no-untyped-call]
                        dpi / 72, dpi / 72
                    ),
                    alpha=False,
                )
                rgb = np.frombuffer(rendered.samples, dtype=np.uint8).reshape(
                    rendered.height, rendered.width, rendered.n
                )
                processed = preprocess_for_ocr(rgb).image
                for block in ir_page.blocks:
                    block_count += 1
                    accepted, reason = accept_silver_label(
                        block,
                        provider.quality_router.score(block.text, block.confidence),
                        minimum_confidence=minimum_confidence,
                        minimum_quality=minimum_quality,
                    )
                    if not accepted:
                        rejected[reason] += 1
                        continue
                    crop = _crop_block(processed, block)
                    if crop is None:
                        rejected["invalid_crop"] += 1
                        continue
                    token = hashlib.sha256(
                        f"{item['sha256']}:{block.source_payload_ref}:{block.text}".encode()
                    ).hexdigest()[:24]
                    relative = Path("images") / f"silver-{token}.png"
                    Image.fromarray(crop).save(destination / relative, format="PNG", optimize=True)
                    split_rows[split].append(f"{relative.as_posix()}\t{block.text}")
                    accepted_by_split[split] += 1

    for split, rows in split_rows.items():
        (destination / f"{split}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
    for name in ("korean_exam_dict.txt", "lexicon.json"):
        source = base_dataset / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "corpus_manifest": str(corpus_manifest.resolve()),
        "base_dataset": str(base_dataset.resolve()),
        "output_directory": str(destination),
        "device": provider.device,
        "dpi": dpi,
        "documents": len(documents),
        "pages": page_count,
        "teacher_blocks": block_count,
        "minimum_confidence": minimum_confidence,
        "minimum_quality": minimum_quality,
        "base_samples": base_counts,
        "silver_samples": dict(accepted_by_split),
        "total_samples": {split: len(rows) for split, rows in split_rows.items()},
        "rejected": dict(rejected.most_common()),
        "guardrails": {
            "gold_source_in_training": False,
            "regression_source_in_training": False,
            "source_test_in_training": False,
            "silver_is_gold": False,
            "promotion_requires_gold_benchmark": True,
        },
    }
    (destination / "adaptation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def accept_silver_label(
    block: Block,
    quality: float,
    *,
    minimum_confidence: float,
    minimum_quality: float,
) -> tuple[bool, str]:
    text = block.text.strip()
    if not 2 <= len(text) <= 60:
        return False, "length"
    if not any(character.isalnum() for character in text):
        return False, "no_alphanumeric"
    if block.confidence < minimum_confidence:
        return False, "confidence"
    if quality < minimum_quality:
        return False, "language_quality"
    if block.style.get("review_reasons"):
        return False, "review_reason"
    x0, y0, x1, y1 = block.bbox.pixel
    if x1 - x0 < 8 or y1 - y0 < 6:
        return False, "tiny_box"
    return True, "accepted"


def write_dictionary_compatible_labels(
    dataset_dir: Path,
    character_dict: Path,
    *,
    suffix: str = "official",
) -> dict[str, Any]:
    characters = set(character_dict.read_text(encoding="utf-8").splitlines())
    characters.add(" ")
    summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source = dataset_dir / f"{split}.txt"
        accepted: list[str] = []
        rejected = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            _, text = line.split("\t", 1)
            if all(character in characters for character in text):
                accepted.append(line)
            else:
                rejected += 1
        output = dataset_dir / f"{split}.{suffix}.txt"
        output.write_text("\n".join(accepted) + ("\n" if accepted else ""), encoding="utf-8")
        summary[split] = {
            "accepted": len(accepted),
            "rejected": rejected,
            "label_file": str(output.resolve()),
        }
    copied_dictionary = dataset_dir / f"character_dict.{suffix}.txt"
    shutil.copy2(character_dict, copied_dictionary)
    report = {
        "schema_version": "1.0",
        "character_dict": str(character_dict.resolve()),
        "copied_character_dict": str(copied_dictionary.resolve()),
        "characters": len(characters) - 1,
        "splits": summary,
    }
    (dataset_dir / f"dictionary_compatibility.{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _crop_block(image: np.ndarray[Any, Any], block: Block) -> np.ndarray[Any, Any] | None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = block.bbox.pixel
    pad_x = max(2, int((y1 - y0) * 0.12))
    pad_y = max(2, int((y1 - y0) * 0.18))
    left = max(0, int(x0) - pad_x)
    top = max(0, int(y0) - pad_y)
    right = min(width, int(x1) + pad_x)
    bottom = min(height, int(y1) + pad_y)
    if right - left < 8 or bottom - top < 6:
        return None
    return np.ascontiguousarray(image[top:bottom, left:right])


def _merge_base_dataset(
    base_dataset: Path,
    destination: Path,
    split_rows: dict[str, list[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, output_rows in split_rows.items():
        label_path = base_dataset / f"{split}.txt"
        if not label_path.is_file():
            counts[split] = 0
            continue
        count = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            relative, text = line.split("\t", 1)
            source = base_dataset / relative
            if not source.is_file():
                continue
            token = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:24]
            target_relative = Path("images") / f"synthetic-{token}{source.suffix.lower()}"
            target = destination / target_relative
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            output_rows.append(f"{target_relative.as_posix()}\t{text}")
            count += 1
        counts[split] = count
    return counts


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        payload
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if isinstance(payload, dict)
    ]
