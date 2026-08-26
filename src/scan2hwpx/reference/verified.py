from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def import_verified_ocr(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if item.get("status") != "verified" or not str(item.get("target", "")).strip():
                raise ValueError("status must be verified and target must not be empty")
            image = Path(str(item["image"]))
            if not image.is_absolute():
                image = (manifest_path.parent / image).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()[:20]
            destination = image_dir / f"{digest}{image.suffix.lower()}"
            if not destination.exists():
                shutil.copy2(image, destination)
            rows.append(
                {
                    **item,
                    "target": str(item["target"]).strip(),
                    "image": destination.relative_to(output_dir).as_posix(),
                }
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            failures.append({"line": line_no, "error": f"{type(exc).__name__}: {exc}"})

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_document[str(row["document_id"])].append(row)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for document_id, document_rows in by_document.items():
        split_rows[_split(document_id)].extend(document_rows)
    for split, samples in split_rows.items():
        labels = [f"{sample['image']}\t{sample['target']}" for sample in samples]
        (output_dir / f"{split}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "verified_samples": len(rows),
        "documents": len(by_document),
        "splits": {split: len(samples) for split, samples in split_rows.items()},
        "failures": failures,
        "trainable_evaluation": len(by_document) >= 3 and all(split_rows.values()),
        "note": (
            "At least three independent PDF documents are required before train/validation/test "
            "metrics are trustworthy."
        ),
    }
    (output_dir / "verified_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _split(document_id: str) -> str:
    bucket = int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"
