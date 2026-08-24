from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerifiedFormulaSample:
    document_id: str
    formula_id: str
    image: Path
    target: str
    format: str


@dataclass
class DatasetAudit:
    total_rows: int = 0
    verified_rows: int = 0
    unique_documents: int = 0
    missing_targets: int = 0
    missing_images: int = 0
    conflicting_duplicates: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def trainable(self) -> bool:
        return (
            self.verified_rows >= 3000
            and self.unique_documents >= 50
            and self.conflicting_duplicates == 0
        )


def load_verified_formula_samples(manifest: Path) -> tuple[list[VerifiedFormulaSample], DatasetAudit]:
    audit = DatasetAudit()
    samples: list[VerifiedFormulaSample] = []
    hashes: dict[str, str] = {}
    documents: set[str] = set()
    if not manifest.exists():
        audit.errors.append(f"manifest를 찾을 수 없습니다: {manifest}")
        return samples, audit

    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        audit.total_rows += 1
        try:
            row: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            audit.errors.append(f"{line_number}? JSON ??: {exc.msg}")
            continue
        target = normalize_latex(str(row.get("target") or ""))
        if not target:
            audit.missing_targets += 1
            continue
        image = _resolve_manifest_image(manifest, str(row.get("image") or ""))
        if not image.is_file():
            audit.missing_images += 1
            audit.errors.append(f"{line_number}행 이미지 없음: {image}")
            continue
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        prior = hashes.get(digest)
        if prior is not None:
            if prior != target:
                audit.conflicting_duplicates += 1
                audit.errors.append(f"{line_number}행 중복 이미지에 서로 다른 정답이 있습니다.")
            continue
        hashes[digest] = target
        document_id = str(row.get("document_id") or "").strip()
        formula_id = str(row.get("formula_id") or f"row-{line_number}").strip()
        documents.add(document_id)
        samples.append(
            VerifiedFormulaSample(
                document_id=document_id,
                formula_id=formula_id,
                image=image,
                target=target,
                format=str(row.get("format") or "latex"),
            )
        )
    audit.verified_rows = len(samples)
    audit.unique_documents = len(documents)
    return samples, audit


def split_by_document(
    samples: list[VerifiedFormulaSample], seed: str = "exam2hwpx-v1"
) -> dict[str, list[VerifiedFormulaSample]]:
    """Deterministic document-level split prevents page/crop leakage."""
    result: dict[str, list[VerifiedFormulaSample]] = {"train": [], "validation": [], "test": []}
    for sample in samples:
        digest = hashlib.sha256(f"{seed}:{sample.document_id}".encode()).digest()[0]
        bucket = "train" if digest < 204 else "validation" if digest < 230 else "test"
        result[bucket].append(sample)
    return result


def write_training_splits(
    splits: dict[str, list[VerifiedFormulaSample]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in splits.items():
        rows = [
            json.dumps(
                {
                    "document_id": sample.document_id,
                    "formula_id": sample.formula_id,
                    "image": sample.image.as_posix(),
                    "target": sample.target,
                    "format": sample.format,
                },
                ensure_ascii=False,
            )
            for sample in samples
        ]
        (output_dir / f"{name}.jsonl").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )


def normalize_latex(expression: str) -> str:
    return " ".join(expression.replace("\r", " ").replace("\n", " ").split())


def _resolve_manifest_image(manifest: Path, value: str) -> Path:
    image = Path(value)
    if image.is_absolute() or image.is_file():
        return image
    return manifest.parent / image
