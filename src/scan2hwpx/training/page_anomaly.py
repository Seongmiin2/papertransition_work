from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image


@dataclass(frozen=True)
class LinearPageAutoencoder:
    mean: np.ndarray[Any, np.dtype[np.float32]]
    components: np.ndarray[Any, np.dtype[np.float32]]
    threshold: float
    thumbnail_size: int

    def score(self, samples: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, Any]:
        centered = samples - self.mean
        reconstruction = (centered @ self.components.T) @ self.components
        return np.asarray(np.mean(np.square(centered - reconstruction), axis=1), dtype=np.float32)


def train_page_anomaly_model(
    corpus_manifest: Path,
    output_dir: Path,
    *,
    rank: int = 32,
    thumbnail_size: int = 32,
) -> dict[str, Any]:
    """Fit a compact linear autoencoder and keep test documents entirely held out."""
    if rank < 1:
        raise ValueError("rank must be positive")
    if thumbnail_size < 16:
        raise ValueError("thumbnail_size must be at least 16")
    rows = _load_manifest(corpus_manifest)
    page_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for item in rows:
        if item.get("role") not in {
            "source_pdf",
            "gold_source_pdf",
            "regression_source_pdf",
        }:
            continue
        source = Path(str(item["source_path"]))
        with pymupdf.open(source) as document:  # type: ignore[no-untyped-call]
            for index, page in enumerate(document):
                feature_rows.append(page_thumbnail(page, thumbnail_size))
                page_rows.append(
                    {
                        "document_id": str(item["id"]),
                        "name": str(item["name"]),
                        "path": str(source.resolve()),
                        "split": str(item["split"]),
                        "role": str(item["role"]),
                        "page": index + 1,
                    }
                )
    if not feature_rows:
        raise ValueError("no PDF pages were found in the corpus manifest")
    features = np.stack(feature_rows).astype(np.float32)
    train_indices = [
        index
        for index, row in enumerate(page_rows)
        if row["split"] == "train" and row["role"] == "source_pdf"
    ]
    validation_indices = [
        index
        for index, row in enumerate(page_rows)
        if row["split"] == "validation" and row["role"] == "source_pdf"
    ]
    if len(train_indices) < 3:
        raise ValueError("at least three training pages are required")
    model = fit_linear_autoencoder(
        features[train_indices],
        features[validation_indices] if validation_indices else None,
        rank=rank,
        thumbnail_size=thumbnail_size,
    )
    scores = model.score(features)
    for row, score in zip(page_rows, scores, strict=True):
        row["reconstruction_mse"] = round(float(score), 8)
        row["anomaly"] = bool(score > model.threshold)

    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "page_anomaly_linear_autoencoder.npz"
    np.savez_compressed(
        weights_path,
        mean=model.mean,
        components=model.components,
        threshold=np.array([model.threshold], dtype=np.float32),
        thumbnail_size=np.array([model.thumbnail_size], dtype=np.int32),
    )
    _write_jsonl(destination / "page_scores.jsonl", page_rows)
    split_summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in page_rows if row["split"] == split]
        split_summary[split] = {
            "pages": len(selected),
            "flagged": sum(bool(row["anomaly"]) for row in selected),
            "mean_mse": round(
                sum(float(row["reconstruction_mse"]) for row in selected) / max(1, len(selected)),
                8,
            ),
        }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model": "linear_page_autoencoder_pca",
        "purpose": "quality routing only; never used to invent OCR or document content",
        "corpus_manifest": str(corpus_manifest.resolve()),
        "weights": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "thumbnail_size": thumbnail_size,
        "requested_rank": rank,
        "effective_rank": int(model.components.shape[0]),
        "threshold": model.threshold,
        "threshold_policy": "max(train_p99_5, validation_p99_5); test is held out",
        "splits": split_summary,
        "documents": _document_summary(page_rows),
    }
    (destination / "anomaly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def fit_linear_autoencoder(
    train: np.ndarray[Any, np.dtype[np.float32]],
    validation: np.ndarray[Any, np.dtype[np.float32]] | None,
    *,
    rank: int,
    thumbnail_size: int,
) -> LinearPageAutoencoder:
    if train.ndim != 2:
        raise ValueError("train must be a two-dimensional sample matrix")
    mean = np.mean(train, axis=0, keepdims=True).astype(np.float32)
    centered = train - mean
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    effective_rank = min(rank, max(1, train.shape[0] - 1), vectors.shape[0])
    components = vectors[:effective_rank].astype(np.float32)
    temporary = LinearPageAutoencoder(mean, components, 0.0, thumbnail_size)
    train_scores = temporary.score(train)
    calibration = train_scores
    if validation is not None and len(validation):
        calibration = np.concatenate([calibration, temporary.score(validation)])
    threshold = float(np.quantile(calibration, 0.995))
    return LinearPageAutoencoder(mean, components, threshold, thumbnail_size)


def load_page_anomaly_model(path: Path) -> LinearPageAutoencoder:
    payload = np.load(path)
    return LinearPageAutoencoder(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        components=np.asarray(payload["components"], dtype=np.float32),
        threshold=float(payload["threshold"][0]),
        thumbnail_size=int(payload["thumbnail_size"][0]),
    )


def page_thumbnail(page: pymupdf.Page, size: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    scale = max(size / max(1.0, page.rect.width), size / max(1.0, page.rect.height))
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),  # type: ignore[no-untyped-call]
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    grayscale = np.asarray(image, dtype=np.float32) / 255.0
    return np.asarray((1.0 - grayscale).reshape(-1), dtype=np.float32)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("duplicate_of") is None:
                result.append(payload)
    return result


def _document_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["document_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for document_id, pages in sorted(grouped.items()):
        scores = sorted(float(page["reconstruction_mse"]) for page in pages)
        result.append(
            {
                "document_id": document_id,
                "name": pages[0]["name"],
                "role": pages[0]["role"],
                "split": pages[0]["split"],
                "pages": len(pages),
                "flagged_pages": sum(bool(page["anomaly"]) for page in pages),
                "max_mse": round(scores[-1], 8),
                "p95_mse": round(float(np.quantile(scores, 0.95)), 8),
            }
        )
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
