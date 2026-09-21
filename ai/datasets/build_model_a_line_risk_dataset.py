from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import tempfile
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol, cast

import yaml  # type: ignore[import-untyped]

from scan2hwpx.evaluation.model_a import (
    MODEL_A_METRIC_DEFINITION_IDS,
    levenshtein_distance,
)
from scan2hwpx.evaluation.ocr_training_rights import (
    AuthenticatedAttestorContext,
    OcrTrainingRightsEvidence,
    verify_ocr_training_dataset_rights,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    publish_directory_create_only as _publish_directory_create_only,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    read_bounded_regular_file,
    remove_staging_directory,
    sha256_bounded_regular_file,
)

SPLITS = ("train", "validation", "test")
_MAX_SOURCE_LABEL_BYTES = 64 * 1024 * 1024
_MAX_HASHED_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
RECORD_FIELDS = (
    "image_ref",
    "image_sha256",
    "observed_text",
    "confidence",
    "target_text",
    "is_error",
)


class LineRecognizer(Protocol):
    def predict(self, input: list[str], *, batch_size: int) -> Iterable[object]: ...

    def close(self) -> None: ...


PredictorFactory = Callable[[Path, str], LineRecognizer]


@dataclass(frozen=True)
class SourceSample:
    split: str
    image_ref: str
    image_path: Path
    image_sha256: str
    target_text: str


@dataclass(frozen=True)
class PredictedSample:
    source: SourceSample
    observed_text: str
    confidence: float
    is_error: bool


@dataclass(frozen=True)
class SourceDataset:
    samples: dict[str, tuple[SourceSample, ...]]
    split_tsv_sha256: dict[str, str]
    split_images_sha256: dict[str, str]
    sha256: str


def main(*, attestor_context: AuthenticatedAttestorContext | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build split-safe Model A OCR line-risk JSONL with PaddleOCR"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--rights-manifest",
        type=Path,
        required=True,
        help="Verified rights manifest used to build the source OCR dataset",
    )
    parser.add_argument(
        "--rights-evidence-root",
        type=Path,
        required=True,
        help="Workspace root containing the attestation evidence artifact:// paths",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if attestor_context is None:
        parser.error(
            "Model A line-risk build rights gate failed: OCR training requires "
            "identities from an authenticated external context"
        )
    report = build_model_a_line_risk_dataset(
        args.dataset,
        args.model,
        args.out,
        rights_manifest=args.rights_manifest,
        rights_evidence_root=args.rights_evidence_root,
        attestor_context=attestor_context,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), flush=True)
    return 0


def build_model_a_line_risk_dataset(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    rights_manifest: str | Path,
    rights_evidence_root: str | Path,
    attestor_context: AuthenticatedAttestorContext | None,
    device: str,
    batch_size: int,
    predictor_factory: PredictorFactory | None = None,
) -> dict[str, Any]:
    if attestor_context is None:
        raise ValueError(
            "Model A line-risk build requires an authenticated external context"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not device.strip():
        raise ValueError("device must not be empty")

    dataset_root = _required_directory(Path(dataset_dir), "dataset")
    model_root = _required_directory(Path(model_dir), "model")
    rights_manifest_path = Path(rights_manifest).absolute()
    evidence_root = Path(rights_evidence_root)
    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if _is_within(destination, dataset_root) or _is_within(destination, model_root):
        raise ValueError("output must be outside the source dataset and model directories")

    rights_evidence = verify_ocr_training_dataset_rights(
        dataset_root,
        rights_manifest_path,
        train_label=dataset_root / "train.txt",
        validation_label=dataset_root / "validation.txt",
        character_dict=dataset_root / "korean_exam_dict.txt",
        evidence_root=evidence_root,
        attestor_context=attestor_context,
    )
    source_dataset = _load_source_dataset(dataset_root)
    model_sha256 = _directory_sha256(model_root)
    leaked_image_hashes = _cross_split_source_image_hashes(source_dataset.samples)
    image_safe_samples, image_removals = _remove_source_image_leaks(
        source_dataset.samples,
        leaked_image_hashes,
    )
    _require_nonempty_splits(image_safe_samples, "cross-split image filtering")

    factory = predictor_factory or _create_predictor
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
        )
    )
    published = False
    try:
        predictor = factory(model_root, device)
        try:
            predictions = _predict_samples(predictor, image_safe_samples, batch_size)
        finally:
            predictor.close()

        leaked_text_pairs = _cross_split_text_pairs(predictions)
        final_samples, text_pair_removals = _remove_text_pair_leaks(
            predictions,
            leaked_text_pairs,
        )
        _require_nonempty_splits(final_samples, "cross-split text-pair filtering")
        _assert_final_split_isolation(final_samples)

        current_rights_evidence = verify_ocr_training_dataset_rights(
            dataset_root,
            rights_manifest_path,
            train_label=dataset_root / "train.txt",
            validation_label=dataset_root / "validation.txt",
            character_dict=dataset_root / "korean_exam_dict.txt",
            evidence_root=evidence_root,
            attestor_context=attestor_context,
        )
        if current_rights_evidence != rights_evidence:
            raise RuntimeError("source dataset rights binding changed during inference")
        if _load_source_dataset(dataset_root).sha256 != source_dataset.sha256:
            raise RuntimeError("source dataset changed during inference")
        if _directory_sha256(model_root) != model_sha256:
            raise RuntimeError("model artifact changed during inference")

        output_hashes: dict[str, str] = {}
        for split in SPLITS:
            output_path = staging / f"{split}.jsonl"
            _write_jsonl(output_path, final_samples[split])
            output_hashes[split] = _sha256(output_path)

        report = _build_report(
            source_dataset,
            rights_evidence,
            model_sha256,
            device,
            batch_size,
            image_safe_samples,
            final_samples,
            leaked_image_hashes,
            leaked_text_pairs,
            image_removals,
            text_pair_removals,
            output_hashes,
        )
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output appeared during build: {destination}")
        _publish_directory_create_only(staging, destination)
        published = True
        return report
    finally:
        if not published and staging.exists():
            remove_staging_directory(staging, parent=destination.parent)


def _load_source_dataset(dataset_root: Path) -> SourceDataset:
    samples: dict[str, tuple[SourceSample, ...]] = {}
    split_tsv_sha256: dict[str, str] = {}
    split_images_sha256: dict[str, str] = {}
    dataset_digest = hashlib.sha256(b"model-a-line-risk-source-dataset/v1\0")

    for split in SPLITS:
        label_path = dataset_root / f"{split}.txt"
        if not label_path.is_file() or label_path.is_symlink():
            raise FileNotFoundError(f"missing regular split file: {split}.txt")
        payload = read_bounded_regular_file(
            label_path,
            max_bytes=_MAX_SOURCE_LABEL_BYTES,
            label=f"source OCR label {split}",
        )
        split_tsv_sha256[split] = hashlib.sha256(payload).hexdigest()
        split_samples = _parse_split(dataset_root, split, payload)
        if not split_samples:
            raise ValueError(f"{split}.txt must contain at least one sample")
        samples[split] = tuple(split_samples)

        image_digest = hashlib.sha256(b"model-a-line-risk-images/v1\0")
        for sample in split_samples:
            image_digest.update(sample.image_ref.encode("utf-8"))
            image_digest.update(b"\0")
            image_digest.update(bytes.fromhex(sample.image_sha256))
        split_images_sha256[split] = image_digest.hexdigest()

        dataset_digest.update(split.encode("ascii"))
        dataset_digest.update(b"\0")
        dataset_digest.update(bytes.fromhex(split_tsv_sha256[split]))
        dataset_digest.update(bytes.fromhex(split_images_sha256[split]))

    return SourceDataset(
        samples=samples,
        split_tsv_sha256=split_tsv_sha256,
        split_images_sha256=split_images_sha256,
        sha256=dataset_digest.hexdigest(),
    )


def _parse_split(dataset_root: Path, split: str, payload: bytes) -> list[SourceSample]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{split}.txt must be valid UTF-8") from exc
    if not text:
        return []
    raw_lines = text.split("\n")
    if raw_lines[-1] == "":
        raw_lines.pop()

    samples: list[SourceSample] = []
    seen_refs: set[str] = set()
    seen_paths: set[str] = set()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.removesuffix("\r")
        if not line or line.count("\t") != 1:
            raise ValueError(f"malformed {split}.txt row {line_number}")
        image_ref, target_text = line.split("\t")
        if not target_text or not target_text.strip():
            raise ValueError(f"empty target text in {split}.txt row {line_number}")
        normalized_ref, image_path = _resolve_image_ref(
            dataset_root,
            image_ref,
            split,
            line_number,
        )
        ref_key = normalized_ref.casefold()
        path_key = str(image_path).casefold()
        if ref_key in seen_refs or path_key in seen_paths:
            raise ValueError(f"duplicate image reference in {split}.txt row {line_number}")
        seen_refs.add(ref_key)
        seen_paths.add(path_key)
        samples.append(
            SourceSample(
                split=split,
                image_ref=normalized_ref,
                image_path=image_path,
                image_sha256=_sha256(image_path),
                target_text=target_text,
            )
        )
    return samples


def _resolve_image_ref(
    dataset_root: Path,
    image_ref: str,
    split: str,
    line_number: int,
) -> tuple[str, Path]:
    invalid = (
        not image_ref
        or image_ref != image_ref.strip()
        or "\\" in image_ref
        or ":" in image_ref
        or any(ord(character) < 32 for character in image_ref)
    )
    pure_path = PurePosixPath(image_ref)
    parts = image_ref.split("/")
    if (
        invalid
        or pure_path.is_absolute()
        or PureWindowsPath(image_ref).drive
        or any(part in {"", ".", ".."} for part in parts)
        or pure_path.as_posix() != image_ref
    ):
        raise ValueError(f"unsafe image reference in {split}.txt row {line_number}")

    unresolved = dataset_root.joinpath(*pure_path.parts)
    cursor = dataset_root
    for part in pure_path.parts:
        cursor /= part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise ValueError(f"missing image in {split}.txt row {line_number}") from exc
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0) & reparse_attribute
        ):
            raise ValueError(f"reparse image reference in {split}.txt row {line_number}")
    try:
        resolved = unresolved.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"missing image in {split}.txt row {line_number}") from exc
    if not _is_within(resolved, dataset_root) or not resolved.is_file():
        raise ValueError(f"unsafe image reference in {split}.txt row {line_number}")
    return pure_path.as_posix(), resolved


def _predict_samples(
    predictor: LineRecognizer,
    samples: dict[str, tuple[SourceSample, ...]],
    batch_size: int,
) -> dict[str, tuple[PredictedSample, ...]]:
    predictions: dict[str, tuple[PredictedSample, ...]] = {}
    for split in SPLITS:
        split_predictions: list[PredictedSample] = []
        split_samples = samples[split]
        for offset in range(0, len(split_samples), batch_size):
            batch = split_samples[offset : offset + batch_size]
            results = tuple(
                predictor.predict(
                    input=[str(sample.image_path) for sample in batch],
                    batch_size=batch_size,
                )
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"predictor returned {len(results)} results for {len(batch)} {split} inputs"
                )
            for sample, result in zip(batch, results, strict=True):
                observed_text, confidence = _prediction_values(result)
                split_predictions.append(
                    PredictedSample(
                        source=sample,
                        observed_text=observed_text,
                        confidence=confidence,
                        is_error=observed_text != sample.target_text,
                    )
                )
        predictions[split] = tuple(split_predictions)
    return predictions


def _prediction_values(result: object) -> tuple[str, float]:
    value = getattr(result, "json", None)
    if not isinstance(value, dict) or not isinstance(value.get("res"), dict):
        raise TypeError("TextRecognition result must contain a JSON res object")
    payload = value["res"]
    observed_text = payload.get("rec_text")
    raw_confidence = payload.get("rec_score")
    if not isinstance(observed_text, str) or isinstance(raw_confidence, bool):
        raise TypeError("TextRecognition result has invalid rec_text or rec_score")
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("TextRecognition result has invalid rec_score") from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("TextRecognition rec_score must be finite and between 0 and 1")
    return observed_text, confidence


def _cross_split_source_image_hashes(
    samples: dict[str, tuple[SourceSample, ...]],
) -> set[str]:
    owners: dict[str, set[str]] = {}
    for split in SPLITS:
        for sample in samples[split]:
            owners.setdefault(sample.image_sha256, set()).add(split)
    return {fingerprint for fingerprint, splits in owners.items() if len(splits) > 1}


def _remove_source_image_leaks(
    samples: dict[str, tuple[SourceSample, ...]],
    fingerprints: set[str],
) -> tuple[dict[str, tuple[SourceSample, ...]], dict[str, int]]:
    filtered: dict[str, tuple[SourceSample, ...]] = {}
    removals: dict[str, int] = {}
    for split in SPLITS:
        filtered[split] = tuple(
            sample for sample in samples[split] if sample.image_sha256 not in fingerprints
        )
        removals[split] = len(samples[split]) - len(filtered[split])
    return filtered, removals


def _cross_split_text_pairs(
    samples: dict[str, tuple[PredictedSample, ...]],
) -> set[tuple[str, str]]:
    owners: dict[tuple[str, str], set[str]] = {}
    for split in SPLITS:
        for sample in samples[split]:
            owners.setdefault(_text_pair_fingerprint(sample), set()).add(split)
    return {fingerprint for fingerprint, splits in owners.items() if len(splits) > 1}


def _remove_text_pair_leaks(
    samples: dict[str, tuple[PredictedSample, ...]],
    fingerprints: set[tuple[str, str]],
) -> tuple[dict[str, tuple[PredictedSample, ...]], dict[str, int]]:
    filtered: dict[str, tuple[PredictedSample, ...]] = {}
    removals: dict[str, int] = {}
    for split in SPLITS:
        filtered[split] = tuple(
            sample
            for sample in samples[split]
            if _text_pair_fingerprint(sample) not in fingerprints
        )
        removals[split] = len(samples[split]) - len(filtered[split])
    return filtered, removals


def _text_pair_fingerprint(sample: PredictedSample) -> tuple[str, str]:
    return (
        unicodedata.normalize("NFC", sample.observed_text),
        unicodedata.normalize("NFC", sample.source.target_text),
    )


def _assert_final_split_isolation(
    samples: dict[str, tuple[PredictedSample, ...]],
) -> None:
    image_owners: dict[str, set[str]] = {}
    for split in SPLITS:
        for sample in samples[split]:
            image_owners.setdefault(sample.source.image_sha256, set()).add(split)
    leaked_images = sum(len(owners) > 1 for owners in image_owners.values())
    leaked_pairs = len(_cross_split_text_pairs(samples))
    if leaked_images or leaked_pairs:
        raise RuntimeError("derived split isolation verification failed")


def _require_nonempty_splits(samples: dict[str, tuple[Any, ...]], stage: str) -> None:
    empty = [split for split in SPLITS if not samples[split]]
    if empty:
        raise ValueError(f"{stage} emptied splits: " + ", ".join(empty))


def _write_jsonl(path: Path, samples: tuple[PredictedSample, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            row = {
                "image_ref": sample.source.image_ref,
                "image_sha256": sample.source.image_sha256,
                "observed_text": sample.observed_text,
                "confidence": sample.confidence,
                "target_text": sample.source.target_text,
                "is_error": sample.is_error,
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _build_report(
    source_dataset: SourceDataset,
    rights_evidence: OcrTrainingRightsEvidence,
    model_sha256: str,
    device: str,
    batch_size: int,
    image_safe_samples: dict[str, tuple[SourceSample, ...]],
    final_samples: dict[str, tuple[PredictedSample, ...]],
    leaked_image_hashes: set[str],
    leaked_text_pairs: set[tuple[str, str]],
    image_removals: dict[str, int],
    text_pair_removals: dict[str, int],
    output_hashes: dict[str, str],
) -> dict[str, Any]:
    split_reports: dict[str, dict[str, Any]] = {}
    all_final: list[PredictedSample] = []
    for split in SPLITS:
        metrics = _metrics(final_samples[split])
        all_final.extend(final_samples[split])
        split_reports[split] = {
            "source_count": len(source_dataset.samples[split]),
            "after_image_filter_count": len(image_safe_samples[split]),
            "final_count": len(final_samples[split]),
            "removed_cross_split_image_sha256": image_removals[split],
            "removed_cross_split_observed_target_pair": text_pair_removals[split],
            "source_tsv_sha256": source_dataset.split_tsv_sha256[split],
            "source_images_sha256": source_dataset.split_images_sha256[split],
            "output_jsonl_sha256": output_hashes[split],
            **metrics,
        }

    derived_digest = hashlib.sha256(b"model-a-line-risk-derived-dataset/v1\0")
    for split in SPLITS:
        derived_digest.update(split.encode("ascii"))
        derived_digest.update(bytes.fromhex(output_hashes[split]))
    total_metrics = _metrics(tuple(all_final))
    return {
        "schema_version": "model-a-line-risk-dataset/1.2",
        "record_fields": list(RECORD_FIELDS),
        "source_dataset_sha256": source_dataset.sha256,
        "source_dataset_digest_definition": "split-tsv-and-referenced-image-bytes/v1",
        "source_rights": {
            "validation": "verified",
            "rights_manifest_sha256": rights_evidence.rights_manifest_sha256,
            "dataset_report_sha256": rights_evidence.dataset_report_sha256,
            "document_count": rights_evidence.document_count,
            "identity_assurance": rights_evidence.identity_assurance,
            "evidence_artifacts_sha256": rights_evidence.evidence_artifacts_sha256,
            "training_images_sha256": rights_evidence.training_images_sha256,
            "derivation_assurance": rights_evidence.derivation_assurance,
            "derivation_verification_scope": (
                rights_evidence.derivation_verification_scope
            ),
            "derivation_attestor_id": rights_evidence.derivation_attestor_id,
        },
        "derived_dataset_sha256": derived_digest.hexdigest(),
        "model_sha256": model_sha256,
        "model_digest_definition": "relative-file-path-and-bytes/v1",
        "device": device,
        "batch_size": batch_size,
        "exact_match_definition": "raw-unicode-codepoint-exact/v1",
        "cer_definition": MODEL_A_METRIC_DEFINITION_IDS["model_a.overall_cer"],
        "filters": {
            "cross_split_image_sha256": {
                "fingerprint_count": len(leaked_image_hashes),
                "removed_by_split": image_removals,
            },
            "cross_split_observed_target_pair": {
                "fingerprint_definition": "nfc-observed-text-and-nfc-target-text/v1",
                "fingerprint_count": len(leaked_text_pairs),
                "removed_by_split": text_pair_removals,
            },
            "final_cross_split_image_sha256_count": 0,
            "final_cross_split_observed_target_pair_count": 0,
        },
        "splits": split_reports,
        "totals": {
            "source_count": sum(len(source_dataset.samples[split]) for split in SPLITS),
            "after_image_filter_count": sum(
                len(image_safe_samples[split]) for split in SPLITS
            ),
            "final_count": len(all_final),
            "removed_cross_split_image_sha256": sum(image_removals.values()),
            "removed_cross_split_observed_target_pair": sum(text_pair_removals.values()),
            **total_metrics,
        },
    }


def _metrics(samples: tuple[PredictedSample, ...]) -> dict[str, int | float]:
    errors = 0
    characters = 0
    exact = 0
    for sample in samples:
        reference = unicodedata.normalize("NFC", sample.source.target_text)
        prediction = unicodedata.normalize("NFC", sample.observed_text)
        errors += levenshtein_distance(reference, prediction)
        characters += max(len(reference), len(prediction))
        exact += not sample.is_error
    if not samples or not characters:
        raise ValueError("metrics require at least one non-empty target")
    return {
        "exact_match_count": exact,
        "exact_match_rate": exact / len(samples),
        "character_errors": errors,
        "character_denominator": characters,
        "cer": errors / characters,
    }


def _required_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise ValueError(f"{label} directory must be a non-reparse directory")
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} directory not found: {path}") from exc
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")
    return resolved


def _directory_sha256(root: Path) -> str:
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if any(
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
        for path in entries
        for metadata in (path.lstat(),)
    ):
        raise ValueError("model directory must not contain reparse paths")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError("model directory must contain at least one file")
    digest = hashlib.sha256(b"model-a-line-risk-model-directory/v1\0")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return sha256_bounded_regular_file(
        path,
        max_bytes=_MAX_HASHED_ARTIFACT_BYTES,
        label=f"Model A line-risk artifact {path}",
    )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _model_name_from_config(model_dir: Path) -> str:
    config_path = model_dir / "inference.yml"
    try:
        metadata = config_path.lstat()
    except OSError as exc:
        raise FileNotFoundError(
            "model directory must contain a regular inference.yml"
        ) from exc
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
    ):
        raise FileNotFoundError("model directory must contain a regular inference.yml")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("model inference.yml is unreadable or malformed") from exc
    if not isinstance(config, dict) or not isinstance(config.get("Global"), dict):
        raise TypeError("model inference.yml must contain a Global mapping")
    model_name = config["Global"].get("model_name")
    if not isinstance(model_name, str):
        raise TypeError("model inference.yml Global.model_name must be a string")
    if not model_name.strip():
        raise ValueError("model inference.yml must declare Global.model_name")
    return model_name


def _create_predictor(model_dir: Path, device: str) -> LineRecognizer:
    from paddleocr import TextRecognition  # type: ignore[import-untyped]

    return cast(
        LineRecognizer,
        TextRecognition(
            model_name=_model_name_from_config(model_dir),
            model_dir=str(model_dir),
            device=device,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
