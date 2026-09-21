from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai.datasets.build_model_a_line_risk_dataset import (
    SourceDataset,
    _load_source_dataset,
)
from scan2hwpx.evaluation.ocr_training_rights import (
    AuthenticatedAttestorContext,
    OcrTrainingRightsError,
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

SPECIAL_TOKENS = ("<pad>", "<unk>")
RECORD_FIELDS = (
    "image_ref",
    "image_sha256",
    "observed_text",
    "confidence",
    "target_text",
    "is_error",
)
EXPECTED_FIELDS = frozenset(RECORD_FIELDS)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
_MAX_HASHED_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class RiskMetrics:
    samples: int
    errors: int
    roc_auc: float
    average_precision: float
    brier_score: float
    threshold: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "samples": self.samples,
            "errors": self.errors,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "brier_score": self.brier_score,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def main(*, attestor_context: AuthenticatedAttestorContext | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a conservative Model A helper that predicts whether an OCR line needs review"
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        required=True,
        help="Rights-bound OCR line dataset from which the risk dataset was derived",
    )
    parser.add_argument("--character-dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="gpu:0")
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
    parser.add_argument("--train-split", choices=SPLITS, required=True)
    parser.add_argument("--validation-split", choices=SPLITS, required=True)
    parser.add_argument(
        "--upstream-training-split",
        choices=("train",),
        default="train",
        help="Fixed by the source OCR trainer contract",
    )
    args = parser.parse_args()

    if attestor_context is None:
        parser.error(
            "Model A risk training rights gate failed: OCR training requires "
            "identities from an authenticated external context"
        )
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if not 0 < args.target_recall <= 1:
        parser.error("--target-recall must be in (0, 1]")

    try:
        _validate_split_selection(
            args.train_split,
            args.validation_split,
            args.upstream_training_split,
        )
    except ValueError as exc:
        parser.error(str(exc))

    project_root = PROJECT_ROOT
    dataset_dir = _resolve(project_root, args.dataset)
    source_dataset_dir = _resolve(project_root, args.source_dataset)
    character_dict = _resolve(project_root, args.character_dict)
    destination = _resolve(project_root, args.out)
    rights_manifest = _resolve(project_root, args.rights_manifest)
    rights_evidence_root = _resolve(project_root, args.rights_evidence_root)
    required = [dataset_dir / f"{split}.jsonl" for split in SPLITS]
    required.extend([dataset_dir / "report.json", character_dict])
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("missing required files: " + ", ".join(str(path) for path in missing))
    if destination.exists() or destination.is_symlink():
        parser.error(f"output already exists: {destination}")

    try:
        rights_evidence = verify_ocr_training_dataset_rights(
            source_dataset_dir,
            rights_manifest,
            train_label=source_dataset_dir / "train.txt",
            validation_label=source_dataset_dir / "validation.txt",
            character_dict=character_dict,
            evidence_root=rights_evidence_root,
            attestor_context=attestor_context,
        )
        source_dataset = _load_source_dataset(source_dataset_dir)
    except (OcrTrainingRightsError, FileNotFoundError, ValueError) as exc:
        parser.error(f"Model A risk training rights gate failed: {exc}")

    source_inputs = [
        rights_manifest,
        source_dataset_dir / "dataset_report.json",
        *(source_dataset_dir / f"{split}.txt" for split in SPLITS),
    ]
    required = list(dict.fromkeys([*required, *source_inputs]))
    snapshots = {path: _snapshot(path) for path in required}
    trainer_snapshot = _snapshot(Path(__file__).resolve())

    vocabulary = _load_vocabulary(character_dict, snapshots[character_dict].payload)
    rows = {
        split: _load_rows(
            dataset_dir / f"{split}.jsonl",
            snapshots[dataset_dir / f"{split}.jsonl"].payload,
        )
        for split in SPLITS
    }
    split_sha256 = {
        split: snapshots[dataset_dir / f"{split}.jsonl"].sha256 for split in SPLITS
    }
    dataset_report = _validate_dataset_report(
        dataset_dir / "report.json",
        snapshots[dataset_dir / "report.json"].payload,
        rows,
        split_sha256,
    )
    _validate_source_authorization(dataset_report, rows, source_dataset, rights_evidence)
    encoded = {
        "train": _encode_rows(rows[args.train_split], vocabulary, args.max_length),
        "validation": _encode_rows(rows[args.validation_split], vocabulary, args.max_length),
    }
    _require_both_classes(encoded["train"][2], "risk train split")
    _require_both_classes(encoded["validation"][2], "risk validation split")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".partial")
    )

    published = False
    try:
        import paddle
        from paddle import nn

        paddle.device.set_device(args.device)
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        model = _create_model(nn, len(vocabulary), args.max_length)
        optimizer = paddle.optimizer.Adam(
            learning_rate=args.learning_rate, parameters=model.parameters()
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_average_precision = -math.inf
        best_epoch = 0
        history: list[dict[str, float | int]] = []
        best_path = temporary / "model.pdparams"
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses: list[float] = []
            for token_ids, numeric, labels in _batches(
                encoded["train"], args.batch_size, shuffle=True, seed=args.seed + epoch
            ):
                logits = model(paddle.to_tensor(token_ids), paddle.to_tensor(numeric))
                loss = loss_fn(logits, paddle.to_tensor(labels).reshape([-1, 1]))
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                epoch_losses.append(float(loss.numpy().item()))

            validation_scores = _predict_scores(model, encoded["validation"], args.batch_size, paddle)
            validation_ap = average_precision(encoded["validation"][2], validation_scores)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(epoch_losses)),
                    "validation_average_precision": validation_ap,
                }
            )
            print(json.dumps(history[-1], ensure_ascii=False), flush=True)
            if validation_ap > best_average_precision:
                best_average_precision = validation_ap
                best_epoch = epoch
                paddle.save(model.state_dict(), str(best_path))

        model.set_state_dict(paddle.load(str(best_path)))
        validation_labels = encoded["validation"][2]
        validation_scores = _predict_scores(model, encoded["validation"], args.batch_size, paddle)
        threshold = select_threshold(validation_labels, validation_scores, args.target_recall)
        model_metrics = {
            "validation": evaluate_risk(validation_labels, validation_scores, threshold).as_dict(),
        }
        confidence_threshold = select_threshold(
            validation_labels, 1.0 - encoded["validation"][1][:, 0], args.target_recall
        )
        confidence_metrics = {
            "validation": evaluate_risk(
                validation_labels,
                1.0 - encoded["validation"][1][:, 0],
                confidence_threshold,
            ).as_dict(),
        }

        vocabulary_path = temporary / "vocabulary.json"
        vocabulary_path.write_text(
            json.dumps({"tokens": vocabulary}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metrics = {
            "schema_version": "1.0",
            "selection_split": args.validation_split,
            "test_evaluation_performed": False,
            "target_error_recall": args.target_recall,
            "model": model_metrics,
            "confidence_only_baseline": confidence_metrics,
        }
        (temporary / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _verify_snapshots_unchanged([*snapshots.values(), trainer_snapshot])
        _verify_source_authorization_unchanged(
            source_dataset_dir,
            rights_manifest,
            rights_evidence_root,
            attestor_context,
            character_dict,
            source_dataset,
            rights_evidence,
        )
        artifact_hashes = {
            "weights_sha256": _sha256(best_path),
            "vocabulary_sha256": _sha256(vocabulary_path),
            "metrics_sha256": _sha256(temporary / "metrics.json"),
        }
        report = {
            "schema_version": "model-a-line-risk-training-run/1.2",
            "model_id": (
                "model-a-line-risk-cnn-"
                f"{dataset_report['derived_dataset_sha256'][:12]}-seed-{args.seed}"
            ),
            "component": "model_a_line_error_review_router",
            "status": "research_only_non_promotable",
            "non_promotion_reasons": [
                "not_a_complete_model_a",
                "table_formula_role_and_reading_order_gates_unmeasured",
                "training_labels_not_human_verified",
                "upstream_ocr_ancestry_not_cryptographically_verified",
            ],
            "dataset_report_sha256": snapshots[dataset_dir / "report.json"].sha256,
            "dataset_derived_sha256": dataset_report["derived_dataset_sha256"],
            "source_dataset_sha256": source_dataset.sha256,
            "source_dataset_report_sha256": rights_evidence.dataset_report_sha256,
            "split_sha256": split_sha256,
            "split_roles": {
                "risk_train": args.train_split,
                "risk_validation": args.validation_split,
            },
            "upstream_ocr_overlap_policy": {
                "policy": "risk_splits_must_differ_from_fixed_source_ocr_train_split",
                "declared_upstream_training_split": args.upstream_training_split,
                "result": "passed",
                "assurance": "fixed_source_ocr_train_split_contract",
            },
            "split_counts": {
                split: {"lines": len(split_rows), "errors": sum(row["is_error"] for row in split_rows)}
                for split, split_rows in rows.items()
            },
            "character_dict_sha256": snapshots[character_dict].sha256,
            "rights_manifest_sha256": rights_evidence.rights_manifest_sha256,
            "rights_manifest_validation": "verified",
            "rights_document_count": rights_evidence.document_count,
            "rights_identity_assurance": rights_evidence.identity_assurance,
            "rights_evidence_artifacts_sha256": rights_evidence.evidence_artifacts_sha256,
            "source_training_images_sha256": rights_evidence.training_images_sha256,
            "source_derivation_assurance": rights_evidence.derivation_assurance,
            "source_derivation_verification_scope": (
                rights_evidence.derivation_verification_scope
            ),
            "source_derivation_attestor_id": rights_evidence.derivation_attestor_id,
            "seed": args.seed,
            "device": args.device,
            "trainer_sha256": trainer_snapshot.sha256,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "target_error_recall": args.target_recall,
            "loss": "binary_cross_entropy_with_logits_unweighted",
            "architecture": {
                "character_embedding": 48,
                "convolution_channels": 64,
                "convolution_kernels": [3, 5],
                "numeric_features": 6,
                "hidden_units": 128,
            },
            "history": history,
            **artifact_hashes,
        }
        (temporary / "training_run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _verify_snapshots_unchanged([*snapshots.values(), trainer_snapshot])
        _verify_source_authorization_unchanged(
            source_dataset_dir,
            rights_manifest,
            rights_evidence_root,
            attestor_context,
            character_dict,
            source_dataset,
            rights_evidence,
        )
        _verify_artifact_hashes(temporary, artifact_hashes)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output appeared during training: {destination}")
        _publish_directory_create_only(temporary, destination)
        published = True
        print(json.dumps({"output": str(destination), **metrics}, ensure_ascii=False, indent=2))
        return 0
    finally:
        if not published and temporary.exists():
            remove_staging_directory(temporary, parent=destination.parent)


def _resolve(project_root: Path, path: Path) -> Path:
    return path.absolute() if path.is_absolute() else (project_root / path).absolute()


def _validate_split_selection(
    train_split: str,
    validation_split: str,
    upstream_training_split: str,
) -> None:
    if upstream_training_split != "train":
        raise ValueError("upstream OCR training split is fixed to train by the source contract")
    if train_split == validation_split:
        raise ValueError("risk train and validation splits must differ")
    overlapping = [
        role
        for role, split in (("risk train", train_split), ("risk validation", validation_split))
        if split == upstream_training_split
    ]
    if overlapping:
        raise ValueError(
            ", ".join(overlapping)
            + " split overlaps the declared upstream OCR training split"
        )


def _snapshot(path: Path) -> FileSnapshot:
    payload = read_bounded_regular_file(
        path,
        max_bytes=_MAX_SNAPSHOT_BYTES,
        label=f"training input {path}",
    )
    return FileSnapshot(path=path, payload=payload, sha256=_sha256_bytes(payload))


def _verify_snapshots_unchanged(snapshots: list[FileSnapshot]) -> None:
    for snapshot in snapshots:
        try:
            current_sha256 = _sha256(snapshot.path)
        except OSError as exc:
            raise RuntimeError(f"snapshotted input became unreadable: {snapshot.path}") from exc
        if current_sha256 != snapshot.sha256:
            raise RuntimeError(f"snapshotted input changed during training: {snapshot.path}")


def _validate_dataset_report(
    path: Path,
    payload: bytes,
    rows: dict[str, list[dict[str, Any]]],
    split_sha256: dict[str, str],
) -> dict[str, Any]:
    report = _load_json_object(path, payload)
    _require_exact_fields(
        report,
        {
            "schema_version",
            "record_fields",
            "source_dataset_sha256",
            "source_dataset_digest_definition",
            "source_rights",
            "derived_dataset_sha256",
            "model_sha256",
            "model_digest_definition",
            "device",
            "batch_size",
            "exact_match_definition",
            "cer_definition",
            "filters",
            "splits",
            "totals",
        },
        "dataset report",
    )
    if report["schema_version"] != "model-a-line-risk-dataset/1.2":
        raise ValueError("dataset report has unsupported schema_version")
    if report["record_fields"] != list(RECORD_FIELDS):
        raise ValueError("dataset report record_fields do not match trainer contract")
    if report["source_dataset_digest_definition"] != (
        "split-tsv-and-referenced-image-bytes/v1"
    ):
        raise ValueError("dataset report has unsupported source dataset digest definition")
    if report["model_digest_definition"] != "relative-file-path-and-bytes/v1":
        raise ValueError("dataset report has unsupported model digest definition")
    if report["exact_match_definition"] != "raw-unicode-codepoint-exact/v1":
        raise ValueError("dataset report has unsupported exact-match definition")
    _require_sha256(report["source_dataset_sha256"], "dataset report source_dataset_sha256")
    _require_sha256(report["model_sha256"], "dataset report model_sha256")
    _require_sha256(report["derived_dataset_sha256"], "dataset report derived_dataset_sha256")
    _require_nonempty_string(report["device"], "dataset report device")
    _require_positive_int(report["batch_size"], "dataset report batch_size")
    _require_nonempty_string(report["cer_definition"], "dataset report cer_definition")
    source_rights = _require_exact_fields(
        report["source_rights"],
        {
            "validation",
            "rights_manifest_sha256",
            "dataset_report_sha256",
            "document_count",
            "identity_assurance",
            "evidence_artifacts_sha256",
            "training_images_sha256",
            "derivation_assurance",
            "derivation_verification_scope",
            "derivation_attestor_id",
        },
        "dataset report source rights",
    )
    if source_rights["validation"] != "verified":
        raise ValueError("dataset report source rights are not verified")
    _require_sha256(
        source_rights["rights_manifest_sha256"],
        "dataset report source rights manifest sha256",
    )
    if source_rights["derivation_assurance"] != (
        "authenticated_external_report_attestation"
    ):
        raise ValueError("dataset report source derivation is not externally trusted")
    if source_rights["derivation_verification_scope"] != (
        "report_digest_and_artifact_integrity_only"
    ):
        raise ValueError("dataset report overstates source derivation verification")
    _require_nonempty_string(
        source_rights["derivation_attestor_id"],
        "dataset report source derivation attestor",
    )
    _require_sha256(
        source_rights["dataset_report_sha256"],
        "dataset report source dataset report sha256",
    )
    _require_positive_int(
        source_rights["document_count"],
        "dataset report source rights document_count",
    )
    if source_rights["identity_assurance"] != "authenticated_external_context":
        raise ValueError("dataset report source rights identity is not authenticated")
    _require_sha256(
        source_rights["evidence_artifacts_sha256"],
        "dataset report rights evidence artifacts sha256",
    )
    _require_sha256(
        source_rights["training_images_sha256"],
        "dataset report source training images sha256",
    )

    split_reports = _require_exact_fields(report["splits"], set(SPLITS), "dataset report splits")
    split_fields = {
        "source_count",
        "after_image_filter_count",
        "final_count",
        "removed_cross_split_image_sha256",
        "removed_cross_split_observed_target_pair",
        "source_tsv_sha256",
        "source_images_sha256",
        "output_jsonl_sha256",
        "exact_match_count",
        "exact_match_rate",
        "character_errors",
        "character_denominator",
        "cer",
    }
    count_fields = {
        "source_count",
        "after_image_filter_count",
        "final_count",
        "removed_cross_split_image_sha256",
        "removed_cross_split_observed_target_pair",
        "exact_match_count",
        "character_errors",
        "character_denominator",
    }
    additive_fields = tuple(count_fields)
    totals_from_splits = {field: 0 for field in additive_fields}
    for split in SPLITS:
        split_report = _require_exact_fields(
            split_reports[split], split_fields, f"dataset report {split} split"
        )
        for field in count_fields:
            _require_nonnegative_int(split_report[field], f"dataset report {split}.{field}")
            totals_from_splits[field] += split_report[field]
        for field in ("source_tsv_sha256", "source_images_sha256", "output_jsonl_sha256"):
            _require_sha256(split_report[field], f"dataset report {split}.{field}")
        _require_probability(split_report["exact_match_rate"], f"dataset report {split}.exact_match_rate")
        _require_probability(split_report["cer"], f"dataset report {split}.cer")

        if split_report["output_jsonl_sha256"] != split_sha256[split]:
            raise ValueError(f"dataset report {split} JSONL sha256 mismatch")
        if split_report["final_count"] != len(rows[split]):
            raise ValueError(f"dataset report {split} final_count mismatch")
        exact_matches = sum(not row["is_error"] for row in rows[split])
        if split_report["exact_match_count"] != exact_matches:
            raise ValueError(f"dataset report {split} exact_match_count mismatch")
        if not math.isclose(
            split_report["exact_match_rate"], exact_matches / len(rows[split]), abs_tol=1e-12
        ):
            raise ValueError(f"dataset report {split} exact_match_rate mismatch")
        if (
            split_report["source_count"]
            - split_report["removed_cross_split_image_sha256"]
            != split_report["after_image_filter_count"]
        ):
            raise ValueError(f"dataset report {split} image-removal counts are inconsistent")
        if (
            split_report["after_image_filter_count"]
            - split_report["removed_cross_split_observed_target_pair"]
            != split_report["final_count"]
        ):
            raise ValueError(f"dataset report {split} pair-removal counts are inconsistent")
        if split_report["character_denominator"] < 1:
            raise ValueError(f"dataset report {split} character denominator must be positive")
        if not math.isclose(
            split_report["cer"],
            split_report["character_errors"] / split_report["character_denominator"],
            abs_tol=1e-12,
        ):
            raise ValueError(f"dataset report {split} CER counts are inconsistent")

    totals = _require_exact_fields(
        report["totals"],
        {
            "source_count",
            "after_image_filter_count",
            "final_count",
            "removed_cross_split_image_sha256",
            "removed_cross_split_observed_target_pair",
            "exact_match_count",
            "exact_match_rate",
            "character_errors",
            "character_denominator",
            "cer",
        },
        "dataset report totals",
    )
    for field in count_fields:
        _require_nonnegative_int(totals[field], f"dataset report totals.{field}")
        if totals[field] != totals_from_splits[field]:
            raise ValueError(f"dataset report totals.{field} mismatch")
    _require_probability(totals["exact_match_rate"], "dataset report totals.exact_match_rate")
    _require_probability(totals["cer"], "dataset report totals.cer")
    if not math.isclose(
        totals["exact_match_rate"], totals["exact_match_count"] / totals["final_count"], abs_tol=1e-12
    ):
        raise ValueError("dataset report totals exact_match_rate mismatch")
    if not math.isclose(
        totals["cer"], totals["character_errors"] / totals["character_denominator"], abs_tol=1e-12
    ):
        raise ValueError("dataset report totals CER mismatch")

    filters = _require_exact_fields(
        report["filters"],
        {
            "cross_split_image_sha256",
            "cross_split_observed_target_pair",
            "final_cross_split_image_sha256_count",
            "final_cross_split_observed_target_pair_count",
        },
        "dataset report filters",
    )
    image_filter = _require_exact_fields(
        filters["cross_split_image_sha256"],
        {"fingerprint_count", "removed_by_split"},
        "dataset report image filter",
    )
    pair_filter = _require_exact_fields(
        filters["cross_split_observed_target_pair"],
        {"fingerprint_definition", "fingerprint_count", "removed_by_split"},
        "dataset report text-pair filter",
    )
    if pair_filter["fingerprint_definition"] != "nfc-observed-text-and-nfc-target-text/v1":
        raise ValueError("dataset report has unsupported text-pair fingerprint definition")
    for label, filter_report in (("image", image_filter), ("text-pair", pair_filter)):
        _require_nonnegative_int(
            filter_report["fingerprint_count"], f"dataset report {label} fingerprint_count"
        )
        removals = _require_exact_fields(
            filter_report["removed_by_split"], set(SPLITS), f"dataset report {label} removals"
        )
        for split in SPLITS:
            _require_nonnegative_int(removals[split], f"dataset report {label} {split} removals")
    if image_filter["removed_by_split"] != {
        split: split_reports[split]["removed_cross_split_image_sha256"] for split in SPLITS
    }:
        raise ValueError("dataset report image-removal details mismatch")
    if pair_filter["removed_by_split"] != {
        split: split_reports[split]["removed_cross_split_observed_target_pair"] for split in SPLITS
    }:
        raise ValueError("dataset report text-pair removal details mismatch")
    for field in (
        "final_cross_split_image_sha256_count",
        "final_cross_split_observed_target_pair_count",
    ):
        if filters[field] != 0 or isinstance(filters[field], bool):
            raise ValueError(f"dataset report {field} must be integer zero")

    image_owners: dict[str, set[str]] = {}
    pair_owners: dict[tuple[str, str], set[str]] = {}
    for split in SPLITS:
        for row in rows[split]:
            image_owners.setdefault(row["image_sha256"], set()).add(split)
            pair = (
                unicodedata.normalize("NFC", row["observed_text"]),
                unicodedata.normalize("NFC", row["target_text"]),
            )
            pair_owners.setdefault(pair, set()).add(split)
    if any(len(owners) > 1 for owners in image_owners.values()):
        raise ValueError("derived dataset has cross-split image sha256 leakage")
    if any(len(owners) > 1 for owners in pair_owners.values()):
        raise ValueError("derived dataset has cross-split observed/target pair leakage")

    derived_digest = hashlib.sha256(b"model-a-line-risk-derived-dataset/v1\0")
    for split in SPLITS:
        derived_digest.update(split.encode("ascii"))
        derived_digest.update(bytes.fromhex(split_sha256[split]))
    if report["derived_dataset_sha256"] != derived_digest.hexdigest():
        raise ValueError("dataset report derived_dataset_sha256 mismatch")
    return report


def _validate_source_authorization(
    report: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    source_dataset: SourceDataset,
    rights_evidence: OcrTrainingRightsEvidence,
) -> None:
    if report["source_dataset_sha256"] != source_dataset.sha256:
        raise ValueError("risk dataset is not bound to the supplied source dataset")
    for split in SPLITS:
        split_report = cast(dict[str, Any], report["splits"])[split]
        if split_report["source_count"] != len(source_dataset.samples[split]):
            raise ValueError(f"risk dataset {split} source count mismatch")
        if split_report["source_tsv_sha256"] != source_dataset.split_tsv_sha256[split]:
            raise ValueError(f"risk dataset {split} source label digest mismatch")
        if split_report["source_images_sha256"] != source_dataset.split_images_sha256[split]:
            raise ValueError(f"risk dataset {split} source image digest mismatch")
        source_by_ref = {
            sample.image_ref: sample for sample in source_dataset.samples[split]
        }
        seen_refs: set[str] = set()
        for row in rows[split]:
            image_ref = cast(str, row["image_ref"])
            if image_ref in seen_refs:
                raise ValueError(f"risk dataset {split} has duplicate source image_ref")
            seen_refs.add(image_ref)
            source = source_by_ref.get(image_ref)
            if source is None:
                raise ValueError(f"risk dataset {split} row is absent from the source dataset")
            if row["image_sha256"] != source.image_sha256:
                raise ValueError(f"risk dataset {split} row source image digest mismatch")
            if row["target_text"] != source.target_text:
                raise ValueError(f"risk dataset {split} row target text mismatch")

    source_rights = cast(dict[str, Any], report["source_rights"])
    expected = {
        "validation": "verified",
        "rights_manifest_sha256": rights_evidence.rights_manifest_sha256,
        "dataset_report_sha256": rights_evidence.dataset_report_sha256,
        "document_count": rights_evidence.document_count,
        "identity_assurance": rights_evidence.identity_assurance,
        "evidence_artifacts_sha256": rights_evidence.evidence_artifacts_sha256,
        "training_images_sha256": rights_evidence.training_images_sha256,
        "derivation_assurance": rights_evidence.derivation_assurance,
        "derivation_verification_scope": rights_evidence.derivation_verification_scope,
        "derivation_attestor_id": rights_evidence.derivation_attestor_id,
    }
    if source_rights != expected:
        raise ValueError("risk dataset source rights binding mismatch")


def _verify_source_authorization_unchanged(
    source_dataset_dir: Path,
    rights_manifest: Path,
    rights_evidence_root: Path,
    attestor_context: AuthenticatedAttestorContext | None,
    character_dict: Path,
    expected_source_dataset: SourceDataset,
    expected_rights_evidence: OcrTrainingRightsEvidence,
) -> None:
    current_rights_evidence = verify_ocr_training_dataset_rights(
        source_dataset_dir,
        rights_manifest,
        train_label=source_dataset_dir / "train.txt",
        validation_label=source_dataset_dir / "validation.txt",
        character_dict=character_dict,
        evidence_root=rights_evidence_root,
        attestor_context=attestor_context,
    )
    if current_rights_evidence != expected_rights_evidence:
        raise RuntimeError("source dataset rights binding changed during training")
    if _load_source_dataset(source_dataset_dir).sha256 != expected_source_dataset.sha256:
        raise RuntimeError("source dataset changed during training")


def _load_json_object(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require_exact_fields(
    value: Any, expected: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has unexpected fields")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    result = _require_nonnegative_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _require_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def _require_both_classes(labels: np.ndarray, label: str) -> None:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError(f"{label} must contain both correct and erroneous OCR lines")


def _verify_artifact_hashes(root: Path, expected: dict[str, str]) -> None:
    paths = {
        "weights_sha256": root / "model.pdparams",
        "vocabulary_sha256": root / "vocabulary.json",
        "metrics_sha256": root / "metrics.json",
    }
    for field, path in paths.items():
        if _sha256(path) != expected[field]:
            raise RuntimeError(f"staged artifact changed before publish: {path.name}")


def _load_vocabulary(path: Path, payload: bytes) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"character dictionary is not UTF-8: {path}") from exc
    characters = [line.rstrip("\r\n") for line in text.splitlines()]
    characters = [character for character in characters if character]
    if not characters:
        raise ValueError("character dictionary is empty")
    if any(character in SPECIAL_TOKENS for character in characters):
        raise ValueError("character dictionary collides with reserved tokens")
    return [*SPECIAL_TOKENS, *dict.fromkeys(characters)]


def _load_rows(path: Path, payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"dataset split is not UTF-8: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or set(row) != EXPECTED_FIELDS:
            raise ValueError(f"{path}:{line_number}: unexpected record fields")
        if not isinstance(row["image_ref"], str) or not row["image_ref"]:
            raise ValueError(f"{path}:{line_number}: invalid image_ref")
        _require_sha256(row["image_sha256"], f"{path}:{line_number}: image_sha256")
        for field in ("observed_text", "target_text"):
            if not isinstance(row[field], str):
                raise TypeError(f"{path}:{line_number}: invalid {field}")
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"{path}:{line_number}: invalid confidence")
        if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"{path}:{line_number}: confidence must be finite and in [0, 1]")
        if not isinstance(row["is_error"], bool):
            raise TypeError(f"{path}:{line_number}: invalid is_error")
        if row["is_error"] != (row["observed_text"] != row["target_text"]):
            raise ValueError(f"{path}:{line_number}: is_error does not match texts")
        rows.append(row)
    if not rows:
        raise ValueError(f"dataset split is empty: {path}")
    return rows


def _encode_rows(
    rows: list[dict[str, Any]], vocabulary: list[str], max_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    tokens = np.zeros((len(rows), max_length), dtype=np.int64)
    numeric = np.zeros((len(rows), 6), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        text = str(row["observed_text"])
        clipped = text[:max_length]
        if clipped:
            tokens[index, : len(clipped)] = [token_to_id.get(character, 1) for character in clipped]
        else:
            tokens[index, 0] = 1
        denominator = max(1, len(text))
        numeric[index] = (
            float(row["confidence"]),
            min(1.0, len(text) / max_length),
            sum("가" <= char <= "힣" for char in text) / denominator,
            sum(char.isdigit() for char in text) / denominator,
            sum(char.isascii() and char.isalpha() for char in text) / denominator,
            sum(not char.isalnum() and not char.isspace() for char in text) / denominator,
        )
        labels[index] = float(row["is_error"])
    return tokens, numeric, labels


def _create_model(nn: Any, vocabulary_size: int, max_length: int) -> Any:
    class LineRiskCNN(nn.Layer):  # type: ignore[misc]  # Paddle is imported lazily.
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, 48, padding_idx=0)
            self.conv3 = nn.Conv1D(48, 64, 3, padding=1)
            self.conv5 = nn.Conv1D(48, 64, 5, padding=2)
            self.hidden = nn.Linear(64 * 4 + 6, 128)
            self.dropout = nn.Dropout(0.2)
            self.output = nn.Linear(128, 1)
            self.max_length = max_length

        def forward(self, token_ids: Any, numeric: Any) -> Any:
            import paddle
            from paddle.nn import functional

            embedded = self.embedding(token_ids).transpose([0, 2, 1])
            mask = (token_ids != 0).astype("float32").unsqueeze(1)
            denominator = paddle.clip(mask.sum(axis=2), min=1.0)
            pooled: list[Any] = []
            for convolution in (self.conv3, self.conv5):
                encoded = functional.relu(convolution(embedded))
                pooled.append((encoded * mask).sum(axis=2) / denominator)
                pooled.append((encoded + (1.0 - mask) * -1e4).max(axis=2))
            features = paddle.concat([*pooled, numeric], axis=1)
            return self.output(self.dropout(functional.relu(self.hidden(features))))

    return LineRiskCNN()


def _batches(
    encoded: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    token_ids, numeric, labels = encoded
    indices = np.arange(len(labels))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [
        (token_ids[selection], numeric[selection], labels[selection])
        for start in range(0, len(indices), batch_size)
        if len(selection := indices[start : start + batch_size])
    ]


def _predict_scores(
    model: Any, encoded: tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int, paddle: Any
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with paddle.no_grad():
        for token_ids, numeric, _ in _batches(encoded, batch_size, shuffle=False, seed=0):
            logits = model(paddle.to_tensor(token_ids), paddle.to_tensor(numeric))
            scores.append(paddle.nn.functional.sigmoid(logits).numpy().reshape(-1))
    return cast(np.ndarray, np.concatenate(scores).astype(np.float64))


def select_threshold(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    _validate_metric_inputs(labels, scores)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("threshold selection needs at least one positive example")
    candidates = np.unique(scores)
    best: tuple[float, float] | None = None
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = int((predicted & (labels == 1)).sum())
        recall = true_positive / positives
        if recall + 1e-12 < target_recall:
            continue
        precision = true_positive / max(1, int(predicted.sum()))
        candidate = (precision, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return float(np.nextafter(scores.min(), -math.inf))
    return best[1]


def evaluate_risk(labels: np.ndarray, scores: np.ndarray, threshold: float) -> RiskMetrics:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    _validate_metric_inputs(labels, scores)
    predicted = scores >= threshold
    true_positive = int((predicted & (labels == 1)).sum())
    false_positive = int((predicted & (labels == 0)).sum())
    false_negative = int(((~predicted) & (labels == 1)).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return RiskMetrics(
        samples=len(labels),
        errors=int(labels.sum()),
        roc_auc=roc_auc(labels, scores),
        average_precision=average_precision(labels, scores),
        brier_score=float(np.mean((scores - labels) ** 2)),
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    _validate_metric_inputs(labels, scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC needs both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    _validate_metric_inputs(labels, scores)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("average precision needs at least one positive example")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative = np.cumsum(sorted_labels)
    result = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_positives = int(sorted_labels[start:end].sum())
        if group_positives:
            result += (group_positives / positives) * (cumulative[end - 1] / end)
        start = end
    return float(result)


def _validate_metric_inputs(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("labels and scores must be equally sized non-empty vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")


def _sha256(path: Path) -> str:
    return sha256_bounded_regular_file(
        path,
        max_bytes=_MAX_HASHED_ARTIFACT_BYTES,
        label=f"training artifact {path}",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
