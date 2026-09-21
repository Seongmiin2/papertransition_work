from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.ocr_training_rights import (
    AuthenticatedAttestorContext,
    OcrTrainingRightsError,
    verify_ocr_training_dataset_rights,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    publish_directory_create_only as _publish_directory_create_only,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    publish_file_create_only as _publish_file_create_only,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    read_bounded_regular_file,
    remove_staging_directory,
    sha256_bounded_regular_file,
)

PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
)
_MAX_MODEL_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
_MAX_DATASET_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_DATASET_IMAGE_BYTES = 64 * 1024 * 1024


def main(*, attestor_context: AuthenticatedAttestorContext | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv5 for Korean exam sheets")
    parser.add_argument("--paddleocr-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("output/ocr-training"))
    parser.add_argument(
        "--rights-manifest",
        type=Path,
        required=True,
        help="Strict per-source rights and split manifest used to build this dataset",
    )
    parser.add_argument(
        "--rights-evidence-root",
        type=Path,
        required=True,
        help="Workspace root containing the attestation evidence artifact:// paths",
    )
    parser.add_argument("--output", type=Path, default=Path("output/trained-korean-ocr"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("ai/modeling/pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("ai/modeling/configs/korean_exam_mobile_rec.yml")
    )
    parser.add_argument("--train-label", type=Path)
    parser.add_argument("--validation-label", type=Path)
    parser.add_argument("--character-dict", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--download-pretrained", action="store_true")
    parser.add_argument(
        "--cpu", action="store_true", help="Allowed for smoke tests; real training needs a GPU"
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if attestor_context is None:
        parser.error(
            "OCR training rights gate failed: OCR training requires identities "
            "from an authenticated external context"
        )
    project_root = PROJECT_ROOT
    paddle_repo = args.paddleocr_repo.resolve()
    dataset = _project_path(project_root, args.dataset)
    output = (project_root / args.output).resolve()
    pretrained = (project_root / args.pretrained).resolve()
    config = (project_root / args.config).resolve()
    train_label = (
        _project_path(project_root, args.train_label) if args.train_label else dataset / "train.txt"
    )
    validation_label = (
        _project_path(project_root, args.validation_label)
        if args.validation_label
        else dataset / "validation.txt"
    )
    character_dict = (
        _project_path(project_root, args.character_dict)
        if args.character_dict
        else dataset / "korean_exam_dict.txt"
    )
    rights_manifest = _project_path(project_root, args.rights_manifest)
    rights_evidence_root = _project_path(project_root, args.rights_evidence_root)
    if output.exists() or output.is_symlink():
        parser.error(f"output already exists: {output}")
    required = [
        paddle_repo / "tools" / "train.py",
        paddle_repo / "tools" / "eval.py",
        paddle_repo / "tools" / "export_model.py",
        train_label,
        validation_label,
        character_dict,
        config,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("missing required files: " + ", ".join(str(path) for path in missing))
    try:
        rights_evidence = verify_ocr_training_dataset_rights(
            dataset,
            rights_manifest,
            train_label=train_label,
            validation_label=validation_label,
            character_dict=character_dict,
            evidence_root=rights_evidence_root,
            attestor_context=attestor_context,
        )
    except OcrTrainingRightsError as exc:
        parser.error(f"OCR training rights gate failed: {exc}")
    try:
        _reject_incomplete_pretrained_download(pretrained)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not pretrained.is_file():
        if not args.download_pretrained:
            parser.error("pretrained weights missing; pass --download-pretrained")
        _download_pretrained(pretrained)
    pretrained_sha256 = _sha256(pretrained)
    current_rights = verify_ocr_training_dataset_rights(
        dataset,
        rights_manifest,
        train_label=train_label,
        validation_label=validation_label,
        character_dict=character_dict,
        evidence_root=rights_evidence_root,
        attestor_context=attestor_context,
    )
    if current_rights != rights_evidence:
        parser.error("OCR training rights inputs changed before training")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            # Keep room for the source-digest directory in snapshotted image
            # paths on Windows, where legacy MAX_PATH may still apply.
            prefix=".ocr.",
            suffix=".partial",
        )
    )
    published = False
    try:
        training_dataset = staging / ".verified-input"
        _copy_verified_dataset(dataset, training_dataset)
        snapshot_rights = verify_ocr_training_dataset_rights(
            training_dataset,
            rights_manifest,
            train_label=training_dataset / "train.txt",
            validation_label=training_dataset / "validation.txt",
            character_dict=training_dataset / "korean_exam_dict.txt",
            evidence_root=rights_evidence_root,
            attestor_context=attestor_context,
        )
        if snapshot_rights != rights_evidence:
            raise RuntimeError("verified OCR training snapshot does not match source inputs")
        training_train_label = training_dataset / "train.txt"
        training_validation_label = training_dataset / "validation.txt"
        training_character_dict = training_dataset / "korean_exam_dict.txt"
        overrides = [
            f"Global.use_gpu={str(not args.cpu).lower()}",
            f"Global.pretrained_model={pretrained}",
            f"Global.save_model_dir={staging}",
            f"Global.character_dict_path={training_character_dict}",
            f"Global.epoch_num={args.epochs}",
            "Global.distributed=false",
            "Global.use_visualdl=false",
            "Global.save_epoch_step=1",
            f"Global.eval_batch_step=[0,{args.eval_every}]",
            f"Optimizer.lr.learning_rate={args.learning_rate:.10f}",
            f"Optimizer.lr.warmup_epoch={args.warmup_epochs}",
            f"Train.dataset.data_dir={training_dataset}",
            f"Train.dataset.label_file_list=['{training_train_label}']",
            f"Train.sampler.first_bs={args.batch_size}",
            f"Train.loader.batch_size_per_card={args.batch_size}",
            f"Train.loader.num_workers={args.workers}",
            f"Eval.dataset.data_dir={training_dataset}",
            f"Eval.dataset.label_file_list=['{training_validation_label}']",
            f"Eval.loader.batch_size_per_card={args.batch_size}",
            f"Eval.loader.num_workers={args.workers}",
        ]
        command = [
            sys.executable,
            str(paddle_repo / "tools" / "train.py"),
            "-c",
            str(config),
            "-o",
            *overrides,
        ]
        print("running:", subprocess.list2cmdline(command))
        training = subprocess.run(command, cwd=paddle_repo, check=False)
        if training.returncode:
            return training.returncode

        checkpoint = staging / "best_accuracy"
        eval_command = [
            sys.executable,
            str(paddle_repo / "tools" / "eval.py"),
            "-c",
            str(config),
            "-o",
            *overrides,
            f"Global.checkpoints={checkpoint}",
        ]
        print("evaluating:", subprocess.list2cmdline(eval_command))
        evaluation = subprocess.run(eval_command, cwd=paddle_repo, check=False)
        if evaluation.returncode:
            return evaluation.returncode

        inference_dir = staging / "inference"
        if not args.no_export:
            export_command = [
                sys.executable,
                str(paddle_repo / "tools" / "export_model.py"),
                "-c",
                str(config),
                "-o",
                *overrides,
                f"Global.pretrained_model={checkpoint}",
                f"Global.save_inference_dir={inference_dir}",
            ]
            print("exporting:", subprocess.list2cmdline(export_command))
            exported = subprocess.run(export_command, cwd=paddle_repo, check=False)
            if exported.returncode:
                return exported.returncode

        final_rights = verify_ocr_training_dataset_rights(
            dataset,
            rights_manifest,
            train_label=train_label,
            validation_label=validation_label,
            character_dict=character_dict,
            evidence_root=rights_evidence_root,
            attestor_context=attestor_context,
        )
        if final_rights != rights_evidence:
            raise RuntimeError("OCR training rights inputs changed during training")
        final_snapshot_rights = verify_ocr_training_dataset_rights(
            training_dataset,
            rights_manifest,
            train_label=training_train_label,
            validation_label=training_validation_label,
            character_dict=training_character_dict,
            evidence_root=rights_evidence_root,
            attestor_context=attestor_context,
        )
        if final_snapshot_rights != rights_evidence:
            raise RuntimeError("verified OCR training snapshot changed during training")
        if _sha256(pretrained) != pretrained_sha256:
            raise RuntimeError("OCR pretrained artifact changed during training")

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paddle_repo,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        report = {
            "schema_version": "1.1",
            "paddleocr_commit": commit,
            "dataset": str(dataset),
            "dataset_report_sha256": rights_evidence.dataset_report_sha256,
            "rights_manifest": str(rights_manifest),
            "rights_manifest_sha256": rights_evidence.rights_manifest_sha256,
            "rights_document_count": rights_evidence.document_count,
            "rights_identity_assurance": rights_evidence.identity_assurance,
            "rights_evidence_artifacts_sha256": rights_evidence.evidence_artifacts_sha256,
            "training_images_sha256": rights_evidence.training_images_sha256,
            "derivation_assurance": rights_evidence.derivation_assurance,
            "derivation_verification_scope": (
                rights_evidence.derivation_verification_scope
            ),
            "derivation_attestor_id": rights_evidence.derivation_attestor_id,
            "train_label": str(train_label),
            "validation_label": str(validation_label),
            "character_dict": str(character_dict),
            "pretrained": str(pretrained),
            "pretrained_sha256": pretrained_sha256,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_epochs": args.warmup_epochs,
            "checkpoint": str(output / "best_accuracy"),
            "inference_dir": str(output / "inference") if not args.no_export else None,
        }
        (staging / "training_run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.rmtree(training_dataset)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during training: {output}")
        _publish_directory_create_only(staging, output)
        published = True
        return 0
    finally:
        if not published and staging.exists():
            remove_staging_directory(staging, parent=output.parent)


def _sha256(path: Path) -> str:
    return sha256_bounded_regular_file(
        path,
        max_bytes=_MAX_MODEL_ARTIFACT_BYTES,
        label=f"OCR model artifact {path}",
    )


def _project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _copy_verified_dataset(source: Path, destination: Path) -> None:
    destination.mkdir()
    for name in (
        "dataset_report.json",
        "train.txt",
        "validation.txt",
        "test.txt",
        "korean_exam_dict.txt",
    ):
        source_path = source / name
        payload = read_bounded_regular_file(
            source_path,
            max_bytes=_MAX_DATASET_ARTIFACT_BYTES,
            label=f"verified OCR input {name}",
        )
        (destination / name).write_bytes(payload)

    image_split_by_sha256: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        label_path = destination / f"{split}.txt"
        payload = read_bounded_regular_file(
            label_path,
            max_bytes=_MAX_DATASET_ARTIFACT_BYTES,
            label=f"verified OCR snapshot label {split}",
        )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("verified OCR label is not UTF-8") from exc
        raw_lines = text.split("\n")
        if raw_lines and raw_lines[-1] == "":
            raw_lines.pop()
        for raw_line in raw_lines:
            if raw_line.count("\t") != 1:
                raise RuntimeError("verified OCR label changed before snapshot")
            image_ref, _ = raw_line.removesuffix("\r").split("\t")
            ref = PurePosixPath(image_ref)
            if (
                ref.is_absolute()
                or ref.as_posix() != image_ref
                or "\\" in image_ref
                or ":" in image_ref
                or any(part in {"", ".", ".."} for part in ref.parts)
                or len(ref.parts) != 4
                or ref.parts[:2] != ("images", split)
                or re.fullmatch(r"[0-9a-f]{64}", ref.parts[2]) is None
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.png", ref.parts[3])
                is None
            ):
                raise RuntimeError("verified OCR lineage ref changed before snapshot")
            source_image = source.joinpath(*ref.parts)
            cursor = source
            for part in ref.parts:
                cursor /= part
                metadata = cursor.lstat()
                reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                if stat.S_ISLNK(metadata.st_mode) or bool(
                    getattr(metadata, "st_file_attributes", 0) & reparse_attribute
                ):
                    raise RuntimeError("verified OCR image became a reparse path")
            target_image = destination.joinpath(*ref.parts)
            target_image.parent.mkdir(parents=True, exist_ok=True)
            image_payload = read_bounded_regular_file(
                source_image,
                max_bytes=_MAX_DATASET_IMAGE_BYTES,
                label=f"verified OCR image {image_ref}",
            )
            image_sha256 = hashlib.sha256(image_payload).hexdigest()
            previous_split = image_split_by_sha256.setdefault(image_sha256, split)
            if previous_split != split:
                raise RuntimeError(
                    "verified OCR snapshot has cross-split image byte leakage"
                )
            target_image.write_bytes(image_payload)


def _download_pretrained(pretrained: Path) -> None:
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    marker = _pretrained_transaction_marker(pretrained)
    marker_descriptor: int | None = None
    staging: Path | None = None
    staging_identity: os.stat_result | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= getattr(os, flag_name, 0)
        try:
            marker_descriptor = os.open(marker, flags, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                f"incomplete pretrained download marker exists: {marker}"
            ) from exc
        _write_download_marker(
            marker_descriptor,
            {
                "schema_version": "ocr-pretrained-download-transaction/1.0",
                "status": "downloading",
                "source_url": PRETRAINED_URL,
                "destination_name": pretrained.name,
                "staged_sha256": None,
            },
        )
        descriptor, staging_name = tempfile.mkstemp(
            dir=pretrained.parent,
            prefix=f".{pretrained.name}.",
            suffix=".partial",
        )
        staging_identity = os.fstat(descriptor)
        os.close(descriptor)
        staging = Path(staging_name)
        print(f"downloading {PRETRAINED_URL} -> {pretrained}")
        urllib.request.urlretrieve(PRETRAINED_URL, staging)
        staged_sha256 = _sha256(staging)
        _write_download_marker(
            marker_descriptor,
            {
                "schema_version": "ocr-pretrained-download-transaction/1.0",
                "status": "downloaded_unpublished",
                "source_url": PRETRAINED_URL,
                "destination_name": pretrained.name,
                "staged_sha256": staged_sha256,
            },
        )
        if pretrained.exists() or pretrained.is_symlink():
            raise FileExistsError(f"pretrained output appeared during download: {pretrained}")
        _publish_file_create_only(staging, pretrained)
        if _sha256(pretrained) != staged_sha256:
            raise RuntimeError("published pretrained artifact digest mismatch")
        _write_download_marker(
            marker_descriptor,
            {
                "schema_version": "ocr-pretrained-download-transaction/1.0",
                "status": "published_verified",
                "source_url": PRETRAINED_URL,
                "destination_name": pretrained.name,
                "staged_sha256": staged_sha256,
            },
        )
        descriptor_metadata = os.fstat(marker_descriptor)
        if not _unlink_regular_file_if_same(marker, descriptor_metadata):
            raise RuntimeError("pretrained download transaction marker was replaced")
    finally:
        if marker_descriptor is not None:
            os.close(marker_descriptor)
        if staging is not None and staging_identity is not None:
            _unlink_regular_file_if_same(staging, staging_identity)


def _pretrained_transaction_marker(pretrained: Path) -> Path:
    return pretrained.with_name(f".{pretrained.name}.download-incomplete.json")


def _reject_incomplete_pretrained_download(pretrained: Path) -> None:
    marker = _pretrained_transaction_marker(pretrained)
    if marker.exists() or marker.is_symlink():
        raise RuntimeError(
            "pretrained artifact is quarantined by an incomplete download marker: "
            f"{marker}"
        )


def _write_download_marker(descriptor: int, value: dict[str, object]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count < 1:
            raise OSError("failed to write pretrained download transaction marker")
        written += count
    os.fsync(descriptor)


def _unlink_regular_file_if_same(path: Path, expected: os.stat_result) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return True
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not os.path.samestat(expected, current)
        or stat.S_ISLNK(current.st_mode)
        or bool(getattr(current, "st_file_attributes", 0) & reparse_attribute)
        or not stat.S_ISREG(current.st_mode)
    ):
        return False
    path.unlink()
    return True


if __name__ == "__main__":
    raise SystemExit(main())
