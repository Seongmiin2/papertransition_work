from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv5 for Korean exam sheets")
    parser.add_argument("--paddleocr-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("output/ocr-training"))
    parser.add_argument("--output", type=Path, default=Path("output/trained-korean-ocr"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("models/korean_PP-OCRv5_mobile_rec_pretrained.pdparams"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("training/paddleocr/korean_exam_mobile_rec.yml")
    )
    parser.add_argument("--train-label", type=Path)
    parser.add_argument("--validation-label", type=Path)
    parser.add_argument("--character-dict", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--download-pretrained", action="store_true")
    parser.add_argument(
        "--cpu", action="store_true", help="Allowed for smoke tests; real training needs a GPU"
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paddle_repo = args.paddleocr_repo.resolve()
    dataset = (project_root / args.dataset).resolve()
    output = (project_root / args.output).resolve()
    pretrained = (project_root / args.pretrained).resolve()
    config = (project_root / args.config).resolve()
    train_label = (
        (project_root / args.train_label).resolve() if args.train_label else dataset / "train.txt"
    )
    validation_label = (
        (project_root / args.validation_label).resolve()
        if args.validation_label
        else dataset / "validation.txt"
    )
    character_dict = (
        (project_root / args.character_dict).resolve()
        if args.character_dict
        else dataset / "korean_exam_dict.txt"
    )
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
    if not pretrained.is_file():
        if not args.download_pretrained:
            parser.error("pretrained weights missing; pass --download-pretrained")
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {PRETRAINED_URL} -> {pretrained}")
        urllib.request.urlretrieve(PRETRAINED_URL, pretrained)
    output.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"Global.use_gpu={str(not args.cpu).lower()}",
        f"Global.pretrained_model={pretrained}",
        f"Global.save_model_dir={output}",
        f"Global.character_dict_path={character_dict}",
        f"Global.epoch_num={args.epochs}",
        "Global.distributed=false",
        "Global.use_visualdl=false",
        "Global.save_epoch_step=1",
        f"Global.eval_batch_step=[0,{args.eval_every}]",
        f"Optimizer.lr.learning_rate={args.learning_rate:.10f}",
        f"Optimizer.lr.warmup_epoch={args.warmup_epochs}",
        f"Train.dataset.data_dir={dataset}",
        f"Train.dataset.label_file_list=['{train_label}']",
        f"Train.sampler.first_bs={args.batch_size}",
        f"Train.loader.batch_size_per_card={args.batch_size}",
        f"Train.loader.num_workers={args.workers}",
        f"Eval.dataset.data_dir={dataset}",
        f"Eval.dataset.label_file_list=['{validation_label}']",
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

    checkpoint = output / "best_accuracy"
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

    inference_dir = output / "inference"
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

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paddle_repo, text=True, capture_output=True, check=False
    ).stdout.strip()
    report = {
        "schema_version": "1.0",
        "paddleocr_commit": commit,
        "dataset": str(dataset),
        "train_label": str(train_label),
        "validation_label": str(validation_label),
        "character_dict": str(character_dict),
        "pretrained": str(pretrained),
        "pretrained_sha256": _sha256(pretrained),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_epochs": args.warmup_epochs,
        "checkpoint": str(checkpoint),
        "inference_dir": str(inference_dir) if not args.no_export else None,
    }
    (output / "training_run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
