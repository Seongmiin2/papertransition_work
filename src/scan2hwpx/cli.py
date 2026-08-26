from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scan2hwpx")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("input", type=Path)
    convert = commands.add_parser("convert")
    convert.add_argument("input", type=Path)
    convert.add_argument(
        "--provider", "--ocr-provider", default="auto", choices=["auto", "fixture", "paddleocr"]
    )
    convert.add_argument("--template", type=Path, default=Path("templates/exam_base.hwpx"))
    convert.add_argument("--out", "--output", dest="out", type=Path, required=True)
    convert.add_argument("--dpi", type=int, default=240)
    convert.add_argument("--formulas", action="store_true")
    convert.add_argument("--formula-model", default="PP-FormulaNet_plus-S")
    convert.add_argument("--formula-layout-model", default="PP-DocLayout-S")
    convert.add_argument("--mode", choices=["fast", "balanced", "accurate"], default="fast")
    convert.add_argument("--lexicon", type=Path)
    convert.add_argument(
        "--renderer",
        choices=["fidelity", "semantic", "editable", "portable", "hancom"],
        default="fidelity",
    )
    convert.add_argument("--device", default="auto", help="auto, cpu, gpu:0, ...")
    convert.add_argument("--recognition-model-dir", type=Path)
    convert.add_argument("--page-anomaly-model", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("input", type=Path)
    export = commands.add_parser("export-ir")
    export.add_argument("input", type=Path)
    export.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("formula-audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--split-out", type=Path)
    benchmark = commands.add_parser("formula-benchmark")
    benchmark.add_argument("input", type=Path)
    benchmark.add_argument("--out", type=Path, default=Path("output/formula-benchmark"))
    benchmark.add_argument(
        "--models",
        nargs="+",
        default=["PP-FormulaNet_plus-S"],
        help="비교할 PaddleOCR 수식 인식 모델 이름",
    )
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--limit", type=int)
    reference = commands.add_parser(
        "reference-audit",
        help="HWP output examples and PDF inputs are inventoried into a training-ready corpus",
    )
    reference.add_argument("input", type=Path)
    reference.add_argument("--out", type=Path, default=Path("output/reference-dataset"))
    synthetic = commands.add_parser(
        "build-ocr-dataset",
        help="Render HWP reference text into PaddleOCR recognition training crops",
    )
    synthetic.add_argument("manifest", type=Path)
    synthetic.add_argument("--out", type=Path, default=Path("output/ocr-training"))
    synthetic.add_argument("--variants", type=int, default=1)
    synthetic.add_argument("--limit", type=int, default=5000)
    synthetic.add_argument("--font", action="append", type=Path, dest="fonts")
    synthetic.add_argument("--seed", type=int, default=20260823)
    synthetic.add_argument("--base-model-dir", type=Path)
    batch = commands.add_parser(
        "batch-convert",
        help="Convert up to 10 PDFs with resumable jobs and two GPU OCR workers",
    )
    batch.add_argument("input", type=Path)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--dpi", type=int, default=120)
    batch.add_argument("--mode", choices=["fast", "balanced", "accurate"], default="fast")
    batch.add_argument("--lexicon", type=Path)
    batch.add_argument("--no-resume", action="store_true")
    batch.add_argument(
        "--renderer",
        choices=["fidelity", "semantic", "editable", "portable", "hancom"],
        default="fidelity",
    )
    batch.add_argument("--device", default="auto", help="auto, cpu, gpu:0, ...")
    batch.add_argument("--recognition-model-dir", type=Path)
    batch.add_argument("--page-anomaly-model", type=Path)
    verified = commands.add_parser(
        "import-ocr-labels",
        help="Validate reviewer-confirmed OCR JSONL and copy crops into document-level splits",
    )
    verified.add_argument("manifest", type=Path)
    verified.add_argument("--out", type=Path, default=Path("output/verified-ocr-training"))
    layout_dataset = commands.add_parser(
        "build-layout-dataset",
        help="Create PP-DocLayout and ruled-line seed labels for human review",
    )
    layout_dataset.add_argument("input", type=Path)
    layout_dataset.add_argument("--out", type=Path, required=True)
    layout_dataset.add_argument("--model", default="PP-DocLayout-S")
    layout_dataset.add_argument("--device", default="auto")
    layout_dataset.add_argument("--dpi", type=int, default=200)
    corpus = commands.add_parser(
        "prepare-training-corpus",
        help="Copy and classify local exam data with document-level leakage-safe splits",
    )
    corpus.add_argument("inputs", nargs="+", type=Path)
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--pairs", type=Path)
    corpus.add_argument("--no-copy", action="store_true")
    anomaly = commands.add_parser(
        "train-page-anomaly",
        help="Fit a leakage-safe page autoencoder used for corruption and anomaly routing",
    )
    anomaly.add_argument("manifest", type=Path)
    anomaly.add_argument("--out", type=Path, required=True)
    anomaly.add_argument("--rank", type=int, default=32)
    anomaly.add_argument("--thumbnail-size", type=int, default=32)
    ocr_benchmark = commands.add_parser(
        "ocr-benchmark",
        help="Measure OCR CER against a held-out native-text transcript PDF",
    )
    ocr_benchmark.add_argument("source", type=Path)
    ocr_benchmark.add_argument("transcript", type=Path)
    ocr_benchmark.add_argument("--out", type=Path, required=True)
    ocr_benchmark.add_argument("--device", default="auto")
    ocr_benchmark.add_argument("--dpi", type=int, default=200)
    ocr_benchmark.add_argument("--mode", choices=["fast", "balanced", "accurate"], default="fast")
    ocr_benchmark.add_argument("--recognition-model-dir", type=Path)
    ocr_benchmark.add_argument("--page-anomaly-model", type=Path)
    adaptation = commands.add_parser(
        "build-ocr-adaptation-dataset",
        help="Extract conservative silver scan crops without touching held-out gold documents",
    )
    adaptation.add_argument("manifest", type=Path)
    adaptation.add_argument("base_dataset", type=Path)
    adaptation.add_argument("--out", type=Path, required=True)
    adaptation.add_argument("--device", default="auto")
    adaptation.add_argument("--dpi", type=int, default=200)
    adaptation.add_argument("--minimum-confidence", type=float, default=0.985)
    adaptation.add_argument("--minimum-quality", type=float, default=0.92)
    dictionary_filter = commands.add_parser(
        "filter-ocr-dataset",
        help="Write labels compatible with a fixed pretrained recognition dictionary",
    )
    dictionary_filter.add_argument("dataset", type=Path)
    dictionary_filter.add_argument("character_dict", type=Path)
    dictionary_filter.add_argument("--suffix", default="official")
    promotion = commands.add_parser(
        "compare-ocr-models",
        help="Apply the fixed gold-test promotion rule to two OCR benchmark reports",
    )
    promotion.add_argument("baseline", type=Path)
    promotion.add_argument("candidate", type=Path)
    promotion.add_argument("--out", type=Path, required=True)
    return parser


def _default_lexicon(value: Path | None) -> Path | None:
    if value is not None:
        return value
    candidate = Path("output/ocr-training/lexicon.json")
    return candidate if candidate.is_file() else None


def main() -> int:
    args = _parser().parse_args()
    if args.command == "inspect":
        from scan2hwpx.pipeline import inspect_pdf

        print(json.dumps(inspect_pdf(args.input), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        from scan2hwpx.hwpx.validate import validate_hwpx

        result = validate_hwpx(args.input)
        for error in result.errors:
            print(f"ERROR: {error}")
        print("valid" if result.valid else "invalid")
        return 0 if result.valid else 1
    if args.command == "formula-audit":
        from scan2hwpx.vision.training_data import (
            load_verified_formula_samples,
            split_by_document,
            write_training_splits,
        )

        samples, audit = load_verified_formula_samples(args.manifest)
        if args.split_out is not None and samples:
            write_training_splits(split_by_document(samples), args.split_out)
        print(json.dumps(audit.__dict__, ensure_ascii=False, indent=2))
        return 0 if audit.trainable else 2
    if args.command == "formula-benchmark":
        from scan2hwpx.vision.benchmark import run_formula_benchmark

        report = run_formula_benchmark(
            args.input,
            args.out,
            args.models,
            device=args.device,
            limit=args.limit,
        )
        print(
            json.dumps(
                {"summary": report["summary"], "output": str(args.out)},
                ensure_ascii=False,
                indent=2,
            )
        )
        successful = sum(int(item["successful"]) for item in report["summary"].values())
        return 0 if successful else 1
    if args.command == "reference-audit":
        from scan2hwpx.reference import audit_reference_directory

        report = audit_reference_directory(args.input, args.out)
        print(
            json.dumps(
                {
                    "summary": report["summary"],
                    "training_readiness": report["training_readiness"],
                    "output": str(args.out.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not report["failures"] else 2
    if args.command == "build-ocr-dataset":
        from scan2hwpx.reference.synthetic import build_ocr_dataset

        report = build_ocr_dataset(
            args.manifest,
            args.out,
            variants=args.variants,
            limit=args.limit,
            fonts=args.fonts,
            seed=args.seed,
            base_model_dir=args.base_model_dir,
        )
        print(
            json.dumps({**report, "output": str(args.out.resolve())}, ensure_ascii=False, indent=2)
        )
        return 0
    if args.command == "batch-convert":
        from scan2hwpx.batch import convert_directory

        report = convert_directory(
            args.input,
            args.out,
            dpi=args.dpi,
            fusion_mode=args.mode,
            lexicon_path=_default_lexicon(args.lexicon),
            resume=not args.no_resume,
            renderer=args.renderer,
            device=args.device,
            recognition_model_dir=args.recognition_model_dir,
            page_anomaly_model=args.page_anomaly_model,
            progress=print,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 1 if report["summary"]["failed"] else 0
    if args.command == "import-ocr-labels":
        from scan2hwpx.reference.verified import import_verified_ocr

        report = import_verified_ocr(args.manifest, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not report["failures"] else 2
    if args.command == "build-layout-dataset":
        from scan2hwpx.vision.layout_dataset import build_layout_seed_dataset

        report = build_layout_seed_dataset(
            args.input,
            args.out,
            model_name=args.model,
            device=args.device,
            dpi=args.dpi,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-training-corpus":
        from scan2hwpx.training import prepare_training_corpus

        report = prepare_training_corpus(
            args.inputs,
            args.out,
            pair_config=args.pairs,
            copy_files=not args.no_copy,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0 if not report["failures"] else 2
    if args.command == "train-page-anomaly":
        from scan2hwpx.training.page_anomaly import train_page_anomaly_model

        report = train_page_anomaly_model(
            args.manifest,
            args.out,
            rank=args.rank,
            thumbnail_size=args.thumbnail_size,
        )
        print(
            json.dumps(
                {
                    "model": report["model"],
                    "weights": report["weights"],
                    "threshold": report["threshold"],
                    "splits": report["splits"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "ocr-benchmark":
        from scan2hwpx.training.ocr_benchmark import run_ocr_benchmark

        report = run_ocr_benchmark(
            args.source,
            args.transcript,
            args.out,
            device=args.device,
            dpi=args.dpi,
            fusion_mode=args.mode,
            recognition_model_dir=args.recognition_model_dir,
            page_anomaly_model=args.page_anomaly_model,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-ocr-adaptation-dataset":
        from scan2hwpx.training.ocr_adaptation import build_ocr_adaptation_dataset

        report = build_ocr_adaptation_dataset(
            args.manifest,
            args.base_dataset,
            args.out,
            device=args.device,
            dpi=args.dpi,
            minimum_confidence=args.minimum_confidence,
            minimum_quality=args.minimum_quality,
            progress=print,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "filter-ocr-dataset":
        from scan2hwpx.training.ocr_adaptation import write_dictionary_compatible_labels

        report = write_dictionary_compatible_labels(
            args.dataset, args.character_dict, suffix=args.suffix
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "compare-ocr-models":
        from scan2hwpx.training.ocr_benchmark import write_ocr_promotion_decision

        decision = write_ocr_promotion_decision(args.baseline, args.candidate, args.out)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0 if decision["promoted"] else 3
    use_pdf = (
        args.input.suffix.lower() == ".pdf" or getattr(args, "provider", "auto") == "paddleocr"
    )
    if args.command == "convert" and use_pdf:
        from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
        from scan2hwpx.pipeline import convert_pdf

        formula_processor = None
        if args.formulas:
            from scan2hwpx.vision.paddle_formula import PaddleFormulaProcessor

            formula_processor = PaddleFormulaProcessor(
                formula_model=args.formula_model,
                layout_model=args.formula_layout_model,
            )
        summary = convert_pdf(
            args.input,
            args.out,
            dpi=args.dpi,
            formula_processor=formula_processor,
            ocr_provider=PaddlePdfOcrProvider(
                dpi=args.dpi,
                fusion_mode=args.mode,
                lexicon_path=_default_lexicon(args.lexicon),
                device=args.device,
                recognition_model_dir=args.recognition_model_dir,
                page_anomaly_model=args.page_anomaly_model,
            ),
            renderer=args.renderer,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    from scan2hwpx.hwpx import render_hwpx, validate_hwpx
    from scan2hwpx.ocr.providers import FixtureOcrProvider

    if use_pdf:
        from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider

    document = (
        PaddlePdfOcrProvider().convert(args.input)
        if use_pdf
        else FixtureOcrProvider().convert(args.input)
    )
    if args.command == "export-ir":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return 0
    render_hwpx(document, args.template, args.out)
    result = validate_hwpx(args.out)
    if not result.valid:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
