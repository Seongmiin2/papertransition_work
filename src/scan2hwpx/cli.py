from __future__ import annotations

import argparse
import json
from pathlib import Path

from scan2hwpx.hwpx import render_hwpx, validate_hwpx
from scan2hwpx.ocr.providers import FixtureOcrProvider
from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.pipeline import convert_pdf, inspect_pdf


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
    convert.add_argument("--remove-red-marks", action="store_true")
    convert.add_argument("--layout", default="exam-two-column", choices=["exam-two-column"])
    convert.add_argument("--dpi", type=int, default=300)
    validate = commands.add_parser("validate")
    validate.add_argument("input", type=Path)
    export = commands.add_parser("export-ir")
    export.add_argument("input", type=Path)
    export.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_pdf(args.input), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        result = validate_hwpx(args.input)
        for error in result.errors:
            print(f"ERROR: {error}")
        print("valid" if result.valid else "invalid")
        return 0 if result.valid else 1
    use_pdf = (
        args.input.suffix.lower() == ".pdf" or getattr(args, "provider", "auto") == "paddleocr"
    )
    if args.command == "convert" and use_pdf:
        summary = convert_pdf(args.input, args.out, dpi=args.dpi)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
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
