from __future__ import annotations

import base64
import html
import json
import mimetypes
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from scan2hwpx.vision.training_data import normalize_latex

FORMULA_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class FormulaPredictor(Protocol):
    def predict(self, input: str, *, batch_size: int = 1) -> Any: ...


PredictorFactory = Callable[[str, str], FormulaPredictor]


@dataclass(frozen=True)
class FormulaBenchmarkSample:
    document_id: str
    formula_id: str
    image: Path
    target: str | None = None
    format: str = "latex"


def load_formula_benchmark_samples(source: Path) -> list[FormulaBenchmarkSample]:
    """Load one crop, a crop directory, or a JSONL review/training manifest."""
    if source.is_dir():
        images = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in FORMULA_IMAGE_SUFFIXES
        )
        return [
            FormulaBenchmarkSample(source.name or "formula-crops", path.stem, path.resolve())
            for path in images
        ]
    if source.suffix.lower() in FORMULA_IMAGE_SUFFIXES and source.is_file():
        return [FormulaBenchmarkSample(source.parent.name or "single-image", source.stem, source.resolve())]
    if source.suffix.lower() != ".jsonl" or not source.is_file():
        raise ValueError("입력은 수식 이미지, 이미지 폴더 또는 JSONL manifest여야 합니다.")

    samples: list[FormulaBenchmarkSample] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest {line_number}행의 JSON이 잘못되었습니다: {exc.msg}") from exc
        image = _resolve_manifest_image(source, str(row.get("image") or ""))
        if not image.is_file():
            raise ValueError(f"manifest {line_number}행의 이미지를 찾을 수 없습니다: {image}")
        target_value = normalize_latex(str(row.get("target") or "")) or None
        samples.append(
            FormulaBenchmarkSample(
                document_id=str(row.get("document_id") or source.stem),
                formula_id=str(row.get("formula_id") or f"row-{line_number}"),
                image=image.resolve(),
                target=target_value,
                format=str(row.get("format") or "latex"),
            )
        )
    return samples


def run_formula_benchmark(
    source: Path,
    output_dir: Path,
    model_names: list[str],
    *,
    device: str = "cpu",
    limit: int | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> dict[str, Any]:
    samples = load_formula_benchmark_samples(source)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit은 1 이상이어야 합니다.")
        samples = samples[:limit]
    if not samples:
        raise ValueError("벤치마크할 수식 이미지가 없습니다.")
    models = list(dict.fromkeys(model_names))
    if not models:
        raise ValueError("하나 이상의 수식 모델을 지정해야 합니다.")

    factory = predictor_factory or _create_official_predictor
    predictors: dict[str, FormulaPredictor] = {}
    load_seconds: dict[str, float] = {}
    model_errors: dict[str, str] = {}
    for model_name in models:
        started = time.perf_counter()
        try:
            predictors[model_name] = factory(model_name, device)
        except Exception as exc:  # noqa: BLE001 - isolate optional model failures in report
            model_errors[model_name] = f"{type(exc).__name__}: {exc}"
        load_seconds[model_name] = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for sample in samples:
        predictions: list[dict[str, Any]] = []
        for model_name in models:
            if model_name in model_errors:
                predictions.append(
                    {
                        "model": model_name,
                        "expression": "",
                        "seconds": 0.0,
                        "error": model_errors[model_name],
                    }
                )
                continue
            started = time.perf_counter()
            try:
                results = list(predictors[model_name].predict(str(sample.image), batch_size=1))
                expression = _formula_expression(results[0].json if results else {})
                error = None if expression else "모델이 빈 수식을 반환했습니다."
            except Exception as exc:  # noqa: BLE001 - continue comparing the remaining models
                expression = ""
                error = f"{type(exc).__name__}: {exc}"
            prediction: dict[str, Any] = {
                "model": model_name,
                "expression": expression,
                "seconds": round(time.perf_counter() - started, 4),
                "error": error,
            }
            if sample.target is not None and expression:
                prediction["exact_match"] = normalize_latex(expression) == sample.target
                prediction["normalized_edit_similarity"] = round(
                    normalized_edit_similarity(expression, sample.target), 6
                )
            predictions.append(prediction)
        expressions = {
            normalize_latex(str(item["expression"]))
            for item in predictions
            if item["expression"]
        }
        rows.append(
            {
                "document_id": sample.document_id,
                "formula_id": sample.formula_id,
                "image": str(sample.image),
                "target": sample.target,
                "format": sample.format,
                "disagreement": len(expressions) > 1,
                "predictions": predictions,
            }
        )

    summary = {
        model_name: _summarize_model(rows, model_name, load_seconds[model_name])
        for model_name in models
    }
    report: dict[str, Any] = {
        "source": str(source.resolve()),
        "device": device,
        "sample_count": len(samples),
        "models": models,
        "summary": summary,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "formula_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "formula_benchmark.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "formula_benchmark.html").write_text(
        _render_review_html(report), encoding="utf-8"
    )
    return report


def normalized_edit_similarity(left: str, right: str) -> float:
    normalized_left = normalize_latex(left)
    normalized_right = normalize_latex(right)
    longest = max(len(normalized_left), len(normalized_right))
    if longest == 0:
        return 1.0
    return 1.0 - _levenshtein_distance(normalized_left, normalized_right) / longest


def _create_official_predictor(model_name: str, device: str) -> FormulaPredictor:
    try:
        from paddleocr import FormulaRecognition  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError('수식 모델 의존성이 없습니다. pip install -e ".[vision]"을 실행하세요.') from exc
    return cast(
        FormulaPredictor,
        FormulaRecognition(model_name=model_name, device=device, enable_mkldnn=False),
    )


def _resolve_manifest_image(manifest: Path, value: str) -> Path:
    image = Path(value)
    if image.is_absolute():
        return image
    if image.is_file():
        return image
    return manifest.parent / image


def _formula_expression(payload: Any) -> str:
    data = payload.get("res", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return ""
    return normalize_latex(str(data.get("rec_formula") or ""))


def _summarize_model(rows: list[dict[str, Any]], model_name: str, load_seconds: float) -> dict[str, Any]:
    predictions = [
        next(item for item in row["predictions"] if item["model"] == model_name)
        for row in rows
    ]
    successful = [item for item in predictions if item["expression"] and not item["error"]]
    evaluated = [item for item in successful if "exact_match" in item]
    return {
        "samples": len(predictions),
        "successful": len(successful),
        "errors": len(predictions) - len(successful),
        "model_load_seconds": round(load_seconds, 4),
        "average_inference_seconds": round(
            sum(float(item["seconds"]) for item in successful) / max(1, len(successful)), 4
        ),
        "evaluated_targets": len(evaluated),
        "exact_match_rate": (
            round(sum(bool(item["exact_match"]) for item in evaluated) / len(evaluated), 6)
            if evaluated
            else None
        ),
        "normalized_edit_similarity": (
            round(
                sum(float(item["normalized_edit_similarity"]) for item in evaluated)
                / len(evaluated),
                6,
            )
            if evaluated
            else None
        ),
    }


def _render_review_html(report: dict[str, Any]) -> str:
    cards: list[str] = []
    review_rows: list[dict[str, Any]] = []
    for index, row in enumerate(report["rows"]):
        image_path = Path(row["image"])
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        predictions = []
        for prediction in row["predictions"]:
            expression = html.escape(str(prediction["expression"]))
            error = html.escape(str(prediction["error"] or ""))
            predictions.append(
                "<div class='prediction'>"
                f"<strong>{html.escape(str(prediction['model']))}</strong> "
                f"<span>{prediction['seconds']:.4f}s</span>"
                f"<code>{expression or error}</code>"
                f"<button type='button' data-index='{index}' "
                f"data-expression='{html.escape(str(prediction['expression']), quote=True)}'>"
                "이 값을 정답으로</button></div>"
            )
        target = html.escape(str(row["target"] or ""))
        warning = "<span class='warning'>모델 간 결과 불일치</span>" if row["disagreement"] else ""
        cards.append(
            "<article class='card'>"
            f"<h2>{html.escape(str(row['formula_id']))} {warning}</h2>"
            f"<img src='data:{mime};base64,{encoded}' alt='수식 crop'>"
            + "".join(predictions)
            + f"<label>확정 LaTeX<textarea id='target-{index}'>{target}</textarea></label>"
            "</article>"
        )
        review_rows.append(
            {
                "document_id": row["document_id"],
                "formula_id": row["formula_id"],
                "image": row["image"],
                "format": row["format"],
            }
        )
    review_json = json.dumps(review_rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Exam2HWPX 수식 검수</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#f5f6f8;color:#17191c}}
header{{position:sticky;top:0;background:#f5f6f8;padding:12px 0;z-index:2}}button{{cursor:pointer}}
.card{{background:white;border:1px solid #dfe3e8;border-radius:12px;padding:20px;margin:16px 0}}
.card img{{max-width:100%;max-height:220px;background:white;border:1px solid #ddd}}
.prediction{{display:grid;grid-template-columns:220px 80px 1fr auto;gap:12px;align-items:center;margin:10px 0}}
code,textarea{{font-family:Consolas,monospace}}code{{overflow-wrap:anywhere}}textarea{{width:100%;min-height:58px;margin-top:6px}}
.warning{{color:#a53b00;font-size:14px}}.muted{{color:#5f6873}}
</style></head><body><header><h1>Exam2HWPX 수식 검수</h1>
<p class="muted">예측을 확인하고 정답 LaTeX를 확정한 뒤 JSONL을 저장하세요.</p>
<button id="download" type="button">확정한 정답 JSONL 저장</button></header>
{''.join(cards)}
<script>
const rows={review_json};
document.querySelectorAll('button[data-expression]').forEach(button=>button.addEventListener('click',()=>{{
  document.getElementById(`target-${{button.dataset.index}}`).value=button.dataset.expression;
}}));
document.getElementById('download').addEventListener('click',()=>{{
  const verified=rows.map((row,index)=>({{...row,target:document.getElementById(`target-${{index}}`).value.trim()}})).filter(row=>row.target);
  const body=verified.map(row=>JSON.stringify(row)).join('\n')+(verified.length?'\n':'');
  const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([body],{{type:'application/x-ndjson'}}));
  link.download='verified_formula_training.jsonl';link.click();URL.revokeObjectURL(link.href);
}});
</script></body></html>"""


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
