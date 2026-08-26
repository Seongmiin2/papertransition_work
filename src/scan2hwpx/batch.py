from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.pipeline import convert_pdf

BatchProgress = Callable[[str], None]
MAX_BATCH_FILES = 10
_WORKER_PROVIDER: PaddlePdfOcrProvider | None = None


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 120,
    fusion_mode: str = "fast",
    lexicon_path: Path | None = None,
    resume: bool = True,
    renderer: str = "fidelity",
    device: str = "auto",
    page_anomaly_model: Path | None = None,
    progress: BatchProgress | None = None,
) -> dict[str, Any]:
    source_dir = input_dir.resolve()
    pdf_files = sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())
    if len(pdf_files) > MAX_BATCH_FILES:
        raise ValueError(f"batch-convert accepts at most {MAX_BATCH_FILES} PDF files")
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = PaddlePdfOcrProvider(
        dpi=dpi,
        fusion_mode=fusion_mode,
        lexicon_path=lexicon_path,
        device=device,
        page_anomaly_model=page_anomaly_model,
    )
    settings = {
        "dpi": dpi,
        "fusion_mode": fusion_mode,
        "renderer": renderer,
        "device": provider.device,
        "page_anomaly_model": str(page_anomaly_model.resolve()) if page_anomaly_model else None,
        "lexicon_sha256": _sha256(lexicon_path)
        if lexicon_path and lexicon_path.is_file()
        else None,
    }
    workers = 2 if provider.device.startswith("gpu") and len(pdf_files) > 1 else 1
    started = time.perf_counter()
    results_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, Path, str, Path, Path]] = []
    for index, source in enumerate(pdf_files, start=1):
        source_hash = _sha256(source)
        job_dir = output_dir / _job_name(source, source_hash)
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / "result.hwpx"
        result_path = job_dir / "conversion.json"
        if resume and output_path.is_file() and result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                previous.get("source_hash") == source_hash
                and previous.get("status") == "completed"
                and previous.get("settings") == settings
            ):
                results_by_index[index] = {**previous, "status": "skipped"}
                if progress:
                    progress(f"[{index}/{len(pdf_files)}] 건너뜀: {source.name}")
                continue
        pending.append((index, source, source_hash, output_path, result_path))

    if workers == 2 and pending:
        for index, source, _source_hash, _output_path, _result_path in pending:
            if progress:
                progress(f"[{index}/{len(pdf_files)}] 변환 시작: {source.name}")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                dpi,
                fusion_mode,
                lexicon_path,
                provider.device,
                page_anomaly_model,
            ),
        ) as executor:
            futures = {
                executor.submit(
                    _convert_one,
                    source,
                    source_hash,
                    output_path,
                    settings,
                    dpi,
                    renderer,
                ): (index, source, result_path)
                for index, source, source_hash, output_path, result_path in pending
            }
            for future in as_completed(futures):
                index, source, result_path = futures[future]
                item = future.result()
                _write_json(result_path, item)
                results_by_index[index] = item
                if progress:
                    progress(f"[{index}/{len(pdf_files)}] {item['status']}: {source.name}")
                _write_report(
                    output_dir,
                    [results_by_index[key] for key in sorted(results_by_index)],
                    started,
                    len(pdf_files),
                    fusion_mode,
                    dpi,
                    provider.device,
                    workers,
                )
    else:
        for index, source, source_hash, output_path, result_path in pending:
            if progress:
                progress(f"[{index}/{len(pdf_files)}] 변환 시작: {source.name}")
            item = _convert_one(
                source,
                source_hash,
                output_path,
                settings,
                dpi,
                renderer,
                provider=provider,
                progress=progress,
            )
            _write_json(result_path, item)
            results_by_index[index] = item
            _write_report(
                output_dir,
                [results_by_index[key] for key in sorted(results_by_index)],
                started,
                len(pdf_files),
                fusion_mode,
                dpi,
                provider.device,
                workers,
            )

    results = [results_by_index[key] for key in sorted(results_by_index)]
    return _write_report(
        output_dir,
        results,
        started,
        len(pdf_files),
        fusion_mode,
        dpi,
        provider.device,
        workers,
    )


def _init_worker(
    dpi: int,
    fusion_mode: str,
    lexicon_path: Path | None,
    device: str,
    page_anomaly_model: Path | None,
) -> None:
    global _WORKER_PROVIDER
    _WORKER_PROVIDER = PaddlePdfOcrProvider(
        dpi=dpi,
        fusion_mode=fusion_mode,
        lexicon_path=lexicon_path,
        device=device,
        page_anomaly_model=page_anomaly_model,
    )


def _convert_one(
    source: Path,
    source_hash: str,
    output_path: Path,
    settings: dict[str, Any],
    dpi: int,
    renderer: str,
    *,
    provider: PaddlePdfOcrProvider | None = None,
    progress: BatchProgress | None = None,
) -> dict[str, Any]:
    active_provider = provider or _WORKER_PROVIDER
    if active_provider is None:
        raise RuntimeError("batch OCR worker was not initialized")
    job_started = time.perf_counter()
    try:
        summary = convert_pdf(
            source,
            output_path,
            dpi=dpi,
            ocr_provider=active_provider,
            renderer=renderer,
            progress=progress,
            write_diagnostics=False,
        )
        return {
            "status": "completed",
            "source": str(source),
            "source_hash": source_hash,
            "output": str(output_path.resolve()),
            "settings": settings,
            "seconds": round(time.perf_counter() - job_started, 3),
            **summary,
        }
    except Exception as exc:  # noqa: BLE001 - batch must continue with remaining customer files
        return {
            "status": "failed",
            "source": str(source),
            "source_hash": source_hash,
            "output": str(output_path.resolve()),
            "settings": settings,
            "seconds": round(time.perf_counter() - job_started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _write_report(
    output_dir: Path,
    results: list[dict[str, Any]],
    started: float,
    total: int,
    fusion_mode: str,
    dpi: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    finished = [item for item in results if item["status"] in {"completed", "skipped"}]
    completed_now = [item for item in results if item["status"] == "completed"]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": fusion_mode,
        "dpi": dpi,
        "device": device,
        "workers": workers,
        "summary": {
            "total_files": total,
            "processed": len(results),
            "completed": len(completed_now),
            "skipped": sum(item["status"] == "skipped" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "elapsed_seconds": round(elapsed, 3),
            "throughput_files_per_hour": round(len(completed_now) * 3600 / max(elapsed, 0.001), 2),
            "pages": sum(int(item.get("pages", 0)) for item in finished),
        },
        "results": results,
    }
    _write_json(output_dir / "batch_report.json", report)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    candidate = path.with_suffix(path.suffix + ".partial")
    candidate.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate.replace(path)


def _job_name(path: Path, source_hash: str) -> str:
    safe_stem = re.sub(r'[<>:"/\\|?*]+', "_", path.stem).strip(" .")[:64] or "document"
    return f"{safe_stem}-{source_hash[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
