from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from scan2hwpx.ir.models import Document
from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider
from scan2hwpx.pipeline import convert_pdf, run_ocr_stage, run_render_stage

BatchProgress = Callable[[str], None]
MAX_BATCH_FILES = 10
STAGE2_MAX_WORKERS = 4
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
    recognition_model_dir: Path | None = None,
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
        recognition_model_dir=recognition_model_dir,
        page_anomaly_model=page_anomaly_model,
    )
    settings = {
        "dpi": dpi,
        "fusion_mode": fusion_mode,
        "renderer": renderer,
        "device": provider.device,
        "recognition_model_dir": str(recognition_model_dir.resolve())
        if recognition_model_dir
        else None,
        "recognition_model_sha256": _sha256(recognition_model_dir / "inference.pdiparams")
        if recognition_model_dir and (recognition_model_dir / "inference.pdiparams").is_file()
        else None,
        "page_anomaly_model": str(page_anomaly_model.resolve()) if page_anomaly_model else None,
        "lexicon_sha256": _sha256(lexicon_path)
        if lexicon_path and lexicon_path.is_file()
        else None,
    }
    stage1_workers = 2 if provider.device.startswith("gpu") and len(pdf_files) > 1 else 1
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

    stage2_workers = 0
    if len(pending) > 1:
        stage2_workers = max(1, min(os.cpu_count() or 1, len(pending), STAGE2_MAX_WORKERS))
        for index, source, _source_hash, _output_path, _result_path in pending:
            if progress:
                progress(f"[{index}/{len(pdf_files)}] 변환 시작: {source.name}")

        def handle_result(index: int, source: Path, result_path: Path, item: dict[str, Any]) -> None:
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
                stage1_workers,
                stage2_workers,
            )

        _run_pipelined(
            pending,
            dpi=dpi,
            fusion_mode=fusion_mode,
            lexicon_path=lexicon_path,
            device=provider.device,
            recognition_model_dir=recognition_model_dir,
            page_anomaly_model=page_anomaly_model,
            renderer=renderer,
            settings=settings,
            stage1_workers=stage1_workers,
            stage2_workers=stage2_workers,
            on_result=handle_result,
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
                stage1_workers,
                stage2_workers,
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
        stage1_workers,
        stage2_workers,
    )


def _run_pipelined(
    pending: list[tuple[int, Path, str, Path, Path]],
    *,
    dpi: int,
    fusion_mode: str,
    lexicon_path: Path | None,
    device: str,
    recognition_model_dir: Path | None,
    page_anomaly_model: Path | None,
    renderer: str,
    settings: dict[str, Any],
    stage1_workers: int,
    stage2_workers: int,
    on_result: Callable[[int, Path, Path, dict[str, Any]], None],
) -> None:
    """Overlap GPU-bound OCR (stage 1) with CPU-bound rendering (stage 2).

    Each stage gets its own worker pool, sized for what that stage actually
    needs (stage1_workers stays GPU-gated exactly as before; stage2_workers
    scales with CPU cores since rendering/validation touch no GPU state). As
    soon as one document finishes OCR, its rendering is submitted to stage 2
    immediately rather than waiting for the whole batch's OCR to finish, so
    rendering document N overlaps with OCR for document N + 1.
    """
    with (
        ProcessPoolExecutor(
            max_workers=stage1_workers,
            initializer=_init_stage1_worker,
            initargs=(
                dpi,
                fusion_mode,
                lexicon_path,
                device,
                recognition_model_dir,
                page_anomaly_model,
            ),
        ) as stage1_pool,
        ProcessPoolExecutor(max_workers=stage2_workers) as stage2_pool,
    ):
        stage1_futures: dict[Future[dict[str, Any]], tuple[int, Path, Path, Path]] = {
            stage1_pool.submit(_run_ocr_stage_job, source, output_path.parent, dpi, renderer): (
                index,
                source,
                output_path,
                result_path,
            )
            for index, source, _source_hash, output_path, result_path in pending
        }
        source_hashes = {index: source_hash for index, _s, source_hash, _o, _r in pending}
        stage2_futures: dict[Future[dict[str, Any]], tuple[int, Path, Path]] = {}

        remaining: set[Future[dict[str, Any]]] = set(stage1_futures)
        while remaining:
            done, remaining = wait(remaining, return_when=FIRST_COMPLETED)
            for future in done:
                if future in stage1_futures:
                    index, source, output_path, result_path = stage1_futures.pop(future)
                    ocr_result = future.result()
                    if ocr_result["status"] != "ok":
                        on_result(
                            index,
                            source,
                            result_path,
                            {
                                "status": "failed",
                                "source": str(source),
                                "source_hash": source_hashes[index],
                                "output": str(output_path.resolve()),
                                "settings": settings,
                                "seconds": ocr_result["seconds"],
                                "error_type": ocr_result["error_type"],
                                "error": ocr_result["error"],
                            },
                        )
                        continue
                    stage2_future = stage2_pool.submit(
                        _run_render_stage_job,
                        output_path.parent,
                        output_path,
                        renderer,
                        str(source),
                        source_hashes[index],
                        settings,
                        float(ocr_result["seconds"]),
                    )
                    stage2_futures[stage2_future] = (index, source, result_path)
                    remaining.add(stage2_future)
                else:
                    index, source, result_path = stage2_futures.pop(future)
                    on_result(index, source, result_path, future.result())


def _init_stage1_worker(
    dpi: int,
    fusion_mode: str,
    lexicon_path: Path | None,
    device: str,
    recognition_model_dir: Path | None,
    page_anomaly_model: Path | None,
) -> None:
    global _WORKER_PROVIDER
    _WORKER_PROVIDER = PaddlePdfOcrProvider(
        dpi=dpi,
        fusion_mode=fusion_mode,
        lexicon_path=lexicon_path,
        device=device,
        recognition_model_dir=recognition_model_dir,
        page_anomaly_model=page_anomaly_model,
    )


def _run_ocr_stage_job(source: Path, job_dir: Path, dpi: int, renderer: str) -> dict[str, Any]:
    if _WORKER_PROVIDER is None:
        raise RuntimeError("batch OCR worker was not initialized")
    started = time.perf_counter()
    try:
        run_ocr_stage(
            source,
            job_dir,
            dpi=dpi,
            ocr_provider=_WORKER_PROVIDER,
            renderer=renderer,
            write_diagnostics=False,
        )
        return {"status": "ok", "seconds": round(time.perf_counter() - started, 3)}
    except Exception as exc:  # noqa: BLE001 - batch must continue with remaining customer files
        return {
            "status": "failed",
            "seconds": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_render_stage_job(
    job_dir: Path,
    output_path: Path,
    renderer: str,
    source: str,
    source_hash: str,
    settings: dict[str, Any],
    ocr_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        document = Document.model_validate_json(
            (job_dir / "document_ir.json").read_text(encoding="utf-8")
        )
        summary = run_render_stage(document, job_dir, output_path, renderer=renderer)
        return {
            "status": "completed",
            "source": source,
            "source_hash": source_hash,
            "output": str(output_path.resolve()),
            "settings": settings,
            "seconds": round(ocr_seconds + time.perf_counter() - started, 3),
            **summary,
        }
    except Exception as exc:  # noqa: BLE001 - batch must continue with remaining customer files
        return {
            "status": "failed",
            "source": source,
            "source_hash": source_hash,
            "output": str(output_path.resolve()),
            "settings": settings,
            "seconds": round(ocr_seconds + time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


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
    stage1_workers: int,
    stage2_workers: int = 0,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    finished = [item for item in results if item["status"] in {"completed", "skipped"}]
    completed_now = [item for item in results if item["status"] == "completed"]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": fusion_mode,
        "dpi": dpi,
        "device": device,
        "workers": stage1_workers,
        "stage2_workers": stage2_workers,
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
