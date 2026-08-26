from __future__ import annotations

import contextlib
import json
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider

PROTOCOL_VERSION = "1.0"
_PROVIDER: PaddlePdfOcrProvider | None = None
_PROVIDER_SETTINGS: tuple[int, str, str, str | None, str | None] | None = None


def emit(request_id: str, event: str, data: dict[str, Any]) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "event": event,
        "data": data,
    }
    output = sys.__stdout__ or sys.stdout
    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
    output.flush()


def _provider_for(
    dpi: int,
    fusion_mode: str,
    device: str,
    recognition_model_dir: Path | None,
    page_anomaly_model: Path | None,
) -> PaddlePdfOcrProvider:
    global _PROVIDER, _PROVIDER_SETTINGS
    settings = (
        dpi,
        fusion_mode,
        device,
        str(recognition_model_dir) if recognition_model_dir else None,
        str(page_anomaly_model) if page_anomaly_model else None,
    )
    if _PROVIDER is None or _PROVIDER_SETTINGS != settings:
        from scan2hwpx.ocr.providers.paddle import PaddlePdfOcrProvider

        _PROVIDER = PaddlePdfOcrProvider(
            dpi=dpi,
            fusion_mode=fusion_mode,
            device=device,
            recognition_model_dir=recognition_model_dir,
            page_anomaly_model=page_anomaly_model,
        )
        _PROVIDER_SETTINGS = settings
    return _PROVIDER


def handle(request: dict[str, Any]) -> None:
    request_id = str(request.get("request_id") or uuid.uuid4())
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    if request.get("method") != "convert":
        raise ValueError("unsupported method")
    from scan2hwpx.pipeline import convert_pdf

    params = request["params"]
    source = Path(params["input_pdf"]).resolve(strict=True)
    work_dir = Path(params["work_dir"]).resolve()
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(work_dir.anchor)
    required = max(source.stat().st_size * 20, 512 * 1024 * 1024)
    if usage.free < required:
        raise OSError(f"insufficient disk space: need {required} bytes")
    output = output_dir / "result.hwpx"
    emit(request_id, "started", {"job_id": params["job_id"]})
    options = params.get("options") or {}
    dpi = int(options.get("dpi", 240))
    fusion_mode = str(options.get("mode", "fast"))
    device = str(options.get("device", "auto"))
    renderer = str(options.get("renderer", "fidelity"))
    write_diagnostics = bool(options.get("write_diagnostics", True))
    recognition_value = options.get("recognition_model_dir")
    recognition_model_dir = Path(str(recognition_value)).resolve() if recognition_value else None
    anomaly_value = options.get("page_anomaly_model")
    page_anomaly_model = Path(str(anomaly_value)).resolve() if anomaly_value else None

    def progress(message: str) -> None:
        emit(request_id, "progress", {"stage": "OCR", "message": message})

    with contextlib.redirect_stdout(sys.stderr):
        provider = _provider_for(
            dpi,
            fusion_mode,
            device,
            recognition_model_dir,
            page_anomaly_model,
        )
        try:
            result = convert_pdf(
                source,
                output,
                dpi=dpi,
                progress=progress,
                ocr_provider=provider,
                renderer=renderer,
                write_diagnostics=write_diagnostics,
            )
        finally:
            if not write_diagnostics:
                for directory in (output_dir / "debug", output_dir / "layout_training"):
                    shutil.rmtree(directory, ignore_errors=True)
    emit(request_id, "completed", {"output_path": str(output), **result})


def run(stdin: TextIO = sys.stdin) -> int:
    for line in stdin:
        if not line.strip():
            continue
        request_id = "unknown"
        try:
            request = json.loads(line)
            request_id = str(request.get("request_id", request_id))
            handle(request)
        except Exception as exc:  # noqa: BLE001 - protocol boundary must report all worker failures
            traceback.print_exc(file=sys.stderr)
            emit(request_id, "failed", {"error_type": type(exc).__name__, "message": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
