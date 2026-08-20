from __future__ import annotations

import contextlib
import json
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, TextIO

from scan2hwpx.pipeline import convert_pdf

PROTOCOL_VERSION = "1.0"


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


def handle(request: dict[str, Any]) -> None:
    request_id = str(request.get("request_id") or uuid.uuid4())
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    if request.get("method") != "convert":
        raise ValueError("unsupported method")
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

    def progress(message: str) -> None:
        emit(request_id, "progress", {"stage": "OCR", "message": message})

    with contextlib.redirect_stdout(sys.stderr):
        result = convert_pdf(source, output, dpi=300, progress=progress)
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
