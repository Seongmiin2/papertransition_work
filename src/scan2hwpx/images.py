from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_image(path: Path, image: np.ndarray) -> Path:
    """Write an OpenCV image through bytes so Windows Unicode paths are reliable."""
    extension = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"이미지를 인코딩할 수 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path
