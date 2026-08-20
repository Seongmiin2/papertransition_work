from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import cv2
import numpy as np


def recognize_windows_image(
    image: np.ndarray, tile_size: int = 1800, overlap: int = 120
) -> list[tuple[tuple[float, float, float, float], str, float]]:
    """Recognize a large page with the offline Windows Korean OCR engine."""
    return asyncio.run(_recognize_tiled(image, tile_size, overlap))


async def _recognize_tiled(
    image: np.ndarray, tile_size: int, overlap: int
) -> list[tuple[tuple[float, float, float, float], str, float]]:
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage import FileAccessMode, StorageFile

    engine = OcrEngine.try_create_from_language(Language("ko-KR"))
    if engine is None:
        return []
    height, width = image.shape[:2]
    step = tile_size - overlap
    recognized: list[tuple[tuple[float, float, float, float], str, float]] = []
    with tempfile.TemporaryDirectory(prefix="exam2hwpx-ocr-") as folder:
        tile_no = 0
        for y0 in range(0, height, step):
            for x0 in range(0, width, step):
                x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
                tile = image[y0:y1, x0:x1]
                tile_path = Path(folder) / f"tile-{tile_no}.png"
                tile_no += 1
                cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                storage_file = await StorageFile.get_file_from_path_async(str(tile_path))
                stream = await storage_file.open_async(FileAccessMode.READ)
                decoder = await BitmapDecoder.create_async(stream)
                bitmap = await decoder.get_software_bitmap_async()
                result = await engine.recognize_async(bitmap)
                for line in result.lines:
                    words = list(line.words)
                    if not words or not line.text.strip():
                        continue
                    left = min(word.bounding_rect.x for word in words)
                    top = min(word.bounding_rect.y for word in words)
                    right = max(word.bounding_rect.x + word.bounding_rect.width for word in words)
                    bottom = max(word.bounding_rect.y + word.bounding_rect.height for word in words)
                    center_x, center_y = (left + right) / 2, (top + bottom) / 2
                    if x1 < width and center_x >= step:
                        continue
                    if y1 < height and center_y >= step:
                        continue
                    recognized.append(
                        ((x0 + left, y0 + top, x0 + right, y0 + bottom), line.text.strip(), 0.9)
                    )
    return recognized
