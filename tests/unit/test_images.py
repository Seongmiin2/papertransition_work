from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scan2hwpx.images import write_image


def test_write_image_supports_korean_path(tmp_path: Path) -> None:
    output = tmp_path / "한글 시험지" / "검수.png"

    write_image(output, np.full((12, 20, 3), 127, dtype=np.uint8))

    assert output.is_file()
    decoded = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (12, 20, 3)
