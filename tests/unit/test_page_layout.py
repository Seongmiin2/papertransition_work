from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from scan2hwpx.vision.page_layout import analyze_page_layout


def _text_columns(*, two_columns: bool) -> np.ndarray:
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    ranges = ((80, 440), (560, 920)) if two_columns else ((80, 920),)
    for y in range(120, 1250, 42):
        for start, end in ranges:
            for x in range(start, end, 28):
                draw.rectangle((x, y, min(x + 17, end), y + 12), fill="black")
    return np.asarray(image)


def test_page_layout_detects_two_columns_before_ocr() -> None:
    layout = analyze_page_layout(_text_columns(two_columns=True))

    assert len(layout.column_boxes) == 2
    assert layout.gutter_x is not None
    assert 450 <= layout.gutter_x <= 550
    assert layout.column_index((100, 200, 400, 240)) == 0
    assert layout.column_index((600, 200, 900, 240)) == 1
    assert layout.column_index((100, 40, 900, 80)) is None
    assert layout.reading_key((100, 40, 900, 80)) < layout.reading_key(
        (100, 200, 400, 240)
    )
    assert layout.reading_key((100, 200, 400, 240)) < layout.reading_key(
        (600, 100, 900, 140)
    )


def test_page_layout_keeps_single_column_when_text_crosses_center() -> None:
    layout = analyze_page_layout(_text_columns(two_columns=False))

    assert len(layout.column_boxes) == 1
    assert layout.gutter_x is None
    assert layout.column_index((100, 200, 900, 240)) == 0


def test_short_table_divider_does_not_turn_single_column_text_into_two_columns() -> None:
    image = Image.fromarray(_text_columns(two_columns=False))
    draw = ImageDraw.Draw(image)
    draw.line((500, 500, 500, 800), fill="black", width=5)

    layout = analyze_page_layout(np.asarray(image))

    assert len(layout.column_boxes) == 1


def test_page_layout_assigns_blocks_inside_ruled_table() -> None:
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 200, 450, 650), outline="black", width=5)
    draw.line((265, 200, 265, 650), fill="black", width=5)
    draw.line((80, 350, 450, 350), fill="black", width=5)
    draw.line((80, 500, 450, 500), fill="black", width=5)

    layout = analyze_page_layout(np.asarray(image))
    style = layout.block_style((120, 240, 220, 300))

    assert style["layout_region"] == "table"
    assert style["layout_region_source"] == "opencv_ruled_region"
