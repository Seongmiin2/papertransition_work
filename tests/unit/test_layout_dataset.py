from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from scan2hwpx.vision.layout_dataset import detect_ruled_regions, detect_table_grids


def test_detect_ruled_regions_distinguishes_grid_from_box(tmp_path: Path) -> None:
    path = tmp_path / "layout.png"
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 450, 500), outline="black", width=5)
    draw.line((250, 50, 250, 500), fill="black", width=5)
    draw.line((50, 200, 450, 200), fill="black", width=5)
    draw.line((50, 350, 450, 350), fill="black", width=5)
    draw.rectangle((550, 100, 950, 400), outline="black", width=5)
    image.save(path)

    labels = {item.label for item in detect_ruled_regions(path)}

    assert "table" in labels
    assert "passage_box" in labels
    grids = detect_table_grids(path)
    assert [(grid.rows, grid.columns) for grid in grids] == [(3, 2)]
