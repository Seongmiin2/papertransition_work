from __future__ import annotations

import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw

from scan2hwpx.hwpx import render_fidelity_hwpx, validate_hwpx


def test_fidelity_renderer_preserves_one_picture_per_page(tmp_path: Path) -> None:
    page_images: list[Path] = []
    for index in range(1, 4):
        path = tmp_path / f"page-{index}.png"
        image = Image.new("RGB", (728, 1032), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((25, 25, 703, 1007), outline="black", width=3)
        draw.text((50, 50), f"page {index}", fill="black")
        image.save(path)
        page_images.append(path)

    output = tmp_path / "result.hwpx"
    stats = render_fidelity_hwpx(page_images, output)

    assert stats.pages == 3
    assert stats.pictures == 3
    assert validate_hwpx(output).valid
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        manifest = etree.fromstring(archive.read("Contents/content.hpf"))
        assert len(section.xpath("//*[local-name()='pic']")) == 3
        assert len(section.xpath("./*[local-name()='p'][@pageBreak='1']")) == 2
        assert len(manifest.xpath("//*[local-name()='item'][starts-with(@id, 'image')]")) == 3
        assert all(f"BinData/image{index}.jpg" in archive.namelist() for index in range(1, 4))
