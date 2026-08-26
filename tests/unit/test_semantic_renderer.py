from __future__ import annotations

import json
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw

from scan2hwpx.hwpx import render_semantic_hwpx, validate_hwpx
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider


def test_semantic_renderer_adds_editable_table_control(tmp_path: Path) -> None:
    image_path = tmp_path / "page-1.png"
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 200, 900, 800), outline="black", width=5)
    draw.line((500, 200, 500, 600), fill="black", width=5)
    draw.line((100, 400, 900, 400), fill="black", width=5)
    draw.line((100, 600, 900, 600), fill="black", width=5)
    image.save(image_path)
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    layout_seed = tmp_path / "layout_seed.json"
    layout_seed.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_no": 1,
                        "width": 1000,
                        "height": 1400,
                        "annotations": [
                            {
                                "label": "table",
                                "confidence": 0.9,
                                "bbox": [100, 200, 900, 800],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "semantic.hwpx"

    stats = render_semantic_hwpx([image_path], document, layout_seed, output)

    assert stats.editable_tables == 1
    assert validate_hwpx(output).valid
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        assert len(section.xpath("//*[local-name()='pic']")) == 1
        tables = section.xpath("//*[local-name()='tbl']")
        assert [(table.get("rowCnt"), table.get("colCnt")) for table in tables] == [("3", "2")]
        rows = tables[0].xpath("./*[local-name()='tr']")
        assert len(rows[2].xpath("./*[local-name()='tc']")) == 1
        span = rows[2].xpath(".//*[local-name()='cellSpan']")[0]
        assert span.get("colSpan") == "2"
