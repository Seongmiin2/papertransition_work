from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw

from scan2hwpx.hwpx import render_semantic_hwpx, validate_hwpx
from scan2hwpx.ir.models import BBox
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
        position = tables[0].xpath("./*[local-name()='pos']")[0]
        assert position.get("vertRelTo") == "PAPER"
        assert position.get("horzRelTo") == "PAPER"
        rows = tables[0].xpath("./*[local-name()='tr']")
        assert len(rows[2].xpath("./*[local-name()='tc']")) == 1
        span = rows[2].xpath(".//*[local-name()='cellSpan']")[0]
        assert span.get("colSpan") == "2"


def test_editable_renderer_keeps_pages_without_source_backgrounds(tmp_path: Path) -> None:
    page_images = []
    for page_no in (1, 2):
        image_path = tmp_path / f"page-{page_no}.png"
        Image.new("RGB", (1000, 1400), "white").save(image_path)
        page_images.append(image_path)
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    second_page = copy.deepcopy(document.pages[0])
    second_page.page_no = 2
    document.pages.append(second_page)
    layout_seed = tmp_path / "layout_seed.json"
    layout_seed.write_text(
        json.dumps(
            {
                "pages": [
                    {"page_no": page_no, "width": 1000, "height": 1400, "annotations": []}
                    for page_no in (1, 2)
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "editable.hwpx"

    stats = render_semantic_hwpx(
        page_images,
        document,
        layout_seed,
        output,
        include_page_backgrounds=False,
    )

    assert stats.pages == 2
    assert stats.background_pictures == 0
    assert stats.editable_text_boxes == 2
    assert validate_hwpx(output).valid
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        paragraphs = section.xpath("./*[local-name()='p']")
        assert [paragraph.get("pageBreak") for paragraph in paragraphs] == ["0", "1"]
        assert not section.xpath("//*[local-name()='pic']")
        assert "BinData/image1.jpg" not in archive.namelist()
        assert "BinData/image2.jpg" not in archive.namelist()
        flow_boxes = section.xpath(
            "//*[local-name()='rect'][./*[local-name()='lineShape'][@style='NONE']]"
        )
        assert len(flow_boxes) == 2
        body_boxes = [
            box
            for box in flow_boxes
            if len(box.xpath(".//*[local-name()='t'][normalize-space()]")) >= 3
        ]
        assert len(body_boxes) == 2


def test_editable_renderer_groups_passage_lines_in_one_continuous_edit_area(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (1000, 1400), "white").save(image_path)
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.width = 1000
    page.height = 1400
    page.blocks[1].text = "<보기>"
    page.blocks[1].bbox = BBox(pixel=(220, 215, 330, 245), normalized=(0.22, 0.154, 0.33, 0.175))
    page.blocks[2].text = "작품의 첫 번째 본문 줄"
    page.blocks[2].bbox = BBox(pixel=(130, 275, 420, 310), normalized=(0.13, 0.196, 0.42, 0.221))
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
                                "confidence": 0.55,
                                "source": "opencv_ruled_region",
                                "bbox": [100, 200, 450, 500],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "passage.hwpx"

    stats = render_semantic_hwpx(
        [image_path],
        document,
        layout_seed,
        output,
        include_page_backgrounds=False,
    )

    assert stats.editable_text_boxes >= 1
    assert validate_hwpx(output).valid
    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        header = etree.fromstring(archive.read("Contents/header.xml"))
        boxed = section.xpath(
            "//*[local-name()='rect'][./*[local-name()='lineShape'][@style='SOLID']]"
        )
        assert len(boxed) == 1
        assert not section.xpath("//*[local-name()='tbl']")
        paragraphs = boxed[0].xpath(".//*[local-name()='subList']/*[local-name()='p']")
        assert len(paragraphs) == 1
        assert not boxed[0].xpath(".//*[local-name()='t']/text()")
        flow_boxes = section.xpath(
            "//*[local-name()='rect'][./*[local-name()='lineShape'][@style='NONE']]"
        )
        passage_flow = [
            box
            for box in flow_boxes
            if "<보기>" in box.xpath("string(.//*[local-name()='subList'])")
        ]
        assert len(passage_flow) == 1
        flow_text = passage_flow[0].xpath("string(.//*[local-name()='subList'])")
        assert "작품의 첫 번째 본문 줄" in flow_text
        wraps = section.xpath(
            "//*[local-name()='rect']/*[local-name()='drawText']/*[local-name()='subList']"
        )
        assert {item.get("lineWrap") for item in wraps} == {"BREAK"}
        style_ids = [
            int(style.get("id"))
            for style in header.xpath(
                "//*[local-name()='charProperties']/*[local-name()='charPr']"
            )
        ]
        assert style_ids == list(range(len(style_ids)))
        ocr_styles = header.xpath(
            "//*[local-name()='charProperties']/*[local-name()='charPr'][@height='800']"
        )
        assert len(ocr_styles) == 2


def test_editable_renderer_rejects_cross_column_passage_outline(tmp_path: Path) -> None:
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (1000, 1400), "white").save(image_path)
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
                                "label": "passage_box",
                                "confidence": 0.5,
                                "source": "opencv_ruled_region",
                                "bbox": [50, 200, 950, 900],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "cross-column.hwpx"

    render_semantic_hwpx(
        [image_path],
        document,
        layout_seed,
        output,
        include_page_backgrounds=False,
    )

    with zipfile.ZipFile(output) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        assert not section.xpath(
            "//*[local-name()='rect'][./*[local-name()='lineShape'][@style='SOLID']]"
        )
