from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from lxml import etree
from PIL import Image, ImageDraw

from scan2hwpx.hwpx import render_semantic_hwpx, semantic, validate_hwpx
from scan2hwpx.hwpx.fidelity import _load_template
from scan2hwpx.ir.models import BBox, BlockKind
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider
from scan2hwpx.vision.layout_dataset import TableGrid


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
    assert stats.editable_text_boxes == 4  # body and footer on each page
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
        assert len(flow_boxes) == 4
        body_boxes = [
            box
            for box in flow_boxes
            if len(box.xpath(".//*[local-name()='t'][normalize-space()]")) >= 3
        ]
        assert len(body_boxes) == 2
        assert section.xpath("string()").count("3학년 국어 1/8") == 2


def test_editable_renderer_groups_passage_lines_in_one_continuous_edit_area(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page-1.png"
    Image.new("RGB", (1000, 1400), "white").save(image_path)
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page = document.pages[0]
    page.width = 1000
    page.height = 1400
    page.blocks[0].bbox = BBox(pixel=(100, 60, 900, 110), normalized=(0.1, 0.043, 0.9, 0.079))
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


def _line(text: str, x0: float, y0: float, x1: float, *, height: float = 26.0) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"{text}@{y0}",
        text=text,
        bbox=SimpleNamespace(pixel=(x0, y0, x1, y0 + height)),
        reading_order=0,
        kind=BlockKind.UNKNOWN,
        confidence=0.99,
        annotation_state=SimpleNamespace(value="printed"),
        style={"layout_column": 0},
    )


def _template_header() -> etree._Element:
    return etree.fromstring(_load_template()["Contents/header.xml"])


def _paragraph_layout(
    header: etree._Element, sublist: etree._Element
) -> list[tuple[str, int, int, int]]:
    """(alignment, left indent, spacing before, pitch) of each paragraph, in HWPUNIT."""
    layout = []
    for paragraph in sublist:
        style = header.xpath(f"//*[local-name()='paraPr'][@id='{paragraph.get('paraPrIDRef')}']")[0]

        def length(name: str, style: etree._Element = style) -> int:
            stored = int(style.xpath(f".//*[local-name()='{name}']")[0].get("value"))
            return stored // semantic.PARA_LENGTH_SCALE

        alignment = style.xpath("./*[local-name()='align']")[0].get("horizontal")
        layout.append((alignment, length("left"), length("prev"), length("lineSpacing")))
    return layout


def test_source_rows_keep_blocks_of_one_visual_line_together_with_their_gap() -> None:
    rows = semantic._source_rows(
        [
            _line("② ㄱ, ㄷ", 300, 101, 360),
            _line("다음 줄", 100, 127, 170),
            _line("① ㄱ, ㄴ", 100, 100, 160),
        ]
    )

    assert [[(gap, text) for gap, text, _bold in row.segments] for row in rows] == [
        [(0.0, "① ㄱ, ㄴ"), (140.0, "② ㄱ, ㄷ")],
        [(0.0, "다음 줄")],
    ]


def test_lines_land_on_their_source_indent_and_height() -> None:
    header = _template_header()
    sublist = etree.Element("subList")
    blocks = [
        _line("1. 첫 문항의 첫째 줄은 단의 오른쪽 끝까지 채운다", 100, 100, 600),
        _line("둘째 줄", 130, 127, 300),
        _line("2. 다음 문항", 100, 208, 400),
    ]

    semantic._write_line_paragraphs(
        sublist, blocks, (100, 100, 600, 400), 1000, 1400, semantic._StyleBook(header), (1000, 1600)
    )

    layout = _paragraph_layout(header, sublist)
    # A full-width line is justified; the others start at their own indent.
    assert [(alignment, left) for alignment, left, _prev, _pitch in layout] == [
        ("DISTRIBUTE_SPACE", 0),
        ("LEFT", round((30 * 59_527 / 1000 - semantic.TEXT_BOX_MARGIN_X) / 100) * 100),
        ("LEFT", 0),
    ]
    tops = []
    cursor = 0
    for _alignment, _left, prev, pitch in layout:
        tops.append(cursor + prev)
        cursor += prev + pitch
    targets = [round(dy * 84_189 / 1400) - semantic.TEXT_BOX_MARGIN_Y for dy in (0, 27, 108)]
    assert all(abs(top - max(0, target)) <= 50 for top, target in zip(tops, targets, strict=True))


def test_table_cell_lines_fit_the_source_row_height() -> None:
    header = _template_header()
    cell = etree.Element(f"{{{semantic.HP}}}tc")
    etree.SubElement(cell, f"{{{semantic.HP}}}subList")

    semantic._write_cell_paragraphs(
        cell,
        [_line("예사", 10, 10, 60, height=12), _line("소리", 10, 24, 60, height=12)],
        centered=True,
        bold=False,
        cell_height=1500,
        x_scale=59.527,
        book=semantic._StyleBook(header),
        metrics=(1000, 1600),
    )

    layout = _paragraph_layout(header, cell[0])
    assert len(layout) == 2
    assert sum(pitch for *_, pitch in layout) <= 1500 - 2 * semantic.TABLE_CELL_MARGIN


def test_boxed_grid_whose_rows_split_text_lines_is_not_a_table() -> None:
    grid = TableGrid(bbox=(0, 0, 400, 200), x_lines=(0, 200, 400), y_lines=(0, 100, 200))

    assert not semantic._grid_cuts_text(grid, [_line("칸 안의 글", 20, 30, 180)])
    assert semantic._grid_cuts_text(grid, [_line("칸막이를 가로지르는 글줄", 20, 88, 380)])


def test_passage_box_whose_edge_runs_through_a_text_line_is_not_drawn() -> None:
    blocks = [_line("상자 윗변 근처의 글줄", 100, 200, 450), _line("상자 안의 글줄", 100, 300, 450)]

    def outlines(top: float) -> int:
        region = semantic._ImageRegion(
            bbox=(80, top, 480, 500), source_width=1000, source_height=1400
        )
        return len(semantic._passage_groups(blocks, [region], [], [], [], 1000, 1400))

    assert outlines(210) == 0
    assert outlines(180) == 1


def test_flow_metrics_reproduce_source_character_width_and_line_pitch() -> None:
    blocks = [_line("가" * 30, 100, 100 + 27 * index, 430) for index in range(12)]

    advances, pitches = semantic._flow_samples(blocks, 1000, 1400)

    # 11 px/char and 27 px/line on a 1000x1400 page mapped to the A4 sheet.
    assert semantic._flow_metrics(advances, pitches, min_samples=5) == (800, 1620)
    assert semantic._flow_metrics(advances[:4], pitches[:4], min_samples=5) is None
