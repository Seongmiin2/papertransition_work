from __future__ import annotations

import base64
import copy
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from lxml import etree  # type: ignore[import-untyped]
from PIL import Image, ImageDraw

from scan2hwpx.ir.models import Document
from scan2hwpx.preprocess import preprocess_for_ocr
from scan2hwpx.vision.layout_dataset import TableGrid, detect_table_grids

from .fidelity import HC, HP, OPF, render_fidelity_hwpx
from .render import MIMETYPE

HH = "http://www.hancom.co.kr/hwpml/2011/head"
PAGE_WIDTH_HWP = 56_693
PAGE_HEIGHT_HWP = 80_221
CENTER_PARA_ID = "20"
LEFT_PARA_ID = "11"
BOLD_CHAR_ID = "7"
REGULAR_CHAR_ID = "2"
OCR_CHAR_STYLE_START = 100
EDITABLE_TEXT_CONFIDENCE = 0.75


@dataclass(frozen=True)
class SemanticRenderStats:
    pages: int
    background_pictures: int
    editable_tables: int
    extracted_pictures: int
    editable_text_boxes: int
    editable_text_characters: int
    package_bytes: int


@dataclass(frozen=True)
class _ImageRegion:
    bbox: tuple[float, float, float, float]
    source_width: float
    source_height: float


def render_semantic_hwpx(
    page_images: list[Path],
    document: Document,
    layout_seed: Path,
    output: Path,
) -> SemanticRenderStats:
    """Overlay editable table cells and extracted pictures on the fidelity background."""
    if len(page_images) != len(document.pages):
        raise ValueError("page image count and document page count differ")
    layout_pages = _read_layout_pages(layout_seed, len(page_images))
    temporary = output.with_name(f".{output.stem}.fidelity{output.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="scan2hwpx-semantic-") as temporary_directory:
            clean_pages = _build_editable_backgrounds(
                page_images,
                document,
                layout_pages,
                Path(temporary_directory),
            )
            render_fidelity_hwpx(clean_pages, temporary)
            members = _read_package(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    section = etree.fromstring(members["Contents/section0.xml"])
    header = etree.fromstring(members["Contents/header.xml"])
    manifest = etree.fromstring(members["Contents/content.hpf"])
    table_template, border_fill = _load_table_parts()
    _install_table_border_fill(header, border_fill)
    _install_semantic_text_styles(header)
    ocr_text_styles = _install_ocr_text_styles(header, document)
    paragraphs = cast(list[etree._Element], section.xpath("./hp:p", namespaces={"hp": HP}))
    background_picture = cast(
        etree._Element,
        section.xpath(".//hp:pic", namespaces={"hp": HP})[0],
    )
    editable_tables = 0
    extracted_pictures = 0
    editable_text_boxes = 0
    editable_text_characters = 0
    binary_assets: dict[str, bytes] = {}
    for page_index, (page_path, ir_page, layout_page) in enumerate(
        zip(page_images, document.pages, layout_pages, strict=True), start=1
    ):
        with Image.open(page_path) as opened:
            page_width, page_height = opened.size
        image_regions = _image_regions(layout_page)
        scaled_images = [
            _scale_region(item, float(page_width), float(page_height)) for item in image_regions
        ]
        table_regions = [
            _scale_region(item, float(page_width), float(page_height))
            for item in _table_regions(layout_page)
        ]
        grids = [
            _align_grid_to_seed(grid, table_regions)
            for grid in detect_table_grids(page_path)
            if _grid_matches_seed(grid, table_regions)
        ]
        page_binary = _binarize_page(page_path) if grids else None
        for table_index, grid in enumerate(grids, start=1):
            if any(_intersection_ratio(grid.bbox, image.bbox) >= 0.45 for image in scaled_images):
                continue
            table = _build_table(
                table_template,
                grid,
                _detect_column_spans(page_binary, grid),
                ir_page.blocks,
                page_width,
                page_height,
                ocr_text_styles,
                control_id=1_300_000_000 + page_index * 100 + table_index,
                z_order=100 + editable_tables,
            )
            _append_control(paragraphs[page_index - 1], table)
            editable_tables += 1
        for image_index, region in enumerate(scaled_images, start=1):
            asset_id = f"semantic_image_{page_index}_{image_index}"
            asset_name = f"BinData/{asset_id}.jpg"
            crop_bytes, crop_size = _crop_image(page_path, region.bbox)
            binary_assets[asset_name] = crop_bytes
            picture = _build_picture(
                background_picture,
                region.bbox,
                page_width,
                page_height,
                crop_size,
                asset_id,
                control_id=1_400_000_000 + page_index * 100 + image_index,
                z_order=200 + extracted_pictures,
            )
            _append_control(paragraphs[page_index - 1], picture)
            _add_manifest_item(manifest, asset_id, asset_name)
            extracted_pictures += 1
        for block_index, block in enumerate(ir_page.blocks, start=1):
            if not _eligible_editable_block(block, scaled_images, table_regions):
                continue
            style_key = _ocr_style_key(block, ir_page.height)
            text_box = _build_text_box(
                block,
                page_width,
                page_height,
                char_style_id=ocr_text_styles[style_key],
                control_id=1_500_000_000 + page_index * 10_000 + block_index,
                z_order=1_000 + editable_text_boxes,
            )
            _append_control(paragraphs[page_index - 1], text_box)
            editable_text_boxes += 1
            editable_text_characters += len(str(block.text))

    members["Contents/section0.xml"] = _xml_bytes(section)
    members["Contents/header.xml"] = _xml_bytes(header)
    members["Contents/content.hpf"] = _xml_bytes(manifest)
    members.update(binary_assets)
    _write_package(output, members)
    return SemanticRenderStats(
        pages=len(page_images),
        background_pictures=len(page_images),
        editable_tables=editable_tables,
        extracted_pictures=extracted_pictures,
        editable_text_boxes=editable_text_boxes,
        editable_text_characters=editable_text_characters,
        package_bytes=output.stat().st_size,
    )


def _read_layout_pages(path: Path, expected_pages: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = cast(list[dict[str, Any]], payload.get("pages", []))
    if len(pages) != expected_pages:
        raise ValueError("layout seed page count differs from source")
    return pages


def _read_package(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _load_table_parts() -> tuple[etree._Element, etree._Element]:
    payload = Path(__file__).with_name("table_template.b64").read_text(encoding="ascii")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        section = etree.fromstring(archive.read("Contents/section0.xml"))
        header = etree.fromstring(archive.read("Contents/header.xml"))
    tables = cast(list[etree._Element], section.xpath(".//hp:tbl", namespaces={"hp": HP}))
    borders = cast(
        list[etree._Element],
        header.xpath(".//hh:borderFill[@id='3']", namespaces={"hh": HH}),
    )
    if len(tables) != 1 or len(borders) != 1:
        raise RuntimeError("editable table template is invalid")
    return tables[0], borders[0]


def _install_table_border_fill(header: etree._Element, source: etree._Element) -> None:
    containers = cast(
        list[etree._Element],
        header.xpath(".//hh:borderFills", namespaces={"hh": HH}),
    )
    if len(containers) != 1:
        raise RuntimeError("HWPX header border-fill list is invalid")
    container = containers[0]
    existing = container.xpath("./hh:borderFill[@id='3']", namespaces={"hh": HH})
    if not existing:
        border = copy.deepcopy(source)
        fill_brush = etree.SubElement(border, f"{{{HC}}}fillBrush")
        brush = etree.SubElement(fill_brush, f"{{{HC}}}winBrush")
        brush.set("faceColor", "#FFFFFF")
        brush.set("hatchColor", "#999999")
        brush.set("alpha", "0")
        container.append(border)
    container.set("itemCnt", str(len(container)))


def _install_semantic_text_styles(header: etree._Element) -> None:
    para_containers = cast(
        list[etree._Element],
        header.xpath(".//hh:paraProperties", namespaces={"hh": HH}),
    )
    char_containers = cast(
        list[etree._Element],
        header.xpath(".//hh:charProperties", namespaces={"hh": HH}),
    )
    if len(para_containers) != 1 or len(char_containers) != 1:
        raise RuntimeError("HWPX text style lists are invalid")
    para_container = para_containers[0]
    if not para_container.xpath(f"./hh:paraPr[@id='{CENTER_PARA_ID}']", namespaces={"hh": HH}):
        source = cast(
            etree._Element,
            para_container.xpath(f"./hh:paraPr[@id='{LEFT_PARA_ID}']", namespaces={"hh": HH})[0],
        )
        centered = copy.deepcopy(source)
        centered.set("id", CENTER_PARA_ID)
        align = cast(etree._Element, centered.xpath("./hh:align", namespaces={"hh": HH})[0])
        align.set("horizontal", "CENTER")
        para_container.append(centered)
    para_container.set("itemCnt", str(len(para_container)))
    char_container = char_containers[0]
    if not char_container.xpath(f"./hh:charPr[@id='{BOLD_CHAR_ID}']", namespaces={"hh": HH}):
        source = cast(
            etree._Element,
            char_container.xpath(f"./hh:charPr[@id='{REGULAR_CHAR_ID}']", namespaces={"hh": HH})[0],
        )
        bold = copy.deepcopy(source)
        bold.set("id", BOLD_CHAR_ID)
        etree.SubElement(bold, f"{{{HH}}}bold")
        char_container.append(bold)
    char_container.set("itemCnt", str(len(char_container)))


def _install_ocr_text_styles(
    header: etree._Element, document: Document
) -> dict[tuple[int, bool], str]:
    containers = cast(
        list[etree._Element],
        header.xpath(".//hh:charProperties", namespaces={"hh": HH}),
    )
    if len(containers) != 1:
        raise RuntimeError("HWPX header character-style list is invalid")
    container = containers[0]
    sources = cast(
        list[etree._Element],
        container.xpath(f"./hh:charPr[@id='{REGULAR_CHAR_ID}']", namespaces={"hh": HH}),
    )
    if len(sources) != 1:
        raise RuntimeError("regular HWPX character style is missing")
    heights = {
        _ocr_style_key(block, page.height)[0]
        for page in document.pages
        for block in page.blocks
        if str(block.text).strip()
    }
    keys = [(height, bold) for height in sorted(heights) for bold in (False, True)]
    result: dict[tuple[int, bool], str] = {}
    for index, (height, bold) in enumerate(keys):
        style_id = str(OCR_CHAR_STYLE_START + index)
        style = copy.deepcopy(sources[0])
        style.set("id", style_id)
        style.set("height", str(height))
        style.set("textColor", "#000000")
        for existing_bold in style.xpath("./hh:bold", namespaces={"hh": HH}):
            style.remove(existing_bold)
        if bold:
            etree.SubElement(style, f"{{{HH}}}bold")
        container.append(style)
        result[(height, bold)] = style_id
    container.set("itemCnt", str(len(container)))
    return result


def _ocr_style_key(block: Any, page_height: float) -> tuple[int, bool]:
    y0 = float(block.bbox.pixel[1])
    y1 = float(block.bbox.pixel[3])
    points = (y1 - y0) / max(1.0, page_height) * 841.89
    height = int(round(min(16.0, max(6.0, points)) * 2) * 50)
    bold = str(block.kind) in {"instruction", "question", "title"} or height >= 1_200
    return height, bold


def _build_editable_backgrounds(
    page_images: list[Path],
    document: Document,
    layout_pages: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    for page_index, (page_path, ir_page, layout_page) in enumerate(
        zip(page_images, document.pages, layout_pages, strict=True), start=1
    ):
        with Image.open(page_path) as opened:
            image = opened.convert("RGB")
        page_width, page_height = image.size
        image_regions = [
            _scale_region(item, float(page_width), float(page_height))
            for item in _image_regions(layout_page)
        ]
        table_regions = [
            _scale_region(item, float(page_width), float(page_height))
            for item in _table_regions(layout_page)
        ]
        array = np.asarray(image, dtype=np.uint8)
        draw = ImageDraw.Draw(image)
        for block in ir_page.blocks:
            if not _eligible_editable_block(block, image_regions, table_regions):
                continue
            x0, y0, x1, y1 = _clamped_pixel_box(block.bbox.pixel, page_width, page_height)
            if x1 <= x0 or y1 <= y0:
                continue
            draw.rectangle((x0, y0, x1, y1), fill=_background_color(array, (x0, y0, x1, y1)))
        destination = output_dir / f"page-{page_index}.png"
        image.save(destination, format="PNG", optimize=True)
        result.append(destination)
    return result


def _eligible_editable_block(
    block: Any,
    image_regions: list[_ImageRegion],
    table_regions: list[_ImageRegion],
) -> bool:
    text = str(block.text).strip()
    if not text or float(block.confidence) < EDITABLE_TEXT_CONFIDENCE:
        return False
    raw_bbox = block.bbox.pixel
    bbox = (
        float(raw_bbox[0]),
        float(raw_bbox[1]),
        float(raw_bbox[2]),
        float(raw_bbox[3]),
    )
    return not any(
        _intersection_ratio(bbox, region.bbox) >= 0.35
        for region in [*image_regions, *table_regions]
    )


def _clamped_pixel_box(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, round(x0) - 1),
        max(0, round(y0) - 1),
        min(width - 1, round(x1) + 1),
        min(height - 1, round(y1) + 1),
    )


def _background_color(
    image: np.ndarray[Any, Any], bbox: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    x0, y0, x1, y1 = bbox
    pad = 5
    height, width = image.shape[:2]
    strips = [
        image[max(0, y0 - pad) : y0, max(0, x0 - pad) : min(width, x1 + pad + 1)],
        image[y1 + 1 : min(height, y1 + pad + 1), max(0, x0 - pad) : min(width, x1 + pad + 1)],
        image[y0 : y1 + 1, max(0, x0 - pad) : x0],
        image[y0 : y1 + 1, x1 + 1 : min(width, x1 + pad + 1)],
    ]
    samples = [strip.reshape(-1, 3) for strip in strips if strip.size]
    if not samples:
        return (255, 255, 255)
    color = np.median(np.concatenate(samples, axis=0), axis=0)
    if float(np.mean(color)) < 180:
        return (255, 255, 255)
    return (round(color[0]), round(color[1]), round(color[2]))


def _build_text_box(
    block: Any,
    page_width: int,
    page_height: int,
    *,
    char_style_id: str,
    control_id: int,
    z_order: int,
) -> etree._Element:
    x0, y0, x1, y1 = (float(value) for value in block.bbox.pixel)
    width = _x_hwp(max(2.0, x1 - x0 + 3.0), page_width)
    height = _y_hwp(max(2.0, y1 - y0 + 2.0), page_height)
    rectangle = etree.Element(f"{{{HP}}}rect")
    for name, value in {
        "id": control_id,
        "zOrder": z_order,
        "numberingType": "NONE",
        "textWrap": "IN_FRONT_OF_TEXT",
        "textFlow": "BOTH_SIDES",
        "lock": 0,
        "dropcapstyle": "None",
        "href": "",
        "groupLevel": 0,
        "instid": control_id + 1,
        "ratio": 0,
    }.items():
        rectangle.set(name, str(value))
    etree.SubElement(rectangle, f"{{{HP}}}offset", x="0", y="0")
    etree.SubElement(rectangle, f"{{{HP}}}orgSz", width=str(width), height=str(height))
    etree.SubElement(rectangle, f"{{{HP}}}curSz", width=str(width), height=str(height))
    etree.SubElement(rectangle, f"{{{HP}}}flip", horizontal="0", vertical="0")
    etree.SubElement(
        rectangle,
        f"{{{HP}}}rotationInfo",
        angle="0",
        centerX=str(width // 2),
        centerY=str(height // 2),
        rotateimage="1",
    )
    rendering = etree.SubElement(rectangle, f"{{{HP}}}renderingInfo")
    for matrix_name in ("transMatrix", "scaMatrix", "rotMatrix"):
        etree.SubElement(
            rendering,
            f"{{{HC}}}{matrix_name}",
            e1="1",
            e2="0",
            e3="0",
            e4="0",
            e5="1",
            e6="0",
        )
    etree.SubElement(
        rectangle,
        f"{{{HP}}}lineShape",
        color="#000000",
        width="0",
        style="NONE",
        endCap="ROUND",
        headStyle="NORMAL",
        tailStyle="NORMAL",
        headfill="0",
        tailfill="0",
        headSz="SMALL_SMALL",
        tailSz="SMALL_SMALL",
        outlineStyle="NORMAL",
        alpha="0",
    )
    etree.SubElement(
        rectangle,
        f"{{{HP}}}shadow",
        type="NONE",
        color="#B2B2B2",
        offsetX="0",
        offsetY="0",
        alpha="0",
    )
    draw_text = etree.SubElement(
        rectangle,
        f"{{{HP}}}drawText",
        lastWidth="4294967295",
        name="",
        editable="1",
    )
    sublist = etree.SubElement(
        draw_text,
        f"{{{HP}}}subList",
        id="",
        textDirection="HORIZONTAL",
        lineWrap="SQUEEZE",
        vertAlign="CENTER",
        linkListIDRef="0",
        linkListNextIDRef="0",
        textWidth=str(width),
        textHeight=str(height),
        hasTextRef="0",
        hasNumRef="0",
    )
    paragraph = etree.SubElement(
        sublist,
        f"{{{HP}}}p",
        id="0",
        paraPrIDRef=LEFT_PARA_ID,
        styleIDRef="0",
        pageBreak="0",
        columnBreak="0",
        merged="0",
    )
    run = etree.SubElement(paragraph, f"{{{HP}}}run", charPrIDRef=char_style_id)
    text = etree.SubElement(run, f"{{{HP}}}t")
    text.text = str(block.text).strip()
    etree.SubElement(draw_text, f"{{{HP}}}textMargin", left="0", right="0", top="0", bottom="0")
    etree.SubElement(rectangle, f"{{{HC}}}pt0", x="0", y="0")
    etree.SubElement(rectangle, f"{{{HC}}}pt1", x=str(width), y="0")
    etree.SubElement(rectangle, f"{{{HC}}}pt2", x=str(width), y=str(height))
    etree.SubElement(rectangle, f"{{{HC}}}pt3", x="0", y=str(height))
    etree.SubElement(
        rectangle,
        f"{{{HP}}}sz",
        width=str(width),
        widthRelTo="ABSOLUTE",
        height=str(height),
        heightRelTo="ABSOLUTE",
        protect="0",
    )
    etree.SubElement(
        rectangle,
        f"{{{HP}}}pos",
        treatAsChar="0",
        affectLSpacing="0",
        flowWithText="0",
        allowOverlap="1",
        holdAnchorAndSO="0",
        vertRelTo="PAPER",
        horzRelTo="PAPER",
        vertAlign="TOP",
        horzAlign="LEFT",
        vertOffset=str(_y_hwp(y0, page_height)),
        horzOffset=str(_x_hwp(x0, page_width)),
    )
    etree.SubElement(rectangle, f"{{{HP}}}outMargin", left="0", right="0", top="0", bottom="0")
    comment = etree.SubElement(rectangle, f"{{{HP}}}shapeComment")
    comment.text = "OCR 좌표 기반 편집 글상자"
    return rectangle


def _build_table(
    template: etree._Element,
    grid: TableGrid,
    cell_layout: list[list[tuple[int, int]]],
    blocks: list[Any],
    page_width: int,
    page_height: int,
    ocr_text_styles: dict[tuple[int, bool], str],
    *,
    control_id: int,
    z_order: int,
) -> etree._Element:
    table = copy.deepcopy(template)
    rows = cast(list[etree._Element], table.xpath("./hp:tr", namespaces={"hp": HP}))
    cell_template = cast(
        etree._Element,
        rows[0].xpath("./hp:tc", namespaces={"hp": HP})[0],
    )
    for row in rows:
        table.remove(row)
    table.set("id", str(control_id))
    table.set("zOrder", str(z_order))
    table.set("rowCnt", str(grid.rows))
    table.set("colCnt", str(grid.columns))
    table.set("textWrap", "IN_FRONT_OF_TEXT")
    table.set("repeatHeader", "0")
    x_lines = grid.x_lines
    y_lines = grid.y_lines
    total_width = _x_hwp(x_lines[-1] - x_lines[0], page_width)
    total_height = _y_hwp(y_lines[-1] - y_lines[0], page_height)
    size = cast(etree._Element, table.xpath("./hp:sz", namespaces={"hp": HP})[0])
    size.set("width", str(total_width))
    size.set("height", str(total_height))
    position = cast(etree._Element, table.xpath("./hp:pos", namespaces={"hp": HP})[0])
    position.set("flowWithText", "0")
    position.set("allowOverlap", "1")
    position.set("horzOffset", str(_x_hwp(x_lines[0], page_width)))
    position.set("vertOffset", str(_y_hwp(y_lines[0], page_height)))
    for row_index in range(grid.rows):
        row = etree.Element(f"{{{HP}}}tr")
        for column_index, column_span in cell_layout[row_index]:
            cell = copy.deepcopy(cell_template)
            address = cast(
                etree._Element,
                cell.xpath("./hp:cellAddr", namespaces={"hp": HP})[0],
            )
            address.set("colAddr", str(column_index))
            address.set("rowAddr", str(row_index))
            span = cast(
                etree._Element,
                cell.xpath("./hp:cellSpan", namespaces={"hp": HP})[0],
            )
            span.set("colSpan", str(column_span))
            cell_size = cast(
                etree._Element,
                cell.xpath("./hp:cellSz", namespaces={"hp": HP})[0],
            )
            cell_size.set(
                "width",
                str(
                    _x_hwp(
                        x_lines[column_index + column_span] - x_lines[column_index],
                        page_width,
                    )
                ),
            )
            cell_size.set(
                "height", str(_y_hwp(y_lines[row_index + 1] - y_lines[row_index], page_height))
            )
            lines = _cell_lines(
                blocks,
                (
                    x_lines[column_index],
                    y_lines[row_index],
                    x_lines[column_index + column_span],
                    y_lines[row_index + 1],
                ),
            )
            _write_cell_paragraphs(
                cell,
                lines,
                centered=(x_lines[column_index + column_span] - x_lines[column_index])
                / (x_lines[-1] - x_lines[0])
                < 0.3
                or grid.columns >= 3,
                bold=column_index == 0,
                page_height=page_height,
                ocr_text_styles=ocr_text_styles,
            )
            row.append(cell)
        table.append(row)
    return table


def _binarize_page(path: Path) -> Any:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    cleaned = preprocess_for_ocr(rgb).image
    gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _detect_column_spans(binary: Any, grid: TableGrid) -> list[list[tuple[int, int]]]:
    rows: list[list[tuple[int, int]]] = []
    for row_index in range(grid.rows):
        y0 = max(0, round(grid.y_lines[row_index] + 3))
        y1 = min(binary.shape[0], round(grid.y_lines[row_index + 1] - 3))
        starts = [0]
        for boundary_index, x_value in enumerate(grid.x_lines[1:-1], start=1):
            x = round(x_value)
            band = binary[y0:y1, max(0, x - 4) : min(binary.shape[1], x + 5)]
            coverage = (
                float(np.mean(np.max(band, axis=1) > 0)) if band.shape[0] and band.shape[1] else 0
            )
            if coverage >= 0.55:
                starts.append(boundary_index)
        starts.append(grid.columns)
        rows.append([(start, end - start) for start, end in pairwise(starts)])
    return rows


def _cell_lines(blocks: list[Any], bbox: tuple[float, float, float, float]) -> list[Any]:
    selected = []
    for block in blocks:
        x0, y0, x1, y1 = block.bbox.pixel
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        if bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]:
            selected.append((y0, x0, block))
    return [
        block
        for _y, _x, block in sorted(selected, key=lambda item: (item[0], item[1]))
        if str(block.text).strip()
    ]


def _write_cell_paragraphs(
    cell: etree._Element,
    lines: list[Any],
    *,
    centered: bool,
    bold: bool,
    page_height: float,
    ocr_text_styles: dict[tuple[int, bool], str],
) -> None:
    sublist = cast(
        etree._Element,
        cell.xpath("./hp:subList", namespaces={"hp": HP})[0],
    )
    for paragraph in sublist.xpath("./hp:p", namespaces={"hp": HP}):
        sublist.remove(paragraph)
    for block in lines or [None]:
        paragraph = etree.SubElement(sublist, f"{{{HP}}}p")
        paragraph.set("id", "0")
        paragraph.set("paraPrIDRef", CENTER_PARA_ID if centered else LEFT_PARA_ID)
        paragraph.set("styleIDRef", "0")
        paragraph.set("pageBreak", "0")
        paragraph.set("columnBreak", "0")
        paragraph.set("merged", "0")
        run = etree.SubElement(paragraph, f"{{{HP}}}run")
        if block is None:
            run.set("charPrIDRef", BOLD_CHAR_ID if bold else REGULAR_CHAR_ID)
        else:
            height, inferred_bold = _ocr_style_key(block, page_height)
            run.set("charPrIDRef", ocr_text_styles[(height, bold or inferred_bold)])
        if block is not None:
            text = etree.SubElement(run, f"{{{HP}}}t")
            text.text = str(block.text).strip()


def _image_regions(page: dict[str, Any]) -> list[_ImageRegion]:
    width = float(page["width"])
    height = float(page["height"])
    regions: list[_ImageRegion] = []
    for annotation in page.get("annotations", []):
        if annotation.get("label") != "image" or float(annotation.get("confidence", 0)) < 0.45:
            continue
        values = [float(value) for value in annotation["bbox"]]
        regions.append(
            _ImageRegion(
                bbox=(values[0], values[1], values[2], values[3]),
                source_width=width,
                source_height=height,
            )
        )
    return regions


def _table_regions(page: dict[str, Any]) -> list[_ImageRegion]:
    width = float(page["width"])
    height = float(page["height"])
    regions: list[_ImageRegion] = []
    for annotation in page.get("annotations", []):
        if annotation.get("label") != "table" or float(annotation.get("confidence", 0)) < 0.45:
            continue
        values = [float(value) for value in annotation["bbox"]]
        regions.append(
            _ImageRegion(
                bbox=(values[0], values[1], values[2], values[3]),
                source_width=width,
                source_height=height,
            )
        )
    return regions


def _align_grid_to_seed(grid: TableGrid, regions: list[_ImageRegion]) -> TableGrid:
    matches = [region for region in regions if _intersection_ratio(grid.bbox, region.bbox) >= 0.65]
    if not matches:
        return grid
    seed = max(matches, key=lambda item: _intersection_ratio(grid.bbox, item.bbox))
    x0, y0, x1, y1 = seed.bbox
    x_lines = _add_seed_boundaries(grid.x_lines, x0, x1)
    y_lines = _add_seed_boundaries(grid.y_lines, y0, y1)
    return TableGrid(
        bbox=(x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]),
        x_lines=x_lines,
        y_lines=y_lines,
    )


def _grid_matches_seed(grid: TableGrid, regions: list[_ImageRegion]) -> bool:
    return any(_intersection_ratio(grid.bbox, region.bbox) >= 0.65 for region in regions)


def _add_seed_boundaries(
    lines: tuple[float, ...], start: float, end: float, tolerance: float = 25
) -> tuple[float, ...]:
    values = list(lines)
    for boundary in (start, end):
        nearest = min(range(len(values)), key=lambda index: abs(values[index] - boundary))
        if abs(values[nearest] - boundary) <= tolerance:
            values[nearest] = (values[nearest] + boundary) / 2
        else:
            values.append(boundary)
    return tuple(sorted(values))


def _scale_region(region: _ImageRegion, width: float, height: float) -> _ImageRegion:
    sx = width / region.source_width
    sy = height / region.source_height
    x0, y0, x1, y1 = region.bbox
    return _ImageRegion(
        bbox=(x0 * sx, y0 * sy, x1 * sx, y1 * sy),
        source_width=width,
        source_height=height,
    )


def _crop_image(
    path: Path, bbox: tuple[float, float, float, float]
) -> tuple[bytes, tuple[int, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        x0, y0, x1, y1 = (round(value) for value in bbox)
        crop = image.crop((x0, y0, x1, y1))
        stream = io.BytesIO()
        crop.save(stream, "JPEG", quality=94, optimize=True)
        return stream.getvalue(), crop.size


def _build_picture(
    template: etree._Element,
    bbox: tuple[float, float, float, float],
    page_width: int,
    page_height: int,
    crop_size: tuple[int, int],
    asset_id: str,
    *,
    control_id: int,
    z_order: int,
) -> etree._Element:
    picture = copy.deepcopy(template)
    x0, y0, x1, y1 = bbox
    width = _x_hwp(x1 - x0, page_width)
    height = _y_hwp(y1 - y0, page_height)
    picture.set("id", str(control_id))
    picture.set("instid", str(control_id + 1))
    picture.set("zOrder", str(z_order))
    picture.set("textWrap", "IN_FRONT_OF_TEXT")
    for name in ("orgSz", "sz"):
        element = cast(
            etree._Element,
            picture.xpath(f"./hp:{name}", namespaces={"hp": HP})[0],
        )
        element.set("width", str(width))
        element.set("height", str(height))
    rotation = cast(
        etree._Element,
        picture.xpath("./hp:rotationInfo", namespaces={"hp": HP})[0],
    )
    rotation.set("centerX", str(width // 2))
    rotation.set("centerY", str(height // 2))
    points = cast(
        list[etree._Element], picture.xpath("./hp:imgRect/hc:*", namespaces={"hp": HP, "hc": HC})
    )
    coordinates = ((0, 0), (width, 0), (width, height), (0, height))
    for point, (point_x, point_y) in zip(points, coordinates, strict=True):
        point.set("x", str(point_x))
        point.set("y", str(point_y))
    clip = cast(etree._Element, picture.xpath("./hp:imgClip", namespaces={"hp": HP})[0])
    clip.set("right", str(crop_size[0] * 75))
    clip.set("bottom", str(crop_size[1] * 75))
    image = cast(etree._Element, picture.xpath("./hc:img", namespaces={"hc": HC})[0])
    image.set("binaryItemIDRef", asset_id)
    position = cast(etree._Element, picture.xpath("./hp:pos", namespaces={"hp": HP})[0])
    position.set("flowWithText", "0")
    position.set("allowOverlap", "1")
    position.set("horzOffset", str(_x_hwp(x0, page_width)))
    position.set("vertOffset", str(_y_hwp(y0, page_height)))
    comments = picture.xpath("./hp:shapeComment", namespaces={"hp": HP})
    if comments:
        comments[0].text = "레이아웃 모델이 분리한 원본 그림 영역"
    return picture


def _append_control(paragraph: etree._Element, control: etree._Element) -> None:
    run = etree.Element(f"{{{HP}}}run")
    run.set("charPrIDRef", "0")
    run.append(control)
    text = etree.SubElement(run, f"{{{HP}}}t")
    text.text = ""
    line_segments = paragraph.xpath("./hp:linesegarray", namespaces={"hp": HP})
    paragraph.insert(paragraph.index(line_segments[0]) if line_segments else len(paragraph), run)


def _add_manifest_item(root: etree._Element, asset_id: str, asset_name: str) -> None:
    manifests = cast(list[etree._Element], root.xpath("./opf:manifest", namespaces={"opf": OPF}))
    manifest = manifests[0]
    item = etree.Element(f"{{{OPF}}}item")
    item.set("id", asset_id)
    item.set("href", asset_name)
    item.set("media-type", "image/jpg")
    item.set("isEmbeded", "1")
    manifest.append(item)


def _x_hwp(value: float, page_width: int) -> int:
    return max(1, round(value / page_width * PAGE_WIDTH_HWP))


def _y_hwp(value: float, page_height: int) -> int:
    return max(1, round(value / page_height * PAGE_HEIGHT_HWP))


def _intersection_ratio(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, first[2] - first[0]) * max(1.0, first[3] - first[1])
    return intersection / area


def _xml_bytes(root: etree._Element) -> bytes:
    return cast(
        bytes,
        etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True),
    )


def _write_package(output: Path, members: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", MIMETYPE, compress_type=zipfile.ZIP_STORED)
        for name, data in members.items():
            if name == "mimetype":
                continue
            archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
