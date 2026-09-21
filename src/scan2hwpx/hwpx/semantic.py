from __future__ import annotations

import base64
import copy
import io
import json
import re
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

from scan2hwpx.ir.models import Document, RegionPlacement
from scan2hwpx.preprocess import preprocess_for_ocr
from scan2hwpx.vision.layout_dataset import TableGrid, detect_table_grids

from .fidelity import HC, HP, OPF, render_fidelity_hwpx
from .render import MIMETYPE

HH = "http://www.hancom.co.kr/hwpml/2011/head"
# Objects are anchored to the paper, so source pixels map onto the whole A4 sheet.
PAGE_WIDTH_HWP = 59_527
PAGE_HEIGHT_HWP = 84_189
LEFT_PARA_ID = "11"
REGULAR_CHAR_ID = "2"
BODY_FONT_ID = "1"  # 함초롬바탕 in the template's font lists; exam body text is serif
EDITABLE_TEXT_CONFIDENCE = 0.75
FLOW_TEXT_HEIGHT = 800
FLOW_LINE_PITCH = 1080
TEXT_BOX_MARGIN_X = 120
TEXT_BOX_MARGIN_Y = 80
TABLE_CELL_MARGIN = 141
# Hancom 2024 advances the template's body font by about 0.82 em per character
# of Korean exam text including spaces (measured from its PDF output at 8 pt).
HANCOM_ADVANCE_PER_EM = 0.82
SPACE_ADVANCE_PER_EM = 0.3  # 함초롬바탕/돋움 space width
# Hancom lays paragraph margins, spacing and FIXED line spacing out at half the
# stored value (measured from its PDF output), so those lengths are stored doubled.
PARA_LENGTH_SCALE = 2


@dataclass(frozen=True)
class SemanticRenderStats:
    pages: int
    background_pictures: int
    editable_tables: int
    extracted_pictures: int
    editable_text_boxes: int
    editable_text_characters: int
    package_bytes: int


@dataclass
class _Row:
    """One visual line of source text.

    segments are (source gap before it in pixels, text, bold) runs, and
    em_width is the pixels per em of its text in 함초롬 fonts, if measurable.
    """

    segments: list[tuple[float, str, bool]]
    x0: float
    y0: float
    x1: float
    y1: float
    em_width: float | None


@dataclass(frozen=True)
class _ImageRegion:
    bbox: tuple[float, float, float, float]
    source_width: float
    source_height: float


@dataclass(frozen=True)
class _PageRegions:
    images: list[_ImageRegion]
    table_seeds: list[_ImageRegion]
    passages: list[_ImageRegion]
    grids: list[TableGrid]  # ruled tables rebuilt as editable tables
    tables: list[_ImageRegion]  # their areas, whose text lives in the table cells


def render_semantic_hwpx(
    page_images: list[Path],
    document: Document,
    layout_seed: Path,
    output: Path,
    *,
    include_page_backgrounds: bool = True,
) -> SemanticRenderStats:
    """Place editable objects at source coordinates, optionally over page backgrounds."""
    if len(page_images) != len(document.pages):
        raise ValueError("page image count and document page count differ")
    layout_pages = _read_layout_pages(layout_seed, len(page_images))
    page_regions = [
        _page_regions(page_path, ir_page, layout_page)
        for page_path, ir_page, layout_page in zip(
            page_images, document.pages, layout_pages, strict=True
        )
    ]
    temporary = output.with_name(f".{output.stem}.fidelity{output.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="scan2hwpx-semantic-") as temporary_directory:
            clean_pages = _build_editable_backgrounds(
                page_images,
                document,
                page_regions,
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
    book = _StyleBook(header)
    advances: list[float] = []
    pitches: list[float] = []
    for ir_page in document.pages:
        page_advances, page_pitches = _flow_samples(ir_page.blocks, ir_page.width, ir_page.height)
        advances.extend(page_advances)
        pitches.extend(page_pitches)
    document_metrics = _flow_metrics(advances, pitches, min_samples=10) or (
        FLOW_TEXT_HEIGHT,
        FLOW_LINE_PITCH,
    )
    paragraphs = cast(list[etree._Element], section.xpath("./hp:p", namespaces={"hp": HP}))
    background_picture = cast(
        etree._Element,
        section.xpath(".//hp:pic", namespaces={"hp": HP})[0],
    )
    if not include_page_backgrounds:
        _remove_page_backgrounds(members, manifest, paragraphs)
    editable_tables = 0
    extracted_pictures = 0
    editable_text_boxes = 0
    editable_text_characters = 0
    binary_assets: dict[str, bytes] = {}
    for page_index, (page_path, ir_page, regions) in enumerate(
        zip(page_images, document.pages, page_regions, strict=True), start=1
    ):
        with Image.open(page_path) as opened:
            page_width, page_height = opened.size
        scaled_images = regions.images
        page_binary = _binarize_page(page_path) if regions.grids else None
        for table_index, grid in enumerate(regions.grids, start=1):
            table = _build_table(
                table_template,
                grid,
                _detect_column_spans(page_binary, grid),
                ir_page.blocks,
                page_width,
                page_height,
                book,
                document_metrics,
                control_id=1_300_000_000 + page_index * 100 + table_index,
                z_order=100 + editable_tables,
            )
            _append_control(paragraphs[page_index - 1], table)
            editable_tables += 1
        passage_groups = _passage_groups(
            ir_page.blocks,
            regions.passages,
            scaled_images,
            regions.table_seeds,
            regions.tables,
            page_width,
            page_height,
        )
        for passage_index, (region, passage_blocks) in enumerate(passage_groups, start=1):
            passage = _build_passage_outline(
                region.bbox,
                passage_blocks,
                page_width,
                page_height,
                book,
                document_metrics,
                control_id=1_450_000_000 + page_index * 100 + passage_index,
                z_order=500 + editable_text_boxes,
            )
            _append_control(paragraphs[page_index - 1], passage)
            editable_text_boxes += 1
        clean_page: Image.Image | None = None
        if scaled_images:
            with Image.open(page_path) as opened:
                source_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            clean_page = Image.fromarray(preprocess_for_ocr(source_rgb).image)
        for image_index, region in enumerate(scaled_images, start=1):
            asset_id = f"semantic_image_{page_index}_{image_index}"
            asset_name = f"BinData/{asset_id}.jpg"
            if clean_page is None:
                raise RuntimeError("clean image page is unavailable")
            crop_bytes, crop_size = _crop_image(clean_page, region.bbox)
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
        if clean_page is not None:
            clean_page.close()
        editable_blocks = [
            block
            for block in ir_page.blocks
            if _eligible_editable_block(block, scaled_images, regions.tables)
        ]
        for group_index, (bbox, flow_blocks) in enumerate(
            _column_flow_groups(ir_page, editable_blocks, page_width, page_height), start=1
        ):
            box_metrics = _flow_metrics(
                *_flow_samples(flow_blocks, page_width, page_height), min_samples=5
            )
            text_box = _build_text_box(
                flow_blocks,
                bbox,
                page_width,
                page_height,
                book=book,
                metrics=box_metrics or document_metrics,
                control_id=1_500_000_000 + page_index * 100 + group_index,
                z_order=1_000 + editable_text_boxes,
            )
            _append_control(paragraphs[page_index - 1], text_box)
            editable_text_boxes += 1
            editable_text_characters += sum(len(str(block.text)) for block in flow_blocks)

    members["Contents/section0.xml"] = _xml_bytes(section)
    members["Contents/header.xml"] = _xml_bytes(header)
    members["Contents/content.hpf"] = _xml_bytes(manifest)
    members.update(binary_assets)
    _write_package(output, members)
    return SemanticRenderStats(
        pages=len(page_images),
        background_pictures=len(page_images) if include_page_backgrounds else 0,
        editable_tables=editable_tables,
        extracted_pictures=extracted_pictures,
        editable_text_boxes=editable_text_boxes,
        editable_text_characters=editable_text_characters,
        package_bytes=output.stat().st_size,
    )


def _page_regions(page_path: Path, ir_page: Any, layout_page: dict[str, Any]) -> _PageRegions:
    with Image.open(page_path) as opened:
        width, height = (float(value) for value in opened.size)

    def scaled(placement: RegionPlacement, fallback: Any) -> list[_ImageRegion]:
        source = _ir_regions(ir_page, placement, width, height)
        if source is None:
            source = fallback(layout_page)
        return [_scale_region(item, width, height) for item in source]

    images = scaled(RegionPlacement.IMAGE, _image_regions)
    table_seeds = scaled(RegionPlacement.TABLE, _table_regions)
    passages = scaled(RegionPlacement.PASSAGE_BOX, _passage_regions)
    grids = []
    for grid in detect_table_grids(page_path):
        # Low-confidence ruled regions arrive as passage boxes; one that holds a
        # ruled grid of at least 2x2 cells, none of them splitting a text line,
        # is a table all the same.
        seeds = table_seeds
        if not _grid_matches_seed(grid, seeds):
            if grid.rows < 2 or grid.columns < 2 or _grid_cuts_text(grid, ir_page.blocks):
                continue
            seeds = [*table_seeds, *passages]
        if not _grid_matches_seed(grid, seeds):
            continue
        aligned = _align_grid_to_seed(grid, seeds)
        if not any(_intersection_ratio(aligned.bbox, image.bbox) >= 0.45 for image in images):
            grids.append(aligned)
    return _PageRegions(
        images=images,
        table_seeds=table_seeds,
        passages=passages,
        grids=grids,
        tables=[_ImageRegion(grid.bbox, width, height) for grid in grids],
    )


def _grid_cuts_text(grid: TableGrid, blocks: list[Any]) -> bool:
    x0, y0, x1, y1 = grid.bbox
    for block in blocks:
        bx0, by0, bx1, by1 = (float(value) for value in block.bbox.pixel)
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        inset = (by1 - by0) * 0.25
        if any(by0 + inset < line < by1 - inset for line in grid.y_lines[1:-1]):
            return True
    return False


def _read_layout_pages(path: Path, expected_pages: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = cast(list[dict[str, Any]], payload.get("pages", []))
    if len(pages) != expected_pages:
        raise ValueError("layout seed page count differs from source")
    return pages


def _read_package(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _remove_page_backgrounds(
    members: dict[str, bytes],
    manifest: etree._Element,
    paragraphs: list[etree._Element],
) -> None:
    for index, paragraph in enumerate(paragraphs, start=1):
        for picture in paragraph.xpath(".//hp:pic", namespaces={"hp": HP}):
            parent = picture.getparent()
            if parent is not None:
                parent.remove(picture)
        members.pop(f"BinData/image{index}.jpg", None)
        for item in manifest.xpath(f".//opf:item[@id='image{index}']", namespaces={"opf": OPF}):
            parent = item.getparent()
            if parent is not None:
                parent.remove(item)


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


def _flow_samples(
    blocks: list[Any], page_width: float, page_height: float
) -> tuple[list[float], list[float]]:
    """Per-character advances and line pitches of the source text, in HWPUNIT."""
    advances: list[float] = []
    pitches: list[float] = []
    x_scale = PAGE_WIDTH_HWP / page_width
    y_scale = PAGE_HEIGHT_HWP / page_height
    columns: dict[object, list[Any]] = {}
    for block in blocks:
        text = str(block.text).strip()
        x0, _y0, x1, _y1 = (float(value) for value in block.bbox.pixel)
        if len(text) > 20:
            advances.append((x1 - x0) / len(text) * x_scale)
        style = block.style if isinstance(block.style, dict) else {}
        columns.setdefault(style.get("layout_column"), []).append(block)
    for column_blocks in columns.values():
        column_blocks.sort(key=lambda block: float(block.bbox.pixel[1]))
        for upper, lower in pairwise(column_blocks):
            height = float(upper.bbox.pixel[3]) - float(upper.bbox.pixel[1])
            step = float(lower.bbox.pixel[1]) - float(upper.bbox.pixel[1])
            if height * 0.8 < step < height * 1.8:
                pitches.append(step * y_scale)
    return advances, pitches


def _flow_metrics(
    advances: list[float], pitches: list[float], *, min_samples: int
) -> tuple[int, int] | None:
    """Font height and line pitch (both HWPUNIT) that reproduce the source text.

    Text density differs between pages and columns, so callers measure per box.
    """
    if len(advances) < min_samples or len(pitches) < min_samples:
        return None
    font_height = float(np.median(advances)) / HANCOM_ADVANCE_PER_EM
    font_height = round(min(1200.0, max(600.0, font_height)) / 10) * 10
    pitch = round(float(np.median(pitches)) / 10) * 10
    return int(font_height), int(min(font_height * 2.5, max(font_height, pitch)))


class _StyleBook:
    """Character and paragraph styles, added to the header on first use."""

    def __init__(self, header: etree._Element) -> None:
        char_containers = cast(
            list[etree._Element],
            header.xpath(".//hh:charProperties", namespaces={"hh": HH}),
        )
        para_containers = cast(
            list[etree._Element],
            header.xpath(".//hh:paraProperties", namespaces={"hh": HH}),
        )
        if len(char_containers) != 1 or len(para_containers) != 1:
            raise RuntimeError("HWPX text style lists are invalid")
        self._chars, self._paras = char_containers[0], para_containers[0]
        char_sources = cast(
            list[etree._Element],
            self._chars.xpath(f"./hh:charPr[@id='{REGULAR_CHAR_ID}']", namespaces={"hh": HH}),
        )
        para_sources = cast(
            list[etree._Element],
            self._paras.xpath(f"./hh:paraPr[@id='{LEFT_PARA_ID}']", namespaces={"hh": HH}),
        )
        if len(char_sources) != 1 or len(para_sources) != 1:
            raise RuntimeError("regular HWPX text styles are missing")
        self._char_source, self._para_source = char_sources[0], para_sources[0]
        self._char_ids: dict[tuple[int, bool], str] = {}
        self._para_ids: dict[tuple[str, int, int, int, int], str] = {}

    def char(self, height: int, bold: bool) -> str:
        key = (height, bold)
        if key not in self._char_ids:
            style = copy.deepcopy(self._char_source)
            style.set("height", str(height))
            style.set("textColor", "#000000")
            font_ref = cast(etree._Element, style.xpath("./hh:fontRef", namespaces={"hh": HH})[0])
            for language in font_ref.attrib:
                font_ref.set(language, BODY_FONT_ID)
            for existing_bold in style.xpath("./hh:bold", namespaces={"hh": HH}):
                style.remove(existing_bold)
            if bold:
                etree.SubElement(style, f"{{{HH}}}bold")
            self._char_ids[key] = _append_style(self._chars, style)
        return self._char_ids[key]

    def line(
        self, *, pitch: int, align: str = "LEFT", left: int = 0, right: int = 0, prev: int = 0
    ) -> str:
        """A paragraph holding one source line at a fixed pitch (all lengths HWPUNIT).

        Hancom squeezes the line rather than wrapping it, so a font that runs a
        little wider than the scan can never push the following lines down.
        """
        key = (align, left, right, prev, pitch)
        if key not in self._para_ids:
            style = copy.deepcopy(self._para_source)
            style.set("snapToGrid", "0")
            align_element = cast(
                etree._Element, style.xpath("./hh:align", namespaces={"hh": HH})[0]
            )
            align_element.set("horizontal", align)
            for setting in style.xpath(".//hh:breakSetting", namespaces={"hh": HH}):
                setting.set("lineWrap", "SQUEEZE")
            for spacing in style.xpath(".//hh:lineSpacing", namespaces={"hh": HH}):
                spacing.set("type", "FIXED")
                spacing.set("value", str(pitch * PARA_LENGTH_SCALE))
            for name, value in (("left", left), ("right", right), ("prev", prev)):
                for margin in style.xpath(
                    f".//hh:margin/hc:{name}", namespaces={"hh": HH, "hc": HC}
                ):
                    margin.set("value", str(value * PARA_LENGTH_SCALE))
            self._para_ids[key] = _append_style(self._paras, style)
        return self._para_ids[key]


def _append_style(container: etree._Element, style: etree._Element) -> str:
    style_id = str(max(int(item.get("id", "-1")) for item in container) + 1)
    style.set("id", style_id)
    container.append(style)
    container.set("itemCnt", str(len(container)))
    return style_id


def _ocr_bold(block: Any) -> bool:
    kind = getattr(block.kind, "value", str(block.kind))
    return kind in {"instruction", "question", "title"}


def _build_editable_backgrounds(
    page_images: list[Path],
    document: Document,
    page_regions: list[_PageRegions],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    for page_index, (page_path, ir_page, regions) in enumerate(
        zip(page_images, document.pages, page_regions, strict=True), start=1
    ):
        with Image.open(page_path) as opened:
            source_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        image = Image.fromarray(preprocess_for_ocr(source_rgb).image)
        page_width, page_height = image.size
        array = np.asarray(image, dtype=np.uint8)
        draw = ImageDraw.Draw(image)
        for block in ir_page.blocks:
            if not _eligible_editable_block(block, regions.images, regions.tables):
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
    if getattr(block.annotation_state, "value", str(block.annotation_state)) != "printed":
        return False
    raw_bbox = block.bbox.pixel
    bbox = (
        float(raw_bbox[0]),
        float(raw_bbox[1]),
        float(raw_bbox[2]),
        float(raw_bbox[3]),
    )
    kind = getattr(block.kind, "value", str(block.kind))
    if kind == "header" and bbox[3] - bbox[1] > (bbox[2] - bbox[0]) * 1.5:
        return False
    if kind == "footer" and "저작권" in text:
        return False
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
    blocks: list[Any],
    bbox: tuple[float, float, float, float],
    page_width: int,
    page_height: int,
    *,
    book: _StyleBook,
    metrics: tuple[int, int],
    control_id: int,
    z_order: int,
) -> etree._Element:
    x0, y0, x1, y1 = bbox
    width = _x_hwp(max(2.0, x1 - x0), page_width)
    height = _y_hwp(max(2.0, y1 - y0), page_height)
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
        lineWrap="BREAK",
        vertAlign="TOP",
        linkListIDRef="0",
        linkListNextIDRef="0",
        textWidth=str(width),
        textHeight=str(height),
        hasTextRef="0",
        hasNumRef="0",
    )
    _write_line_paragraphs(sublist, blocks, bbox, page_width, page_height, book, metrics)
    etree.SubElement(
        draw_text,
        f"{{{HP}}}textMargin",
        left=str(TEXT_BOX_MARGIN_X),
        right=str(TEXT_BOX_MARGIN_X),
        top=str(TEXT_BOX_MARGIN_Y),
        bottom=str(TEXT_BOX_MARGIN_Y),
    )
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
    comment.text = "OCR 연속 편집 영역"
    return rectangle


def _build_passage_outline(
    bbox: tuple[float, float, float, float],
    blocks: list[Any],
    page_width: int,
    page_height: int,
    book: _StyleBook,
    metrics: tuple[int, int],
    *,
    control_id: int,
    z_order: int,
) -> etree._Element:
    ordered = sorted(
        blocks,
        key=lambda block: (float(block.bbox.pixel[1]), float(block.bbox.pixel[0])),
    )
    anchor_pixel = ordered[0].bbox.pixel
    rectangle = _build_text_box(
        [ordered[0]],
        (
            float(anchor_pixel[0]),
            float(anchor_pixel[1]),
            float(anchor_pixel[2]),
            float(anchor_pixel[3]),
        ),
        page_width,
        page_height,
        book=book,
        metrics=metrics,
        control_id=control_id,
        z_order=z_order,
    )
    x0, y0, x1, y1 = bbox
    width = _x_hwp(max(2.0, x1 - x0), page_width)
    height = _y_hwp(max(2.0, y1 - y0), page_height)
    for name in ("orgSz", "curSz"):
        size = cast(
            etree._Element,
            rectangle.xpath(f"./hp:{name}", namespaces={"hp": HP})[0],
        )
        size.set("width", str(width))
        size.set("height", str(height))
    rotation = cast(
        etree._Element,
        rectangle.xpath("./hp:rotationInfo", namespaces={"hp": HP})[0],
    )
    rotation.set("centerX", str(width // 2))
    rotation.set("centerY", str(height // 2))
    line = cast(
        etree._Element,
        rectangle.xpath("./hp:lineShape", namespaces={"hp": HP})[0],
    )
    line.set("color", "#5A5A5A")
    line.set("width", "100")
    line.set("style", "SOLID")
    sublist = cast(
        etree._Element,
        rectangle.xpath("./hp:drawText/hp:subList", namespaces={"hp": HP})[0],
    )
    sublist.set("lineWrap", "BREAK")
    sublist.set("vertAlign", "TOP")
    sublist.set("textWidth", str(max(1, width - 700)))
    sublist.set("textHeight", str(max(1, height - 500)))
    texts = cast(
        list[etree._Element],
        sublist.xpath(".//hp:t", namespaces={"hp": HP}),
    )
    for text in texts:
        text.text = ""
    margin = cast(
        etree._Element,
        rectangle.xpath("./hp:drawText/hp:textMargin", namespaces={"hp": HP})[0],
    )
    for name, value in {
        "left": "350",
        "right": "350",
        "top": "250",
        "bottom": "250",
    }.items():
        margin.set(name, value)
    for name, point_x, point_y in (
        ("pt0", 0, 0),
        ("pt1", width, 0),
        ("pt2", width, height),
        ("pt3", 0, height),
    ):
        point = cast(
            etree._Element,
            rectangle.xpath(f"./hc:{name}", namespaces={"hc": HC})[0],
        )
        point.set("x", str(point_x))
        point.set("y", str(point_y))
    shape_size = cast(
        etree._Element,
        rectangle.xpath("./hp:sz", namespaces={"hp": HP})[0],
    )
    shape_size.set("width", str(width))
    shape_size.set("height", str(height))
    position = cast(
        etree._Element,
        rectangle.xpath("./hp:pos", namespaces={"hp": HP})[0],
    )
    position.set("horzOffset", str(_x_hwp(x0, page_width)))
    position.set("vertOffset", str(_y_hwp(y0, page_height)))
    comments = rectangle.xpath("./hp:shapeComment", namespaces={"hp": HP})
    if comments:
        comments[0].text = "OCR 작품/보기 테두리"
    return rectangle


def _column_flow_groups(
    page: Any,
    blocks: list[Any],
    page_width: int,
    page_height: int,
) -> list[tuple[tuple[float, float, float, float], list[Any]]]:
    if not blocks:
        return []
    columns = sorted(page.columns, key=lambda column: int(column.index))
    assigned: set[str] = set()
    groups: list[tuple[tuple[float, float, float, float], list[Any]]] = []
    body_bottom = page_height * 0.92
    for column in columns:
        cx0, _cy0, cx1, _cy1 = (float(value) for value in column.bbox.pixel)
        selected = []
        for block in blocks:
            bx0, by0, bx1, _by1 = (float(value) for value in block.bbox.pixel)
            style = block.style if isinstance(block.style, dict) else {}
            in_column = style.get("layout_column") == int(column.index)
            if len(columns) == 1:
                in_column = by0 < body_bottom
            elif style.get("layout_column") is None:
                in_column = cx0 <= bx0 and bx1 <= cx1 and by0 < body_bottom
            if in_column:
                selected.append(block)
        if not selected:
            continue
        assigned.update(str(block.id) for block in selected)
        left = max(
            0.0,
            min(float(block.bbox.pixel[0]) for block in selected)
            - TEXT_BOX_MARGIN_X * page_width / PAGE_WIDTH_HWP,
        )
        right = min(
            cx1 - page_width * 0.012,
            max(
                max(float(block.bbox.pixel[2]) for block in selected),
                cx1 - page_width * 0.03,
            ),
        )
        top = max(
            0.0,
            min(float(block.bbox.pixel[1]) for block in selected)
            - TEXT_BOX_MARGIN_Y * page_height / PAGE_HEIGHT_HWP,
        )
        bottom = max(
            max(float(block.bbox.pixel[3]) for block in selected) + page_height * 0.01,
            page_height * 0.925,
        )
        groups.append(((left, top, right, min(page_height * 0.95, bottom)), selected))

    for block in blocks:
        if str(block.id) in assigned:
            continue
        x0, y0, x1, y1 = (float(value) for value in block.bbox.pixel)
        groups.append(
            (
                (
                    max(0.0, x0 - 4.0),
                    max(0.0, y0 - 2.0),
                    min(float(page_width), x1 + 8.0),
                    min(float(page_height), y1 + 4.0),
                ),
                [block],
            )
        )
    return groups


def _write_line_paragraphs(
    sublist: etree._Element,
    blocks: list[Any],
    bbox: tuple[float, float, float, float],
    page_width: int,
    page_height: int,
    book: _StyleBook,
    metrics: tuple[int, int],
) -> None:
    """One paragraph per source line, each placed at its source indent and height.

    A line advances to the next source line by its own fixed pitch, and a
    paragraph's spacing-before absorbs larger gaps. Both are measured from where
    Hancom put the previous line, so rounding errors never accumulate.
    """
    font_height, pitch = metrics
    x_scale = PAGE_WIDTH_HWP / page_width
    y_scale = PAGE_HEIGHT_HWP / page_height
    rows = _source_rows(blocks)
    typical_height = float(np.median([row.y1 - row.y0 for row in rows]))
    em_widths = [row.em_width for row in rows if row.em_width is not None]
    typical_em = float(np.median(em_widths)) if em_widths else None
    box_width = bbox[2] - bbox[0]
    wide = [row.x1 for row in rows if row.x1 - row.x0 >= box_width * 0.6]
    text_right = float(np.percentile(wide, 90)) if wide else float("inf")
    targets = [round((row.y0 - bbox[1]) * y_scale) - TEXT_BOX_MARGIN_Y for row in rows]
    cursor = 0
    for index, row in enumerate(rows):
        height = font_height
        if row.em_width is not None and typical_em is not None:
            # Titles and small print keep their relative size. OCR boxes widened by
            # pen marks are not taller, so width alone does not resize a line.
            ratio = row.em_width / typical_em
            taller = (row.y1 - row.y0) / max(1.0, typical_height)
            if (ratio > 1.15 and taller >= 1.1) or (ratio < 0.85 and taller <= 0.9):
                height = round(font_height * min(2.5, max(0.6, ratio)) / 50) * 50
        prev = max(0, round((targets[index] - cursor) / 50) * 50)
        top = cursor + prev
        line_pitch = pitch
        if index + 1 < len(rows):
            step = targets[index + 1] - top
            if step <= pitch * 1.5:
                line_pitch = max(round(height * 0.5), round(step / 50) * 50)
        cursor = top + line_pitch
        # A single run that reaches the right edge is justified text in the source.
        justified = (
            len(row.segments) == 1
            and row.x1 - row.x0 >= box_width * 0.6
            and row.x1 >= text_right - typical_height * 0.5
        )
        left = round(((row.x0 - bbox[0]) * x_scale - TEXT_BOX_MARGIN_X) / 100) * 100
        right = round(((bbox[2] - row.x1) * x_scale - TEXT_BOX_MARGIN_X) / 100) * 100
        paragraph = etree.SubElement(
            sublist,
            f"{{{HP}}}p",
            id="0",
            paraPrIDRef=book.line(
                pitch=line_pitch,
                align="DISTRIBUTE_SPACE" if justified else "LEFT",
                left=max(0, left),
                right=max(0, right) if justified else 0,
                prev=prev,
            ),
            styleIDRef="0",
            pageBreak="0",
            columnBreak="0",
            merged="0",
        )
        _append_runs(paragraph, row.segments, book, height, SPACE_ADVANCE_PER_EM * height / x_scale)


def _append_runs(
    paragraph: etree._Element,
    segments: list[tuple[float, str, bool]],
    book: _StyleBook,
    height: int,
    space_width: float,
    *,
    bold: bool = False,
) -> None:
    """Write a line's runs, filling each source gap with spaces of space_width px."""
    for index, (gap, text_value, segment_bold) in enumerate(segments):
        if index:
            text_value = " " * max(1, round(gap / max(1.0, space_width))) + text_value
        run = etree.SubElement(
            paragraph, f"{{{HP}}}run", charPrIDRef=book.char(height, bold or segment_bold)
        )
        if not text_value:
            continue
        text = etree.SubElement(run, f"{{{HP}}}t")
        text.text = text_value
        if text_value != text_value.strip():
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def _source_rows(blocks: list[Any]) -> list[_Row]:
    """Group OCR blocks into visual lines, top to bottom and left to right."""
    ordered = sorted(
        blocks,
        key=lambda block: (float(block.bbox.pixel[1]) + float(block.bbox.pixel[3])) / 2,
    )
    heights = [
        max(1.0, float(block.bbox.pixel[3]) - float(block.bbox.pixel[1])) for block in ordered
    ]
    typical_height = float(np.median(heights)) if heights else 24.0
    lines: list[list[Any]] = []
    for block in ordered:
        center = (float(block.bbox.pixel[1]) + float(block.bbox.pixel[3])) / 2
        if lines:
            line_center = float(
                np.mean([(float(b.bbox.pixel[1]) + float(b.bbox.pixel[3])) / 2 for b in lines[-1]])
            )
            if abs(center - line_center) <= typical_height * 0.45:
                lines[-1].append(block)
                continue
        lines.append([block])
    rows: list[_Row] = []
    for line in lines:
        line.sort(key=lambda block: float(block.bbox.pixel[0]))
        segments: list[tuple[float, str, bool]] = []
        previous_x1: float | None = None
        ink = 0.0
        ems = 0.0
        for block in line:
            text = str(block.text).strip()
            if not text:
                continue
            x0, _y0, x1, _y1 = (float(value) for value in block.bbox.pixel)
            gap = 0.0 if previous_x1 is None else x0 - previous_x1
            segments.append((gap, text, _ocr_bold(block)))
            previous_x1 = x1
            ink += x1 - x0
            ems += _em_width(text)
        if segments:
            rows.append(
                _Row(
                    segments=segments,
                    x0=min(float(block.bbox.pixel[0]) for block in line),
                    y0=min(float(block.bbox.pixel[1]) for block in line),
                    x1=max(float(block.bbox.pixel[2]) for block in line),
                    y1=max(float(block.bbox.pixel[3]) for block in line),
                    em_width=ink / ems if ems >= 5 else None,
                )
            )
    return rows


def _em_width(text: str) -> float:
    """Advance of text in 함초롬바탕/돋움, in em (measured from the font files)."""
    width = 0.0
    for char in text:
        if not char.isascii():
            width += 0.97
        elif char == " ":
            width += SPACE_ADVANCE_PER_EM
        elif char in "[](),.:;!'\"":
            width += 0.32
        elif char.isupper():
            width += 0.7
        elif char.islower():
            width += 0.52
        else:  # digits and the remaining symbols
            width += 0.55
    return width


def _build_table(
    template: etree._Element,
    grid: TableGrid,
    cell_layout: list[list[tuple[int, int]]],
    blocks: list[Any],
    page_width: int,
    page_height: int,
    book: _StyleBook,
    metrics: tuple[int, int],
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
    # The template's 5 mm side margins squeeze narrow label columns.
    in_margin = cast(etree._Element, table.xpath("./hp:inMargin", namespaces={"hp": HP})[0])
    for side in ("left", "right", "top", "bottom"):
        in_margin.set(side, str(TABLE_CELL_MARGIN))
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
    position.set("vertRelTo", "PAPER")
    position.set("horzRelTo", "PAPER")
    position.set("vertAlign", "TOP")
    position.set("horzAlign", "LEFT")
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
            cell_height = _y_hwp(y_lines[row_index + 1] - y_lines[row_index], page_height)
            cell_size.set("height", str(cell_height))
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
                cell_height=cell_height,
                x_scale=PAGE_WIDTH_HWP / page_width,
                book=book,
                metrics=metrics,
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
    cell_height: int,
    x_scale: float,
    book: _StyleBook,
    metrics: tuple[int, int],
) -> None:
    """Write a cell's source lines so they fit its source height; Hancom grows
    a row whose text does not fit, which pushes the table over what follows."""
    sublist = cast(
        etree._Element,
        cell.xpath("./hp:subList", namespaces={"hp": HP})[0],
    )
    for paragraph in sublist.xpath("./hp:p", namespaces={"hp": HP}):
        sublist.remove(paragraph)
    rows = _source_rows(lines)
    font_height, pitch = metrics
    pitch = min(pitch, (cell_height - 2 * TABLE_CELL_MARGIN) // max(1, len(rows)))
    pitch = max(100, pitch // 10 * 10)
    height = max(500, min(font_height, round(pitch * 0.9 / 50) * 50))
    paragraph_style = book.line(pitch=pitch, align="CENTER" if centered else "LEFT")
    for segments in [row.segments for row in rows] or [[(0.0, "", False)]]:
        paragraph = etree.SubElement(sublist, f"{{{HP}}}p")
        paragraph.set("id", "0")
        paragraph.set("paraPrIDRef", paragraph_style)
        paragraph.set("styleIDRef", "0")
        paragraph.set("pageBreak", "0")
        paragraph.set("columnBreak", "0")
        paragraph.set("merged", "0")
        _append_runs(
            paragraph, segments, book, height, SPACE_ADVANCE_PER_EM * height / x_scale, bold=bold
        )


def _ir_regions(
    ir_page: Any, placement: RegionPlacement, width: float, height: float
) -> list[_ImageRegion] | None:
    """Blueprint-classified regions for one page, scaled to (width, height).

    Returns None when the page carries no blueprint regions (legacy IR, or
    the rule-based Model B stand-in was not run), so callers fall back to
    parsing layout_seed.json directly, unchanged from before this existed.
    """
    regions = getattr(ir_page, "regions", None)
    if not regions:
        return None
    return [
        _ImageRegion(
            bbox=(
                region.bbox.normalized[0] * width,
                region.bbox.normalized[1] * height,
                region.bbox.normalized[2] * width,
                region.bbox.normalized[3] * height,
            ),
            source_width=width,
            source_height=height,
        )
        for region in regions
        if region.placement == placement
    ]


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
        confidence = float(annotation.get("confidence", 0))
        if annotation.get("label") != "table" or confidence < 0.45:
            continue
        if annotation.get("source") == "opencv_ruled_region" and confidence < 0.7:
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


def _passage_regions(page: dict[str, Any]) -> list[_ImageRegion]:
    width = float(page["width"])
    height = float(page["height"])
    regions: list[_ImageRegion] = []
    for annotation in page.get("annotations", []):
        confidence = float(annotation.get("confidence", 0))
        fallback_box = (
            annotation.get("label") == "table"
            and annotation.get("source") == "opencv_ruled_region"
            and confidence < 0.7
        )
        if (annotation.get("label") != "passage_box" and not fallback_box) or confidence < 0.45:
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


def _passage_groups(
    blocks: list[Any],
    regions: list[_ImageRegion],
    image_regions: list[_ImageRegion],
    table_regions: list[_ImageRegion],
    editable_tables: list[_ImageRegion],
    page_width: int,
    page_height: int,
) -> list[tuple[_ImageRegion, list[Any]]]:
    candidates = [
        region
        for region in regions
        if region.bbox[2] - region.bbox[0] >= page_width * 0.22
        and region.bbox[2] - region.bbox[0] <= page_width * 0.52
        and region.bbox[3] - region.bbox[1] >= page_height * 0.045
        and region.bbox[1] >= page_height * 0.09
        and region.bbox[3] <= page_height * 0.95
        and not any(
            _intersection_ratio(region.bbox, table.bbox) >= 0.5
            for table in [*table_regions, *editable_tables]
        )
    ]
    candidates.sort(
        key=lambda region: (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]),
        reverse=True,
    )
    deduplicated: list[_ImageRegion] = []
    for region in candidates:
        if any(_intersection_ratio(region.bbox, kept.bbox) >= 0.65 for kept in deduplicated):
            continue
        deduplicated.append(region)

    eligible = [
        block for block in blocks if _eligible_editable_block(block, image_regions, editable_tables)
    ]

    def cuts_text(region: _ImageRegion) -> bool:
        x0, y0, x1, y1 = region.bbox
        for block in eligible:
            bx0, by0, bx1, by1 = (float(value) for value in block.bbox.pixel)
            inset = (by1 - by0) * 0.25
            if bx0 < x1 and bx1 > x0 and any(by0 + inset < edge < by1 - inset for edge in (y0, y1)):
                return True
        return False

    deduplicated = [region for region in deduplicated if not cuts_text(region)]
    assigned: set[str] = set()
    result: list[tuple[_ImageRegion, list[Any]]] = []
    for region in deduplicated:
        x0, y0, x1, y1 = region.bbox
        selected = []
        for block in eligible:
            if str(block.id) in assigned:
                continue
            bx0, by0, bx1, by1 = (float(value) for value in block.bbox.pixel)
            center_x = (bx0 + bx1) / 2
            center_y = (by0 + by1) / 2
            if x0 - 8 <= center_x <= x1 + 8 and y0 - 8 <= center_y <= y1 + 8:
                selected.append(block)
        headings = [
            block
            for block in eligible
            if str(block.id) not in assigned
            and re.fullmatch(r"-?\s*<보기>\s*", str(block.text).strip()) is not None
            and x0 <= (float(block.bbox.pixel[0]) + float(block.bbox.pixel[2])) / 2 <= x1
            and 0 <= y0 - float(block.bbox.pixel[3]) <= page_height * 0.06
        ]
        selected.extend(block for block in headings if block not in selected)
        if not selected or sum(len(str(block.text).strip()) for block in selected) < 8:
            continue
        adjusted_y0 = min([y0, *(float(block.bbox.pixel[1]) - 8 for block in headings)])
        adjusted = _ImageRegion(
            bbox=(x0, max(0.0, adjusted_y0), x1, y1),
            source_width=region.source_width,
            source_height=region.source_height,
        )
        assigned.update(str(block.id) for block in selected)
        result.append((adjusted, selected))
    return result


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
    image: Image.Image, bbox: tuple[float, float, float, float]
) -> tuple[bytes, tuple[int, int]]:
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
    position.set("vertRelTo", "PAPER")
    position.set("horzRelTo", "PAPER")
    position.set("vertAlign", "TOP")
    position.set("horzAlign", "LEFT")
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
