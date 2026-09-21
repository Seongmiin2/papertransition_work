from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal, Self, cast

from lxml import etree  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scan2hwpx.hwpx.validate import validate_hwpx

PROJECTION_SOURCE_VERSION: Literal["hwpx-projection-source/1.0"] = (
    "hwpx-projection-source/1.0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECTION = re.compile(r"^Contents/section(\d+)\.xml$")
_MAX_ENTRY_BYTES = 128 * 1024 * 1024
_MAX_PACKAGE_BYTES = 512 * 1024 * 1024
_MAX_COMPRESSED_PACKAGE_BYTES = 256 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1_000
_MAX_ARCHIVE_ENTRIES = 10_000
_MAX_TABLE_ROWS = 1_000
_MAX_TABLE_COLUMNS = 256
_MAX_TABLE_CELLS = 100_000
_MAX_TABLE_AREA = 100_000
_HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProjectionIssue(_StrictModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    severity: Literal["review", "block"]
    locator: str | None = None


class ProjectedStyle(_StrictModel):
    id: str = Field(min_length=1)
    source_para_pr_id: str | None = None
    source_char_pr_id: str | None = None
    font_family: str | None = None
    font_size_pt: float | None = Field(default=None, gt=0, le=200)
    bold: bool = False
    italic: bool = False
    alignment: Literal["left", "center", "right", "justify"] = "left"
    line_spacing: float | None = Field(default=None, ge=0.5, le=5)
    raw_alignment: str | None = None
    raw_line_spacing_type: str | None = None
    raw_line_spacing_value: int | None = None
    text_color: str | None = None
    underline_type: str | None = None
    underline_shape: str | None = None
    underline_color: str | None = None
    hangul_spacing: int | None = None
    hangul_ratio: int | None = None
    margin_indent_hwp: int | None = None
    margin_left_hwp: int | None = None
    margin_right_hwp: int | None = None
    margin_previous_hwp: int | None = None
    margin_next_hwp: int | None = None


class ProjectedCell(_StrictModel):
    locator: str = Field(min_length=1)
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str = ""
    style_id: str | None = None
    paragraph_refs: tuple[str, ...] = ()


class ProjectedNode(_StrictModel):
    id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    kind: Literal["text", "table", "image", "formula"]
    text: str = ""
    role_hint: Literal["header", "footer", "other"] | None = None
    style_id: str | None = None
    source_para_pr_id: str | None = None
    source_char_pr_id: str | None = None
    page_break: bool = False
    column_break: bool = False
    rows: int | None = Field(default=None, ge=1)
    columns: int | None = Field(default=None, ge=1)
    cells: tuple[ProjectedCell, ...] = ()
    asset_id: str | None = None
    formula_script: str | None = None
    width_hwp: int | None = Field(default=None, ge=0)
    height_hwp: int | None = Field(default=None, ge=0)
    review_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_kind_payload(self) -> Self:
        if self.kind == "text" and not self.text.strip():
            raise ValueError("projected text node must contain visible text")
        if self.kind == "table":
            if self.rows is None or self.columns is None or not self.cells:
                raise ValueError("projected table requires rows, columns, and cells")
            _validate_cell_topology(self.rows, self.columns, self.cells)
        elif self.cells or self.rows is not None or self.columns is not None:
            raise ValueError("only projected tables may contain table topology")
        if self.kind == "image" and self.asset_id is None:
            raise ValueError("projected image requires an asset id")
        if self.kind == "formula" and not (self.formula_script or "").strip():
            raise ValueError("projected formula requires a source script")
        return self


class ProjectionAsset(_StrictModel):
    id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)


class ProjectedPageLayout(_StrictModel):
    width_hwp: int = Field(gt=0)
    height_hwp: int = Field(gt=0)
    margin_top_hwp: int = Field(ge=0)
    margin_right_hwp: int = Field(ge=0)
    margin_bottom_hwp: int = Field(ge=0)
    margin_left_hwp: int = Field(ge=0)
    margin_header_hwp: int = Field(ge=0)
    margin_footer_hwp: int = Field(ge=0)
    margin_gutter_hwp: int = Field(ge=0)
    columns: int = Field(default=1, ge=1)
    column_gap_hwp: int = Field(default=0, ge=0)


class ProjectedColumnChange(_StrictModel):
    locator: str = Field(min_length=1)
    columns: int = Field(ge=1)
    column_gap_hwp: int = Field(ge=0)


class ProjectedSection(_StrictModel):
    index: int = Field(ge=0)
    layout: ProjectedPageLayout
    column_changes: tuple[ProjectedColumnChange, ...] = ()
    node_refs: tuple[str, ...]


class HwpxProjectionSource(_StrictModel):
    schema_version: Literal["hwpx-projection-source/1.0"] = PROJECTION_SOURCE_VERSION
    source_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hwpx_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(gt=0)
    sections: tuple[ProjectedSection, ...] = Field(min_length=1)
    nodes: tuple[ProjectedNode, ...]
    styles: tuple[ProjectedStyle, ...]
    assets: tuple[ProjectionAsset, ...]
    issues: tuple[ProjectionIssue, ...]
    stats: dict[str, int]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        _reject_duplicates(node_ids, "projected node ids")
        style_ids = [style.id for style in self.styles]
        asset_ids = [asset.id for asset in self.assets]
        _reject_duplicates(style_ids, "projected style ids")
        _reject_duplicates(asset_ids, "projected asset ids")
        unknown_styles = sorted(
            {
                style_id
                for node in self.nodes
                for style_id in (
                    node.style_id,
                    *(cell.style_id for cell in node.cells),
                )
                if style_id is not None and style_id not in set(style_ids)
            }
        )
        if unknown_styles:
            raise ValueError("unknown projected style refs: " + ", ".join(unknown_styles))
        unknown_assets = sorted(
            {
                node.asset_id
                for node in self.nodes
                if node.asset_id is not None and node.asset_id not in set(asset_ids)
            }
        )
        if unknown_assets:
            raise ValueError("unknown projected asset refs: " + ", ".join(unknown_assets))
        section_refs = [ref for section in self.sections for ref in section.node_refs]
        if section_refs != node_ids:
            raise ValueError("section node refs must preserve every projected node exactly once")
        return self


class _StyleResolver:
    def __init__(self, header: etree._Element) -> None:
        self._fonts = _font_faces(header)
        self._char_properties = {
            str(node.get("id")): node
            for node in header.xpath("//*[local-name()='charPr']")
            if node.get("id") is not None
        }
        self._para_properties = {
            str(node.get("id")): node
            for node in header.xpath("//*[local-name()='paraPr']")
            if node.get("id") is not None
        }
        self.styles: dict[str, ProjectedStyle] = {}

    def resolve(self, para_ref: str | None, char_ref: str | None) -> str | None:
        if para_ref is None and char_ref is None:
            return None
        para = self._para_properties.get(str(para_ref))
        char = self._char_properties.get(str(char_ref))
        font_family: str | None = None
        font_size_pt: float | None = None
        bold = False
        italic = False
        text_color: str | None = None
        underline_type: str | None = None
        underline_shape: str | None = None
        underline_color: str | None = None
        hangul_spacing: int | None = None
        hangul_ratio: int | None = None
        if char is not None:
            font_ref = _first(char.xpath("./*[local-name()='fontRef']"))
            if font_ref is not None:
                font_family = self._fonts.get(str(font_ref.get("hangul")))
            raw_height = _optional_int(char.get("height"))
            if raw_height is not None and 0 < raw_height <= 20_000:
                font_size_pt = raw_height / 100
            bold = bool(char.xpath("./*[local-name()='bold']"))
            italic = bool(char.xpath("./*[local-name()='italic']"))
            text_color = char.get("textColor")
            underline = _first(char.xpath("./*[local-name()='underline']"))
            if underline is not None:
                underline_type = underline.get("type")
                underline_shape = underline.get("shape")
                underline_color = underline.get("color")
            spacing = _first(char.xpath("./*[local-name()='spacing']"))
            ratio = _first(char.xpath("./*[local-name()='ratio']"))
            if spacing is not None:
                hangul_spacing = _optional_int(spacing.get("hangul"))
            if ratio is not None:
                hangul_ratio = _optional_int(ratio.get("hangul"))

        raw_alignment: str | None = None
        alignment: Literal["left", "center", "right", "justify"] = "left"
        raw_spacing_type: str | None = None
        raw_spacing_value: int | None = None
        line_spacing: float | None = None
        margin_indent_hwp: int | None = None
        margin_left_hwp: int | None = None
        margin_right_hwp: int | None = None
        margin_previous_hwp: int | None = None
        margin_next_hwp: int | None = None
        if para is not None:
            align = _first(para.xpath("./*[local-name()='align']"))
            if align is not None:
                raw_alignment = align.get("horizontal")
                alignment = _alignment(raw_alignment)
            spacing = _first(para.xpath(".//*[local-name()='lineSpacing']"))
            if spacing is not None:
                raw_spacing_type = spacing.get("type")
                raw_spacing_value = _optional_int(spacing.get("value"))
                if raw_spacing_type == "PERCENT" and raw_spacing_value is not None:
                    candidate = raw_spacing_value / 100
                    if 0.5 <= candidate <= 5:
                        line_spacing = candidate
            margin = _first(para.xpath(".//*[local-name()='margin']"))
            if margin is not None:
                margin_indent_hwp = _named_child_int(margin, "intent")
                margin_left_hwp = _named_child_int(margin, "left")
                margin_right_hwp = _named_child_int(margin, "right")
                margin_previous_hwp = _named_child_int(margin, "prev")
                margin_next_hwp = _named_child_int(margin, "next")

        facts = {
            "source_para_pr_id": para_ref,
            "source_char_pr_id": char_ref,
            "font_family": font_family,
            "font_size_pt": font_size_pt,
            "bold": bold,
            "italic": italic,
            "alignment": alignment,
            "line_spacing": line_spacing,
            "raw_alignment": raw_alignment,
            "raw_line_spacing_type": raw_spacing_type,
            "raw_line_spacing_value": raw_spacing_value,
            "text_color": text_color,
            "underline_type": underline_type,
            "underline_shape": underline_shape,
            "underline_color": underline_color,
            "hangul_spacing": hangul_spacing,
            "hangul_ratio": hangul_ratio,
            "margin_indent_hwp": margin_indent_hwp,
            "margin_left_hwp": margin_left_hwp,
            "margin_right_hwp": margin_right_hwp,
            "margin_previous_hwp": margin_previous_hwp,
            "margin_next_hwp": margin_next_hwp,
        }
        signature_facts = {
            key: value
            for key, value in facts.items()
            if key not in {"source_para_pr_id", "source_char_pr_id"}
        }
        signature = hashlib.sha256(
            json.dumps(
                signature_facts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        style_id = f"style-{signature}"
        self.styles.setdefault(style_id, ProjectedStyle(id=style_id, **facts))
        return style_id


class _ProjectionBuilder:
    def __init__(
        self,
        *,
        source_sha256: str,
        archive: zipfile.ZipFile,
        media_by_id: dict[str, str],
        style_resolver: _StyleResolver,
    ) -> None:
        self.source_sha256 = source_sha256
        self.archive = archive
        self.media_by_id = media_by_id
        self.style_resolver = style_resolver
        self.assets: dict[str, ProjectionAsset] = {}
        self.issues: list[ProjectionIssue] = []
        self.counts: Counter[str] = Counter()

    def parse_section(
        self, root: etree._Element, section_index: int
    ) -> tuple[ProjectedSection, list[ProjectedNode]]:
        section_locator = f"s{section_index:03d}"
        layout = _page_layout(root)
        column_changes = _column_changes(root, section_locator)
        if len({(item.columns, item.column_gap_hwp) for item in column_changes}) > 1:
            self._issue("section_column_layout_changes_flattened", section_locator)
        if layout.margin_header_hwp or layout.margin_footer_hwp or layout.margin_gutter_hwp:
            self._issue("header_footer_gutter_margins_not_representable", section_locator)
        nodes: list[ProjectedNode] = []
        paragraphs = cast(
            list[etree._Element],
            root.xpath("./hp:p", namespaces={"hp": _HP}),
        )
        self.counts["paragraphs"] += len(paragraphs)
        for paragraph_index, paragraph in enumerate(paragraphs):
            locator = f"{section_locator}/p{paragraph_index:06d}"
            created = self._parse_paragraph(paragraph, locator)
            if not created:
                self.counts["empty_paragraphs"] += 1
                self._issue("empty_paragraph_layout_not_projected", locator)
                if _xml_bool(paragraph.get("pageBreak")) or _xml_bool(
                    paragraph.get("columnBreak")
                ):
                    self._issue("break_on_empty_paragraph_not_projected", locator)
            elif len({node.kind for node in created}) > 1:
                self._issue("mixed_content_paragraph_layout_flattened", locator)
            nodes.extend(created)
        return (
            ProjectedSection(
                index=section_index,
                layout=layout,
                column_changes=column_changes,
                node_refs=tuple(node.id for node in nodes),
            ),
            nodes,
        )

    def _parse_paragraph(
        self, paragraph: etree._Element, locator: str
    ) -> list[ProjectedNode]:
        nodes: list[ProjectedNode] = []
        para_ref = paragraph.get("paraPrIDRef")
        page_break = _xml_bool(paragraph.get("pageBreak"))
        column_break = _xml_bool(paragraph.get("columnBreak"))
        first_node = True
        for run_index, run in enumerate(paragraph.xpath("./hp:run", namespaces={"hp": _HP})):
            char_ref = run.get("charPrIDRef")
            style_id = self.style_resolver.resolve(para_ref, char_ref)
            text_index = 0
            object_counts: Counter[str] = Counter()
            for child in run:
                name = _local_name(child)
                child_locator = f"{locator}/r{run_index:03d}/{name}{object_counts[name]:03d}"
                object_counts[name] += 1
                created: list[ProjectedNode] = []
                if name == "t":
                    text = _text_value(child)
                    if text.strip():
                        text_locator = f"{locator}/r{run_index:03d}/t{text_index:03d}"
                        text_index += 1
                        created = [
                            self._text_node(
                                text_locator,
                                text,
                                style_id,
                                para_ref,
                                char_ref,
                            )
                        ]
                elif name == "tbl":
                    created = [self._table_node(child, child_locator, para_ref, char_ref)]
                elif name == "pic":
                    created = [self._image_node(child, child_locator)]
                elif name == "equation":
                    created = [self._formula_node(child, child_locator)]
                elif name in {"ctrl", "container", "rect", "polygon", "compose"}:
                    created = self._parse_complex(child, child_locator, para_ref, char_ref, None)
                elif name not in {"secPr", "colPr"}:
                    visible = _visible_text(child)
                    if visible.strip():
                        created = [
                            self._text_node(
                                child_locator,
                                visible,
                                style_id,
                                para_ref,
                                char_ref,
                                reasons=("flattened_unsupported_control",),
                            )
                        ]
                    self._issue("unsupported_run_control", child_locator)
                if created and first_node:
                    created[0] = created[0].model_copy(
                        update={"page_break": page_break, "column_break": column_break}
                    )
                    first_node = False
                nodes.extend(created)
        return nodes

    def _parse_complex(
        self,
        element: etree._Element,
        locator: str,
        para_ref: str | None,
        char_ref: str | None,
        role_hint: Literal["header", "footer", "other"] | None,
    ) -> list[ProjectedNode]:
        name = _local_name(element)
        if name == "shapeComment":
            return []
        if name == "pageNum":
            self._issue("page_number_control_not_projected", locator)
            self.counts["page_number_controls"] += 1
            return []
        if name in {"header", "footer"}:
            role_hint = cast(Literal["header", "footer"], name)
            self.counts[name] += 1
            self._issue("header_footer_layout_requires_review", locator)
        if name in {"rect", "polygon", "compose"}:
            role_hint = role_hint or "other"
            self._issue("shape_layout_flattened", locator)
        if name == "tbl":
            return [self._table_node(element, locator, para_ref, char_ref)]
        if name == "pic":
            return [self._image_node(element, locator)]
        if name == "equation":
            return [self._formula_node(element, locator)]
        if name == "t":
            text = _text_value(element)
            if not text.strip():
                return []
            style_id = self.style_resolver.resolve(para_ref, char_ref)
            return [
                self._text_node(
                    locator,
                    text,
                    style_id,
                    para_ref,
                    char_ref,
                    role_hint=role_hint,
                    reasons=("flattened_complex_text",),
                )
            ]

        nodes: list[ProjectedNode] = []
        counters: Counter[str] = Counter()
        for child in element:
            child_name = _local_name(child)
            if child_name in {
                "sz",
                "pos",
                "offset",
                "orgSz",
                "curSz",
                "flip",
                "rotationInfo",
                "renderingInfo",
                "imgRect",
                "imgClip",
                "inMargin",
                "outMargin",
                "lineShape",
                "fillBrush",
                "shadow",
                "shapeComment",
                "linesegarray",
            }:
                continue
            child_locator = f"{locator}/{child_name}{counters[child_name]:03d}"
            counters[child_name] += 1
            child_para_ref = child.get("paraPrIDRef", para_ref)
            child_char_ref = child.get("charPrIDRef", char_ref)
            nodes.extend(
                self._parse_complex(
                    child,
                    child_locator,
                    child_para_ref,
                    child_char_ref,
                    role_hint,
                )
            )
        return nodes

    def _text_node(
        self,
        locator: str,
        text: str,
        style_id: str | None,
        para_ref: str | None,
        char_ref: str | None,
        *,
        role_hint: Literal["header", "footer", "other"] | None = None,
        reasons: tuple[str, ...] = (),
    ) -> ProjectedNode:
        self.counts["text_nodes"] += 1
        return ProjectedNode(
            id=f"{self.source_sha256}:{locator}",
            locator=locator,
            kind="text",
            text=unicodedata.normalize("NFC", text),
            role_hint=role_hint,
            style_id=style_id,
            source_para_pr_id=para_ref,
            source_char_pr_id=char_ref,
            review_reasons=("machine_projected", *reasons),
        )

    def _table_node(
        self,
        table: etree._Element,
        locator: str,
        para_ref: str | None,
        char_ref: str | None,
    ) -> ProjectedNode:
        rows = _required_positive_int(table.get("rowCnt"), "table rowCnt")
        columns = _required_positive_int(table.get("colCnt"), "table colCnt")
        if rows > _MAX_TABLE_ROWS or columns > _MAX_TABLE_COLUMNS:
            raise ValueError("projected table dimensions exceed the safety limit")
        if rows * columns > _MAX_TABLE_AREA:
            raise ValueError("projected table area exceeds the safety limit")
        raw_cells = table.xpath("./hp:tr/hp:tc", namespaces={"hp": _HP})
        if len(raw_cells) > _MAX_TABLE_CELLS:
            raise ValueError("projected table cell count exceeds the safety limit")
        cells: list[ProjectedCell] = []
        for cell_index, cell in enumerate(raw_cells):
            address = _required_one(cell.xpath("./hp:cellAddr", namespaces={"hp": _HP}))
            span = _required_one(cell.xpath("./hp:cellSpan", namespaces={"hp": _HP}))
            row = _required_nonnegative_int(address.get("rowAddr"), "table cell row")
            column = _required_nonnegative_int(address.get("colAddr"), "table cell column")
            row_span = _required_positive_int(span.get("rowSpan"), "table cell rowSpan")
            column_span = _required_positive_int(span.get("colSpan"), "table cell colSpan")
            cell_locator = f"{locator}/cell-r{row:03d}-c{column:03d}"
            paragraphs = cast(
                list[etree._Element],
                cell.xpath("./hp:subList/hp:p", namespaces={"hp": _HP}),
            )
            paragraph_refs = tuple(
                f"{cell_locator}/p{index:03d}" for index in range(len(paragraphs))
            )
            text = "\n".join(
                part
                for paragraph in paragraphs
                if (part := _visible_text(paragraph)).strip()
            )
            cell_para_ref, cell_char_ref = _first_style_refs(paragraphs)
            style_id = self.style_resolver.resolve(
                cell_para_ref or para_ref,
                cell_char_ref or char_ref,
            )
            cells.append(
                ProjectedCell(
                    locator=cell_locator,
                    row=row,
                    column=column,
                    row_span=row_span,
                    column_span=column_span,
                    text=unicodedata.normalize("NFC", text),
                    style_id=style_id,
                    paragraph_refs=paragraph_refs,
                )
            )
            nested = cell.xpath(".//*[local-name()='tbl' or local-name()='pic' or local-name()='equation']")
            if nested:
                self._issue("table_nested_object_not_projected", cell_locator)
            self.counts["table_cells"] += 1
        _validate_cell_topology(rows, columns, tuple(cells))
        size = _first(table.xpath("./hp:sz", namespaces={"hp": _HP}))
        self.counts["tables"] += 1
        return ProjectedNode(
            id=f"{self.source_sha256}:{locator}",
            locator=locator,
            kind="table",
            text=_table_text(rows, columns, cells),
            source_para_pr_id=para_ref,
            source_char_pr_id=char_ref,
            rows=rows,
            columns=columns,
            cells=tuple(cells),
            width_hwp=_optional_int(size.get("width")) if size is not None else None,
            height_hwp=_optional_int(size.get("height")) if size is not None else None,
            review_reasons=("machine_projected", "table_visual_style_requires_review"),
        )

    def _image_node(self, picture: etree._Element, locator: str) -> ProjectedNode:
        image = _first(picture.xpath(".//*[local-name()='img']"))
        if image is None or not image.get("binaryItemIDRef"):
            raise ValueError(f"picture has no binaryItemIDRef at {locator}")
        package_id = cast(str, image.get("binaryItemIDRef"))
        member = self.media_by_id.get(package_id)
        if member is None:
            raise ValueError(f"unresolved HWPX image reference at {locator}")
        payload = self.archive.read(member)
        if not payload:
            raise ValueError(f"empty HWPX image payload at {locator}")
        digest = hashlib.sha256(payload).hexdigest()
        asset_id = f"asset-{digest[:24]}"
        media_type = mimetypes.guess_type(member)[0] or "application/octet-stream"
        self.assets.setdefault(
            asset_id,
            ProjectionAsset(
                id=asset_id,
                sha256=digest,
                media_type=media_type,
                size_bytes=len(payload),
            ),
        )
        size = _first(picture.xpath("./hp:sz", namespaces={"hp": _HP}))
        self.counts["pictures"] += 1
        return ProjectedNode(
            id=f"{self.source_sha256}:{locator}",
            locator=locator,
            kind="image",
            asset_id=asset_id,
            width_hwp=_optional_int(size.get("width")) if size is not None else None,
            height_hwp=_optional_int(size.get("height")) if size is not None else None,
            review_reasons=("machine_projected", "image_anchor_requires_review"),
        )

    def _formula_node(self, equation: etree._Element, locator: str) -> ProjectedNode:
        script = _first(equation.xpath("./*[local-name()='script']"))
        expression = "" if script is None else "".join(script.itertext())
        if not expression.strip():
            raise ValueError(f"equation has no source script at {locator}")
        size = _first(equation.xpath("./hp:sz", namespaces={"hp": _HP}))
        self._issue("hwp_equation_script_requires_conversion", locator)
        self.counts["formulas"] += 1
        return ProjectedNode(
            id=f"{self.source_sha256}:{locator}",
            locator=locator,
            kind="formula",
            text=unicodedata.normalize("NFC", expression),
            formula_script=unicodedata.normalize("NFC", expression),
            width_hwp=_optional_int(size.get("width")) if size is not None else None,
            height_hwp=_optional_int(size.get("height")) if size is not None else None,
            review_reasons=("machine_projected", "hwp_equation_script_requires_conversion"),
        )

    def _issue(self, code: str, locator: str | None = None) -> None:
        self.issues.append(ProjectionIssue(code=code, severity="review", locator=locator))


def read_hwpx_projection(
    path: str | Path,
    *,
    source_document_sha256: str,
    expected_hwpx_sha256: str,
    expected_page_count: int,
) -> HwpxProjectionSource:
    """Read HWPX target facts into a review-only, non-golden projection."""
    if _SHA256.fullmatch(source_document_sha256) is None:
        raise ValueError("source_document_sha256 must be lowercase SHA-256")
    if _SHA256.fullmatch(expected_hwpx_sha256) is None:
        raise ValueError("expected_hwpx_sha256 must be lowercase SHA-256")
    if (
        isinstance(expected_page_count, bool)
        or not isinstance(expected_page_count, int)
        or expected_page_count < 1
    ):
        raise ValueError("expected_page_count must be a positive integer")
    source = _required_file(Path(path))
    actual_sha256 = _sha256(source)
    if actual_sha256 != expected_hwpx_sha256:
        raise ValueError("HWPX SHA-256 differs from the bound pair manifest")
    try:
        with zipfile.ZipFile(source) as preflight_archive:
            _validate_archive(preflight_archive)
    except zipfile.BadZipFile as exc:
        raise ValueError("HWPX input is not a valid ZIP package") from exc
    validation = validate_hwpx(source)
    if not validation.valid:
        raise ValueError("HWPX package validation failed")

    with zipfile.ZipFile(source) as archive:
        names = _validate_archive(archive)
        header = _read_xml(archive, "Contents/header.xml")
        media_by_id = _manifest_media(archive, names)
        resolver = _StyleResolver(header)
        builder = _ProjectionBuilder(
            source_sha256=source_document_sha256,
            archive=archive,
            media_by_id=media_by_id,
            style_resolver=resolver,
        )
        section_members = sorted(
            (
                (int(match.group(1)), name)
                for name in names
                if (match := _SECTION.fullmatch(name)) is not None
            ),
            key=lambda item: item[0],
        )
        if [index for index, _ in section_members] != list(range(len(section_members))):
            raise ValueError("HWPX section members must be consecutively numbered")
        if not section_members:
            raise ValueError("HWPX package contains no section")
        sections: list[ProjectedSection] = []
        nodes: list[ProjectedNode] = []
        for section_index, member in section_members:
            section, section_nodes = builder.parse_section(
                _read_xml(archive, member), section_index
            )
            sections.append(section)
            nodes.extend(section_nodes)
        builder._issue("page_assignment_requires_pdf_alignment")
        unsupported_styles = sum(
            _style_has_unrepresented_plan_facts(style) for style in resolver.styles.values()
        )
        if unsupported_styles:
            builder.counts["styles_with_unrepresented_plan_facts"] = unsupported_styles
            builder._issue("style_facts_not_representable_in_plan")

    stats = dict(sorted(builder.counts.items()))
    stats.setdefault("paragraphs", 0)
    stats.setdefault("text_nodes", 0)
    stats.setdefault("tables", 0)
    stats.setdefault("table_cells", 0)
    stats.setdefault("pictures", 0)
    stats.setdefault("formulas", 0)
    stats.setdefault("empty_paragraphs", 0)
    stats.setdefault("page_number_controls", 0)
    stats.setdefault("styles_with_unrepresented_plan_facts", 0)
    return HwpxProjectionSource(
        source_document_sha256=source_document_sha256,
        hwpx_sha256=actual_sha256,
        page_count=expected_page_count,
        sections=tuple(sections),
        nodes=tuple(nodes),
        styles=tuple(sorted(resolver.styles.values(), key=lambda style: style.id)),
        assets=tuple(sorted(builder.assets.values(), key=lambda asset: asset.id)),
        issues=tuple(builder.issues),
        stats=stats,
    )


def _manifest_media(archive: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    root = _read_xml(archive, "Contents/content.hpf")
    result: dict[str, str] = {}
    for item in root.xpath("//*[local-name()='manifest']/*[local-name()='item']"):
        item_id = item.get("id")
        href = item.get("href")
        if not item_id or not href:
            continue
        member = _resolve_member(href, names)
        if member.startswith("BinData/"):
            if item_id in result:
                raise ValueError(f"duplicate HWPX media id: {item_id}")
            result[item_id] = member
    return result


def _resolve_member(href: str, names: set[str]) -> str:
    if "\\" in href:
        raise ValueError("HWPX manifest href must use POSIX separators")
    path = PurePosixPath(href)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("HWPX manifest href escapes the package")
    candidates = (path.as_posix(), (PurePosixPath("Contents") / path).as_posix())
    matches = [candidate for candidate in candidates if candidate in names]
    if len(matches) != 1:
        raise ValueError(f"unresolved or ambiguous HWPX manifest href: {href}")
    return matches[0]


def _validate_archive(archive: zipfile.ZipFile) -> set[str]:
    infos = archive.infolist()
    if len(infos) > _MAX_ARCHIVE_ENTRIES:
        raise ValueError("HWPX package contains too many members")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
        raise ValueError("HWPX package contains duplicate member names")
    total = 0
    compressed_total = 0
    for info in infos:
        if "\\" in info.filename:
            raise ValueError("HWPX member names must use POSIX separators")
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts or not member.parts:
            raise ValueError("HWPX package contains an unsafe member name")
        if info.file_size > _MAX_ENTRY_BYTES:
            raise ValueError("HWPX member exceeds the extraction safety limit")
        total += info.file_size
        compressed_total += info.compress_size
        if info.flag_bits & 0x1:
            raise ValueError("HWPX package must not contain encrypted members")
        if info.compress_size == 0:
            if info.file_size > 0:
                raise ValueError("HWPX member has an invalid compression size")
        elif info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
            raise ValueError("HWPX member exceeds the compression-ratio safety limit")
    if total > _MAX_PACKAGE_BYTES:
        raise ValueError("HWPX package exceeds the uncompressed safety limit")
    if compressed_total > _MAX_COMPRESSED_PACKAGE_BYTES:
        raise ValueError("HWPX package exceeds the compressed safety limit")
    return set(names)


def _read_xml(archive: zipfile.ZipFile, member: str) -> etree._Element:
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError(f"HWPX package is missing {member}") from exc
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError(f"HWPX XML must not contain a DTD or entity declaration: {member}")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=False,
    )
    try:
        return cast(etree._Element, etree.fromstring(payload, parser=parser))
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"malformed HWPX XML: {member}") from exc


def _page_layout(root: etree._Element) -> ProjectedPageLayout:
    page = _first(root.xpath(".//*[local-name()='secPr']/*[local-name()='pagePr']"))
    if page is None:
        raise ValueError("HWPX section has no pagePr")
    margin = _first(page.xpath("./*[local-name()='margin']"))
    if margin is None:
        raise ValueError("HWPX pagePr has no margin")
    columns = [
        _required_positive_int(node.get("colCount"), "column count")
        for node in root.xpath(".//*[local-name()='colPr']")
        if node.get("colCount") is not None
    ]
    gaps = [
        _required_nonnegative_int(node.get("sameGap"), "column gap")
        for node in root.xpath(".//*[local-name()='colPr']")
        if node.get("sameGap") is not None
    ]
    return ProjectedPageLayout(
        width_hwp=_required_positive_int(page.get("width"), "page width"),
        height_hwp=_required_positive_int(page.get("height"), "page height"),
        margin_top_hwp=_required_nonnegative_int(margin.get("top"), "top margin"),
        margin_right_hwp=_required_nonnegative_int(margin.get("right"), "right margin"),
        margin_bottom_hwp=_required_nonnegative_int(margin.get("bottom"), "bottom margin"),
        margin_left_hwp=_required_nonnegative_int(margin.get("left"), "left margin"),
        margin_header_hwp=_required_nonnegative_int(margin.get("header"), "header margin"),
        margin_footer_hwp=_required_nonnegative_int(margin.get("footer"), "footer margin"),
        margin_gutter_hwp=_required_nonnegative_int(margin.get("gutter"), "gutter margin"),
        columns=max(columns, default=1),
        column_gap_hwp=max(gaps, default=0),
    )


def _column_changes(
    root: etree._Element,
    section_locator: str,
) -> tuple[ProjectedColumnChange, ...]:
    result: list[ProjectedColumnChange] = []
    paragraphs = root.xpath("./hp:p", namespaces={"hp": _HP})
    for paragraph_index, paragraph in enumerate(paragraphs):
        column_nodes = paragraph.xpath(".//*[local-name()='colPr']")
        for column_index, node in enumerate(column_nodes):
            gap = _optional_int(node.get("sameGap"))
            if gap is not None and gap < 0:
                raise ValueError("column gap must be non-negative")
            result.append(
                ProjectedColumnChange(
                    locator=(
                        f"{section_locator}/p{paragraph_index:06d}/colPr{column_index:03d}"
                    ),
                    columns=_required_positive_int(node.get("colCount"), "column count"),
                    column_gap_hwp=gap or 0,
                )
            )
    return tuple(result)


def _font_faces(header: etree._Element) -> dict[str, str]:
    result: dict[str, str] = {}
    faces = header.xpath("//*[local-name()='fontface'][@lang='HANGUL']")
    if not faces:
        return result
    for font in faces[0].xpath("./*[local-name()='font']"):
        font_id = font.get("id")
        face = font.get("face")
        if font_id is not None and face:
            result[str(font_id)] = unicodedata.normalize("NFC", face)
    return result


def _named_child_int(parent: etree._Element, local_name: str) -> int | None:
    child = _first(parent.xpath(f"./*[local-name()='{local_name}']"))
    return _optional_int(child.get("value")) if child is not None else None


def _style_has_unrepresented_plan_facts(style: ProjectedStyle) -> bool:
    return bool(
        style.text_color not in {None, "#000000", "0", "none"}
        or style.underline_type not in {None, "NONE"}
        or style.hangul_spacing not in {None, 0}
        or style.hangul_ratio not in {None, 100}
        or any(
            value not in {None, 0}
            for value in (
                style.margin_indent_hwp,
                style.margin_left_hwp,
                style.margin_right_hwp,
                style.margin_previous_hwp,
                style.margin_next_hwp,
            )
        )
    )


def _visible_text(element: etree._Element) -> str:
    pieces: list[str] = []
    for text_node in element.xpath(".//*[local-name()='t']"):
        if text_node.xpath("ancestor::*[local-name()='shapeComment']"):
            continue
        pieces.append(_text_value(text_node))
    return "".join(pieces)


def _text_value(element: etree._Element) -> str:
    pieces: list[str] = []

    def visit(node: etree._Element) -> None:
        if node.text:
            pieces.append(node.text)
        for child in node:
            name = _local_name(child)
            if name == "tab":
                pieces.append("\t")
            elif name == "lineBreak":
                pieces.append("\n")
            elif name == "nbSpace":
                pieces.append("\u00a0")
            elif name == "fwSpace":
                pieces.append("\u3000")
            elif name in {"hyphen", "hypen"}:
                pieces.append("-")
            else:
                visit(child)
            if child.tail:
                pieces.append(child.tail)

    visit(element)
    return "".join(pieces)


def _first_style_refs(
    paragraphs: list[etree._Element],
) -> tuple[str | None, str | None]:
    for paragraph in paragraphs:
        para_ref = paragraph.get("paraPrIDRef")
        run = _first(paragraph.xpath("./hp:run", namespaces={"hp": _HP}))
        if run is not None:
            return para_ref, run.get("charPrIDRef")
    return None, None


def _table_text(rows: int, columns: int, cells: list[ProjectedCell]) -> str:
    values = [["" for _ in range(columns)] for _ in range(rows)]
    for cell in cells:
        values[cell.row][cell.column] = cell.text
    return "\n".join("\t".join(row) for row in values)


def _validate_cell_topology(
    rows: int,
    columns: int,
    cells: tuple[ProjectedCell, ...],
) -> None:
    occupied: set[tuple[int, int]] = set()
    for cell in cells:
        if cell.row + cell.row_span > rows or cell.column + cell.column_span > columns:
            raise ValueError("projected table cell span exceeds declared bounds")
        positions = {
            (row, column)
            for row in range(cell.row, cell.row + cell.row_span)
            for column in range(cell.column, cell.column + cell.column_span)
        }
        if occupied & positions:
            raise ValueError("projected table cells overlap")
        occupied.update(positions)
    expected = {(row, column) for row in range(rows) for column in range(columns)}
    if occupied != expected:
        raise ValueError("projected table cells do not cover the declared grid")


def _alignment(value: str | None) -> Literal["left", "center", "right", "justify"]:
    return {
        "LEFT": "left",
        "CENTER": "center",
        "RIGHT": "right",
        "JUSTIFY": "justify",
        "DISTRIBUTE": "justify",
        "DISTRIBUTE_SPACE": "justify",
    }.get(value or "", "left")  # type: ignore[return-value]


def _local_name(element: etree._Element) -> str:
    return cast(str, etree.QName(element).localname)


def _first(values: list[etree._Element]) -> etree._Element | None:
    return values[0] if values else None


def _required_one(values: list[etree._Element]) -> etree._Element:
    if len(values) != 1:
        raise ValueError("HWPX element must contain exactly one required child")
    return values[0]


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("HWPX numeric attribute must be an integer") from exc


def _required_positive_int(value: str | None, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed < 1:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _required_nonnegative_int(value: str | None, label: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _xml_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def _required_file(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("HWPX input must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"HWPX input not found: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"HWPX input is not a regular file: {path}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicates(values: list[str], label: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: " + ", ".join(duplicates))
