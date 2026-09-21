from __future__ import annotations

import hashlib
import os
import tempfile
import unicodedata
import zipfile
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from io import BytesIO
from math import isclose, isfinite
from pathlib import Path
from typing import TypeVar, cast

from lxml import etree  # type: ignore[import-untyped]

from scan2hwpx.contracts import (
    ColumnBreakPlanItem,
    ContentIR,
    ContentPlanItem,
    EvidenceIR,
    HwpDocumentPlan,
    ImageContentNode,
    LayoutIntent,
    PageBreakPlanItem,
    PageLayoutIntent,
    StyleIntent,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.contracts.models import TextAlignment

from .assets import (
    ImageAssetBundle,
    ImageAssetBundleError,
    PngImageAsset,
    image_asset_bundle_sha256,
    validate_image_asset_bundle,
)
from .render import HH, HP, RenderParagraph, _append_text_run, _base_entries, _replace_body
from .validate import validate_hwpx

COMPILER_VERSION = "scan2hwpx-plan-styled-table-image/5.0"
SUPPORTED_CAPABILITY_PROFILE_ID = "exam2hwpx-authoring-v1"
SUPPORTED_DESIGN_PROFILE_ID = "scan2hwpx-canonical-styled-text-v2"
LEGACY_DESIGN_PROFILE_ID = "scan2hwpx-canonical-text-v1"
TABLE_DESIGN_PROFILE_ID = "scan2hwpx-canonical-styled-table-v3"
IMAGE_DESIGN_PROFILE_ID = "scan2hwpx-canonical-styled-table-image-v4"

HC = "http://www.hancom.co.kr/hwpml/2011/core"
OPF = "http://www.idpf.org/2007/opf/"

_CANONICAL_PAGE_LAYOUT = PageLayoutIntent(
    width_mm=210.0,
    height_mm=297.0,
    margin_top_mm=20.0,
    margin_right_mm=30.0,
    margin_bottom_mm=15.0,
    margin_left_mm=30.0,
    columns=2,
    column_gap_mm=8.0,
)
_LAYOUT_TOLERANCE_MM = 0.02
_HWPUNIT_PER_INCH = Decimal(7_200)
_MM_PER_INCH = Decimal("25.4")
_MIN_USABLE_LAYOUT_MM = Decimal(1)
_CANONICAL_PAGE_WIDTH_HWP = 59_528
_CANONICAL_PAGE_HEIGHT_HWP = 84_186
_CANONICAL_MARGIN_TOP_HWP = 5_668
_CANONICAL_MARGIN_RIGHT_HWP = 8_504
_CANONICAL_MARGIN_BOTTOM_HWP = 4_252
_CANONICAL_MARGIN_LEFT_HWP = 8_504
_CANONICAL_COLUMN_GAP_HWP = 2_267
_CANONICAL_COLUMN_WIDTH_HWP = 20_126
_CANONICAL_HEADER_HWP = 4_252
_CANONICAL_FOOTER_HWP = 4_252
_CANONICAL_GUTTER_HWP = 0
_MAX_PARAGRAPHS = 100_000
_MAX_CONTENT_NODES = 100_000
_MAX_TEXT_UTF8_BYTES = 32 * 1024 * 1024
_MAX_TABLE_CELL_TEXT_UTF8_BYTES = 64 * 1024
_MAX_TABLE_TEXT_CONTROLS = 100_000
_MAX_TABLES = 1_000
_MAX_IMAGE_CONTROLS = 10_000
_MAX_IMAGE_OCCURRENCE_PIXELS = 64_000_000
_MAX_TABLE_ROWS = 1_000
_MAX_TABLE_COLUMNS = 256
_MAX_TABLE_GRID_AREA = 10_000
_MAX_TOTAL_TABLE_GRID_AREA = 100_000
_MAX_TABLE_CELLS = 10_000
_MAX_STYLE_INTENTS = 4_096
_MAX_STYLE_METADATA_UTF8_BYTES = 4 * 1024 * 1024
_MAX_FONT_FAMILY_UTF8_BYTES = 1_024
_MAX_HEADER_XML_BYTES = 32 * 1024 * 1024
_MAX_SECTION_XML_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_XML_BYTES = 4 * 1024 * 1024
_MAX_TABLE_TEMPLATE_B64_BYTES = 4 * 1024 * 1024
_MAX_IMAGE_TEMPLATE_B64_BYTES = 4 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_CANONICAL_TEMPLATE_ENTRIES_SHA256 = (
    "f452a9ec7e8966deecb00feabaf84fecca1ea8f6e2b653eb4dedd7c7561ab227"
)
_TABLE_TEMPLATE = Path(__file__).with_name("table_template.b64")
_TABLE_TEMPLATE_PACKAGE_SHA256 = "bb791e580f9b9fcc5520a1bc628f31eb8a94f72805ff0c88718133b1a3928414"
_TABLE_ROW_HEIGHT_HWP = 282
_TABLE_CONTROL_ID_BASE = 1_300_000_000
_IMAGE_TEMPLATE = Path(__file__).with_name("fidelity_template.b64")
_IMAGE_TEMPLATE_PACKAGE_SHA256 = "96a9f8a6343f9940cc4005e8b2a66ce7fe91b2d3d95e2f14f6acdbc4a9b0926a"
_IMAGE_TEMPLATE_PICTURE_C14N_SHA256 = (
    "fb40ba27f916c9d2b696aa1ac2ecdc257bd3fc1360ebf6a3d02aebe9414fe720"
)
_IMAGE_CONTROL_ID_BASE = 1_400_000_000
_IMAGE_PREVIEW_MARKER = "[IMAGE]"
_IMAGE_CLIP_UNITS_PER_PIXEL = 75
ContractT = TypeVar("ContractT", EvidenceIR, ContentIR, HwpDocumentPlan)
_FONT_LANGUAGES = (
    ("HANGUL", "hangul"),
    ("LATIN", "latin"),
    ("HANJA", "hanja"),
    ("JAPANESE", "japanese"),
    ("OTHER", "other"),
    ("SYMBOL", "symbol"),
    ("USER", "user"),
)
_ALIGNMENTS = {
    TextAlignment.LEFT: "LEFT",
    TextAlignment.CENTER: "CENTER",
    TextAlignment.RIGHT: "RIGHT",
    TextAlignment.JUSTIFY: "JUSTIFY",
}
_HWP_SPECIAL_TEXT_VALUES = {
    "tab": "\t",
    "lineBreak": "\n",
    "nbSpace": "\u00a0",
    "fwSpace": "\u3000",
}
_HWP_SPECIAL_TEXT_CHARACTERS = frozenset(_HWP_SPECIAL_TEXT_VALUES.values())


class PlanCompilationError(RuntimeError):
    """The supplied contracts cannot be compiled without losing declared intent."""


@dataclass(frozen=True)
class PlanCompileResult:
    compiler_version: str
    content_ir_sha256: str
    hwp_document_plan_sha256: str
    artifact_sha256: str
    paragraph_count: int
    page_break_count: int
    image_asset_bundle_sha256: str | None = None


@dataclass(frozen=True)
class _CompiledStyle:
    source_id: str
    semantic_role: str
    char_pr_id: int
    para_pr_id: int
    style_id: int
    font_size_hwp: int
    font_refs: tuple[tuple[str, int, str], ...]
    bold: bool
    italic: bool
    alignment: str
    line_spacing_percent: int


@dataclass(frozen=True)
class _CompiledPageLayout:
    width_hwp: int
    height_hwp: int
    margin_top_hwp: int
    margin_right_hwp: int
    margin_bottom_hwp: int
    margin_left_hwp: int
    columns: int
    column_gap_hwp: int
    column_width_hwp: int
    usable_height_hwp: int


@dataclass(frozen=True)
class _CompiledTable:
    node: TableContentNode
    paragraph_index: int
    control_id: int
    z_order: int
    width_hwp: int


@dataclass(frozen=True)
class _CompiledImage:
    node: ImageContentNode
    asset: PngImageAsset
    paragraph_index: int
    control_id: int
    instance_id: int
    z_order: int
    binary_id: str
    package_path: str
    width_hwp: int
    height_hwp: int


@dataclass(frozen=True)
class _CompiledImageBinary:
    binary_id: str
    package_path: str
    asset: PngImageAsset


def compile_plan_hwpx(
    content_ir: ContentIR,
    plan: HwpDocumentPlan,
    output: Path,
    *,
    evidence_ir: EvidenceIR | None = None,
    image_assets: ImageAssetBundle | None = None,
) -> PlanCompileResult:
    """Compile narrow text/table/image Plan profiles to deterministic HWPX.

    This compiler does not certify review, release, official-reference grounding, or
    Hancom round-trip status. Those remain separate gates. It rejects every Plan intent
    that the bundled canonical template cannot faithfully materialize.
    """
    if type(content_ir) is not ContentIR:
        raise TypeError("ContentIR must be a ContentIR")
    _preflight_content_contract(content_ir)
    content = _revalidate_contract(content_ir, ContentIR, "ContentIR")
    validated_plan = _revalidate_contract(plan, HwpDocumentPlan, "HwpDocumentPlan")
    try:
        validated_plan.assert_content_integrity(content)
    except ValueError as exc:
        raise PlanCompilationError(f"contract lineage is invalid: {exc}") from exc
    _assert_supported_profiles(validated_plan)
    validated_image_assets = _validate_compile_image_inputs(
        content,
        validated_plan,
        evidence_ir,
        image_assets,
    )
    if validated_image_assets is not None:
        _preflight_image_occurrence_pixels(content, validated_image_assets)
    image_bundle_digest = (
        image_asset_bundle_sha256(validated_image_assets)
        if validated_image_assets is not None
        else None
    )
    compiled_layout = _compile_page_layout(validated_plan.page_layout)

    entries = _base_entries()
    _assert_template_version(entries)
    style_bindings = _materialize_style_intents(entries, validated_plan)
    paragraphs, tables, images, preview_lines = _compile_paragraphs(
        content,
        validated_plan,
        style_bindings,
        validated_image_assets,
        compiled_layout,
    )
    _replace_body(entries, paragraphs)
    _materialize_page_layout(entries, compiled_layout)
    if tables:
        _materialize_tables(entries, tables)
    if images:
        _materialize_images(entries, images)
    entries["Preview/PrvText.txt"] = "\n".join(preview_lines).encode("utf-8")
    package = _deterministic_package(entries)
    package_sha256 = hashlib.sha256(package).hexdigest()

    if output.suffix.lower() != ".hwpx":
        raise PlanCompilationError("output path must use the .hwpx suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=".candidate.hwpx",
    )
    candidate = Path(candidate_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(package)
            stream.flush()
            os.fsync(stream.fileno())
        validation = validate_hwpx(candidate)
        if not validation.valid:
            raise PlanCompilationError(
                "compiled HWPX failed package validation: " + "; ".join(validation.errors)
            )
        _verify_compiled_content(
            candidate,
            paragraphs,
            tables,
            images,
            preview_lines,
            tuple(style_bindings.values()),
            compiled_layout,
        )
        _verify_candidate_digest(candidate, package_sha256)
        os.replace(candidate, output)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise

    return PlanCompileResult(
        compiler_version=COMPILER_VERSION,
        content_ir_sha256=contract_sha256(content),
        hwp_document_plan_sha256=contract_sha256(validated_plan),
        artifact_sha256=package_sha256,
        paragraph_count=len(paragraphs),
        page_break_count=sum(paragraph.page_break for paragraph in paragraphs),
        image_asset_bundle_sha256=image_bundle_digest,
    )


def _revalidate_contract(
    value: ContractT,
    expected_type: type[ContractT],
    label: str,
) -> ContractT:
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    try:
        payload = value.model_dump(mode="python", round_trip=True, warnings=False)
        return expected_type.model_validate(payload, strict=True)
    except (TypeError, ValueError) as exc:
        raise PlanCompilationError(f"{label} is not a valid strict contract: {exc}") from exc


def _validate_compile_image_inputs(
    content: ContentIR,
    plan: HwpDocumentPlan,
    evidence_ir: EvidenceIR | None,
    image_assets: ImageAssetBundle | None,
) -> ImageAssetBundle | None:
    image_nodes = tuple(node for node in content.nodes if isinstance(node, ImageContentNode))
    supplied = evidence_ir is not None or image_assets is not None
    if plan.design_profile_id != IMAGE_DESIGN_PROFILE_ID:
        if supplied:
            raise PlanCompilationError("image evidence/assets are unneeded for this design profile")
        return None
    if not image_nodes:
        if supplied:
            raise PlanCompilationError("image evidence/assets are unneeded without image nodes")
        return None
    if image_assets is None:
        raise PlanCompilationError("image compilation requires an image asset bundle")
    if evidence_ir is None:
        raise PlanCompilationError("image compilation requires EvidenceIR")
    if type(evidence_ir) is not EvidenceIR:
        raise TypeError("EvidenceIR must be an EvidenceIR")
    if type(image_assets) is not ImageAssetBundle:
        raise TypeError("image_assets must be an ImageAssetBundle")

    evidence = _revalidate_contract(evidence_ir, EvidenceIR, "EvidenceIR")
    try:
        content.assert_evidence_integrity(evidence)
    except ValueError as exc:
        raise PlanCompilationError(f"image EvidenceIR lineage is invalid: {exc}") from exc
    try:
        return validate_image_asset_bundle(evidence, content, image_assets)
    except (ImageAssetBundleError, ValueError) as exc:
        raise PlanCompilationError(str(exc)) from exc


def _preflight_content_contract(content: ContentIR) -> None:
    raw_nodes: object = content.nodes
    if not isinstance(raw_nodes, (list, tuple)):
        return
    if len(raw_nodes) > _MAX_CONTENT_NODES:
        raise PlanCompilationError("content node count exceeds the compiler limit")

    table_count = 0
    image_count = 0
    total_cells = 0
    total_grid_area = 0
    for raw_node in raw_nodes:
        raw_kind = _contract_field(raw_node, "kind")
        if raw_kind == "image":
            image_count += 1
            if image_count > _MAX_IMAGE_CONTROLS:
                raise PlanCompilationError("image control count exceeds the compiler limit")
        if raw_kind != "table":
            continue
        table_count += 1
        if table_count > _MAX_TABLES:
            raise PlanCompilationError("table count exceeds the compiler limit")

        raw_rows = _contract_field(raw_node, "rows")
        raw_columns = _contract_field(raw_node, "columns")
        rows = raw_rows if isinstance(raw_rows, int) and not isinstance(raw_rows, bool) else None
        columns = (
            raw_columns
            if isinstance(raw_columns, int) and not isinstance(raw_columns, bool)
            else None
        )
        if rows is not None and rows > _MAX_TABLE_ROWS:
            raise PlanCompilationError("table row count exceeds the compiler limit")
        if columns is not None and columns > _MAX_TABLE_COLUMNS:
            raise PlanCompilationError("table column count exceeds the compiler limit")
        if rows is not None and columns is not None and rows > 0 and columns > 0:
            grid_area = rows * columns
            if grid_area > _MAX_TABLE_GRID_AREA:
                raise PlanCompilationError("table grid area exceeds the compiler limit")
            total_grid_area += grid_area
            if total_grid_area > _MAX_TOTAL_TABLE_GRID_AREA:
                raise PlanCompilationError("total table grid area exceeds the compiler limit")

        raw_cells = _contract_field(raw_node, "cells")
        if not isinstance(raw_cells, (list, tuple)):
            continue
        if len(raw_cells) > _MAX_TABLE_CELLS:
            raise PlanCompilationError("table cell count exceeds the compiler limit")
        total_cells += len(raw_cells)
        if total_cells > _MAX_TABLE_CELLS:
            raise PlanCompilationError("total table cell count exceeds the compiler limit")

        occupied_area = 0
        for raw_cell in raw_cells:
            raw_row_span = _contract_field(raw_cell, "row_span")
            raw_column_span = _contract_field(raw_cell, "column_span")
            row_span = (
                raw_row_span
                if isinstance(raw_row_span, int) and not isinstance(raw_row_span, bool)
                else None
            )
            column_span = (
                raw_column_span
                if isinstance(raw_column_span, int) and not isinstance(raw_column_span, bool)
                else None
            )
            if row_span is not None and row_span > _MAX_TABLE_ROWS:
                raise PlanCompilationError("table cell row span exceeds the compiler limit")
            if column_span is not None and column_span > _MAX_TABLE_COLUMNS:
                raise PlanCompilationError("table cell column span exceeds the compiler limit")
            if (
                row_span is not None
                and column_span is not None
                and row_span > 0
                and column_span > 0
            ):
                occupied_area += row_span * column_span
                if occupied_area > _MAX_TABLE_GRID_AREA:
                    raise PlanCompilationError("table cell spans exceed the compiler grid limit")


def _preflight_image_occurrence_pixels(
    content: ContentIR,
    bundle: ImageAssetBundle,
) -> None:
    pixels_by_ref = {asset.asset_ref: asset.width_px * asset.height_px for asset in bundle.assets}
    total_pixels = 0
    for node in content.nodes:
        if not isinstance(node, ImageContentNode):
            continue
        pixels = pixels_by_ref.get(node.asset_ref)
        if pixels is None:
            raise PlanCompilationError(f"missing validated image asset: {node.asset_ref}")
        total_pixels += pixels
        if total_pixels > _MAX_IMAGE_OCCURRENCE_PIXELS:
            raise PlanCompilationError("image occurrence pixels exceed the compiler limit")


def _contract_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value).get(name)
    return getattr(value, name, None)


def _materialize_style_intents(
    entries: dict[str, bytes],
    plan: HwpDocumentPlan,
) -> dict[str, _CompiledStyle]:
    _assert_supported_profiles(plan)
    if plan.design_profile_id == LEGACY_DESIGN_PROFILE_ID and plan.styles:
        raise PlanCompilationError(
            "style intents require the scan2hwpx-canonical-styled-text-v2 design profile"
        )
    if len(plan.styles) > _MAX_STYLE_INTENTS:
        raise PlanCompilationError("style intent count exceeds the compiler limit")
    if not plan.styles:
        return {}

    metadata_bytes = 0
    for style in plan.styles:
        _assert_xml_attribute(f"style id {style.id}", style.id)
        _assert_xml_attribute(f"style semantic role {style.id}", style.semantic_role)
        metadata_bytes += len(style.id.encode("utf-8"))
        metadata_bytes += len(style.semantic_role.encode("utf-8"))
        if style.font_family is not None:
            _assert_xml_attribute(f"font family {style.id}", style.font_family)
            font_bytes = len(style.font_family.encode("utf-8"))
            if font_bytes > _MAX_FONT_FAMILY_UTF8_BYTES:
                raise PlanCompilationError(
                    f"font family exceeds the compiler size limit: {style.id}"
                )
            metadata_bytes += font_bytes
        if metadata_bytes > _MAX_STYLE_METADATA_UTF8_BYTES:
            raise PlanCompilationError("style metadata exceeds the compiler size limit")

    root = _parse_header(entries["Contents/header.xml"])
    fontfaces = _header_collection(root, "fontfaces")
    char_properties = _header_collection(root, "charProperties")
    para_properties = _header_collection(root, "paraProperties")
    style_properties = _header_collection(root, "styles")
    _assert_declared_count(fontfaces, "fontface", "fontfaces")
    _assert_declared_count(char_properties, "charPr", "charProperties")
    _assert_declared_count(para_properties, "paraPr", "paraProperties")
    _assert_declared_count(style_properties, "style", "styles")

    fontfaces_by_language = _fontfaces_by_language(fontfaces)
    base_char = _element_with_id(char_properties, "charPr", 3)
    base_para = _element_with_id(para_properties, "paraPr", 20)
    next_char_id = _next_element_id(char_properties, "charPr", "charProperties")
    next_para_id = _next_element_id(para_properties, "paraPr", "paraProperties")
    next_style_id = _next_element_id(style_properties, "style", "styles")

    compiled: dict[str, _CompiledStyle] = {}
    for offset, intent in enumerate(sorted(plan.styles, key=lambda item: item.id)):
        char_pr_id = next_char_id + offset
        para_pr_id = next_para_id + offset
        style_id = next_style_id + offset
        font_refs = _resolve_font_refs(fontfaces_by_language, base_char, intent)

        char_pr = deepcopy(base_char)
        char_pr.set("id", str(char_pr_id))
        font_size_hwp = _materialize_char_properties(char_pr, intent, font_refs)
        char_properties.append(char_pr)

        para_pr = deepcopy(base_para)
        para_pr.set("id", str(para_pr_id))
        line_spacing_percent = _materialize_para_properties(para_pr, intent)
        para_properties.append(para_pr)

        etree.SubElement(
            style_properties,
            f"{{{HH}}}style",
            id=str(style_id),
            type="PARA",
            name=intent.id,
            engName=intent.semantic_role,
            paraPrIDRef=str(para_pr_id),
            charPrIDRef=str(char_pr_id),
            nextStyleIDRef=str(style_id),
            langID="1042",
            lockForm="0",
        )
        compiled[intent.id] = _CompiledStyle(
            source_id=intent.id,
            semantic_role=intent.semantic_role,
            char_pr_id=char_pr_id,
            para_pr_id=para_pr_id,
            style_id=style_id,
            font_size_hwp=font_size_hwp,
            font_refs=font_refs,
            bold=intent.bold,
            italic=intent.italic,
            alignment=_ALIGNMENTS[intent.alignment],
            line_spacing_percent=line_spacing_percent,
        )

    _set_declared_count(fontfaces, "fontface")
    for fontface in fontfaces_by_language.values():
        _set_declared_count(fontface, "font", attribute="fontCnt")
    _set_declared_count(char_properties, "charPr")
    _set_declared_count(para_properties, "paraPr")
    _set_declared_count(style_properties, "style")
    header_payload = _serialize_xml(root)
    if len(header_payload) > _MAX_HEADER_XML_BYTES:
        raise PlanCompilationError("compiled header XML exceeds the compiler size limit")
    entries["Contents/header.xml"] = header_payload
    return compiled


def _assert_supported_profiles(plan: HwpDocumentPlan) -> None:
    if plan.capability_profile_id != SUPPORTED_CAPABILITY_PROFILE_ID:
        raise PlanCompilationError(f"unsupported capability profile: {plan.capability_profile_id}")
    if plan.design_profile_id not in {
        LEGACY_DESIGN_PROFILE_ID,
        SUPPORTED_DESIGN_PROFILE_ID,
        TABLE_DESIGN_PROFILE_ID,
        IMAGE_DESIGN_PROFILE_ID,
    }:
        raise PlanCompilationError(f"unsupported design profile: {plan.design_profile_id}")


def _parse_header(payload: bytes) -> etree._Element:
    if len(payload) > _MAX_HEADER_XML_BYTES:
        raise PlanCompilationError("header XML exceeds the compiler size limit")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise PlanCompilationError("header XML contains a forbidden declaration")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=False,
    )
    try:
        root = cast(etree._Element, etree.fromstring(payload, parser=parser))
    except etree.XMLSyntaxError as exc:
        raise PlanCompilationError(f"header XML is malformed: {exc}") from exc
    name = etree.QName(root)
    if name.namespace != HH or name.localname != "head":
        raise PlanCompilationError("header XML uses an unsupported root element")
    return root


def _serialize_xml(root: etree._Element) -> bytes:
    return cast(
        bytes,
        etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        ),
    )


def _header_collection(root: etree._Element, local_name: str) -> etree._Element:
    nodes = cast(
        list[etree._Element],
        root.xpath(
            f"./hh:refList/hh:{local_name}",
            namespaces={"hh": HH},
        ),
    )
    if len(nodes) != 1:
        raise PlanCompilationError(
            f"canonical header must contain exactly one {local_name} collection"
        )
    return nodes[0]


def _children(parent: etree._Element, local_name: str) -> list[etree._Element]:
    tag = f"{{{HH}}}{local_name}"
    return [cast(etree._Element, child) for child in parent if child.tag == tag]


def _assert_declared_count(
    parent: etree._Element,
    child_name: str,
    label: str,
    *,
    attribute: str = "itemCnt",
) -> None:
    value = parent.get(attribute)
    try:
        declared = int(value) if value is not None else -1
    except ValueError as exc:
        raise PlanCompilationError(f"{label} has an invalid {attribute}") from exc
    if declared != len(_children(parent, child_name)):
        raise PlanCompilationError(f"{label} {attribute} does not match its children")


def _set_declared_count(
    parent: etree._Element,
    child_name: str,
    *,
    attribute: str = "itemCnt",
) -> None:
    parent.set(attribute, str(len(_children(parent, child_name))))


def _element_with_id(
    parent: etree._Element,
    child_name: str,
    element_id: int,
) -> etree._Element:
    matches = [
        child for child in _children(parent, child_name) if child.get("id") == str(element_id)
    ]
    if len(matches) != 1:
        raise PlanCompilationError(
            f"canonical header must contain exactly one {child_name} id {element_id}"
        )
    return matches[0]


def _next_element_id(
    parent: etree._Element,
    child_name: str,
    label: str,
) -> int:
    ids: list[int] = []
    for child in _children(parent, child_name):
        value = child.get("id")
        try:
            parsed = int(value) if value is not None else -1
        except ValueError as exc:
            raise PlanCompilationError(f"{label} contains an invalid id") from exc
        if parsed < 0:
            raise PlanCompilationError(f"{label} contains an invalid id")
        ids.append(parsed)
    if len(ids) != len(set(ids)):
        raise PlanCompilationError(f"{label} contains duplicate ids")
    return max(ids, default=-1) + 1


def _fontfaces_by_language(
    fontfaces: etree._Element,
) -> dict[str, etree._Element]:
    result: dict[str, etree._Element] = {}
    for fontface in _children(fontfaces, "fontface"):
        language = fontface.get("lang")
        if language is None or language in result:
            raise PlanCompilationError("fontfaces contains a missing or duplicate language")
        _assert_declared_count(
            fontface,
            "font",
            f"fontface {language}",
            attribute="fontCnt",
        )
        _next_element_id(fontface, "font", f"fontface {language}")
        result[language] = fontface
    expected = {language for language, _attribute in _FONT_LANGUAGES}
    if set(result) != expected:
        raise PlanCompilationError("canonical header has unsupported fontface languages")
    return result


def _resolve_font_refs(
    fontfaces: dict[str, etree._Element],
    base_char: etree._Element,
    intent: StyleIntent,
) -> tuple[tuple[str, int, str], ...]:
    base_refs = _single_child(base_char, "fontRef")
    resolved: list[tuple[str, int, str]] = []
    for language, attribute in _FONT_LANGUAGES:
        fontface = fontfaces[language]
        raw_base_font_id = base_refs.get(attribute)
        try:
            base_font_id = int(raw_base_font_id) if raw_base_font_id is not None else -1
        except ValueError as exc:
            raise PlanCompilationError("canonical fontRef contains an invalid id") from exc
        base_font = _element_with_id(fontface, "font", base_font_id)
        if intent.font_family is None:
            font_id = base_font_id
            font = base_font
        else:
            matching = [
                font
                for font in _children(fontface, "font")
                if font.get("face") == intent.font_family
            ]
            if len(matching) > 1:
                raise PlanCompilationError(
                    f"fontface {language} contains duplicate font family names"
                )
            if matching:
                font = matching[0]
                raw_font_id = font.get("id")
                try:
                    font_id = int(raw_font_id) if raw_font_id is not None else -1
                except ValueError as exc:
                    raise PlanCompilationError(
                        f"fontface {language} contains an invalid font id"
                    ) from exc
            else:
                font_id = _next_element_id(fontface, "font", f"fontface {language}")
                font = deepcopy(base_font)
                font.set("id", str(font_id))
                font.set("face", intent.font_family)
                font.set("type", "TTF")
                font.set("isEmbedded", "0")
                font.attrib.pop("binaryItemIDRef", None)
                fontface.append(font)
        face = font.get("face")
        if font_id < 0 or face is None:
            raise PlanCompilationError(f"fontface {language} contains an invalid font")
        resolved.append((attribute, font_id, face))
    return tuple(resolved)


def _single_child(parent: etree._Element, local_name: str) -> etree._Element:
    children = _children(parent, local_name)
    if len(children) != 1:
        raise PlanCompilationError(
            f"canonical {etree.QName(parent).localname} must contain one {local_name}"
        )
    return children[0]


def _materialize_char_properties(
    char_pr: etree._Element,
    intent: StyleIntent,
    font_refs: tuple[tuple[str, int, str], ...],
) -> int:
    if intent.font_size_pt is None:
        raw_height = char_pr.get("height")
        try:
            height = int(raw_height) if raw_height is not None else -1
        except ValueError as exc:
            raise PlanCompilationError("canonical charPr contains an invalid height") from exc
        if height <= 0:
            raise PlanCompilationError("canonical charPr contains an invalid height")
    else:
        height = _exact_scaled_integer(
            f"font_size_pt for style {intent.id}",
            intent.font_size_pt,
            100,
        )
        char_pr.set("height", str(height))

    font_ref = _single_child(char_pr, "fontRef")
    for attribute, font_id, _face in font_refs:
        font_ref.set(attribute, str(font_id))
    for child_name in ("italic", "bold"):
        for child in _children(char_pr, child_name):
            char_pr.remove(child)
    if intent.italic:
        etree.SubElement(char_pr, f"{{{HH}}}italic")
    if intent.bold:
        etree.SubElement(char_pr, f"{{{HH}}}bold")
    return height


def _materialize_para_properties(para_pr: etree._Element, intent: StyleIntent) -> int:
    align = _single_child(para_pr, "align")
    align.set("horizontal", _ALIGNMENTS[intent.alignment])
    line_spacing = _single_child(para_pr, "lineSpacing")
    line_spacing.set("type", "PERCENT")
    if intent.line_spacing is None:
        raw_value = line_spacing.get("value")
        try:
            value = int(raw_value) if raw_value is not None else -1
        except ValueError as exc:
            raise PlanCompilationError("canonical paraPr contains an invalid line spacing") from exc
        if value < 0:
            raise PlanCompilationError("canonical paraPr contains an invalid line spacing")
    else:
        value = _exact_scaled_integer(
            f"line_spacing for style {intent.id}",
            intent.line_spacing,
            100,
        )
        line_spacing.set("value", str(value))
    return value


def _exact_scaled_integer(label: str, value: float, scale: int) -> int:
    decimal_value = Decimal(str(value)) * scale
    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise PlanCompilationError(f"{label} cannot be represented exactly by the HWPX profile")
    return int(decimal_value)


def _compile_paragraphs(
    content: ContentIR,
    plan: HwpDocumentPlan,
    style_bindings: dict[str, _CompiledStyle],
    image_assets: ImageAssetBundle | None,
    page_layout: _CompiledPageLayout,
) -> tuple[
    list[RenderParagraph],
    tuple[_CompiledTable, ...],
    tuple[_CompiledImage, ...],
    tuple[str, ...],
]:
    _assert_supported_profiles(plan)

    content_by_id = {node.id: node for node in content.nodes}
    image_assets_by_ref = (
        {asset.asset_ref: asset for asset in image_assets.assets}
        if image_assets is not None
        else {}
    )
    paragraphs: list[RenderParagraph] = []
    tables: list[_CompiledTable] = []
    images: list[_CompiledImage] = []
    preview_lines: list[str] = []
    image_binary_by_sha256: dict[str, tuple[str, str, bytes]] = {}
    page_break_pending = False
    object_ordinal = 0
    total_text_bytes = 0
    total_table_text_controls = 0
    for item in plan.flow:
        if isinstance(item, ColumnBreakPlanItem):
            raise PlanCompilationError("column breaks are unsupported by the compiler")
        if isinstance(item, PageBreakPlanItem):
            if not paragraphs:
                raise PlanCompilationError("a leading page break is unsupported")
            if page_break_pending:
                raise PlanCompilationError("consecutive page breaks are unsupported")
            page_break_pending = True
            continue
        if not isinstance(item, ContentPlanItem):
            raise PlanCompilationError(f"unsupported Plan flow item: {type(item).__name__}")

        node = content_by_id[item.content_ref]
        if not isinstance(node, ImageContentNode) and item.layout != LayoutIntent():
            raise PlanCompilationError(
                f"layout intent is unsupported by the canonical compiler: {item.id}"
            )
        if isinstance(node, TableContentNode):
            if plan.design_profile_id not in {
                TABLE_DESIGN_PROFILE_ID,
                IMAGE_DESIGN_PROFILE_ID,
            }:
                raise PlanCompilationError(f"unsupported content kind for {node.id}: {node.kind}")
            if item.render_as != "table":
                raise PlanCompilationError(f"unsupported renderer for {node.id}: {item.render_as}")
            if item.style_ref is not None:
                raise PlanCompilationError(f"table style refs are unsupported: {item.id}")
            for cell in node.cells:
                if len(cell.text) > _MAX_TABLE_CELL_TEXT_UTF8_BYTES:
                    raise PlanCompilationError("table cell text exceeds the compiler size limit")
                cell_text_bytes = _assert_xml_text(
                    f"{node.id}[{cell.row},{cell.column}]",
                    cell.text,
                )
                if cell.text and all(character.isspace() for character in cell.text):
                    raise PlanCompilationError(
                        "whitespace-only table cell text cannot be preserved by projection: "
                        f"{node.id}[{cell.row},{cell.column}]"
                    )
                if not unicodedata.is_normalized("NFC", cell.text):
                    raise PlanCompilationError(
                        "table cell text must be NFC-normalized for exact projection: "
                        f"{node.id}[{cell.row},{cell.column}]"
                    )
                if cell_text_bytes > _MAX_TABLE_CELL_TEXT_UTF8_BYTES:
                    raise PlanCompilationError("table cell text exceeds the compiler size limit")
                total_text_bytes += cell_text_bytes
                if total_text_bytes > _MAX_TEXT_UTF8_BYTES:
                    raise PlanCompilationError("text content exceeds the compiler size limit")
                total_table_text_controls += sum(
                    character in _HWP_SPECIAL_TEXT_CHARACTERS for character in cell.text
                )
                if total_table_text_controls > _MAX_TABLE_TEXT_CONTROLS:
                    raise PlanCompilationError(
                        "table text control count exceeds the compiler limit"
                    )
            table_ordinal = len(tables) + 1
            paragraphs.append(RenderParagraph("", page_break=page_break_pending))
            tables.append(
                _CompiledTable(
                    node=node,
                    paragraph_index=len(paragraphs) - 1,
                    control_id=_TABLE_CONTROL_ID_BASE + table_ordinal,
                    z_order=object_ordinal,
                    width_hwp=page_layout.column_width_hwp,
                )
            )
            object_ordinal += 1
            preview_lines.append(_table_preview_text(node))
        elif isinstance(node, ImageContentNode):
            if plan.design_profile_id != IMAGE_DESIGN_PROFILE_ID:
                raise PlanCompilationError(f"unsupported content kind for {node.id}: {node.kind}")
            if item.render_as != "image":
                raise PlanCompilationError(f"unsupported renderer for {node.id}: {item.render_as}")
            if item.style_ref is not None:
                raise PlanCompilationError(f"image style refs are unsupported: {item.id}")
            width_hwp, height_hwp = _compile_image_dimensions(
                item,
                node,
                image_assets_by_ref,
                max_width_hwp=page_layout.column_width_hwp,
                max_height_hwp=page_layout.usable_height_hwp,
            )
            asset = image_assets_by_ref[node.asset_ref]
            binary = image_binary_by_sha256.get(asset.sha256)
            if binary is None:
                binary_ordinal = len(image_binary_by_sha256) + 1
                binary = (
                    f"image{binary_ordinal}",
                    f"BinData/image{binary_ordinal}.png",
                    asset.payload,
                )
                image_binary_by_sha256[asset.sha256] = binary
            elif binary[2] != asset.payload:
                raise PlanCompilationError("distinct image payloads share the same SHA-256 digest")
            image_ordinal = len(images)
            control_id = _IMAGE_CONTROL_ID_BASE + image_ordinal * 2
            paragraphs.append(
                RenderParagraph(
                    "",
                    page_break=page_break_pending,
                    char_pr_id=0,
                )
            )
            images.append(
                _CompiledImage(
                    node=node,
                    asset=asset,
                    paragraph_index=len(paragraphs) - 1,
                    control_id=control_id,
                    instance_id=control_id + 1,
                    z_order=object_ordinal,
                    binary_id=binary[0],
                    package_path=binary[1],
                    width_hwp=width_hwp,
                    height_hwp=height_hwp,
                )
            )
            object_ordinal += 1
            preview_lines.append(_IMAGE_PREVIEW_MARKER)
        elif isinstance(node, TextContentNode):
            if item.render_as != "paragraph":
                raise PlanCompilationError(f"unsupported renderer for {node.id}: {item.render_as}")
            if len(node.text) > _MAX_TEXT_UTF8_BYTES:
                raise PlanCompilationError("text content exceeds the compiler size limit")
            text_bytes = _assert_xml_text(node.id, node.text)
            if not any(not character.isspace() for character in node.text):
                raise PlanCompilationError(
                    f"whitespace-only text content cannot be preserved by projection: {node.id}"
                )
            if not unicodedata.is_normalized("NFC", node.text):
                raise PlanCompilationError(
                    f"text content must be NFC-normalized for exact projection: {node.id}"
                )
            if text_bytes > _MAX_TEXT_UTF8_BYTES:
                raise PlanCompilationError("text content exceeds the compiler size limit")
            total_text_bytes += text_bytes
            style = style_bindings.get(item.style_ref) if item.style_ref is not None else None
            if item.style_ref is not None and style is None:
                raise PlanCompilationError(f"unknown style ref: {item.style_ref}")
            paragraphs.append(
                RenderParagraph(
                    node.text,
                    page_break=page_break_pending,
                    char_pr_id=style.char_pr_id if style is not None else 3,
                    para_pr_id=style.para_pr_id if style is not None else 20,
                    style_id=style.style_id if style is not None else 0,
                )
            )
            preview_lines.append(node.text)
        else:
            raise PlanCompilationError(f"unsupported content kind for {node.id}: {node.kind}")

        if total_text_bytes > _MAX_TEXT_UTF8_BYTES:
            raise PlanCompilationError("text content exceeds the compiler size limit")
        page_break_pending = False
        if len(paragraphs) > _MAX_PARAGRAPHS:
            raise PlanCompilationError("paragraph count exceeds the compiler limit")

    if page_break_pending:
        raise PlanCompilationError("a trailing page break is unsupported")
    return paragraphs, tuple(tables), tuple(images), tuple(preview_lines)


def _compile_image_dimensions(
    item: ContentPlanItem,
    node: ImageContentNode,
    assets_by_ref: dict[str, PngImageAsset],
    *,
    max_width_hwp: int,
    max_height_hwp: int,
) -> tuple[int, int]:
    if item.layout.width_fraction is None:
        raise PlanCompilationError(f"image width_fraction is required: {item.id}")
    if item.layout.column_span != 1:
        raise PlanCompilationError(f"image column_span must be 1: {item.id}")
    if item.layout.keep_with_next:
        raise PlanCompilationError(f"image keep_with_next is unsupported: {item.id}")
    asset = assets_by_ref.get(node.asset_ref)
    if asset is None:
        raise PlanCompilationError(f"missing validated image asset: {node.asset_ref}")

    scaled_width = Decimal(str(item.layout.width_fraction)) * max_width_hwp
    if not scaled_width.is_finite():
        raise PlanCompilationError(f"image width_fraction is not finite: {item.id}")
    width_hwp = int(scaled_width.to_integral_value(rounding=ROUND_HALF_UP))
    if width_hwp <= 0 or width_hwp > max_width_hwp:
        raise PlanCompilationError(f"computed image width is unsupported: {item.id}")

    height_hwp = (width_hwp * asset.height_px + asset.width_px // 2) // asset.width_px
    if height_hwp <= 0:
        raise PlanCompilationError(f"computed image height is unsupported: {item.id}")
    if height_hwp > max_height_hwp:
        raise PlanCompilationError(f"image height exceeds the usable page region: {item.id}")
    return width_hwp, height_hwp


def _table_preview_text(table: TableContentNode) -> str:
    values = [["" for _column in range(table.columns)] for _row in range(table.rows)]
    for cell in table.cells:
        values[cell.row][cell.column] = cell.text
    return "\n".join("\t".join(row) for row in values)


def _materialize_images(
    entries: dict[str, bytes],
    images: tuple[_CompiledImage, ...],
) -> None:
    picture_template = _load_picture_template()
    binaries = _compiled_image_binaries(images)

    manifest_root = _parse_manifest(entries["Contents/content.hpf"])
    manifests = cast(
        list[etree._Element],
        manifest_root.xpath("./opf:manifest", namespaces={"opf": OPF}),
    )
    if len(manifests) != 1:
        raise PlanCompilationError("canonical OPF must contain exactly one manifest")
    manifest = manifests[0]
    existing_image_items = [
        cast(etree._Element, child)
        for child in manifest
        if child.get("id", "").startswith("image") or child.get("href", "").startswith("BinData/")
    ]
    if existing_image_items or any(name.startswith("BinData/") for name in entries):
        raise PlanCompilationError("canonical package contains unexpected image resources")
    section_items = [
        (index, cast(etree._Element, child))
        for index, child in enumerate(manifest)
        if child.get("id") == "section0"
    ]
    if len(section_items) != 1:
        raise PlanCompilationError("canonical OPF must contain one section0 item")
    section_index = section_items[0][0]
    for offset, binary in enumerate(binaries):
        item = etree.Element(f"{{{OPF}}}item")
        item.attrib.update(
            {
                "id": binary.binary_id,
                "href": binary.package_path,
                "media-type": "image/png",
                "isEmbeded": "1",
            }
        )
        manifest.insert(section_index + offset, item)
        entries[binary.package_path] = binary.asset.payload

    section = _parse_section(entries["Contents/section0.xml"])
    paragraphs = cast(
        list[etree._Element],
        section.xpath("./hp:p", namespaces={"hp": HP}),
    )
    for image in images:
        if image.paragraph_index >= len(paragraphs):
            raise PlanCompilationError("image anchor is outside the compiled section")
        anchor = paragraphs[image.paragraph_index]
        runs = cast(
            list[etree._Element],
            anchor.xpath("./hp:run", namespaces={"hp": HP}),
        )
        if not runs or len(runs[-1]) or runs[-1].text:
            raise PlanCompilationError("compiled image anchor has an unsupported structure")
        runs[-1].append(_build_picture(picture_template, image))
        etree.SubElement(runs[-1], f"{{{HP}}}t")

    manifest_payload = _serialize_xml(manifest_root)
    section_payload = _serialize_xml(section)
    if len(manifest_payload) > _MAX_MANIFEST_XML_BYTES:
        raise PlanCompilationError("compiled OPF manifest exceeds the compiler size limit")
    if len(section_payload) > _MAX_SECTION_XML_BYTES:
        raise PlanCompilationError("compiled section XML exceeds the compiler size limit")
    entries["Contents/content.hpf"] = manifest_payload
    entries["Contents/section0.xml"] = section_payload


def _compiled_image_binaries(
    images: tuple[_CompiledImage, ...],
) -> tuple[_CompiledImageBinary, ...]:
    binaries: dict[str, _CompiledImageBinary] = {}
    for image in images:
        current = binaries.get(image.binary_id)
        candidate = _CompiledImageBinary(
            binary_id=image.binary_id,
            package_path=image.package_path,
            asset=image.asset,
        )
        if current is None:
            binaries[image.binary_id] = candidate
        elif (
            current.package_path != candidate.package_path
            or current.asset.sha256 != candidate.asset.sha256
            or current.asset.payload != candidate.asset.payload
        ):
            raise PlanCompilationError("compiled image binary binding is inconsistent")
    return tuple(binaries.values())


def _load_picture_template() -> etree._Element:
    try:
        encoded = _IMAGE_TEMPLATE.read_bytes()
    except OSError as exc:
        raise PlanCompilationError(f"cannot read the canonical image template: {exc}") from exc
    if len(encoded) > _MAX_IMAGE_TEMPLATE_B64_BYTES:
        raise PlanCompilationError("canonical image template exceeds the compiler size limit")
    try:
        package = b64decode(b"".join(encoded.split()), validate=True)
    except (Base64Error, ValueError) as exc:
        raise PlanCompilationError("canonical image template is not valid base64") from exc
    if hashlib.sha256(package).hexdigest() != _IMAGE_TEMPLATE_PACKAGE_SHA256:
        raise PlanCompilationError(
            "canonical image template changed without a compiler version update"
        )
    try:
        with zipfile.ZipFile(BytesIO(package)) as archive:
            section_payload = archive.read("Contents/section0.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise PlanCompilationError("canonical image template package is invalid") from exc

    section = _parse_section(section_payload)
    pictures = cast(
        list[etree._Element],
        section.xpath(".//hp:pic", namespaces={"hp": HP}),
    )
    if len(pictures) != 8:
        raise PlanCompilationError("canonical image template has unsupported picture resources")
    canonical = etree.tostring(pictures[0], method="c14n", with_comments=False)
    if hashlib.sha256(canonical).hexdigest() != _IMAGE_TEMPLATE_PICTURE_C14N_SHA256:
        raise PlanCompilationError(
            "canonical image picture changed without a compiler version update"
        )
    return pictures[0]


def _build_picture(
    template: etree._Element,
    compiled: _CompiledImage,
) -> etree._Element:
    picture = deepcopy(template)
    picture.attrib.clear()
    picture.attrib.update(
        {
            "id": str(compiled.control_id),
            "zOrder": str(compiled.z_order),
            "numberingType": "PICTURE",
            "textWrap": "TOP_AND_BOTTOM",
            "textFlow": "BOTH_SIDES",
            "lock": "0",
            "dropcapstyle": "None",
            "href": "",
            "groupLevel": "0",
            "instid": str(compiled.instance_id),
            "reverse": "0",
        }
    )
    _set_exact_attributes(_single_hp_child(picture, "offset"), {"x": "0", "y": "0"})
    size_attributes = {
        "width": str(compiled.width_hwp),
        "height": str(compiled.height_hwp),
    }
    _set_exact_attributes(_single_hp_child(picture, "orgSz"), size_attributes)
    _set_exact_attributes(_single_hp_child(picture, "curSz"), {"width": "0", "height": "0"})
    _set_exact_attributes(
        _single_hp_child(picture, "flip"),
        {"horizontal": "0", "vertical": "0"},
    )
    _set_exact_attributes(
        _single_hp_child(picture, "rotationInfo"),
        {
            "angle": "0",
            "centerX": str(compiled.width_hwp // 2),
            "centerY": str(compiled.height_hwp // 2),
            "rotateimage": "1",
        },
    )
    image_rectangle = _single_hp_child(picture, "imgRect")
    points = cast(
        list[etree._Element],
        image_rectangle.xpath("./hc:*", namespaces={"hc": HC}),
    )
    coordinates = (
        ("pt0", 0, 0),
        ("pt1", compiled.width_hwp, 0),
        ("pt2", compiled.width_hwp, compiled.height_hwp),
        ("pt3", 0, compiled.height_hwp),
    )
    if len(points) != len(coordinates):
        raise PlanCompilationError("canonical image template has an invalid image rectangle")
    for point, (name, x, y) in zip(points, coordinates, strict=True):
        qname = etree.QName(point)
        if qname.namespace != HC or qname.localname != name:
            raise PlanCompilationError("canonical image template has an invalid image rectangle")
        _set_exact_attributes(point, {"x": str(x), "y": str(y)})
    _set_exact_attributes(
        _single_hp_child(picture, "imgClip"),
        {
            "left": "0",
            "right": str(compiled.asset.width_px * _IMAGE_CLIP_UNITS_PER_PIXEL),
            "top": "0",
            "bottom": str(compiled.asset.height_px * _IMAGE_CLIP_UNITS_PER_PIXEL),
        },
    )
    margin_attributes = {"left": "0", "right": "0", "top": "0", "bottom": "0"}
    _set_exact_attributes(_single_hp_child(picture, "inMargin"), margin_attributes)
    image_nodes = cast(
        list[etree._Element],
        picture.xpath("./hc:img", namespaces={"hc": HC}),
    )
    if len(image_nodes) != 1:
        raise PlanCompilationError("canonical image template must contain one image binding")
    _set_exact_attributes(
        image_nodes[0],
        {
            "binaryItemIDRef": compiled.binary_id,
            "bright": "0",
            "contrast": "0",
            "effect": "REAL_PIC",
            "alpha": "0",
        },
    )
    _set_exact_attributes(
        _single_hp_child(picture, "sz"),
        {
            "width": str(compiled.width_hwp),
            "widthRelTo": "ABSOLUTE",
            "height": str(compiled.height_hwp),
            "heightRelTo": "ABSOLUTE",
            "protect": "0",
        },
    )
    _set_exact_attributes(
        _single_hp_child(picture, "pos"),
        {
            "treatAsChar": "1",
            "affectLSpacing": "0",
            "flowWithText": "1",
            "allowOverlap": "0",
            "holdAnchorAndSO": "0",
            "vertRelTo": "PARA",
            "horzRelTo": "COLUMN",
            "vertAlign": "TOP",
            "horzAlign": "LEFT",
            "vertOffset": "0",
            "horzOffset": "0",
        },
    )
    _set_exact_attributes(_single_hp_child(picture, "outMargin"), margin_attributes)
    comments = _hp_children(picture, "shapeComment")
    if len(comments) != 1:
        raise PlanCompilationError("canonical image template must contain one shape comment")
    comments[0].text = _IMAGE_PREVIEW_MARKER
    return picture


def _set_exact_attributes(element: etree._Element, attributes: dict[str, str]) -> None:
    element.attrib.clear()
    element.attrib.update(attributes)


def _parse_manifest(payload: bytes) -> etree._Element:
    if len(payload) > _MAX_MANIFEST_XML_BYTES:
        raise PlanCompilationError("OPF manifest exceeds the compiler size limit")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise PlanCompilationError("OPF manifest contains a forbidden declaration")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=False,
    )
    try:
        root = cast(etree._Element, etree.fromstring(payload, parser=parser))
    except etree.XMLSyntaxError as exc:
        raise PlanCompilationError(f"OPF manifest is malformed: {exc}") from exc
    name = etree.QName(root)
    if name.namespace != OPF or name.localname != "package":
        raise PlanCompilationError("OPF manifest uses an unsupported root element")
    return root


def _materialize_tables(
    entries: dict[str, bytes],
    tables: tuple[_CompiledTable, ...],
) -> None:
    table_template, border_fill = _load_table_parts()

    header = _parse_header(entries["Contents/header.xml"])
    border_fills = _header_collection(header, "borderFills")
    _assert_declared_count(border_fills, "borderFill", "borderFills")
    if _next_element_id(border_fills, "borderFill", "borderFills") != 3:
        raise PlanCompilationError("canonical header has an unsupported border-fill table")
    border_fills.append(deepcopy(border_fill))
    _set_declared_count(border_fills, "borderFill")

    section = _parse_section(entries["Contents/section0.xml"])
    paragraphs = cast(
        list[etree._Element],
        section.xpath("./hp:p", namespaces={"hp": HP}),
    )
    for table in tables:
        if table.paragraph_index >= len(paragraphs):
            raise PlanCompilationError("table anchor is outside the compiled section")
        anchor = paragraphs[table.paragraph_index]
        runs = cast(
            list[etree._Element],
            anchor.xpath("./hp:run", namespaces={"hp": HP}),
        )
        if not runs or anchor.xpath("./hp:run/hp:tbl", namespaces={"hp": HP}):
            raise PlanCompilationError("compiled table anchor has an unsupported structure")
        runs[-1].append(_build_table(table_template, table))

    header_payload = _serialize_xml(header)
    section_payload = _serialize_xml(section)
    if len(header_payload) > _MAX_HEADER_XML_BYTES:
        raise PlanCompilationError("compiled header XML exceeds the compiler size limit")
    if len(section_payload) > _MAX_SECTION_XML_BYTES:
        raise PlanCompilationError("compiled section XML exceeds the compiler size limit")
    entries["Contents/header.xml"] = header_payload
    entries["Contents/section0.xml"] = section_payload


def _load_table_parts() -> tuple[etree._Element, etree._Element]:
    try:
        encoded = _TABLE_TEMPLATE.read_bytes()
    except OSError as exc:
        raise PlanCompilationError(f"cannot read the canonical table template: {exc}") from exc
    if len(encoded) > _MAX_TABLE_TEMPLATE_B64_BYTES:
        raise PlanCompilationError("canonical table template exceeds the compiler size limit")
    try:
        package = b64decode(b"".join(encoded.split()), validate=True)
    except (Base64Error, ValueError) as exc:
        raise PlanCompilationError("canonical table template is not valid base64") from exc
    if hashlib.sha256(package).hexdigest() != _TABLE_TEMPLATE_PACKAGE_SHA256:
        raise PlanCompilationError(
            "canonical table template changed without a compiler version update"
        )
    try:
        with zipfile.ZipFile(BytesIO(package)) as archive:
            section_payload = archive.read("Contents/section0.xml")
            header_payload = archive.read("Contents/header.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise PlanCompilationError("canonical table template package is invalid") from exc

    section = _parse_section(section_payload)
    header = _parse_header(header_payload)
    table_nodes = cast(
        list[etree._Element],
        section.xpath(".//hp:tbl", namespaces={"hp": HP}),
    )
    border_nodes = cast(
        list[etree._Element],
        header.xpath(
            "./hh:refList/hh:borderFills/hh:borderFill[@id='3']",
            namespaces={"hh": HH},
        ),
    )
    if len(table_nodes) != 1 or len(border_nodes) != 1:
        raise PlanCompilationError("canonical table template has unsupported resources")
    return table_nodes[0], border_nodes[0]


def _parse_section(payload: bytes) -> etree._Element:
    if len(payload) > _MAX_SECTION_XML_BYTES:
        raise PlanCompilationError("section XML exceeds the compiler size limit")
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise PlanCompilationError("section XML contains a forbidden declaration")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=False,
    )
    try:
        root = cast(etree._Element, etree.fromstring(payload, parser=parser))
    except etree.XMLSyntaxError as exc:
        raise PlanCompilationError(f"section XML is malformed: {exc}") from exc
    if etree.QName(root).localname != "sec":
        raise PlanCompilationError("section XML uses an unsupported root element")
    return root


def _build_table(template: etree._Element, compiled: _CompiledTable) -> etree._Element:
    table = deepcopy(template)
    rows = _hp_children(table, "tr")
    if not rows:
        raise PlanCompilationError("canonical table template has no rows")
    source_cells = _hp_children(rows[0], "tc")
    if not source_cells:
        raise PlanCompilationError("canonical table template has no cells")
    cell_template = source_cells[0]
    for row in rows:
        table.remove(row)

    table.set("id", str(compiled.control_id))
    table.set("zOrder", str(compiled.z_order))
    table.set("rowCnt", str(compiled.node.rows))
    table.set("colCnt", str(compiled.node.columns))
    table.set("textWrap", "SQUARE")
    table.set("repeatHeader", "0")
    size = _single_hp_child(table, "sz")
    size.set("width", str(compiled.width_hwp))
    size.set("height", "0")
    position = _single_hp_child(table, "pos")
    position.attrib.clear()
    position.attrib.update(
        {
            "treatAsChar": "1",
            "affectLSpacing": "0",
            "flowWithText": "1",
            "allowOverlap": "0",
            "holdAnchorAndSO": "0",
            "vertRelTo": "PARA",
            "horzRelTo": "PARA",
            "vertAlign": "TOP",
            "horzAlign": "LEFT",
            "vertOffset": "0",
            "horzOffset": "0",
        }
    )

    cells_by_row: dict[int, list[TableCell]] = {
        row_index: [] for row_index in range(compiled.node.rows)
    }
    for cell in sorted(compiled.node.cells, key=lambda item: (item.row, item.column)):
        cells_by_row[cell.row].append(cell)
    for row_index in range(compiled.node.rows):
        row = etree.Element(f"{{{HP}}}tr")
        for cell in cells_by_row[row_index]:
            row.append(
                _build_table_cell(
                    cell_template,
                    compiled.node.columns,
                    cell,
                    compiled.width_hwp,
                )
            )
        table.append(row)
    return table


def _build_table_cell(
    template: etree._Element,
    columns: int,
    source: TableCell,
    table_width_hwp: int,
) -> etree._Element:
    cell = deepcopy(template)
    address = _single_hp_child(cell, "cellAddr")
    address.set("rowAddr", str(source.row))
    address.set("colAddr", str(source.column))
    span = _single_hp_child(cell, "cellSpan")
    span.set("rowSpan", str(source.row_span))
    span.set("colSpan", str(source.column_span))
    cell_size = _single_hp_child(cell, "cellSz")
    left = _table_axis_offset(table_width_hwp, source.column, columns)
    right = _table_axis_offset(
        table_width_hwp,
        source.column + source.column_span,
        columns,
    )
    cell_size.set("width", str(right - left))
    cell_size.set("height", str(_TABLE_ROW_HEIGHT_HWP * source.row_span))

    sub_list = _single_hp_child(cell, "subList")
    paragraph = _single_hp_child(sub_list, "p")
    for child in list(paragraph):
        paragraph.remove(child)
    _append_text_run(paragraph, source.text, 0)
    return cell


def _table_axis_offset(total: int, index: int, parts: int) -> int:
    return total * index // parts


def _hp_children(parent: etree._Element, local_name: str) -> list[etree._Element]:
    tag = f"{{{HP}}}{local_name}"
    return [cast(etree._Element, child) for child in parent if child.tag == tag]


def _single_hp_child(parent: etree._Element, local_name: str) -> etree._Element:
    children = _hp_children(parent, local_name)
    if len(children) != 1:
        raise PlanCompilationError(
            f"canonical {etree.QName(parent).localname} must contain one {local_name}"
        )
    return children[0]


def _compile_page_layout(layout: PageLayoutIntent) -> _CompiledPageLayout:
    numeric_fields = (
        "width_mm",
        "height_mm",
        "margin_top_mm",
        "margin_right_mm",
        "margin_bottom_mm",
        "margin_left_mm",
        "column_gap_mm",
    )
    non_finite = [field for field in numeric_fields if not isfinite(getattr(layout, field))]
    if non_finite:
        raise PlanCompilationError(
            "page layout values must be finite: " + ", ".join(non_finite)
        )
    if not (
        isclose(
            layout.width_mm,
            _CANONICAL_PAGE_LAYOUT.width_mm,
            rel_tol=0.0,
            abs_tol=_LAYOUT_TOLERANCE_MM,
        )
        and isclose(
            layout.height_mm,
            _CANONICAL_PAGE_LAYOUT.height_mm,
            rel_tol=0.0,
            abs_tol=_LAYOUT_TOLERANCE_MM,
        )
        and layout.width_mm < layout.height_mm
    ):
        raise PlanCompilationError(
            "page layout must be A4 portrait within the supported tolerance"
        )
    if not 1 <= layout.columns <= 4:
        raise PlanCompilationError("page layout column count must be between 1 and 4")

    values = {field: Decimal(str(getattr(layout, field))) for field in numeric_fields}
    usable_width_mm = (
        values["width_mm"] - values["margin_left_mm"] - values["margin_right_mm"]
    )
    usable_height_mm = (
        values["height_mm"] - values["margin_top_mm"] - values["margin_bottom_mm"]
    )
    total_gap_mm = values["column_gap_mm"] * (layout.columns - 1)
    column_width_mm = (usable_width_mm - total_gap_mm) / layout.columns
    if usable_width_mm < _MIN_USABLE_LAYOUT_MM:
        raise PlanCompilationError("page layout usable width must be at least 1 mm")
    if usable_height_mm < _MIN_USABLE_LAYOUT_MM:
        raise PlanCompilationError("page layout usable height must be at least 1 mm")
    if values["column_gap_mm"] > usable_width_mm:
        raise PlanCompilationError("page layout column gap exceeds the usable width")
    if column_width_mm < _MIN_USABLE_LAYOUT_MM:
        raise PlanCompilationError("page layout column width must be at least 1 mm")

    if _is_canonical_page_layout(layout):
        return _CompiledPageLayout(
            width_hwp=_CANONICAL_PAGE_WIDTH_HWP,
            height_hwp=_CANONICAL_PAGE_HEIGHT_HWP,
            margin_top_hwp=_CANONICAL_MARGIN_TOP_HWP,
            margin_right_hwp=_CANONICAL_MARGIN_RIGHT_HWP,
            margin_bottom_hwp=_CANONICAL_MARGIN_BOTTOM_HWP,
            margin_left_hwp=_CANONICAL_MARGIN_LEFT_HWP,
            columns=_CANONICAL_PAGE_LAYOUT.columns,
            column_gap_hwp=_CANONICAL_COLUMN_GAP_HWP,
            column_width_hwp=_CANONICAL_COLUMN_WIDTH_HWP,
            usable_height_hwp=(
                _CANONICAL_PAGE_HEIGHT_HWP
                - _CANONICAL_MARGIN_TOP_HWP
                - _CANONICAL_MARGIN_BOTTOM_HWP
            ),
        )

    width_hwp = _mm_to_hwpunit("width_mm", layout.width_mm)
    height_hwp = _mm_to_hwpunit("height_mm", layout.height_mm)
    margin_top_hwp = _mm_to_hwpunit("margin_top_mm", layout.margin_top_mm)
    margin_right_hwp = _mm_to_hwpunit("margin_right_mm", layout.margin_right_mm)
    margin_bottom_hwp = _mm_to_hwpunit("margin_bottom_mm", layout.margin_bottom_mm)
    margin_left_hwp = _mm_to_hwpunit("margin_left_mm", layout.margin_left_mm)
    column_gap_hwp = _mm_to_hwpunit("column_gap_mm", layout.column_gap_mm)
    usable_width_hwp = width_hwp - margin_left_hwp - margin_right_hwp
    usable_height_hwp = height_hwp - margin_top_hwp - margin_bottom_hwp
    column_width_hwp = (
        usable_width_hwp - column_gap_hwp * (layout.columns - 1)
    ) // layout.columns
    minimum_hwp = _mm_to_hwpunit("minimum usable layout", float(_MIN_USABLE_LAYOUT_MM))
    if usable_width_hwp < minimum_hwp:
        raise PlanCompilationError("page layout usable width cannot be represented safely")
    if usable_height_hwp < minimum_hwp:
        raise PlanCompilationError("page layout usable height cannot be represented safely")
    if column_width_hwp < minimum_hwp:
        raise PlanCompilationError("page layout column width cannot be represented safely")
    return _CompiledPageLayout(
        width_hwp=width_hwp,
        height_hwp=height_hwp,
        margin_top_hwp=margin_top_hwp,
        margin_right_hwp=margin_right_hwp,
        margin_bottom_hwp=margin_bottom_hwp,
        margin_left_hwp=margin_left_hwp,
        columns=layout.columns,
        column_gap_hwp=column_gap_hwp,
        column_width_hwp=column_width_hwp,
        usable_height_hwp=usable_height_hwp,
    )


def _is_canonical_page_layout(layout: PageLayoutIntent) -> bool:
    numeric_fields = (
        "width_mm",
        "height_mm",
        "margin_top_mm",
        "margin_right_mm",
        "margin_bottom_mm",
        "margin_left_mm",
        "column_gap_mm",
    )
    return layout.columns == _CANONICAL_PAGE_LAYOUT.columns and all(
        isclose(
            getattr(layout, field),
            getattr(_CANONICAL_PAGE_LAYOUT, field),
            rel_tol=0.0,
            abs_tol=_LAYOUT_TOLERANCE_MM,
        )
        for field in numeric_fields
    )


def _mm_to_hwpunit(label: str, value: float) -> int:
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite() or decimal_value < 0:
        raise PlanCompilationError(f"page layout {label} is invalid")
    scaled = decimal_value * _HWPUNIT_PER_INCH / _MM_PER_INCH
    result = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    if value > 0 and result <= 0:
        raise PlanCompilationError(f"page layout {label} cannot be represented")
    return result


def _materialize_page_layout(
    entries: dict[str, bytes],
    layout: _CompiledPageLayout,
) -> None:
    section = _parse_section(entries["Contents/section0.xml"])
    page, margin, columns = _section_layout_nodes(section)
    expected_page = {
        "width": str(layout.width_hwp),
        "height": str(layout.height_hwp),
    }
    expected_margin = {
        "top": str(layout.margin_top_hwp),
        "right": str(layout.margin_right_hwp),
        "bottom": str(layout.margin_bottom_hwp),
        "left": str(layout.margin_left_hwp),
    }
    expected_columns = {
        "colCount": str(layout.columns),
        "sameGap": str(layout.column_gap_hwp),
    }
    if (
        all(page.get(name) == value for name, value in expected_page.items())
        and all(margin.get(name) == value for name, value in expected_margin.items())
        and all(columns.get(name) == value for name, value in expected_columns.items())
    ):
        return
    for name, value in expected_page.items():
        page.set(name, value)
    for name, value in expected_margin.items():
        margin.set(name, value)
    for name, value in expected_columns.items():
        columns.set(name, value)
    section_payload = _serialize_xml(section)
    if len(section_payload) > _MAX_SECTION_XML_BYTES:
        raise PlanCompilationError("compiled section XML exceeds the compiler size limit")
    entries["Contents/section0.xml"] = section_payload


def _section_layout_nodes(
    section: etree._Element,
) -> tuple[etree._Element, etree._Element, etree._Element]:
    pages = cast(
        list[etree._Element],
        section.xpath("./hp:p/hp:run/hp:secPr/hp:pagePr", namespaces={"hp": HP}),
    )
    columns = cast(
        list[etree._Element],
        section.xpath("./hp:p/hp:run/hp:ctrl/hp:colPr", namespaces={"hp": HP}),
    )
    if len(pages) != 1 or len(columns) != 1:
        raise PlanCompilationError("canonical section has an unsupported page layout structure")
    page = pages[0]
    margin = _single_hp_child(page, "margin")
    fixed_page = {"landscape": "WIDELY", "gutterType": "LEFT_ONLY"}
    fixed_margin = {
        "header": str(_CANONICAL_HEADER_HWP),
        "footer": str(_CANONICAL_FOOTER_HWP),
        "gutter": str(_CANONICAL_GUTTER_HWP),
    }
    fixed_columns = {
        "id": "",
        "type": "NEWSPAPER",
        "layout": "LEFT",
        "sameSz": "1",
    }
    if (
        any(page.get(name) != value for name, value in fixed_page.items())
        or any(margin.get(name) != value for name, value in fixed_margin.items())
        or any(columns[0].get(name) != value for name, value in fixed_columns.items())
    ):
        raise PlanCompilationError("canonical section has unsupported fixed layout properties")
    return page, margin, columns[0]


def _assert_xml_text(node_id: str, text: str) -> int:
    utf8_bytes = 0
    for character in text:
        codepoint = ord(character)
        if not _is_xml_10_character(codepoint):
            raise PlanCompilationError(f"text contains an XML 1.0-forbidden character: {node_id}")
        utf8_bytes += (
            1 if codepoint <= 0x7F else 2 if codepoint <= 0x7FF else 3 if codepoint <= 0xFFFF else 4
        )
    if "\r" in text:
        raise PlanCompilationError(
            f"text contains an unsupported carriage return; use LF line breaks: {node_id}"
        )
    return utf8_bytes


def _assert_xml_attribute(label: str, value: str) -> None:
    _assert_xml_text(label, value)
    if any(character in "\t\n\r" for character in value):
        raise PlanCompilationError(
            f"{label} contains whitespace that XML attributes would normalize"
        )


def _is_xml_10_character(codepoint: int) -> bool:
    return (
        codepoint in (0x09, 0x0A, 0x0D)
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _deterministic_package(entries: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
        names = ["mimetype", *sorted(name for name in entries if name != "mimetype")]
        for name in names:
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0o600 << 16
            archive.writestr(info, entries[name])
    return stream.getvalue()


def _verify_candidate_digest(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PlanCompilationError("cannot verify the compiled HWPX candidate") from exc
    if digest.hexdigest() != expected_sha256:
        raise PlanCompilationError("compiled HWPX candidate changed during validation")


def _assert_template_version(entries: dict[str, bytes]) -> None:
    digest = hashlib.sha256()
    for name in sorted(entries):
        encoded_name = name.encode("utf-8")
        payload = entries[name]
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    if digest.hexdigest() != _CANONICAL_TEMPLATE_ENTRIES_SHA256:
        raise PlanCompilationError(
            "canonical HWPX template changed without a compiler version update"
        )


def _verify_compiled_content(
    path: Path,
    expected: list[RenderParagraph],
    expected_tables: tuple[_CompiledTable, ...],
    expected_images: tuple[_CompiledImage, ...],
    expected_preview_lines: tuple[str, ...],
    expected_styles: tuple[_CompiledStyle, ...],
    expected_layout: _CompiledPageLayout,
) -> None:
    with zipfile.ZipFile(path) as archive:
        header_payload = archive.read("Contents/header.xml")
        section_payload = archive.read("Contents/section0.xml")
        preview = archive.read("Preview/PrvText.txt")
    section = _parse_section(section_payload)
    nodes = cast(
        list[etree._Element],
        section.xpath("./hp:p", namespaces={"hp": HP}),
    )
    actual_text = tuple(
        "".join(
            _compiled_text_value(text_node)
            for text_node in cast(
                list[etree._Element],
                node.xpath("./hp:run/hp:t", namespaces={"hp": HP}),
            )
        )
        for node in nodes
    )
    expected_text = tuple(paragraph.text for paragraph in expected)
    if actual_text != expected_text:
        raise PlanCompilationError("compiled HWPX failed content-conservation validation")
    actual_breaks = tuple(node.get("pageBreak") == "1" for node in nodes)
    expected_breaks = tuple(paragraph.page_break for paragraph in expected)
    if actual_breaks != expected_breaks:
        raise PlanCompilationError("compiled HWPX failed page-break validation")
    actual_para_refs = tuple(node.get("paraPrIDRef") for node in nodes)
    expected_para_refs = tuple(str(paragraph.para_pr_id) for paragraph in expected)
    if actual_para_refs != expected_para_refs:
        raise PlanCompilationError("compiled HWPX failed paragraph-style validation")
    actual_style_refs = tuple(node.get("styleIDRef") for node in nodes)
    expected_style_refs = tuple(str(paragraph.style_id) for paragraph in expected)
    if actual_style_refs != expected_style_refs:
        raise PlanCompilationError("compiled HWPX failed named-style validation")
    actual_char_refs: list[str | None] = []
    for node in nodes:
        runs = cast(
            list[etree._Element],
            node.xpath("./hp:run", namespaces={"hp": HP}),
        )
        if not runs:
            raise PlanCompilationError("compiled HWPX paragraph has no text run")
        actual_char_refs.append(runs[-1].get("charPrIDRef"))
    expected_char_refs = tuple(str(paragraph.char_pr_id) for paragraph in expected)
    if tuple(actual_char_refs) != expected_char_refs:
        raise PlanCompilationError("compiled HWPX failed character-style validation")
    expected_preview = "\n".join(expected_preview_lines).encode("utf-8")
    if preview != expected_preview:
        raise PlanCompilationError("compiled HWPX failed preview-text validation")
    header = _parse_header(header_payload)
    _verify_compiled_page_layout(section, expected_layout)
    _verify_compiled_tables(section, header, nodes, expected_tables)
    _verify_compiled_images(path, section, nodes, expected_images)
    _verify_control_identity(section)
    _verify_materialized_styles(header, expected_styles)


def _verify_compiled_page_layout(
    section: etree._Element,
    expected: _CompiledPageLayout,
) -> None:
    page, margin, columns = _section_layout_nodes(section)
    expected_page = {
        "width": str(expected.width_hwp),
        "height": str(expected.height_hwp),
    }
    expected_margin = {
        "top": str(expected.margin_top_hwp),
        "right": str(expected.margin_right_hwp),
        "bottom": str(expected.margin_bottom_hwp),
        "left": str(expected.margin_left_hwp),
    }
    expected_columns = {
        "colCount": str(expected.columns),
        "sameGap": str(expected.column_gap_hwp),
    }
    if (
        any(page.get(name) != value for name, value in expected_page.items())
        or any(margin.get(name) != value for name, value in expected_margin.items())
        or any(columns.get(name) != value for name, value in expected_columns.items())
    ):
        raise PlanCompilationError("compiled HWPX failed page-layout validation")


def _verify_compiled_images(
    path: Path,
    section: etree._Element,
    paragraphs: list[etree._Element],
    expected_images: tuple[_CompiledImage, ...],
) -> None:
    expected_binaries = _compiled_image_binaries(expected_images)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            manifest_payload = archive.read("Contents/content.hpf")
            actual_payloads = {
                name: archive.read(name) for name in names if name.startswith("BinData/")
            }
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise PlanCompilationError("compiled HWPX failed image-package validation") from exc

    expected_paths = {binary.package_path for binary in expected_binaries}
    if set(actual_payloads) != expected_paths:
        raise PlanCompilationError("compiled HWPX failed image-entry validation")
    for binary in expected_binaries:
        payload = actual_payloads[binary.package_path]
        if (
            payload != binary.asset.payload
            or hashlib.sha256(payload).hexdigest() != binary.asset.sha256
        ):
            raise PlanCompilationError("compiled HWPX failed image-payload validation")

    manifest_root = _parse_manifest(manifest_payload)
    manifests = cast(
        list[etree._Element],
        manifest_root.xpath("./opf:manifest", namespaces={"opf": OPF}),
    )
    if len(manifests) != 1:
        raise PlanCompilationError("compiled HWPX failed image-manifest validation")
    image_items = [
        cast(etree._Element, child)
        for child in manifests[0]
        if child.get("id", "").startswith("image")
        or child.get("href", "").startswith("BinData/")
        or child.get("media-type", "").startswith("image/")
    ]
    expected_items = [
        {
            "id": binary.binary_id,
            "href": binary.package_path,
            "media-type": "image/png",
            "isEmbeded": "1",
        }
        for binary in expected_binaries
    ]
    if [dict(item.attrib) for item in image_items] != expected_items:
        raise PlanCompilationError("compiled HWPX failed image-manifest validation")

    actual_pictures = cast(
        list[etree._Element],
        section.xpath(".//hp:pic", namespaces={"hp": HP}),
    )
    if len(actual_pictures) != len(expected_images):
        raise PlanCompilationError("compiled HWPX failed image-count validation")
    picture_template = _load_picture_template() if expected_images else None
    anchored_pictures: list[etree._Element] = []
    for expected in expected_images:
        if expected.paragraph_index >= len(paragraphs):
            raise PlanCompilationError("compiled HWPX failed image-anchor validation")
        anchor_runs = cast(
            list[etree._Element],
            paragraphs[expected.paragraph_index].xpath(
                "./hp:run[hp:pic]",
                namespaces={"hp": HP},
            ),
        )
        if len(anchor_runs) != 1:
            raise PlanCompilationError("compiled HWPX failed image-anchor validation")
        children = list(anchor_runs[0])
        if (
            len(children) != 2
            or etree.QName(children[0]).namespace != HP
            or etree.QName(children[0]).localname != "pic"
            or etree.QName(children[1]).namespace != HP
            or etree.QName(children[1]).localname != "t"
            or children[1].attrib
            or children[1].text
            or len(children[1])
        ):
            raise PlanCompilationError("compiled HWPX failed image-run validation")
        picture = children[0]
        anchored_pictures.append(picture)
        if picture_template is None:
            raise PlanCompilationError("compiled HWPX failed image-template validation")
        expected_picture = _build_picture(picture_template, expected)
        actual_xml = _canonical_control_xml(picture)
        expected_xml = _canonical_control_xml(expected_picture)
        if actual_xml != expected_xml:
            raise PlanCompilationError("compiled HWPX failed image-control validation")
    if any(actual is not anchored for actual, anchored in zip(actual_pictures, anchored_pictures)):
        raise PlanCompilationError("compiled HWPX failed image-order validation")


def _verify_control_identity(section: etree._Element) -> None:
    controls = cast(
        list[etree._Element],
        section.xpath(".//hp:tbl | .//hp:pic", namespaces={"hp": HP}),
    )
    control_ids = [control.get("id") for control in controls]
    z_orders = [control.get("zOrder") for control in controls]
    if (
        any(value is None for value in control_ids)
        or len(set(control_ids)) != len(control_ids)
        or any(value is None for value in z_orders)
        or len(set(z_orders)) != len(z_orders)
    ):
        raise PlanCompilationError("compiled HWPX failed control-identity validation")
    try:
        parsed_z_orders = sorted(int(cast(str, value)) for value in z_orders)
    except ValueError as exc:
        raise PlanCompilationError("compiled HWPX failed control-identity validation") from exc
    if parsed_z_orders != list(range(len(controls))):
        raise PlanCompilationError("compiled HWPX failed control-order validation")
    pictures = [control for control in controls if etree.QName(control).localname == "pic"]
    instance_ids = [picture.get("instid") for picture in pictures]
    if (
        any(value is None for value in instance_ids)
        or len(set(instance_ids)) != len(instance_ids)
        or not set(instance_ids).isdisjoint(control_ids)
    ):
        raise PlanCompilationError("compiled HWPX failed control-instance validation")


def _canonical_control_xml(element: etree._Element) -> bytes:
    detached = deepcopy(element)
    detached.tail = None
    return cast(
        bytes,
        etree.tostring(
            detached,
            method="c14n",
            exclusive=True,
            with_comments=False,
        ),
    )


def _verify_compiled_tables(
    section: etree._Element,
    header: etree._Element,
    paragraphs: list[etree._Element],
    expected_tables: tuple[_CompiledTable, ...],
) -> None:
    actual_tables = cast(
        list[etree._Element],
        section.xpath(".//hp:tbl", namespaces={"hp": HP}),
    )
    if len(actual_tables) != len(expected_tables):
        raise PlanCompilationError("compiled HWPX failed table-count validation")
    if not expected_tables:
        return

    border_fills = _header_collection(header, "borderFills")
    _assert_declared_count(border_fills, "borderFill", "borderFills")
    compiled_borders = [
        border for border in _children(border_fills, "borderFill") if border.get("id") == "3"
    ]
    if len(compiled_borders) != 1:
        raise PlanCompilationError("compiled HWPX failed table-border-ref validation")
    _template, expected_border = _load_table_parts()
    actual_border_xml = etree.tostring(compiled_borders[0], method="c14n", with_comments=False)
    expected_border_xml = etree.tostring(expected_border, method="c14n", with_comments=False)
    if actual_border_xml != expected_border_xml:
        raise PlanCompilationError("compiled HWPX failed table-border validation")

    for expected in expected_tables:
        anchor_tables = cast(
            list[etree._Element],
            paragraphs[expected.paragraph_index].xpath(
                "./hp:run/hp:tbl",
                namespaces={"hp": HP},
            ),
        )
        if len(anchor_tables) != 1:
            raise PlanCompilationError("compiled HWPX failed table-anchor validation")
        table = anchor_tables[0]
        expected_attributes = {
            "id": str(expected.control_id),
            "zOrder": str(expected.z_order),
            "numberingType": "TABLE",
            "textWrap": "SQUARE",
            "textFlow": "BOTH_SIDES",
            "lock": "0",
            "dropcapstyle": "None",
            "pageBreak": "CELL",
            "repeatHeader": "0",
            "rowCnt": str(expected.node.rows),
            "colCnt": str(expected.node.columns),
            "cellSpacing": "0",
            "borderFillIDRef": "3",
            "noAdjust": "0",
        }
        if any(table.get(name) != value for name, value in expected_attributes.items()):
            raise PlanCompilationError("compiled HWPX failed table-attribute validation")
        size = _single_hp_child(table, "sz")
        if size.get("width") != str(expected.width_hwp) or size.get("height") != "0":
            raise PlanCompilationError("compiled HWPX failed table-size validation")
        position = _single_hp_child(table, "pos")
        expected_position = {
            "treatAsChar": "1",
            "affectLSpacing": "0",
            "flowWithText": "1",
            "allowOverlap": "0",
            "holdAnchorAndSO": "0",
            "vertRelTo": "PARA",
            "horzRelTo": "PARA",
            "vertAlign": "TOP",
            "horzAlign": "LEFT",
            "vertOffset": "0",
            "horzOffset": "0",
        }
        if dict(position.attrib) != expected_position:
            raise PlanCompilationError("compiled HWPX failed table-position validation")
        _verify_compiled_table_cells(table, expected.node, expected.width_hwp)


def _verify_compiled_table_cells(
    table: etree._Element,
    expected: TableContentNode,
    table_width_hwp: int,
) -> None:
    rows = _hp_children(table, "tr")
    if len(rows) != expected.rows:
        raise PlanCompilationError("compiled HWPX failed table-row validation")
    actual_cells: list[tuple[int, etree._Element]] = []
    for row_index, row in enumerate(rows):
        actual_cells.extend((row_index, cell) for cell in _hp_children(row, "tc"))
    expected_cells = sorted(expected.cells, key=lambda item: (item.row, item.column))
    if len(actual_cells) != len(expected_cells):
        raise PlanCompilationError("compiled HWPX failed table-cell-count validation")

    for (row_index, cell), source in zip(actual_cells, expected_cells, strict=True):
        address = _single_hp_child(cell, "cellAddr")
        span = _single_hp_child(cell, "cellSpan")
        cell_size = _single_hp_child(cell, "cellSz")
        expected_width = _table_axis_offset(
            table_width_hwp,
            source.column + source.column_span,
            expected.columns,
        ) - _table_axis_offset(table_width_hwp, source.column, expected.columns)
        if (
            row_index != source.row
            or address.get("rowAddr") != str(source.row)
            or address.get("colAddr") != str(source.column)
            or span.get("rowSpan") != str(source.row_span)
            or span.get("colSpan") != str(source.column_span)
            or cell_size.get("width") != str(expected_width)
            or cell_size.get("height") != str(_TABLE_ROW_HEIGHT_HWP * source.row_span)
            or cell.get("borderFillIDRef") != "3"
        ):
            raise PlanCompilationError("compiled HWPX failed table-cell topology validation")

        sub_list = _single_hp_child(cell, "subList")
        paragraph = _single_hp_child(sub_list, "p")
        runs = _hp_children(paragraph, "run")
        if len(runs) != 1 or runs[0].get("charPrIDRef") != "0":
            raise PlanCompilationError("compiled HWPX failed table-cell style validation")
        text_nodes = _hp_children(runs[0], "t")
        if len(text_nodes) > 1:
            raise PlanCompilationError("compiled HWPX failed table-cell text validation")
        actual_text = _compiled_text_value(text_nodes[0]) if text_nodes else ""
        if actual_text != source.text:
            raise PlanCompilationError("compiled HWPX failed table-cell text validation")


def _compiled_text_value(text_node: etree._Element) -> str:
    pieces = [cast(str, text_node.text or "")]
    for child in text_node:
        name = etree.QName(child)
        special = _HWP_SPECIAL_TEXT_VALUES.get(name.localname) if name.namespace == HP else None
        if special is None or child.attrib or child.text or len(child):
            raise PlanCompilationError("compiled HWPX contains an unsupported text element")
        pieces.append(special)
        pieces.append(cast(str, child.tail or ""))
    return "".join(pieces)


def _verify_materialized_styles(
    header: etree._Element,
    expected_styles: tuple[_CompiledStyle, ...],
) -> None:
    fontfaces = _header_collection(header, "fontfaces")
    char_properties = _header_collection(header, "charProperties")
    para_properties = _header_collection(header, "paraProperties")
    style_properties = _header_collection(header, "styles")
    _assert_declared_count(fontfaces, "fontface", "fontfaces")
    _assert_declared_count(char_properties, "charPr", "charProperties")
    _assert_declared_count(para_properties, "paraPr", "paraProperties")
    _assert_declared_count(style_properties, "style", "styles")
    fontfaces_by_language = _fontfaces_by_language(fontfaces)

    language_by_attribute = {attribute: language for language, attribute in _FONT_LANGUAGES}
    for expected in expected_styles:
        char_pr = _element_with_id(char_properties, "charPr", expected.char_pr_id)
        if char_pr.get("height") != str(expected.font_size_hwp):
            raise PlanCompilationError("compiled HWPX failed font-size validation")
        if bool(_children(char_pr, "bold")) != expected.bold:
            raise PlanCompilationError("compiled HWPX failed bold-style validation")
        if bool(_children(char_pr, "italic")) != expected.italic:
            raise PlanCompilationError("compiled HWPX failed italic-style validation")
        font_ref = _single_child(char_pr, "fontRef")
        for attribute, font_id, face in expected.font_refs:
            if font_ref.get(attribute) != str(font_id):
                raise PlanCompilationError("compiled HWPX failed font-ref validation")
            language = language_by_attribute[attribute]
            font = _element_with_id(fontfaces_by_language[language], "font", font_id)
            if font.get("face") != face:
                raise PlanCompilationError("compiled HWPX failed font-family validation")

        para_pr = _element_with_id(para_properties, "paraPr", expected.para_pr_id)
        align = _single_child(para_pr, "align")
        if align.get("horizontal") != expected.alignment:
            raise PlanCompilationError("compiled HWPX failed alignment validation")
        line_spacing = _single_child(para_pr, "lineSpacing")
        if line_spacing.get("type") != "PERCENT" or line_spacing.get("value") != str(
            expected.line_spacing_percent
        ):
            raise PlanCompilationError("compiled HWPX failed line-spacing validation")

        named_style = _element_with_id(style_properties, "style", expected.style_id)
        expected_attributes = {
            "type": "PARA",
            "name": expected.source_id,
            "engName": expected.semantic_role,
            "paraPrIDRef": str(expected.para_pr_id),
            "charPrIDRef": str(expected.char_pr_id),
            "nextStyleIDRef": str(expected.style_id),
            "langID": "1042",
            "lockForm": "0",
        }
        if any(named_style.get(key) != value for key, value in expected_attributes.items()):
            raise PlanCompilationError("compiled HWPX failed style-table validation")
