from __future__ import annotations

import hashlib
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from lxml import etree  # type: ignore[import-untyped]
from PIL import Image

import scan2hwpx.hwpx.compiler as compiler_module
import scan2hwpx.hwpx.render as render_module
from scan2hwpx.contracts import (
    BBox,
    ColumnBreakPlanItem,
    ContentIR,
    ContentPlanItem,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    FormulaContentNode,
    FormulaFormat,
    HwpDocumentPlan,
    ImageContentNode,
    LayoutIntent,
    ObservationKind,
    PageBreakPlanItem,
    PageLayoutIntent,
    StyleIntent,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.contracts.models import TextAlignment
from scan2hwpx.hwpx.assets import (
    ImageAssetBundle,
    PngImageAsset,
    build_image_asset_bundle,
    image_asset_bundle_sha256,
)
from scan2hwpx.hwpx.compiler import (
    COMPILER_VERSION,
    IMAGE_DESIGN_PROFILE_ID,
    PlanCompilationError,
    compile_plan_hwpx,
)
from scan2hwpx.hwpx.validate import ValidationResult, validate_hwpx
from scan2hwpx.reference.hwpx import read_hwpx_projection

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HH = "http://www.hancom.co.kr/hwpml/2011/head"
HC = "http://www.hancom.co.kr/hwpml/2011/core"
OPF = "http://www.idpf.org/2007/opf/"


def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), color).save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def _bound_image_content(
    nodes: tuple[Any, ...],
    payloads: dict[str, bytes],
) -> tuple[EvidenceIR, ContentIR]:
    source_payloads = payloads or {"page-source": b"source"}
    sources = tuple(
        EvidenceSource(
            id=asset_ref,
            kind=EvidenceSourceKind.PAGE_IMAGE,
            artifact_ref=f"artifact://images/{asset_ref}",
            producer="test",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for asset_ref, payload in sorted(source_payloads.items())
    )
    observation_refs: set[str] = set()
    for node in nodes:
        observation_refs.update(node.evidence_refs)
        if isinstance(node, TableContentNode):
            for cell in node.cells:
                observation_refs.update(cell.evidence_refs)
    observation_ids = sorted(observation_refs)
    source_refs_by_observation: dict[str, set[str]] = {
        observation_id: set() for observation_id in observation_ids
    }
    image_observation_refs = {
        observation_id
        for node in nodes
        if isinstance(node, ImageContentNode)
        for observation_id in node.evidence_refs
    }
    default_source_ref = sources[0].id
    for node in nodes:
        node_source_ref = (
            node.asset_ref if isinstance(node, ImageContentNode) else default_source_ref
        )
        for observation_id in node.evidence_refs:
            source_refs_by_observation[observation_id].add(node_source_ref)
        if isinstance(node, TableContentNode):
            for cell in node.cells:
                for observation_id in cell.evidence_refs:
                    source_refs_by_observation[observation_id].add(default_source_ref)
    observations = tuple(
        EvidenceObservation(
            id=observation_id,
            kind=(
                ObservationKind.IMAGE
                if observation_id in image_observation_refs
                else ObservationKind.REGION
            ),
            bbox=BBox(pixel=(0, 0, 1, 1), normalized=(0, 0, 1, 1)),
            confidence=1.0,
            source_refs=tuple(sorted(source_refs_by_observation[observation_id])),
        )
        for observation_id in observation_ids
    )
    evidence = EvidenceIR(
        id="evidence-1",
        source_document_sha256="d" * 64,
        sources=sources,
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=1,
                height=1,
                image_source_ref=sources[0].id,
                observations=observations,
            ),
        ),
    )
    content = ContentIR(
        id="content-1",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        nodes=nodes,
        reading_order=tuple(node.id for node in nodes),
    )
    return evidence, content


def _image_bundle(
    evidence: EvidenceIR,
    content: ContentIR,
    payloads: dict[str, bytes],
) -> ImageAssetBundle:
    assets = tuple(
        PngImageAsset(
            asset_ref=asset_ref,
            media_type="image/png",
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=payload,
        )
        for asset_ref, payload in sorted(payloads.items())
    )
    return build_image_asset_bundle(evidence, content, assets)


def _content(nodes: tuple[Any, ...] | None = None) -> ContentIR:
    actual_nodes = nodes or (
        TextContentNode(
            id="text-1",
            role=ContentRole.QUESTION,
            text="첫 문단",
            evidence_refs=("observation-1",),
            confidence=0.99,
        ),
        TextContentNode(
            id="text-2",
            role=ContentRole.PASSAGE,
            text="둘째 문단",
            evidence_refs=("observation-2",),
            confidence=0.98,
        ),
    )
    return ContentIR(
        id="content-1",
        evidence_ir_id="evidence-1",
        evidence_ir_sha256="a" * 64,
        nodes=actual_nodes,
        reading_order=tuple(node.id for node in actual_nodes),
    )


def _plan(
    content: ContentIR,
    *,
    flow: tuple[Any, ...] | None = None,
    styles: tuple[StyleIntent, ...] = (),
    page_layout: PageLayoutIntent | None = None,
    design_profile_id: str = "scan2hwpx-canonical-text-v1",
) -> HwpDocumentPlan:
    actual_flow = flow or tuple(
        ContentPlanItem(
            id=f"flow-{index}",
            render_as="paragraph",
            content_ref=node.id,
        )
        for index, node in enumerate(content.nodes, start=1)
    )
    return HwpDocumentPlan(
        id="plan-1",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        capability_profile_id="exam2hwpx-authoring-v1",
        design_profile_id=design_profile_id,
        official_spec_refs=("hancom-hwpx-format:fixture",),
        page_layout=page_layout
        or PageLayoutIntent(
            width_mm=210.0,
            height_mm=297.0,
            margin_top_mm=20.0,
            margin_right_mm=30.0,
            margin_bottom_mm=15.0,
            margin_left_mm=30.0,
            columns=2,
            column_gap_mm=8.0,
        ),
        styles=styles,
        flow=actual_flow,
    )


def _section(path: Path) -> etree._Element:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("Contents/section0.xml")
    return etree.fromstring(payload)


def _header(path: Path) -> etree._Element:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("Contents/header.xml")
    return etree.fromstring(payload)


def test_compiles_text_plan_in_flow_order_with_page_break(tmp_path: Path) -> None:
    content = _content()
    plan = _plan(
        content,
        flow=(
            ContentPlanItem(id="flow-1", render_as="paragraph", content_ref="text-2"),
            PageBreakPlanItem(id="break-1"),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-1"),
        ),
    )
    output = tmp_path / "result.hwpx"

    result = compile_plan_hwpx(content, plan, output)

    assert result.compiler_version == COMPILER_VERSION
    assert result.paragraph_count == 2
    assert result.page_break_count == 1
    assert result.content_ir_sha256 == contract_sha256(content)
    assert result.hwp_document_plan_sha256 == contract_sha256(plan)
    assert validate_hwpx(output).valid
    section = _section(output)
    paragraphs = section.xpath("./hp:p", namespaces={"hp": HP})
    assert ["".join(node.itertext()) for node in paragraphs] == ["둘째 문단", "첫 문단"]
    assert [node.get("pageBreak") for node in paragraphs] == ["0", "1"]
    with zipfile.ZipFile(output) as archive:
        assert archive.read("Preview/PrvText.txt") == "둘째 문단\n첫 문단".encode()


def test_compiler_output_is_byte_deterministic(tmp_path: Path) -> None:
    content = _content()
    plan = _plan(content)
    first = tmp_path / "first.hwpx"
    second = tmp_path / "second.hwpx"

    first_result = compile_plan_hwpx(content, plan, first)
    second_result = compile_plan_hwpx(content, plan, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    with zipfile.ZipFile(first) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_v3_compiles_editable_merged_table_with_exact_projection(
    tmp_path: Path,
) -> None:
    special_text = "\t\t병합\n\n\u00a0\u00a0셀\u3000\u3000"
    before = TextContentNode(
        id="text-before",
        role=ContentRole.PASSAGE,
        text="표 앞",
        evidence_refs=("observation-1",),
        confidence=0.99,
    )
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-2",),
        confidence=0.98,
        rows=3,
        columns=3,
        cells=(
            TableCell(
                row=0,
                column=0,
                row_span=2,
                column_span=2,
                text=special_text,
                evidence_refs=("observation-2",),
            ),
            TableCell(
                row=0,
                column=2,
                text="",
                evidence_refs=("observation-2",),
            ),
            TableCell(
                row=1,
                column=2,
                text="오른쪽",
                evidence_refs=("observation-2",),
            ),
            TableCell(
                row=2,
                column=0,
                column_span=3,
                text="아래",
                evidence_refs=("observation-2",),
            ),
        ),
    )
    after = TextContentNode(
        id="text-after",
        role=ContentRole.PASSAGE,
        text="표 뒤",
        evidence_refs=("observation-3",),
        confidence=0.97,
    )
    content = _content((before, table, after))
    body_style = StyleIntent(id="body", semantic_role="body", bold=True)
    plan = _plan(
        content,
        styles=(body_style,),
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(
                id="flow-before",
                render_as="paragraph",
                content_ref=before.id,
                style_ref=body_style.id,
            ),
            PageBreakPlanItem(id="page-break"),
            ContentPlanItem(id="flow-table", render_as="table", content_ref=table.id),
            ContentPlanItem(id="flow-after", render_as="paragraph", content_ref=after.id),
        ),
    )
    first = tmp_path / "table-first.hwpx"
    second = tmp_path / "table-second.hwpx"

    first_result = compile_plan_hwpx(content, plan, first)
    second_result = compile_plan_hwpx(content, plan, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.artifact_sha256 == second_result.artifact_sha256
    assert first_result.paragraph_count == 3
    assert first_result.page_break_count == 1
    assert validate_hwpx(first).valid

    section = _section(first)
    paragraphs = section.xpath("./hp:p", namespaces={"hp": HP})
    assert [paragraph.get("pageBreak") for paragraph in paragraphs] == ["0", "1", "0"]
    tables = paragraphs[1].xpath("./hp:run/hp:tbl", namespaces={"hp": HP})
    assert len(tables) == 1
    compiled_table = tables[0]
    assert compiled_table.get("id") == "1300000001"
    assert compiled_table.get("zOrder") == "0"
    assert compiled_table.get("rowCnt") == "3"
    assert compiled_table.get("colCnt") == "3"
    assert compiled_table.get("borderFillIDRef") == "3"
    assert compiled_table.xpath("./hp:sz/@width", namespaces={"hp": HP}) == ["20126"]
    assert compiled_table.xpath("./hp:sz/@height", namespaces={"hp": HP}) == ["0"]
    page = section.xpath("./hp:p/hp:run/hp:secPr/hp:pagePr", namespaces={"hp": HP})[0]
    margin = page.xpath("./hp:margin", namespaces={"hp": HP})[0]
    columns = section.xpath("./hp:p/hp:run/hp:ctrl/hp:colPr", namespaces={"hp": HP})[0]
    canonical_column_width = (
        int(page.get("width"))
        - int(margin.get("left"))
        - int(margin.get("right"))
        - int(columns.get("sameGap"))
    ) // int(columns.get("colCount"))
    assert canonical_column_width == 20126
    position = compiled_table.xpath("./hp:pos", namespaces={"hp": HP})[0]
    assert position.attrib == {
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
    assert [len(row) for row in compiled_table.xpath("./hp:tr", namespaces={"hp": HP})] == [
        2,
        1,
        1,
    ]
    compiled_cells = compiled_table.xpath("./hp:tr/hp:tc", namespaces={"hp": HP})
    assert [
        (
            cell.xpath("./hp:cellAddr/@rowAddr", namespaces={"hp": HP})[0],
            cell.xpath("./hp:cellAddr/@colAddr", namespaces={"hp": HP})[0],
            cell.xpath("./hp:cellSpan/@rowSpan", namespaces={"hp": HP})[0],
            cell.xpath("./hp:cellSpan/@colSpan", namespaces={"hp": HP})[0],
            cell.xpath("./hp:cellSz/@width", namespaces={"hp": HP})[0],
            cell.xpath("./hp:cellSz/@height", namespaces={"hp": HP})[0],
        )
        for cell in compiled_cells
    ] == [
        ("0", "0", "2", "2", "13417", "564"),
        ("0", "2", "1", "1", "6709", "282"),
        ("1", "2", "1", "1", "6709", "282"),
        ("2", "0", "1", "3", "20126", "282"),
    ]
    text_element = compiled_cells[0].xpath("./hp:subList/hp:p/hp:run/hp:t", namespaces={"hp": HP})[
        0
    ]
    assert [etree.QName(child).localname for child in text_element] == [
        "tab",
        "tab",
        "lineBreak",
        "lineBreak",
        "nbSpace",
        "nbSpace",
        "fwSpace",
        "fwSpace",
    ]
    assert text_element.text is None
    assert [child.tail for child in text_element] == [
        None,
        "병합",
        None,
        None,
        None,
        "셀",
        None,
        None,
    ]

    header = _header(first)
    border_fills = header.xpath("./hh:refList/hh:borderFills", namespaces={"hh": HH})[0]
    assert border_fills.get("itemCnt") == "3"
    assert len(border_fills.xpath("./hh:borderFill[@id='3']", namespaces={"hh": HH})) == 1
    with zipfile.ZipFile(first) as archive:
        expected_preview = f"표 앞\n{special_text}\t\t\n\t\t오른쪽\n아래\t\t\n표 뒤"
        assert archive.read("Preview/PrvText.txt") == expected_preview.encode("utf-8")

    artifact_sha256 = hashlib.sha256(first.read_bytes()).hexdigest()
    projection = read_hwpx_projection(
        first,
        source_document_sha256="b" * 64,
        expected_hwpx_sha256=artifact_sha256,
        expected_page_count=1,
    )
    assert [node.kind for node in projection.nodes] == ["text", "table", "text"]
    projected_table = next(node for node in projection.nodes if node.kind == "table")
    assert projected_table.rows == 3
    assert projected_table.columns == 3
    assert projected_table.width_hwp == 20126
    assert projected_table.page_break is True
    assert [
        (cell.row, cell.column, cell.row_span, cell.column_span, cell.text)
        for cell in projected_table.cells
    ] == [
        (0, 0, 2, 2, special_text),
        (0, 2, 1, 1, ""),
        (1, 2, 1, 1, "오른쪽"),
        (2, 0, 1, 3, "아래"),
    ]
    assert projected_table.text == f"{special_text}\t\t\n\t\t오른쪽\n아래\t\t"


def test_v2_profile_remains_text_only_when_given_a_table(tmp_path: Path) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, text="셀", evidence_refs=("observation-1",)),),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    output = tmp_path / "v2-table.hwpx"

    with pytest.raises(PlanCompilationError, match="unsupported content kind.*table"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "layout",
    [
        LayoutIntent(width_fraction=0.5),
        LayoutIntent(column_span=2),
        LayoutIntent(keep_with_next=True),
    ],
)
def test_v3_rejects_nondefault_table_layout(
    tmp_path: Path,
    layout: LayoutIntent,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, evidence_refs=("observation-1",)),),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="table",
                content_ref=table.id,
                layout=layout,
            ),
        ),
    )
    output = tmp_path / "nondefault-layout.hwpx"

    with pytest.raises(PlanCompilationError, match="layout intent is unsupported"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


def test_v3_rejects_table_style_ref(tmp_path: Path) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, evidence_refs=("observation-1",)),),
    )
    content = _content((table,))
    style = StyleIntent(id="table-style", semantic_role="table")
    plan = _plan(
        content,
        styles=(style,),
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="table",
                content_ref=table.id,
                style_ref=style.id,
            ),
        ),
    )
    output = tmp_path / "table-style.hwpx"

    with pytest.raises(PlanCompilationError, match="table style refs are unsupported"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("node", "render_as", "kind"),
    [
        (
            ImageContentNode(
                id="image-1",
                evidence_refs=("observation-1",),
                confidence=0.9,
                asset_ref="page-image-1",
            ),
            "image",
            "image",
        ),
        (
            FormulaContentNode(
                id="formula-1",
                evidence_refs=("observation-1",),
                confidence=0.9,
                expression="x^2",
                format=FormulaFormat.LATEX,
            ),
            "formula",
            "formula",
        ),
    ],
)
def test_v3_still_rejects_image_and_formula_content(
    tmp_path: Path,
    node: Any,
    render_as: str,
    kind: str,
) -> None:
    content = _content((node,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as=render_as, content_ref=node.id),),
    )
    output = tmp_path / f"v3-{kind}.hwpx"

    with pytest.raises(PlanCompilationError, match=f"unsupported content kind.*{kind}"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


def test_v3_still_rejects_column_breaks(tmp_path: Path) -> None:
    content = _content()
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(id="flow-1", render_as="paragraph", content_ref="text-1"),
            ColumnBreakPlanItem(id="column-1"),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )

    with pytest.raises(PlanCompilationError, match="column breaks are unsupported"):
        compile_plan_hwpx(content, plan, tmp_path / "v3-column.hwpx")


def test_table_ids_resources_and_cell_order_are_deterministic(tmp_path: Path) -> None:
    spanning = TableContentNode(
        id="table-spanning",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=2,
        columns=1,
        cells=(
            TableCell(
                row=0,
                column=0,
                row_span=2,
                text="세로 병합",
                evidence_refs=("observation-1",),
            ),
        ),
    )
    ordered_cells = (
        TableCell(row=0, column=0, text="왼쪽", evidence_refs=("observation-2",)),
        TableCell(row=0, column=1, text="오른쪽", evidence_refs=("observation-2",)),
    )
    horizontal = TableContentNode(
        id="table-horizontal",
        evidence_refs=("observation-2",),
        confidence=0.9,
        rows=1,
        columns=2,
        cells=ordered_cells,
    )
    reversed_horizontal = horizontal.model_copy(update={"cells": tuple(reversed(ordered_cells))})
    content = _content((spanning, horizontal))
    reversed_content = _content((spanning, reversed_horizontal))
    first_plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(id="flow-1", render_as="table", content_ref=spanning.id),
            ContentPlanItem(id="flow-2", render_as="table", content_ref=horizontal.id),
        ),
    )
    second_plan = _plan(
        reversed_content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(id="flow-1", render_as="table", content_ref=spanning.id),
            ContentPlanItem(id="flow-2", render_as="table", content_ref=horizontal.id),
        ),
    )
    first = tmp_path / "tables-ordered.hwpx"
    second = tmp_path / "tables-reversed.hwpx"

    compile_plan_hwpx(content, first_plan, first)
    compile_plan_hwpx(reversed_content, second_plan, second)

    assert first.read_bytes() == second.read_bytes()
    assert validate_hwpx(first).valid
    section = _section(first)
    tables = section.xpath("./hp:p/hp:run/hp:tbl", namespaces={"hp": HP})
    assert [table.get("id") for table in tables] == ["1300000001", "1300000002"]
    assert [table.get("zOrder") for table in tables] == ["0", "1"]
    assert [len(row) for row in tables[0].xpath("./hp:tr", namespaces={"hp": HP})] == [1, 0]
    header = _header(first)
    border_fills = header.xpath("./hh:refList/hh:borderFills", namespaces={"hh": HH})[0]
    assert border_fills.get("itemCnt") == "3"
    assert len(border_fills.xpath("./hh:borderFill[@id='3']", namespaces={"hh": HH})) == 1


def test_forged_table_bounds_are_rejected_before_contract_materialization(
    tmp_path: Path,
) -> None:
    cell = TableCell(row=0, column=0, evidence_refs=("observation-1",))
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(cell,),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    forged_rows = table.model_copy(update={"rows": compiler_module._MAX_TABLE_ROWS + 1})
    forged_row_content = content.model_copy(update={"nodes": (forged_rows,)})
    forged_span = cell.model_copy(update={"row_span": compiler_module._MAX_TABLE_ROWS + 1})
    forged_cells = table.model_copy(update={"cells": (forged_span,)})
    forged_cell_content = content.model_copy(update={"nodes": (forged_cells,)})

    with pytest.raises(PlanCompilationError, match="table row count exceeds"):
        compile_plan_hwpx(forged_row_content, plan, tmp_path / "forged-rows.hwpx")
    with pytest.raises(PlanCompilationError, match="table cell row span exceeds"):
        compile_plan_hwpx(forged_cell_content, plan, tmp_path / "forged-span.hwpx")


def test_table_text_limit_and_validation_failure_preserve_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, text="가나", evidence_refs=("observation-1",)),),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    output = tmp_path / "preserved-table.hwpx"
    output.write_bytes(b"previous-output")

    monkeypatch.setattr(compiler_module, "_MAX_TABLE_CELL_TEXT_UTF8_BYTES", 5)
    with pytest.raises(PlanCompilationError, match="table cell text exceeds"):
        compile_plan_hwpx(content, plan, output)
    assert output.read_bytes() == b"previous-output"

    monkeypatch.setattr(compiler_module, "_MAX_TABLE_CELL_TEXT_UTF8_BYTES", 64 * 1024)
    monkeypatch.setattr(
        compiler_module,
        "validate_hwpx",
        lambda _path: ValidationResult(errors=["forced table failure"], warnings=[]),
    )
    with pytest.raises(PlanCompilationError, match="forced table failure"):
        compile_plan_hwpx(content, plan, output)
    assert output.read_bytes() == b"previous-output"


def test_table_text_control_limit_is_checked_before_xml_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(
            TableCell(
                row=0,
                column=0,
                text="셀\t\t",
                evidence_refs=("observation-1",),
            ),
        ),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    monkeypatch.setattr(compiler_module, "_MAX_TABLE_TEXT_CONTROLS", 1)
    output = tmp_path / "too-many-table-controls.hwpx"

    with pytest.raises(PlanCompilationError, match="table text control count exceeds"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (" \t\n\u00a0\u3000", "whitespace-only table cell text"),
        ("unsafe\u0001text", "XML 1.0-forbidden"),
    ],
)
def test_v3_rejects_table_cell_text_that_cannot_round_trip(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, text=text, evidence_refs=("observation-1",)),),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    output = tmp_path / "unprojectable-cell.hwpx"

    with pytest.raises(PlanCompilationError, match=message):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


def test_v3_enforces_table_grid_and_cell_limits_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, evidence_refs=("observation-1",)),),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )

    monkeypatch.setattr(compiler_module, "_MAX_TABLE_GRID_AREA", 0)
    with pytest.raises(PlanCompilationError, match="table grid area exceeds"):
        compile_plan_hwpx(content, plan, tmp_path / "grid-limit.hwpx")

    monkeypatch.setattr(compiler_module, "_MAX_TABLE_GRID_AREA", 10_000)
    monkeypatch.setattr(compiler_module, "_MAX_TABLE_CELLS", 0)
    with pytest.raises(PlanCompilationError, match="table cell count exceeds"):
        compile_plan_hwpx(content, plan, tmp_path / "cell-limit.hwpx")


def test_hwpx_package_exports_all_compiler_profile_ids() -> None:
    from scan2hwpx import hwpx

    assert hwpx.SUPPORTED_CAPABILITY_PROFILE_ID == "exam2hwpx-authoring-v1"
    assert hwpx.LEGACY_DESIGN_PROFILE_ID == "scan2hwpx-canonical-text-v1"
    assert hwpx.SUPPORTED_DESIGN_PROFILE_ID == "scan2hwpx-canonical-styled-text-v2"
    assert hwpx.TABLE_DESIGN_PROFILE_ID == "scan2hwpx-canonical-styled-table-v3"
    assert hwpx.IMAGE_DESIGN_PROFILE_ID == "scan2hwpx-canonical-styled-table-image-v4"


def test_v3_rejects_text_that_projection_cannot_preserve(tmp_path: Path) -> None:
    whitespace = TextContentNode(
        id="whitespace",
        role=ContentRole.OTHER,
        text=" \t\n\u00a0\u3000",
        evidence_refs=("observation-1",),
        confidence=0.9,
    )
    whitespace_content = _content((whitespace,))
    whitespace_plan = _plan(
        whitespace_content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
    )
    whitespace_output = tmp_path / "whitespace-text.hwpx"

    with pytest.raises(PlanCompilationError, match="whitespace-only text content"):
        compile_plan_hwpx(whitespace_content, whitespace_plan, whitespace_output)
    assert not whitespace_output.exists()


def test_v3_rejects_nfd_table_text_that_projection_normalizes(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "\u00e9")
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(
            TableCell(
                row=0,
                column=0,
                text=decomposed,
                evidence_refs=("observation-1",),
            ),
        ),
    )
    table_content = _content((table,))
    table_plan = _plan(
        table_content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    table_output = tmp_path / "decomposed-table-text.hwpx"

    with pytest.raises(PlanCompilationError, match="NFC-normalized"):
        compile_plan_hwpx(table_content, table_plan, table_output)
    assert not table_output.exists()


def test_table_aggregate_text_limit_stops_before_later_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        rows=1,
        columns=2,
        cells=(
            TableCell(row=0, column=0, text="aa", evidence_refs=("observation-1",)),
            TableCell(row=0, column=1, text="bb", evidence_refs=("observation-1",)),
        ),
    )
    content = _content((table,))
    plan = _plan(
        content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(ContentPlanItem(id="flow-1", render_as="table", content_ref=table.id),),
    )
    visited: list[str] = []
    original_assert_xml_text = compiler_module._assert_xml_text

    def record_cell_visit(label: str, text: str) -> int:
        visited.append(label)
        return original_assert_xml_text(label, text)

    monkeypatch.setattr(compiler_module, "_MAX_TEXT_UTF8_BYTES", 1)
    monkeypatch.setattr(compiler_module, "_assert_xml_text", record_cell_visit)

    with pytest.raises(PlanCompilationError, match="text content exceeds"):
        compile_plan_hwpx(content, plan, tmp_path / "aggregate-limit.hwpx")
    assert visited == ["table-1[0,0]"]


def test_candidate_mutation_is_rejected_and_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "preserved-after-mutation.hwpx"
    output.write_bytes(b"previous-output")

    original_verify = compiler_module._verify_compiled_content

    def mutate_after_content_verification(path: Path, *args: Any) -> None:
        original_verify(path, *args)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("unexpected-entry", b"mutated")

    monkeypatch.setattr(
        compiler_module,
        "_verify_compiled_content",
        mutate_after_content_verification,
    )

    with pytest.raises(PlanCompilationError, match="changed during validation"):
        compile_plan_hwpx(_content(), _plan(_content()), output)
    assert output.read_bytes() == b"previous-output"


def test_text_utf8_limit_is_counted_without_materializing_encoded_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoEncode(str):
        def encode(self, *_args: Any, **_kwargs: Any) -> bytes:
            raise AssertionError("must not materialize a full UTF-8 copy")

    assert compiler_module._assert_xml_text("probe", NoEncode("\U0001f600")) == 4

    node = TextContentNode(
        id="multibyte",
        role=ContentRole.OTHER,
        text="\U0001f600",
        evidence_refs=("observation-1",),
        confidence=0.9,
    )
    content = _content((node,))
    output = tmp_path / "multibyte-limit.hwpx"
    monkeypatch.setattr(compiler_module, "_MAX_TEXT_UTF8_BYTES", 3)

    with pytest.raises(PlanCompilationError, match="text content exceeds"):
        compile_plan_hwpx(content, _plan(content), output)
    assert not output.exists()


def test_rejects_contract_subclasses_before_dumping_or_compiling(tmp_path: Path) -> None:
    content = _content()
    plan = _plan(content)

    class ContentIRSubclass(ContentIR):
        pass

    class HwpDocumentPlanSubclass(HwpDocumentPlan):
        pass

    forged_content = ContentIRSubclass.model_validate(content.model_dump(mode="python"))
    forged_plan = HwpDocumentPlanSubclass.model_validate(plan.model_dump(mode="python"))

    with pytest.raises(TypeError, match="ContentIR must be a ContentIR"):
        compile_plan_hwpx(forged_content, plan, tmp_path / "content-subclass.hwpx")
    with pytest.raises(TypeError, match="HwpDocumentPlan must be a HwpDocumentPlan"):
        compile_plan_hwpx(content, forged_plan, tmp_path / "plan-subclass.hwpx")


def test_materializes_used_and_unused_styles_into_header_and_section(
    tmp_path: Path,
) -> None:
    content = _content()
    question = StyleIntent(
        id="question",
        semantic_role="question-title",
        font_family="Noto Sans KR",
        font_size_pt=12.5,
        bold=True,
        italic=True,
        alignment=TextAlignment.CENTER,
        line_spacing=1.75,
    )
    unused = StyleIntent(
        id="unused",
        semantic_role="unused-role",
        font_family="함초롬바탕",
        font_size_pt=10,
        alignment=TextAlignment.RIGHT,
        line_spacing=0.9,
    )
    plan = _plan(
        content,
        styles=(unused, question),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="paragraph",
                content_ref="text-1",
                style_ref="question",
            ),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )
    output = tmp_path / "styled.hwpx"

    compile_plan_hwpx(content, plan, output)

    assert validate_hwpx(output).valid
    section = _section(output)
    paragraphs = section.xpath("./hp:p", namespaces={"hp": HP})
    assert [node.get("paraPrIDRef") for node in paragraphs] == ["23", "20"]
    assert [node.get("styleIDRef") for node in paragraphs] == ["22", "0"]
    assert [
        node.xpath("./hp:run[last()]/@charPrIDRef", namespaces={"hp": HP})[0] for node in paragraphs
    ] == ["10", "3"]

    header = _header(output)
    namespaces = {"hh": HH}
    fontfaces = header.xpath("./hh:refList/hh:fontfaces", namespaces=namespaces)[0]
    char_properties = header.xpath("./hh:refList/hh:charProperties", namespaces=namespaces)[0]
    para_properties = header.xpath("./hh:refList/hh:paraProperties", namespaces=namespaces)[0]
    styles = header.xpath("./hh:refList/hh:styles", namespaces=namespaces)[0]
    assert fontfaces.get("itemCnt") == "7"
    assert char_properties.get("itemCnt") == "12"
    assert para_properties.get("itemCnt") == "25"
    assert styles.get("itemCnt") == "24"

    for fontface in fontfaces.xpath("./hh:fontface", namespaces=namespaces):
        assert fontface.get("fontCnt") == "3"
        custom_font = fontface.xpath("./hh:font[@id='2']", namespaces=namespaces)
        assert len(custom_font) == 1
        assert custom_font[0].get("face") == "Noto Sans KR"

    question_char = char_properties.xpath("./hh:charPr[@id='10']", namespaces=namespaces)[0]
    assert question_char.get("height") == "1250"
    question_font_ref = question_char.xpath("./hh:fontRef", namespaces=namespaces)[0]
    assert set(question_font_ref.attrib.values()) == {"2"}
    assert len(question_char.xpath("./hh:bold", namespaces=namespaces)) == 1
    assert len(question_char.xpath("./hh:italic", namespaces=namespaces)) == 1
    question_para = para_properties.xpath("./hh:paraPr[@id='23']", namespaces=namespaces)[0]
    assert question_para.xpath("./hh:align/@horizontal", namespaces=namespaces) == ["CENTER"]
    assert question_para.xpath("./hh:lineSpacing/@type", namespaces=namespaces) == ["PERCENT"]
    assert question_para.xpath("./hh:lineSpacing/@value", namespaces=namespaces) == ["175"]
    question_style = styles.xpath("./hh:style[@id='22']", namespaces=namespaces)[0]
    assert question_style.attrib == {
        "id": "22",
        "type": "PARA",
        "name": "question",
        "engName": "question-title",
        "paraPrIDRef": "23",
        "charPrIDRef": "10",
        "nextStyleIDRef": "22",
        "langID": "1042",
        "lockForm": "0",
    }

    unused_char = char_properties.xpath("./hh:charPr[@id='11']", namespaces=namespaces)[0]
    assert unused_char.get("height") == "1000"
    assert set(unused_char.xpath("./hh:fontRef", namespaces=namespaces)[0].attrib.values()) == {"1"}
    unused_para = para_properties.xpath("./hh:paraPr[@id='24']", namespaces=namespaces)[0]
    assert unused_para.xpath("./hh:align/@horizontal", namespaces=namespaces) == ["RIGHT"]
    assert unused_para.xpath("./hh:lineSpacing/@value", namespaces=namespaces) == ["90"]
    unused_style = styles.xpath("./hh:style[@id='23']", namespaces=namespaces)[0]
    assert unused_style.get("name") == "unused"


def test_styled_compiler_output_is_byte_deterministic(tmp_path: Path) -> None:
    content = _content()
    style = StyleIntent(
        id="body",
        semantic_role="body",
        font_family="테스트 글꼴",
        font_size_pt=9.5,
        bold=True,
        alignment=TextAlignment.JUSTIFY,
        line_spacing=1.55,
    )
    flow = tuple(
        ContentPlanItem(
            id=f"flow-{index}",
            render_as="paragraph",
            content_ref=node.id,
            style_ref="body",
        )
        for index, node in enumerate(content.nodes, start=1)
    )
    plan = _plan(
        content,
        styles=(style,),
        flow=flow,
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
    )
    first = tmp_path / "first-styled.hwpx"
    second = tmp_path / "second-styled.hwpx"

    first_result = compile_plan_hwpx(content, plan, first)
    second_result = compile_plan_hwpx(content, plan, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.artifact_sha256 == second_result.artifact_sha256


def test_renders_hwp_special_spaces_as_ordered_mixed_content(tmp_path: Path) -> None:
    text = "\t시작\t\t중간\n\u00a0끝\u3000"
    content = _content(
        (
            TextContentNode(
                id="text-special",
                role=ContentRole.PASSAGE,
                text=text,
                evidence_refs=("observation-1",),
                confidence=0.99,
            ),
        )
    )
    style = StyleIntent(id="body", semantic_role="body", bold=True)
    plan = _plan(
        content,
        styles=(style,),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="paragraph",
                content_ref="text-special",
                style_ref="body",
            ),
        ),
    )
    output = tmp_path / "special-spaces.hwpx"

    compile_plan_hwpx(content, plan, output)

    section = _section(output)
    text_element = section.xpath("./hp:p/hp:run[last()]/hp:t", namespaces={"hp": HP})[0]
    assert text_element.text is None
    assert [etree.QName(child).localname for child in text_element] == [
        "tab",
        "tab",
        "tab",
        "lineBreak",
        "nbSpace",
        "fwSpace",
    ]
    assert [child.tail for child in text_element] == ["시작", None, "중간", None, "끝", None]

    artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    projection = read_hwpx_projection(
        output,
        source_document_sha256="b" * 64,
        expected_hwpx_sha256=artifact_sha256,
        expected_page_count=1,
    )
    assert [node.text for node in projection.nodes if node.kind == "text"] == [text]


def test_rejects_stale_content_binding_without_creating_output(tmp_path: Path) -> None:
    content = _content()
    plan = _plan(content)
    changed_payload = content.model_dump(mode="json")
    changed_payload["nodes"][0]["text"] = "changed"
    changed = ContentIR.model_validate(changed_payload)
    output = tmp_path / "result.hwpx"

    with pytest.raises(PlanCompilationError, match="ContentIR digest mismatch"):
        compile_plan_hwpx(changed, plan, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("node", "render_as", "kind"),
    [
        (
            TableContentNode(
                id="table-1",
                evidence_refs=("observation-1",),
                confidence=0.9,
                rows=1,
                columns=1,
                cells=(TableCell(row=0, column=0, text="cell", evidence_refs=("observation-1",)),),
            ),
            "table",
            "table",
        ),
        (
            ImageContentNode(
                id="image-1",
                evidence_refs=("observation-1",),
                confidence=0.9,
                asset_ref="page-image-1",
            ),
            "image",
            "image",
        ),
        (
            FormulaContentNode(
                id="formula-1",
                evidence_refs=("observation-1",),
                confidence=0.9,
                expression="x^2",
                format=FormulaFormat.LATEX,
            ),
            "formula",
            "formula",
        ),
    ],
)
def test_rejects_nontext_content_instead_of_flattening_it(
    tmp_path: Path,
    node: Any,
    render_as: str,
    kind: str,
) -> None:
    content = _content((node,))
    plan_payload = _plan(content).model_dump(mode="json")
    plan_payload["flow"][0]["render_as"] = render_as
    plan = HwpDocumentPlan.model_validate(plan_payload)
    output = tmp_path / f"{kind}.hwpx"

    with pytest.raises(PlanCompilationError, match=f"unsupported content kind.*{kind}"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


def test_legacy_design_profile_remains_text_only_and_rejects_styles(
    tmp_path: Path,
) -> None:
    content = _content()
    style = StyleIntent(id="question", semantic_role="question", bold=True)
    styled_plan = _plan(
        content,
        styles=(style,),
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="paragraph",
                content_ref="text-1",
                style_ref="question",
            ),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )
    with pytest.raises(PlanCompilationError, match="styled-text-v2 design profile"):
        compile_plan_hwpx(content, styled_plan, tmp_path / "styled.hwpx")


def test_rejects_unmaterialized_layout_intent(tmp_path: Path) -> None:
    content = _content()

    layout_plan = _plan(
        content,
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="paragraph",
                content_ref="text-1",
                layout=LayoutIntent(width_fraction=0.5),
            ),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )
    with pytest.raises(PlanCompilationError, match="layout intent is unsupported"):
        compile_plan_hwpx(content, layout_plan, tmp_path / "layout.hwpx")


@pytest.mark.parametrize(
    "style",
    [
        StyleIntent(
            id="font-precision",
            semantic_role="body",
            font_size_pt=10.001,
        ),
        StyleIntent(
            id="spacing-precision",
            semantic_role="body",
            line_spacing=1.555,
        ),
    ],
)
def test_rejects_style_values_that_hwpx_cannot_represent_exactly(
    tmp_path: Path,
    style: StyleIntent,
) -> None:
    content = _content()
    plan = _plan(
        content,
        styles=(style,),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="paragraph",
                content_ref="text-1",
                style_ref=style.id,
            ),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )
    output = tmp_path / "precision.hwpx"

    with pytest.raises(PlanCompilationError, match="cannot be represented exactly"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("font_size_pt", True),
        ("line_spacing", float("nan")),
    ],
)
def test_rejects_forged_optional_style_numbers_before_serialization_coercion(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    content = _content()
    style = StyleIntent(id="body", semantic_role="body")
    plan = _plan(
        content,
        styles=(style,),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
    )
    forged_style = style.model_copy(update={field: value})
    forged_plan = plan.model_copy(update={"styles": (forged_style,)})
    output = tmp_path / f"forged-{field}.hwpx"

    with pytest.raises(PlanCompilationError, match="not a valid strict contract"):
        compile_plan_hwpx(content, forged_plan, output)

    assert not output.exists()


def test_rejects_style_count_and_font_family_size_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _content()
    style = StyleIntent(id="body", semantic_role="body", font_family="long-font")
    plan = _plan(
        content,
        styles=(style,),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
    )

    monkeypatch.setattr(compiler_module, "_MAX_STYLE_INTENTS", 0)
    with pytest.raises(PlanCompilationError, match="style intent count"):
        compile_plan_hwpx(content, plan, tmp_path / "too-many.hwpx")

    monkeypatch.setattr(compiler_module, "_MAX_STYLE_INTENTS", 1)
    monkeypatch.setattr(compiler_module, "_MAX_FONT_FAMILY_UTF8_BYTES", 3)
    with pytest.raises(PlanCompilationError, match="font family exceeds"):
        compile_plan_hwpx(content, plan, tmp_path / "font-too-long.hwpx")


def test_rejects_style_attribute_normalization_and_invalid_local_refs(
    tmp_path: Path,
) -> None:
    content = _content()
    newline_style = StyleIntent(
        id="body",
        semantic_role="body",
        font_family="font\nname",
    )
    newline_plan = _plan(
        content,
        styles=(newline_style,),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
    )
    with pytest.raises(PlanCompilationError, match="XML attributes would normalize"):
        compile_plan_hwpx(content, newline_plan, tmp_path / "normalized.hwpx")

    valid_style = StyleIntent(id="body", semantic_role="body")
    valid_plan = _plan(
        content,
        styles=(valid_style,),
        design_profile_id="scan2hwpx-canonical-styled-text-v2",
    )
    invalid_flow = (
        valid_plan.flow[0].model_copy(update={"style_ref": "missing"}),
        valid_plan.flow[1],
    )
    unknown_ref_plan = valid_plan.model_copy(update={"flow": invalid_flow})
    with pytest.raises(PlanCompilationError, match="unknown plan style refs"):
        compile_plan_hwpx(content, unknown_ref_plan, tmp_path / "unknown-ref.hwpx")

    duplicate_style_plan = valid_plan.model_copy(update={"styles": (valid_style, valid_style)})
    with pytest.raises(PlanCompilationError, match="duplicate plan style ids"):
        compile_plan_hwpx(content, duplicate_style_plan, tmp_path / "duplicate.hwpx")


def test_rejects_column_break_and_non_a4_portrait_page_layout(tmp_path: Path) -> None:
    content = _content()
    column_plan = _plan(
        content,
        flow=(
            ContentPlanItem(id="flow-1", render_as="paragraph", content_ref="text-1"),
            ColumnBreakPlanItem(id="column-1"),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )
    with pytest.raises(PlanCompilationError, match="column breaks are unsupported"):
        compile_plan_hwpx(content, column_plan, tmp_path / "column.hwpx")

    page_layout = PageLayoutIntent(
        width_mm=297.0,
        height_mm=210.0,
        margin_top_mm=15.0,
        margin_right_mm=15.0,
        margin_bottom_mm=15.0,
        margin_left_mm=15.0,
        columns=2,
        column_gap_mm=8.0,
    )
    with pytest.raises(PlanCompilationError, match="A4 portrait"):
        compile_plan_hwpx(
            content,
            _plan(content, page_layout=page_layout),
            tmp_path / "page-layout.hwpx",
        )


def test_v5_preserves_canonical_layout_xml_bytes_and_dimensions() -> None:
    entries = compiler_module._base_entries()
    original_section = entries["Contents/section0.xml"]
    layout = compiler_module._compile_page_layout(
        PageLayoutIntent(
            width_mm=210.0,
            height_mm=297.0,
            margin_top_mm=20.0,
            margin_right_mm=30.0,
            margin_bottom_mm=15.0,
            margin_left_mm=30.0,
            columns=2,
            column_gap_mm=8.0,
        )
    )

    compiler_module._materialize_page_layout(entries, layout)

    assert entries["Contents/section0.xml"] == original_section
    assert (
        layout.width_hwp,
        layout.height_hwp,
        layout.margin_top_hwp,
        layout.margin_right_hwp,
        layout.margin_bottom_hwp,
        layout.margin_left_hwp,
        layout.column_gap_hwp,
        layout.column_width_hwp,
    ) == (59528, 84186, 5668, 8504, 4252, 8504, 2267, 20126)


def test_v5_materializes_variable_margins_and_scales_table_and_image(
    tmp_path: Path,
) -> None:
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-1",),
        confidence=0.98,
        rows=1,
        columns=2,
        cells=(
            TableCell(row=0, column=0, text="left", evidence_refs=("observation-1",)),
            TableCell(row=0, column=1, text="right", evidence_refs=("observation-1",)),
        ),
    )
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-2",),
        confidence=0.97,
        asset_ref="image-ref",
    )
    payloads = {image.asset_ref: _png(2, 1, (10, 20, 30))}
    evidence, content = _bound_image_content((table, image), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    page_layout = PageLayoutIntent(
        width_mm=210.0,
        height_mm=297.0,
        margin_top_mm=14.0,
        margin_right_mm=20.0,
        margin_bottom_mm=14.0,
        margin_left_mm=20.0,
        columns=2,
        column_gap_mm=8.0,
    )
    plan = _plan(
        content,
        page_layout=page_layout,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(id="flow-table", render_as="table", content_ref=table.id),
            ContentPlanItem(
                id="flow-image",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )
    first = tmp_path / "variable-layout-first.hwpx"
    second = tmp_path / "variable-layout-second.hwpx"

    first_result = compile_plan_hwpx(
        content,
        plan,
        first,
        evidence_ir=evidence,
        image_assets=bundle,
    )
    second_result = compile_plan_hwpx(
        content,
        plan,
        second,
        evidence_ir=evidence,
        image_assets=bundle,
    )

    assert first_result.compiler_version == "scan2hwpx-plan-styled-table-image/5.0"
    assert second_result == first_result
    assert first.read_bytes() == second.read_bytes()
    assert validate_hwpx(first).valid
    section = _section(first)
    page = section.xpath(".//hp:pagePr", namespaces={"hp": HP})[0]
    margin = page.xpath("./hp:margin", namespaces={"hp": HP})[0]
    columns = section.xpath(".//hp:colPr", namespaces={"hp": HP})[0]
    assert (page.get("width"), page.get("height")) == ("59528", "84189")
    assert {
        name: margin.get(name)
        for name in ("top", "right", "bottom", "left", "header", "footer", "gutter")
    } == {
        "top": "3969",
        "right": "5669",
        "bottom": "3969",
        "left": "5669",
        "header": "4252",
        "footer": "4252",
        "gutter": "0",
    }
    assert (columns.get("colCount"), columns.get("sameGap")) == ("2", "2268")
    column_width_hwp = (
        int(page.get("width"))
        - int(margin.get("left"))
        - int(margin.get("right"))
        - int(columns.get("sameGap"))
    ) // int(columns.get("colCount"))
    assert column_width_hwp == 22961
    assert column_width_hwp * 25.4 / 7200 == pytest.approx(81.0, abs=0.01)
    compiled_table = section.xpath(".//hp:tbl", namespaces={"hp": HP})[0]
    assert compiled_table.xpath("./hp:sz/@width", namespaces={"hp": HP}) == ["22961"]
    assert compiled_table.xpath(
        "./hp:tr/hp:tc/hp:cellSz/@width",
        namespaces={"hp": HP},
    ) == ["11480", "11481"]
    picture = section.xpath(".//hp:pic", namespaces={"hp": HP})[0]
    assert picture.xpath("./hp:sz/@width", namespaces={"hp": HP}) == ["22961"]
    assert picture.xpath("./hp:sz/@height", namespaces={"hp": HP}) == ["11481"]


@pytest.mark.parametrize(
    "page_layout, error",
    [
        (
            PageLayoutIntent(
                width_mm=210.0,
                height_mm=297.0,
                margin_top_mm=148.1,
                margin_right_mm=20.0,
                margin_bottom_mm=148.1,
                margin_left_mm=20.0,
                columns=2,
                column_gap_mm=8.0,
            ),
            "usable height",
        ),
        (
            PageLayoutIntent(
                width_mm=210.0,
                height_mm=297.0,
                margin_top_mm=14.0,
                margin_right_mm=20.0,
                margin_bottom_mm=14.0,
                margin_left_mm=20.0,
                columns=2,
                column_gap_mm=169.0,
            ),
            "column width",
        ),
    ],
)
def test_v5_rejects_page_layout_without_practical_usable_region(
    tmp_path: Path,
    page_layout: PageLayoutIntent,
    error: str,
) -> None:
    content = _content()
    output = tmp_path / "invalid-layout.hwpx"

    with pytest.raises(PlanCompilationError, match=error):
        compile_plan_hwpx(
            content,
            _plan(content, page_layout=page_layout),
            output,
        )

    assert not output.exists()


def test_v5_direct_verifier_rejects_page_layout_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _content()
    plan = _plan(content)
    original_package = compiler_module._deterministic_package

    def tampered_package(entries: dict[str, bytes]) -> bytes:
        changed = dict(entries)
        section = etree.fromstring(changed["Contents/section0.xml"])
        margin = section.xpath(".//hp:pagePr/hp:margin", namespaces={"hp": HP})[0]
        margin.set("left", "1")
        changed["Contents/section0.xml"] = etree.tostring(
            section,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )
        return original_package(changed)

    monkeypatch.setattr(compiler_module, "_deterministic_package", tampered_package)
    output = tmp_path / "tampered-layout.hwpx"

    with pytest.raises(PlanCompilationError, match="page-layout validation"):
        compile_plan_hwpx(content, plan, output)

    assert not output.exists()


def test_rejects_xml_forbidden_text_without_silent_content_loss(tmp_path: Path) -> None:
    content = _content(
        (
            TextContentNode(
                id="text-1",
                role=ContentRole.OTHER,
                text="unsafe\u0001text",
                evidence_refs=("observation-1",),
                confidence=0.9,
            ),
        )
    )

    with pytest.raises(PlanCompilationError, match="XML 1.0-forbidden"):
        compile_plan_hwpx(content, _plan(content), tmp_path / "unsafe.hwpx")


def test_low_level_renderer_rejects_instead_of_dropping_forbidden_text() -> None:
    entries = render_module._base_entries()

    with pytest.raises(ValueError, match="XML 1.0-forbidden"):
        render_module._replace_body(
            entries,
            [render_module.RenderParagraph("앞\u0001뒤")],
        )


@pytest.mark.parametrize("text", ["앞\r뒤", "앞\r\n뒤"])
def test_rejects_carriage_returns_that_have_no_exact_hwp_text_mapping(
    tmp_path: Path,
    text: str,
) -> None:
    content = _content(
        (
            TextContentNode(
                id="text-1",
                role=ContentRole.OTHER,
                text=text,
                evidence_refs=("observation-1",),
                confidence=0.9,
            ),
        )
    )
    output = tmp_path / "carriage-return.hwpx"

    with pytest.raises(PlanCompilationError, match="unsupported carriage return"):
        compile_plan_hwpx(content, _plan(content), output)

    assert not output.exists()


def test_validation_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.hwpx"
    output.write_bytes(b"previous-output")
    monkeypatch.setattr(
        compiler_module,
        "validate_hwpx",
        lambda _path: ValidationResult(errors=["forced failure"], warnings=[]),
    )

    with pytest.raises(PlanCompilationError, match="forced failure"):
        compile_plan_hwpx(_content(), _plan(_content()), output)

    assert output.read_bytes() == b"previous-output"


def test_rejects_unversioned_template_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_entries = compiler_module._base_entries()
    changed_entries["settings.xml"] += b" "
    monkeypatch.setattr(compiler_module, "_base_entries", lambda: changed_entries)
    output = tmp_path / "result.hwpx"

    with pytest.raises(PlanCompilationError, match="compiler version update"):
        compile_plan_hwpx(_content(), _plan(_content()), output)

    assert not output.exists()


def test_v4_compiles_mixed_content_with_deduplicated_png_assets_and_exact_anchors(
    tmp_path: Path,
) -> None:
    text = TextContentNode(
        id="text-1",
        role=ContentRole.PASSAGE,
        text="본문",
        evidence_refs=("observation-1",),
        confidence=0.99,
    )
    table = TableContentNode(
        id="table-1",
        evidence_refs=("observation-2",),
        confidence=0.98,
        rows=1,
        columns=1,
        cells=(TableCell(row=0, column=0, text="셀", evidence_refs=("observation-2",)),),
    )
    alpha = ImageContentNode(
        id="image-alpha",
        evidence_refs=("observation-3",),
        confidence=0.97,
        asset_ref="private-alpha-ref",
    )
    beta = ImageContentNode(
        id="image-beta",
        evidence_refs=("observation-4",),
        confidence=0.96,
        asset_ref="private-beta-ref",
    )
    gamma = ImageContentNode(
        id="image-gamma",
        evidence_refs=("observation-5",),
        confidence=0.95,
        asset_ref="private-gamma-ref",
    )
    red_square = _png(1, 1, (255, 0, 0))
    blue_wide = _png(2, 1, (0, 0, 255))
    payloads = {
        alpha.asset_ref: red_square,
        beta.asset_ref: blue_wide,
        gamma.asset_ref: blue_wide,
    }
    evidence, content = _bound_image_content((alpha, text, beta, table, gamma), payloads)
    bundle = _image_bundle(
        evidence,
        content,
        payloads,
    )
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(
                id="flow-beta",
                render_as="image",
                content_ref=beta.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
            ContentPlanItem(id="flow-text", render_as="paragraph", content_ref=text.id),
            ContentPlanItem(id="flow-table", render_as="table", content_ref=table.id),
            PageBreakPlanItem(id="page-break-alpha"),
            ContentPlanItem(
                id="flow-alpha",
                render_as="image",
                content_ref=alpha.id,
                layout=LayoutIntent(width_fraction=0.5),
            ),
            ContentPlanItem(
                id="flow-gamma",
                render_as="image",
                content_ref=gamma.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )
    first = tmp_path / "image-first.hwpx"
    second = tmp_path / "image-second.hwpx"

    first_result = compile_plan_hwpx(
        content,
        plan,
        first,
        evidence_ir=evidence,
        image_assets=bundle,
    )
    second_result = compile_plan_hwpx(
        content,
        plan,
        second,
        evidence_ir=evidence,
        image_assets=bundle,
    )

    assert first_result.compiler_version == "scan2hwpx-plan-styled-table-image/5.0"
    assert first_result.image_asset_bundle_sha256 == image_asset_bundle_sha256(bundle)
    assert first_result.paragraph_count == 5
    assert first_result.page_break_count == 1
    assert second_result == first_result
    assert first.read_bytes() == second.read_bytes()
    assert validate_hwpx(first).valid

    with zipfile.ZipFile(first) as archive:
        assert archive.read("BinData/image1.png") == blue_wide
        assert archive.read("BinData/image2.png") == red_square
        assert sorted(name for name in archive.namelist() if name.startswith("BinData/image")) == [
            "BinData/image1.png",
            "BinData/image2.png",
        ]
        manifest = etree.fromstring(archive.read("Contents/content.hpf"))
        section_payload = archive.read("Contents/section0.xml")
        preview = archive.read("Preview/PrvText.txt")

    image_items = manifest.xpath(
        "./opf:manifest/opf:item[starts-with(@id, 'image')]",
        namespaces={"opf": OPF},
    )
    assert [dict(item.attrib) for item in image_items] == [
        {
            "id": "image1",
            "href": "BinData/image1.png",
            "media-type": "image/png",
            "isEmbeded": "1",
        },
        {
            "id": "image2",
            "href": "BinData/image2.png",
            "media-type": "image/png",
            "isEmbeded": "1",
        },
    ]
    assert preview == "[IMAGE]\n본문\n셀\n[IMAGE]\n[IMAGE]".encode()
    assert all(asset.asset_ref.encode() not in section_payload for asset in bundle.assets)

    section = etree.fromstring(section_payload)
    pictures = section.xpath("./hp:p/hp:run/hp:pic", namespaces={"hp": HP})
    assert [picture.get("id") for picture in pictures] == [
        "1400000000",
        "1400000002",
        "1400000004",
    ]
    assert [picture.get("instid") for picture in pictures] == [
        "1400000001",
        "1400000003",
        "1400000005",
    ]
    assert [picture.get("zOrder") for picture in pictures] == ["0", "2", "3"]
    tables = section.xpath(".//hp:tbl", namespaces={"hp": HP})
    assert [table_node.get("zOrder") for table_node in tables] == ["1"]
    controls = section.xpath(".//hp:tbl | .//hp:pic", namespaces={"hp": HP})
    assert len({control.get("id") for control in controls}) == len(controls)
    assert len({control.get("zOrder") for control in controls}) == len(controls)
    assert [
        picture.xpath("./hc:img/@binaryItemIDRef", namespaces={"hc": HC})[0] for picture in pictures
    ] == ["image1", "image2", "image1"]
    assert [
        (
            picture.xpath("./hp:sz/@width", namespaces={"hp": HP})[0],
            picture.xpath("./hp:sz/@height", namespaces={"hp": HP})[0],
        )
        for picture in pictures
    ] == [("20126", "10063"), ("10063", "10063"), ("20126", "10063")]
    expected_position = {
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
    }
    assert all(
        dict(picture.xpath("./hp:pos", namespaces={"hp": HP})[0].attrib) == expected_position
        for picture in pictures
    )
    assert all(picture.get("textWrap") == "TOP_AND_BOTTOM" for picture in pictures)
    assert all(
        [etree.QName(child).localname for child in picture.getparent()] == ["pic", "t"]
        for picture in pictures
    )
    assert [picture.getparent().getparent().get("pageBreak") for picture in pictures] == [
        "0",
        "1",
        "0",
    ]
    assert [
        picture.xpath("./hp:shapeComment/text()", namespaces={"hp": HP})[0] for picture in pictures
    ] == ["[IMAGE]", "[IMAGE]", "[IMAGE]"]

    projection = read_hwpx_projection(
        first,
        source_document_sha256=evidence.source_document_sha256,
        expected_hwpx_sha256=first_result.artifact_sha256,
        expected_page_count=2,
    )
    assert [node.kind for node in projection.nodes] == [
        "image",
        "text",
        "table",
        "image",
        "image",
    ]
    projected_images = [node for node in projection.nodes if node.kind == "image"]
    assert [(node.width_hwp, node.height_hwp) for node in projected_images] == [
        (20_126, 10_063),
        (10_063, 10_063),
        (20_126, 10_063),
    ]
    assert [node.page_break for node in projected_images] == [False, True, False]
    assert len(projection.assets) == 2
    assert {asset.sha256 for asset in projection.assets} == {
        hashlib.sha256(blue_wide).hexdigest(),
        hashlib.sha256(red_square).hexdigest(),
    }
    projected_asset_ids = [node.asset_id for node in projected_images]
    assert projected_asset_ids[0] == projected_asset_ids[2]


def test_nonimage_compile_result_has_no_image_bundle_provenance(tmp_path: Path) -> None:
    content = _content()

    result = compile_plan_hwpx(content, _plan(content), tmp_path / "text-only.hwpx")

    assert result.image_asset_bundle_sha256 is None


def test_v4_requires_explicit_image_width_and_an_exact_bound_bundle(tmp_path: Path) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    payloads = {image.asset_ref: _png(1, 1, (1, 2, 3))}
    evidence, content = _bound_image_content((image,), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    no_width_plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(ContentPlanItem(id="flow-1", render_as="image", content_ref=image.id),),
    )

    with pytest.raises(PlanCompilationError, match="image width_fraction is required"):
        compile_plan_hwpx(
            content,
            no_width_plan,
            tmp_path / "no-width.hwpx",
            evidence_ir=evidence,
            image_assets=bundle,
        )
    with pytest.raises(PlanCompilationError, match="requires an image asset bundle"):
        compile_plan_hwpx(content, no_width_plan, tmp_path / "no-bundle.hwpx")
    with pytest.raises(PlanCompilationError, match="requires EvidenceIR"):
        compile_plan_hwpx(
            content,
            no_width_plan,
            tmp_path / "no-evidence.hwpx",
            image_assets=bundle,
        )

    other_content = content.model_copy(update={"id": "other-content"})
    other_bundle = _image_bundle(
        evidence,
        other_content,
        {image.asset_ref: _png(1, 1, (1, 2, 3))},
    )
    with pytest.raises(PlanCompilationError, match="image asset bundle lineage mismatch"):
        compile_plan_hwpx(
            content,
            no_width_plan,
            tmp_path / "wrong-bundle.hwpx",
            evidence_ir=evidence,
            image_assets=other_bundle,
        )


def test_rejects_unneeded_bundle_and_old_profile_cannot_be_enabled_by_assets(
    tmp_path: Path,
) -> None:
    text_nodes = _content().nodes
    text_evidence, text_content = _bound_image_content(text_nodes, {})
    empty_bundle = _image_bundle(text_evidence, text_content, {})
    v4_text_plan = _plan(text_content, design_profile_id=IMAGE_DESIGN_PROFILE_ID)
    with pytest.raises(PlanCompilationError, match="unneeded without image nodes"):
        compile_plan_hwpx(
            text_content,
            v4_text_plan,
            tmp_path / "unneeded.hwpx",
            evidence_ir=text_evidence,
            image_assets=empty_bundle,
        )

    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    image_payloads = {image.asset_ref: _png(1, 1, (2, 3, 4))}
    image_evidence, image_content = _bound_image_content((image,), image_payloads)
    image_bundle = _image_bundle(image_evidence, image_content, image_payloads)
    v3_plan = _plan(
        image_content,
        design_profile_id="scan2hwpx-canonical-styled-table-v3",
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )
    with pytest.raises(PlanCompilationError, match="unneeded for this design profile"):
        compile_plan_hwpx(
            image_content,
            v3_plan,
            tmp_path / "v3-with-assets.hwpx",
            evidence_ir=image_evidence,
            image_assets=image_bundle,
        )


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (LayoutIntent(width_fraction=1.0, column_span=2), "image column_span must be 1"),
        (LayoutIntent(width_fraction=1.0, keep_with_next=True), "image keep_with_next"),
    ],
)
def test_v4_rejects_unrepresentable_image_layout(
    tmp_path: Path,
    layout: LayoutIntent,
    message: str,
) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    payloads = {image.asset_ref: _png(1, 1, (3, 4, 5))}
    evidence, content = _bound_image_content((image,), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                layout=layout,
            ),
        ),
    )

    with pytest.raises(PlanCompilationError, match=message):
        compile_plan_hwpx(
            content,
            plan,
            tmp_path / "bad-layout.hwpx",
            evidence_ir=evidence,
            image_assets=bundle,
        )


def test_v4_rejects_image_style_ref(tmp_path: Path) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    payloads = {image.asset_ref: _png(1, 1, (3, 4, 5))}
    evidence, content = _bound_image_content((image,), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    style = StyleIntent(id="image-style", semantic_role="image")
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        styles=(style,),
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                style_ref=style.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )

    with pytest.raises(PlanCompilationError, match="image style refs are unsupported"):
        compile_plan_hwpx(
            content,
            plan,
            tmp_path / "image-style.hwpx",
            evidence_ir=evidence,
            image_assets=bundle,
        )


def test_v4_preflights_image_control_count_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    content = _content((image,))
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )
    monkeypatch.setattr(compiler_module, "_MAX_IMAGE_CONTROLS", 0)

    with pytest.raises(PlanCompilationError, match="image control count exceeds"):
        compile_plan_hwpx(content, plan, tmp_path / "too-many-images.hwpx")


def test_v4_preflights_repeated_image_occurrence_pixels_before_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = tuple(
        ImageContentNode(
            id=f"image-{index}",
            evidence_refs=(f"observation-{index}",),
            confidence=0.9,
            asset_ref="shared-image-ref",
        )
        for index in range(1, 6)
    )
    payloads = {"shared-image-ref": _png(4, 4, (3, 4, 5))}
    evidence, content = _bound_image_content(images, payloads)
    bundle = _image_bundle(evidence, content, payloads)
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=tuple(
            ContentPlanItem(
                id=f"flow-{index}",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=1.0),
            )
            for index, image in enumerate(images, start=1)
        ),
    )

    def unexpected_compile(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("paragraph compilation must not run past pixel preflight")

    monkeypatch.setattr(compiler_module, "_MAX_IMAGE_OCCURRENCE_PIXELS", 64)
    monkeypatch.setattr(compiler_module, "_compile_paragraphs", unexpected_compile)
    output = tmp_path / "repeated-pixel-limit.hwpx"

    with pytest.raises(PlanCompilationError, match="image occurrence pixels exceed"):
        compile_plan_hwpx(
            content,
            plan,
            output,
            evidence_ir=evidence,
            image_assets=bundle,
        )

    assert not output.exists()


def test_v4_quantizes_explicit_image_width_and_height_half_up(tmp_path: Path) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    payloads = {image.asset_ref: _png(7, 11, (3, 4, 5))}
    evidence, content = _bound_image_content((image,), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=0.059618),
            ),
        ),
    )
    output = tmp_path / "quantized.hwpx"

    compile_plan_hwpx(
        content,
        plan,
        output,
        evidence_ir=evidence,
        image_assets=bundle,
    )

    picture = _section(output).xpath(".//hp:pic", namespaces={"hp": HP})[0]
    assert picture.xpath("./hp:sz/@width", namespaces={"hp": HP}) == ["1200"]
    assert picture.xpath("./hp:sz/@height", namespaces={"hp": HP}) == ["1886"]


def test_v4_rejects_image_that_exceeds_the_usable_page_height(tmp_path: Path) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="tall-image",
    )
    payloads = {image.asset_ref: _png(1, 4, (4, 5, 6))}
    evidence, content = _bound_image_content((image,), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )

    with pytest.raises(PlanCompilationError, match="image height exceeds"):
        compile_plan_hwpx(
            content,
            plan,
            tmp_path / "too-tall.hwpx",
            evidence_ir=evidence,
            image_assets=bundle,
        )


def test_v4_still_rejects_formulas_and_column_breaks(tmp_path: Path) -> None:
    formula = FormulaContentNode(
        id="formula-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        expression="x^2",
        format=FormulaFormat.LATEX,
    )
    formula_content = _content((formula,))
    formula_plan = _plan(
        formula_content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(ContentPlanItem(id="flow-1", render_as="formula", content_ref=formula.id),),
    )
    with pytest.raises(PlanCompilationError, match="unsupported content kind.*formula"):
        compile_plan_hwpx(formula_content, formula_plan, tmp_path / "formula.hwpx")

    text_content = _content()
    column_plan = _plan(
        text_content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(id="flow-1", render_as="paragraph", content_ref="text-1"),
            ColumnBreakPlanItem(id="column-1"),
            ContentPlanItem(id="flow-2", render_as="paragraph", content_ref="text-2"),
        ),
    )
    with pytest.raises(PlanCompilationError, match="column breaks are unsupported"):
        compile_plan_hwpx(text_content, column_plan, tmp_path / "column.hwpx")


@pytest.mark.parametrize("mutation", ["payload", "manifest", "anchor", "extra"])
def test_v4_direct_verifier_rejects_image_package_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    image = ImageContentNode(
        id="image-1",
        evidence_refs=("observation-1",),
        confidence=0.9,
        asset_ref="image-ref",
    )
    payloads = {image.asset_ref: _png(1, 1, (10, 20, 30))}
    evidence, content = _bound_image_content((image,), payloads)
    bundle = _image_bundle(evidence, content, payloads)
    plan = _plan(
        content,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as="image",
                content_ref=image.id,
                layout=LayoutIntent(width_fraction=1.0),
            ),
        ),
    )
    original_package = compiler_module._deterministic_package

    def tampered_package(entries: dict[str, bytes]) -> bytes:
        changed = dict(entries)
        if mutation == "payload":
            changed["BinData/image1.png"] = _png(1, 1, (30, 20, 10))
        elif mutation == "manifest":
            manifest = etree.fromstring(changed["Contents/content.hpf"])
            item = manifest.xpath(
                "./opf:manifest/opf:item[@id='image1']",
                namespaces={"opf": OPF},
            )[0]
            item.set("media-type", "image/jpeg")
            changed["Contents/content.hpf"] = etree.tostring(
                manifest,
                xml_declaration=True,
                encoding="UTF-8",
                standalone=True,
            )
        elif mutation == "anchor":
            section = etree.fromstring(changed["Contents/section0.xml"])
            position = section.xpath(".//hp:pic/hp:pos", namespaces={"hp": HP})[0]
            position.set("treatAsChar", "0")
            changed["Contents/section0.xml"] = etree.tostring(
                section,
                xml_declaration=True,
                encoding="UTF-8",
                standalone=True,
            )
        else:
            changed["BinData/extra.png"] = _png(1, 1, (1, 1, 1))
        return original_package(changed)

    monkeypatch.setattr(compiler_module, "_deterministic_package", tampered_package)
    output = tmp_path / f"tampered-{mutation}.hwpx"

    with pytest.raises(PlanCompilationError, match="compiled HWPX failed image"):
        compile_plan_hwpx(
            content,
            plan,
            output,
            evidence_ir=evidence,
            image_assets=bundle,
        )

    assert not output.exists()
