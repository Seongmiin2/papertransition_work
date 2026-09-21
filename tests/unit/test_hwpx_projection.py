from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from lxml import etree  # type: ignore[import-untyped]

from scan2hwpx.hwpx import ensure_minimal_template
from scan2hwpx.reference.hwpx import HwpxProjectionSource, read_hwpx_projection

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            entries["mimetype"],
            compress_type=zipfile.ZIP_STORED,
        )
        for name in sorted(entries):
            if name != "mimetype":
                archive.writestr(name, entries[name], compress_type=zipfile.ZIP_DEFLATED)


def _projection_fixture(path: Path, *, oversized_table: bool = False) -> None:
    ensure_minimal_template(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}

    section = etree.fromstring(entries["Contents/section0.xml"])
    paragraph = etree.Element(
        f"{{{HP}}}p",
        id="999",
        paraPrIDRef="20",
        styleIDRef="0",
        pageBreak="0",
        columnBreak="0",
        merged="0",
    )
    first_run = etree.SubElement(paragraph, f"{{{HP}}}run", charPrIDRef="3")
    first_text = etree.SubElement(first_run, f"{{{HP}}}t")
    first_text.text = "Question"
    tab = etree.SubElement(first_text, f"{{{HP}}}tab")
    tab.tail = " "
    control = etree.SubElement(first_run, f"{{{HP}}}ctrl")
    etree.SubElement(
        control,
        f"{{{HP}}}pageNum",
        pos="BOTTOM_CENTER",
        formatType="DIGIT",
        sideChar="-",
    )
    etree.SubElement(
        control,
        f"{{{HP}}}colPr",
        id="",
        type="NEWSPAPER",
        layout="LEFT",
        colCount="1",
        sameSz="1",
        sameGap="0",
    )
    second_run = etree.SubElement(paragraph, f"{{{HP}}}run", charPrIDRef="4")
    second_text = etree.SubElement(second_run, f"{{{HP}}}t")
    second_text.text = "continued"
    if oversized_table:
        etree.SubElement(
            second_run,
            f"{{{HP}}}tbl",
            rowCnt="1001",
            colCnt="1",
        )
    section.append(paragraph)
    section.append(
        etree.Element(
            f"{{{HP}}}p",
            id="1000",
            paraPrIDRef="20",
            styleIDRef="0",
            pageBreak="0",
            columnBreak="0",
            merged="0",
        )
    )
    entries["Contents/section0.xml"] = etree.tostring(
        section,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )

    header = etree.fromstring(entries["Contents/header.xml"])
    style = header.xpath("//*[local-name()='charPr'][@id='3']")[0]
    style.set("textColor", "#0000FF")
    underline = style.xpath("./*[local-name()='underline']")
    if underline:
        underline[0].set("type", "BOTTOM")
    entries["Contents/header.xml"] = etree.tostring(
        header,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    _rewrite(path, entries)


def test_projects_structural_facts_and_explicitly_flags_losses(tmp_path: Path) -> None:
    path = tmp_path / "projection.hwpx"
    _projection_fixture(path)

    projection = read_hwpx_projection(
        path,
        source_document_sha256="a" * 64,
        expected_hwpx_sha256=_sha256(path),
        expected_page_count=3,
    )

    assert [node.text for node in projection.nodes] == ["Question\t ", "continued"]
    assert all(node.id.startswith("a" * 64 + ":s000/p") for node in projection.nodes)
    assert projection.page_count == 3
    assert projection.sections[0].layout.margin_header_hwp > 0
    assert [
        (item.columns, item.column_gap_hwp) for item in projection.sections[0].column_changes
    ] == [
        (2, 2267),
        (1, 0),
    ]
    assert any(style.text_color == "#0000FF" for style in projection.styles)
    issue_codes = {issue.code for issue in projection.issues}
    assert {
        "empty_paragraph_layout_not_projected",
        "header_footer_gutter_margins_not_representable",
        "page_assignment_requires_pdf_alignment",
        "page_number_control_not_projected",
        "section_column_layout_changes_flattened",
        "style_facts_not_representable_in_plan",
    } <= issue_codes

    round_tripped = HwpxProjectionSource.model_validate_json(projection.model_dump_json())
    assert round_tripped == projection
    serialized = projection.model_dump_json()
    assert str(tmp_path) not in serialized
    assert "Contents/section0.xml" not in serialized


def test_rejects_wrong_digest_duplicate_member_and_oversized_table(tmp_path: Path) -> None:
    path = tmp_path / "projection.hwpx"
    _projection_fixture(path)

    with pytest.raises(ValueError, match="differs from the bound pair manifest"):
        read_hwpx_projection(
            path,
            source_document_sha256="b" * 64,
            expected_hwpx_sha256="0" * 64,
            expected_page_count=1,
        )

    duplicate = tmp_path / "duplicate.hwpx"
    duplicate.write_bytes(path.read_bytes())
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(duplicate, "a") as archive,
    ):
        archive.writestr("Contents/section0.xml", b"duplicate")
    with pytest.raises(ValueError, match="duplicate member names"):
        read_hwpx_projection(
            duplicate,
            source_document_sha256="b" * 64,
            expected_hwpx_sha256=_sha256(duplicate),
            expected_page_count=1,
        )

    oversized = tmp_path / "oversized.hwpx"
    _projection_fixture(oversized, oversized_table=True)
    with pytest.raises(ValueError, match="dimensions exceed the safety limit"):
        read_hwpx_projection(
            oversized,
            source_document_sha256="b" * 64,
            expected_hwpx_sha256=_sha256(oversized),
            expected_page_count=1,
        )
