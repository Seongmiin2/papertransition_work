from __future__ import annotations

import base64
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

from lxml import etree  # type: ignore[import-untyped]

from scan2hwpx.ir.models import Document

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS = "http://www.hancom.co.kr/hwpml/2011/section"
HH = "http://www.hancom.co.kr/hwpml/2011/head"
XML = "http://www.w3.org/XML/1998/namespace"
MIMETYPE = b"application/hwp+zip"
_CANONICAL_TEMPLATE = Path(__file__).with_name("canonical_template.b64")


@dataclass(frozen=True)
class RenderParagraph:
    text: str
    page_break: bool = False


def _serialize(root: etree._Element) -> bytes:
    return cast(
        bytes,
        etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        ),
    )


def _base_entries() -> dict[str, bytes]:
    encoded = "".join(_CANONICAL_TEMPLATE.read_text(encoding="ascii").split())
    package = base64.b64decode(encoded, validate=True)
    with zipfile.ZipFile(BytesIO(package)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    if entries.get("mimetype") != MIMETYPE or not _has_hancom_structure(entries):
        raise RuntimeError("bundled HWPX template is invalid")
    return entries


def _has_hancom_structure(entries: dict[str, bytes]) -> bool:
    try:
        header = etree.fromstring(entries["Contents/header.xml"])
        section = etree.fromstring(entries["Contents/section0.xml"])
    except (KeyError, etree.XMLSyntaxError):
        return False
    header_name = etree.QName(header)
    section_name = etree.QName(section)
    return (
        header_name.namespace == HH
        and header_name.localname == "head"
        and bool(header.xpath("./*[local-name()='refList']"))
        and section_name.namespace == HS
        and section_name.localname == "sec"
        and bool(section.xpath("./hp:p/hp:run/hp:secPr", namespaces={"hp": HP}))
    )


def _section_xml(paragraphs: list[RenderParagraph], template: bytes) -> bytes:
    source = etree.fromstring(template)
    if etree.QName(source).namespace != HS:
        raise ValueError("HWPX section template uses an unsupported namespace")
    source_paragraphs = source.xpath("./hp:p", namespaces={"hp": HP})
    if not source_paragraphs:
        raise ValueError("HWPX section template has no layout paragraph")

    layout = deepcopy(source_paragraphs[0])
    for child in list(layout):
        keep_layout = child.tag == f"{{{HP}}}run" and child.find(f"{{{HP}}}secPr") is not None
        if not keep_layout:
            layout.remove(child)
    if not layout.xpath("./hp:run/hp:secPr", namespaces={"hp": HP}):
        raise ValueError("HWPX section template has no section properties")

    section = etree.Element(source.tag, nsmap=source.nsmap)
    first = paragraphs[0] if paragraphs else RenderParagraph("")
    layout.attrib.update(
        {
            "id": "1",
            "paraPrIDRef": "20",
            "styleIDRef": "0",
            "pageBreak": "1" if first.page_break else "0",
            "columnBreak": "0",
            "merged": "0",
        }
    )
    _append_text_run(layout, first.text)
    section.append(layout)

    for index, paragraph in enumerate(paragraphs[1:], start=2):
        node = etree.SubElement(
            section,
            f"{{{HP}}}p",
            id=str(index),
            paraPrIDRef="20",
            styleIDRef="0",
            pageBreak="1" if paragraph.page_break else "0",
            columnBreak="0",
            merged="0",
        )
        _append_text_run(node, paragraph.text)
    return _serialize(section)


def _append_text_run(paragraph: etree._Element, text: str) -> None:
    run = etree.SubElement(paragraph, f"{{{HP}}}run", charPrIDRef="3")
    if not text:
        return
    node = etree.SubElement(run, f"{{{HP}}}t")
    node.set(f"{{{XML}}}space", "preserve")
    node.text = _xml_safe_text(text)


def _xml_safe_text(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in "\t\n\r" or ord(character) >= 0x20
    )


def _replace_body(entries: dict[str, bytes], paragraphs: list[RenderParagraph]) -> None:
    entries["Contents/section0.xml"] = _section_xml(
        paragraphs,
        entries["Contents/section0.xml"],
    )
    entries["Preview/PrvText.txt"] = "\n".join(
        paragraph.text for paragraph in paragraphs
    ).encode("utf-8")


def _write_package(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", entries["mimetype"], compress_type=zipfile.ZIP_STORED)
        for name in sorted(name for name in entries if name != "mimetype"):
            archive.writestr(name, entries[name], compress_type=zipfile.ZIP_DEFLATED)


def ensure_minimal_template(path: Path) -> None:
    """Create a Hancom-authored, empty HWPX template when one is absent."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_package(path, _base_entries())


def _read_template(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return _base_entries()
    try:
        with zipfile.ZipFile(path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
    except (OSError, zipfile.BadZipFile):
        return _base_entries()
    return entries if _has_hancom_structure(entries) else _base_entries()


def render_hwpx(document: Document, template: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = _read_template(template)
    paragraphs: list[RenderParagraph] = []
    for page_index, page in enumerate(document.pages):
        page_blocks = [
            block
            for block in sorted(page.blocks, key=lambda item: item.reading_order)
            if block.annotation_state.value == "printed"
        ]
        for block_index, block in enumerate(page_blocks):
            paragraphs.append(
                RenderParagraph(
                    text=block.text,
                    page_break=page_index > 0 and block_index == 0,
                )
            )
    _replace_body(entries, paragraphs)
    _write_package(output, entries)


def render_clean_hwpx(document: Document, output: Path) -> None:
    """Render an editable HWPX from a sanitized Hancom-authored package template."""
    from scan2hwpx.clean_layout import build_clean_items

    output.parent.mkdir(parents=True, exist_ok=True)
    items = build_clean_items(document)
    paragraphs = [
        RenderParagraph(
            text=item.text,
            page_break=index > 0 and item.page_no != items[index - 1].page_no,
        )
        for index, item in enumerate(items)
    ]
    if not paragraphs:
        paragraphs = [
            RenderParagraph(text=block.text, page_break=page_index > 0 and block_index == 0)
            for page_index, page in enumerate(document.pages)
            for block_index, block in enumerate(
                sorted(page.blocks, key=lambda item: item.reading_order)
            )
            if block.annotation_state.value == "printed" and block.text.strip()
        ]
    entries = _base_entries()
    _replace_body(entries, paragraphs)
    _write_package(output, entries)
