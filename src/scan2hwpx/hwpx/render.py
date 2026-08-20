from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from lxml import etree  # type: ignore[import-untyped]

from scan2hwpx.ir.models import Document

H = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
XML = "http://www.w3.org/XML/1998/namespace"
MIMETYPE = b"application/hwp+zip"


@dataclass(frozen=True)
class RenderParagraph:
    text: str
    page_break: bool = False


def _xml(root: etree._Element) -> bytes:
    return cast(
        bytes,
        etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True),
    )


def _base_entries() -> dict[str, bytes]:
    container_ns: dict[Any, str] = {None: "urn:oasis:names:tc:opendocument:xmlns:container"}
    container = etree.Element("container", nsmap=container_ns, version="1.0")
    roots = etree.SubElement(container, "rootfiles")
    etree.SubElement(
        roots, "rootfile", {"full-path": "Contents/content.hpf", "media-type": "application/xml"}
    )
    manifest_ns: dict[Any, str] = {None: "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"}
    manifest = etree.Element("manifest", nsmap=manifest_ns)
    for path, media in [
        ("/", "application/hwp+zip"),
        ("Contents/content.hpf", "application/xml"),
        ("Contents/header.xml", "application/xml"),
        ("Contents/section0.xml", "application/xml"),
    ]:
        etree.SubElement(manifest, "file-entry", {"full-path": path, "media-type": media})
    content_ns: dict[Any, str] = {None: "http://www.idpf.org/2007/opf"}
    content = etree.Element("package", nsmap=content_ns, version="3.0", unique_identifier="uuid")
    manifest_el = etree.SubElement(content, "manifest")
    etree.SubElement(
        manifest_el, "item", id="header", href="header.xml", media_type="application/xml"
    )
    etree.SubElement(
        manifest_el, "item", id="section0", href="section0.xml", media_type="application/xml"
    )
    spine = etree.SubElement(content, "spine")
    etree.SubElement(spine, "itemref", idref="section0")
    header = etree.Element(
        "head", nsmap={"hh": "http://www.hancom.co.kr/hwpml/2011/head"}, version="1.4", secCnt="1"
    )
    return {
        "mimetype": MIMETYPE,
        "META-INF/container.xml": _xml(container),
        "META-INF/manifest.xml": _xml(manifest),
        "Contents/content.hpf": _xml(content),
        "Contents/header.xml": _xml(header),
        "Contents/section0.xml": _section_xml([]),
        "settings.xml": _xml(etree.Element("settings")),
        "version.xml": _xml(
            etree.Element(
                "HCFVersion",
                TargetApplication="WORDPROCESSOR",
                Major="5",
                Minor="1",
                Micro="0",
                BuildNumber="0",
            )
        ),
    }


def _section_xml(paragraphs: list[RenderParagraph]) -> bytes:
    section = etree.Element(f"{{{H}}}sec", nsmap={"hp": HP})
    for index, paragraph in enumerate(paragraphs):
        p = etree.SubElement(
            section,
            f"{{{HP}}}p",
            id=str(index),
            paraPrIDRef="0",
            styleIDRef="0",
            pageBreak="1" if paragraph.page_break else "0",
            columnBreak="0",
            merged="0",
        )
        run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef="0")
        node = etree.SubElement(run, f"{{{HP}}}t")
        node.set(f"{{{XML}}}space", "preserve")
        node.text = paragraph.text
    return _xml(section)


def _write_package(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", entries["mimetype"], compress_type=zipfile.ZIP_STORED)
        for name in sorted(name for name in entries if name != "mimetype"):
            archive.writestr(name, entries[name], compress_type=zipfile.ZIP_DEFLATED)


def ensure_minimal_template(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_package(path, _base_entries())


def render_hwpx(document: Document, template: Path, output: Path) -> None:
    ensure_minimal_template(template)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scan2hwpx-") as temp_dir:
        working = Path(temp_dir) / "working.hwpx"
        shutil.copy2(template, working)
        with zipfile.ZipFile(working) as source:
            entries = {name: source.read(name) for name in source.namelist()}
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
        entries["Contents/section0.xml"] = _section_xml(paragraphs)
        _write_package(output, entries)


def render_clean_hwpx(document: Document, output: Path) -> None:
    """Render a valid editable HWPX package without requiring Hancom automation."""
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
    entries["Contents/section0.xml"] = _section_xml(paragraphs)
    _write_package(output, entries)
