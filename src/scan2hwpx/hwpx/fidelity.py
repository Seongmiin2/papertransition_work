from __future__ import annotations

import base64
import copy
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lxml import etree  # type: ignore[import-untyped]
from PIL import Image, ImageOps

from .render import MIMETYPE

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HC = "http://www.hancom.co.kr/hwpml/2011/core"
OPF = "http://www.idpf.org/2007/opf/"
TARGET_SIZE = (1651, 2339)


@dataclass(frozen=True)
class FidelityRenderStats:
    pages: int
    pictures: int
    package_bytes: int


def render_fidelity_hwpx(page_images: list[Path], output: Path) -> FidelityRenderStats:
    """Write one source-page picture per HWPX page using a Hancom-authored package."""
    if not page_images:
        raise ValueError("at least one page image is required")
    missing = [path for path in page_images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"page image does not exist: {missing[0]}")

    members = _load_template()
    encoded_pages = [_encode_page(path) for path in page_images]
    members["Contents/section0.xml"] = _build_section(
        members["Contents/section0.xml"], len(encoded_pages)
    )
    members["Contents/content.hpf"] = _build_manifest(
        members["Contents/content.hpf"], len(encoded_pages)
    )
    members["Preview/PrvImage.png"] = _build_preview(page_images[0])
    members["Preview/PrvText.txt"] = "원본 페이지 외형 보존 변환본".encode()
    for name in list(members):
        if name.startswith("BinData/image"):
            del members[name]
    for index, page in enumerate(encoded_pages, start=1):
        members[f"BinData/image{index}.jpg"] = page

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", MIMETYPE, compress_type=zipfile.ZIP_STORED)
        for name, data in members.items():
            if name == "mimetype":
                continue
            archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    return FidelityRenderStats(
        pages=len(encoded_pages),
        pictures=len(encoded_pages),
        package_bytes=output.stat().st_size,
    )


def _load_template() -> dict[str, bytes]:
    payload = Path(__file__).with_name("fidelity_template.b64").read_text(encoding="ascii")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _encode_page(path: Path) -> bytes:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        fitted = ImageOps.contain(image, TARGET_SIZE, method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", TARGET_SIZE, "white")
        x = (TARGET_SIZE[0] - fitted.width) // 2
        y = (TARGET_SIZE[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        stream = io.BytesIO()
        canvas.save(stream, "JPEG", quality=92, optimize=True, dpi=(200, 200))
        return stream.getvalue()


def _build_preview(path: Path) -> bytes:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((250, 354), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        image.save(stream, "PNG", optimize=True)
        return stream.getvalue()


def _build_section(template: bytes, pages: int) -> bytes:
    root = etree.fromstring(template)
    paragraphs = cast(list[etree._Element], root.xpath("./hp:p", namespaces={"hp": HP}))
    if len(paragraphs) < 2:
        raise RuntimeError("fidelity template must contain two picture paragraphs")
    first_template = copy.deepcopy(paragraphs[0])
    continuation_template = copy.deepcopy(paragraphs[1])
    for paragraph in paragraphs:
        root.remove(paragraph)

    for index in range(1, pages + 1):
        paragraph = copy.deepcopy(first_template if index == 1 else continuation_template)
        paragraph.set("pageBreak", "0" if index == 1 else "1")
        if index > 1:
            paragraph.set("id", str(3_121_190_098 + index - 1))
        pictures = cast(
            list[etree._Element], paragraph.xpath(".//hp:pic", namespaces={"hp": HP})
        )
        images = cast(
            list[etree._Element], paragraph.xpath(".//hc:img", namespaces={"hc": HC})
        )
        if len(pictures) != 1 or len(images) != 1:
            raise RuntimeError("fidelity template picture paragraph is invalid")
        pictures[0].set("id", str(1_175_785_464 + (index - 1) * 2))
        pictures[0].set("instid", str(102_043_641 + (index - 1) * 2))
        pictures[0].set("zOrder", str(index - 1))
        images[0].set("binaryItemIDRef", f"image{index}")
        comments = cast(
            list[etree._Element],
            paragraph.xpath(".//hp:shapeComment", namespaces={"hp": HP}),
        )
        if comments:
            comments[0].text = f"원본 PDF {index}페이지 외형 보존 그림"
        root.append(paragraph)
    return cast(
        bytes,
        etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True),
    )


def _build_manifest(template: bytes, pages: int) -> bytes:
    root = etree.fromstring(template)
    manifests = cast(
        list[etree._Element], root.xpath("./opf:manifest", namespaces={"opf": OPF})
    )
    if len(manifests) != 1:
        raise RuntimeError("fidelity template OPF manifest is invalid")
    manifest = manifests[0]
    for item in list(manifest):
        if item.get("id", "").startswith("image"):
            manifest.remove(item)
    section_index = next(
        (index for index, item in enumerate(manifest) if item.get("id") == "section0"),
        len(manifest),
    )
    for index in range(1, pages + 1):
        item = etree.Element(f"{{{OPF}}}item")
        item.set("id", f"image{index}")
        item.set("href", f"BinData/image{index}.jpg")
        item.set("media-type", "image/jpg")
        item.set("isEmbeded", "1")
        manifest.insert(section_index + index - 1, item)
    return cast(
        bytes,
        etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True),
    )
