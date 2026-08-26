from __future__ import annotations

import hashlib
import re
import struct
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import olefile  # type: ignore[import-untyped]

HWP_SIGNATURE = b"HWP Document File"
HWPTAG_BEGIN = 0x10
PARA_HEADER = HWPTAG_BEGIN + 50
PARA_TEXT = HWPTAG_BEGIN + 51
CTRL_HEADER = HWPTAG_BEGIN + 55
TABLE = HWPTAG_BEGIN + 61
PICTURE = HWPTAG_BEGIN + 69
EQEDIT = HWPTAG_BEGIN + 72

_INLINE_CONTROLS = frozenset((*range(4, 10), 19, 20))
_EXTENDED_CONTROLS = frozenset((*range(1, 4), 11, 12, *range(14, 19), *range(21, 24)))
_QUESTION_PATTERN = re.compile(r"^\s*(?:문제\s*)?(\d{1,3})\s*[.)번]\s*")


@dataclass(frozen=True)
class HwpRecord:
    tag_id: int
    level: int
    payload: bytes


@dataclass
class HwpParagraph:
    text: str = ""
    para_shape_id: int | None = None
    style_id: int | None = None
    break_flags: int | None = None


@dataclass
class HwpReferenceDocument:
    path: str
    sha256: str
    version: str
    compressed: bool
    preview_text: str
    paragraphs: list[HwpParagraph] = field(default_factory=list)
    section_count: int = 0
    table_count: int = 0
    picture_count: int = 0
    equation_count: int = 0
    control_count: int = 0
    record_count: int = 0
    question_numbers: list[int] = field(default_factory=list)
    para_shape_counts: dict[str, int] = field(default_factory=dict)
    style_counts: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(paragraph.text for paragraph in self.paragraphs if paragraph.text)

    @property
    def character_count(self) -> int:
        return len(self.text)

    def to_dict(self, *, include_paragraphs: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["character_count"] = self.character_count
        result["paragraph_count"] = len(self.paragraphs)
        if not include_paragraphs:
            result.pop("paragraphs", None)
            result["preview_excerpt"] = self.preview_text[:400]
            result.pop("preview_text", None)
        return result


def iter_hwp_records(data: bytes) -> list[HwpRecord]:
    records: list[HwpRecord] = []
    offset = 0
    while offset + 4 <= len(data):
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            if offset + 4 > len(data):
                raise ValueError("truncated extended HWP record size")
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        end = offset + size
        if end > len(data):
            raise ValueError(f"truncated HWP record payload: wanted {size} bytes")
        records.append(HwpRecord(tag_id=tag_id, level=level, payload=data[offset:end]))
        offset = end
    if offset != len(data):
        raise ValueError("trailing bytes after final HWP record")
    return records


def decode_paragraph_text(payload: bytes) -> str:
    if len(payload) % 2:
        payload = payload[:-1]
    units = list(struct.unpack(f"<{len(payload) // 2}H", payload)) if payload else []
    pieces: list[str] = []
    normal: list[int] = []

    def flush() -> None:
        if not normal:
            return
        raw = struct.pack(f"<{len(normal)}H", *normal)
        pieces.append(raw.decode("utf-16le", errors="replace"))
        normal.clear()

    index = 0
    while index < len(units):
        code = units[index]
        if code > 31:
            normal.append(code)
            index += 1
            continue
        flush()
        if code in _INLINE_CONTROLS or code in _EXTENDED_CONTROLS:
            if code == 9:
                pieces.append("\t")
            index += min(8, len(units) - index)
            continue
        if code == 10:
            pieces.append("\n")
        elif code == 24:
            pieces.append("-")
        elif code in (30, 31):
            pieces.append(" ")
        index += 1
    flush()
    return "".join(pieces).replace("\x00", "").strip()


def read_hwp_reference(path: Path) -> HwpReferenceDocument:
    source = path.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    ole = olefile.OleFileIO(str(source))
    try:
        header = ole.openstream("FileHeader").read()
        if not header.startswith(HWP_SIGNATURE) or len(header) < 40:
            raise ValueError(f"not a supported HWP 5 document: {source.name}")
        version = ".".join(str(value) for value in reversed(header[32:36]))
        flags = struct.unpack_from("<I", header, 36)[0]
        if flags & 0b10:
            raise ValueError(f"password-protected HWP is not supported: {source.name}")
        compressed = bool(flags & 0b1)
        preview = ""
        if ole.exists("PrvText"):
            preview = ole.openstream("PrvText").read().decode("utf-16le", errors="replace").rstrip("\x00")

        stream_names = ["/".join(parts) for parts in ole.listdir(streams=True, storages=False)]
        sections = sorted(
            (name for name in stream_names if re.fullmatch(r"BodyText/Section\d+", name)),
            key=lambda value: int(value.rsplit("Section", 1)[1]),
        )
        paragraphs: list[HwpParagraph] = []
        tag_counts: Counter[int] = Counter()
        for section in sections:
            section_data = ole.openstream(section).read()
            if compressed:
                section_data = zlib.decompress(section_data, -15)
            current: HwpParagraph | None = None
            for record in iter_hwp_records(section_data):
                tag_counts[record.tag_id] += 1
                if record.tag_id == PARA_HEADER:
                    if current is not None:
                        paragraphs.append(current)
                    current = _paragraph_from_header(record.payload)
                elif record.tag_id == PARA_TEXT:
                    if current is None:
                        current = HwpParagraph()
                    text = decode_paragraph_text(record.payload)
                    current.text = f"{current.text}{text}"
            if current is not None:
                paragraphs.append(current)

        shape_counts = Counter(
            str(paragraph.para_shape_id)
            for paragraph in paragraphs
            if paragraph.para_shape_id is not None
        )
        style_counts = Counter(
            str(paragraph.style_id) for paragraph in paragraphs if paragraph.style_id is not None
        )
        question_numbers = [
            int(match.group(1))
            for paragraph in paragraphs
            if (match := _QUESTION_PATTERN.match(paragraph.text)) is not None
        ]
        return HwpReferenceDocument(
            path=str(source),
            sha256=digest,
            version=version,
            compressed=compressed,
            preview_text=preview,
            paragraphs=paragraphs,
            section_count=len(sections),
            table_count=tag_counts[TABLE],
            picture_count=tag_counts[PICTURE],
            equation_count=tag_counts[EQEDIT],
            control_count=tag_counts[CTRL_HEADER],
            record_count=sum(tag_counts.values()),
            question_numbers=question_numbers,
            para_shape_counts=dict(shape_counts),
            style_counts=dict(style_counts),
        )
    finally:
        ole.close()


def _paragraph_from_header(payload: bytes) -> HwpParagraph:
    if len(payload) < 12:
        return HwpParagraph()
    return HwpParagraph(
        para_shape_id=struct.unpack_from("<H", payload, 8)[0],
        style_id=payload[10],
        break_flags=payload[11],
    )
