from __future__ import annotations

import struct

import pytest

from scan2hwpx.reference.dataset import normalize_exam_name
from scan2hwpx.reference.hwp5 import decode_paragraph_text, iter_hwp_records


def _record(tag: int, level: int, payload: bytes) -> bytes:
    header = tag | (level << 10) | (len(payload) << 20)
    return struct.pack("<I", header) + payload


def test_iter_hwp_records_decodes_standard_header() -> None:
    records = iter_hwp_records(_record(67, 1, b"abcd") + _record(77, 2, b"xy"))

    assert [(record.tag_id, record.level, record.payload) for record in records] == [
        (67, 1, b"abcd"),
        (77, 2, b"xy"),
    ]


def test_iter_hwp_records_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="truncated"):
        iter_hwp_records(_record(67, 1, b"abcd")[:-1])


def test_decode_paragraph_text_preserves_korean_and_useful_controls() -> None:
    values = [ord("문"), ord("제"), 9, 0, 0, 0, 0, 0, 0, 0, ord("답"), 10, ord("끝"), 13]
    payload = struct.pack(f"<{len(values)}H", *values)

    assert decode_paragraph_text(payload) == "문제\t답\n끝"


def test_normalize_exam_name_removes_publisher_and_copy_suffix() -> None:
    assert normalize_exam_name("(천재박)홍천중3_1학기 기말고사 (1)") == "홍천중31학기기말고사"
