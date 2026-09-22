from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from scan2hwpx import pipeline
from scan2hwpx.hwpx import hancom
from scan2hwpx.ir.models import Document
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider


class FakeHwp:
    instances: ClassVar[list[FakeHwp]] = []

    def __init__(self, *, open_results: list[bool], page_count: int = 1) -> None:
        self.open_results = open_results
        self.PageCount = page_count
        self.opened: list[str] = []
        self.quit_called = False
        FakeHwp.instances.append(self)

    def SetMessageBoxMode(self, mode: int) -> int:
        return 0

    def open(self, path: str, arg: str = "") -> bool:
        assert "forceopen" not in arg
        self.opened.append(path)
        return self.open_results.pop(0)

    def save_as(self, path: str, format: str = "") -> bool:
        Path(path).write_bytes(b"resaved")
        return True

    def close(self, is_dirty: bool = False) -> None:
        pass

    def quit(self) -> None:
        self.quit_called = True


def _install_fake_pyhwpx(monkeypatch: pytest.MonkeyPatch, factory: object) -> None:
    module = types.ModuleType("pyhwpx")
    module.Hwp = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyhwpx", module)
    monkeypatch.setattr(hancom, "_ensure_hancom_security_module", lambda: None)
    FakeHwp.instances.clear()


def test_roundtrip_opens_saves_and_reopens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pyhwpx(
        monkeypatch, lambda **_kwargs: FakeHwp(open_results=[True, True], page_count=2)
    )
    source = tmp_path / "result.hwpx"
    source.write_bytes(b"hwpx")

    assert hancom.verify_hancom_roundtrip(source, 2) is True

    hwp = FakeHwp.instances[0]
    assert hwp.opened[0] == str(source.resolve())
    assert hwp.opened[1].endswith("roundtrip.hwpx")
    assert hwp.quit_called


def test_roundtrip_raises_when_hancom_rejects_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pyhwpx(monkeypatch, lambda **_kwargs: FakeHwp(open_results=[False]))

    with pytest.raises(hancom.HancomRoundTripError):
        hancom.verify_hancom_roundtrip(tmp_path / "broken.hwpx", 1)
    assert FakeHwp.instances[0].quit_called


def test_roundtrip_raises_on_page_count_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pyhwpx(monkeypatch, lambda **_kwargs: FakeHwp(open_results=[True], page_count=3))

    with pytest.raises(hancom.HancomRoundTripError, match="3쪽"):
        hancom.verify_hancom_roundtrip(tmp_path / "result.hwpx", 2)


def test_roundtrip_is_skipped_when_hancom_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(**_kwargs: object) -> None:
        raise OSError("Hancom is not installed")

    _install_fake_pyhwpx(monkeypatch, unavailable)

    assert hancom.verify_hancom_roundtrip(tmp_path / "result.hwpx", 1) is False


def _document_with_page_image(tmp_path: Path) -> Document:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    page_image = tmp_path / "debug" / "original" / "page-1.png"
    page_image.parent.mkdir(parents=True)
    Image.new("RGB", (248, 351), "white").save(page_image)
    return document


def test_render_stage_keeps_previous_output_when_hancom_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document_with_page_image(tmp_path)
    output = tmp_path / "result.hwpx"
    output.write_bytes(b"previous result")

    def reject(path: Path, expected_pages: int) -> bool:
        assert expected_pages == 1
        raise hancom.HancomRoundTripError("rejected")

    monkeypatch.setattr(pipeline, "verify_hancom_roundtrip", reject)

    with pytest.raises(hancom.HancomRoundTripError):
        pipeline.run_render_stage(
            document, tmp_path, output, renderer="fidelity", verify_hancom=True
        )

    assert output.read_bytes() == b"previous result"
    assert not (tmp_path / ".result.candidate.hwpx").exists()


def test_render_stage_reports_hancom_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document_with_page_image(tmp_path)
    monkeypatch.setattr(pipeline, "verify_hancom_roundtrip", lambda *_args: True)

    summary = pipeline.run_render_stage(
        document, tmp_path, tmp_path / "result.hwpx", renderer="fidelity", verify_hancom=True
    )

    assert summary["hancom_reopened"] is True
