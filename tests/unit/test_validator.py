import zipfile
from pathlib import Path

from scan2hwpx.hwpx import ensure_minimal_template, validate_hwpx


def test_minimal_template_validates(tmp_path: Path) -> None:
    template = tmp_path / "exam_base.hwpx"
    ensure_minimal_template(template)
    assert validate_hwpx(template).valid


def test_validator_rejects_missing_entries(tmp_path: Path) -> None:
    broken = tmp_path / "broken.hwpx"
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr("mimetype", "wrong")
    result = validate_hwpx(broken)
    assert not result.valid
    assert any("missing required entry" in error for error in result.errors)


def test_validator_rejects_well_formed_but_non_hancom_xml(tmp_path: Path) -> None:
    path = tmp_path / "structurally-invalid.hwpx"
    ensure_minimal_template(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["Contents/header.xml"] = (
        b'<head xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head" version="1.4"/>'
    )
    entries["Contents/section0.xml"] = (
        b'<hp:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"/>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", entries["mimetype"], compress_type=zipfile.ZIP_STORED)
        for name, data in entries.items():
            if name != "mimetype":
                archive.writestr(name, data)

    result = validate_hwpx(path)

    assert not result.valid
    assert any("header.xml missing refList" in error for error in result.errors)
    assert any("invalid root element in Contents/section0.xml" in error for error in result.errors)
