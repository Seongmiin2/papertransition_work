import zipfile
from pathlib import Path

from scan2hwpx.hwpx import validate_hwpx


def test_validator_accepts_root_and_contents_relative_hrefs(tmp_path: Path) -> None:
    path = tmp_path / "hancom-style.hwpx"
    entries = {
        "mimetype": b"application/hwp+zip",
        "META-INF/container.xml": b"<container/>",
        "META-INF/manifest.xml": b"<manifest/>",
        "Contents/header.xml": b"<head/>",
        "Contents/section0.xml": b"<sec/>",
        "settings.xml": b"<settings/>",
        "Contents/content.hpf": (
            b"<package><manifest>"
            b"<item href='Contents/header.xml'/><item href='section0.xml'/>"
            b"<item href='settings.xml'/></manifest></package>"
        ),
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", entries.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name, data in entries.items():
            archive.writestr(name, data)
    assert validate_hwpx(path).valid
