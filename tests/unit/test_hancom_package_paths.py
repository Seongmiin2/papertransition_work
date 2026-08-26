import zipfile
from pathlib import Path

from lxml import etree

from scan2hwpx.hwpx import ensure_minimal_template, validate_hwpx


def test_validator_accepts_root_and_contents_relative_hrefs(tmp_path: Path) -> None:
    path = tmp_path / "hancom-style.hwpx"
    ensure_minimal_template(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    content = etree.fromstring(entries["Contents/content.hpf"])
    header = content.xpath("//*[local-name()='item'][@id='header']")[0]
    header.set("href", "Contents/header.xml")
    entries["Contents/content.hpf"] = etree.tostring(
        content, xml_declaration=True, encoding="UTF-8"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", entries["mimetype"], compress_type=zipfile.ZIP_STORED)
        for name, data in entries.items():
            if name != "mimetype":
                archive.writestr(name, data)
    assert validate_hwpx(path).valid
