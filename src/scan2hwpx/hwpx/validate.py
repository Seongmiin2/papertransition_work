from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lxml import etree  # type: ignore[import-untyped]

from .render import MIMETYPE

REQUIRED = {
    "mimetype",
    "META-INF/container.xml",
    "META-INF/manifest.xml",
    "Contents/content.hpf",
    "Contents/header.xml",
    "Contents/section0.xml",
}


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_hwpx(path: Path) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not zipfile.is_zipfile(path):
        return ValidationResult(["not a ZIP package"], warnings)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        errors.extend(f"missing required entry: {name}" for name in sorted(REQUIRED - names))
        if archive.namelist() and archive.namelist()[0] != "mimetype":
            errors.append("mimetype must be the first ZIP entry")
        if "mimetype" in names:
            info = archive.getinfo("mimetype")
            if info.compress_type != zipfile.ZIP_STORED:
                errors.append("mimetype must be stored without compression")
            if archive.read("mimetype") != MIMETYPE:
                errors.append("invalid mimetype value")
        parsed: dict[str, etree._Element] = {}
        for name in sorted(name for name in names if name.endswith((".xml", ".hpf"))):
            try:
                parsed[name] = etree.fromstring(archive.read(name))
            except etree.XMLSyntaxError as exc:
                errors.append(f"malformed XML {name}: {exc}")
        content = parsed.get("Contents/content.hpf")
        if content is not None:
            hrefs = cast(list[str], content.xpath("//*[local-name()='item']/@href"))
            for href in hrefs:
                target = href if href in names else f"Contents/{href}"
                if target not in names:
                    errors.append(f"unresolved content reference: {target}")
        manifest = parsed.get("META-INF/manifest.xml")
        if manifest is not None:
            paths = cast(list[str], manifest.xpath("//*[local-name()='file-entry']/@full-path"))
            for item in paths:
                if item != "/" and item not in names:
                    errors.append(f"unresolved manifest path: {item}")
    return ValidationResult(errors, warnings)
