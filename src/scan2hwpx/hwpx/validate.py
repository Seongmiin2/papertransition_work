from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
    "settings.xml",
    "version.xml",
}

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS = "http://www.hancom.co.kr/hwpml/2011/section"
HH = "http://www.hancom.co.kr/hwpml/2011/head"
HV = "http://www.hancom.co.kr/hwpml/2011/version"
HA = "http://www.hancom.co.kr/hwpml/2011/app"
OPF = "http://www.idpf.org/2007/opf/"
OCF = "urn:oasis:names:tc:opendocument:xmlns:container"
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ENTRY_BYTES = 128 * 1024 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_COMPRESSED_PACKAGE_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000


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
    try:
        archive_context = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return ValidationResult(["not a readable ZIP package"], warnings)
    with archive_context as archive:
        errors.extend(_archive_safety_errors(archive))
        if errors:
            return ValidationResult(errors, warnings)
        try:
            broken_entry = archive.testzip()
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            errors.append(f"failed to verify ZIP entries: {exc}")
            return ValidationResult(errors, warnings)
        if broken_entry is not None:
            errors.append(f"corrupt ZIP entry: {broken_entry}")
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
                payload = archive.read(name)
                lowered = payload.lower()
                if b"<!doctype" in lowered or b"<!entity" in lowered:
                    errors.append(f"DTD or entity declaration is forbidden in {name}")
                    continue
                parser = etree.XMLParser(
                    resolve_entities=False,
                    no_network=True,
                    load_dtd=False,
                    recover=False,
                    huge_tree=False,
                )
                parsed[name] = cast(etree._Element, etree.fromstring(payload, parser=parser))
            except (OSError, RuntimeError, etree.XMLSyntaxError) as exc:
                errors.append(f"malformed XML {name}: {exc}")
        content = parsed.get("Contents/content.hpf")
        if content is not None:
            _require_root(content, OPF, "package", "Contents/content.hpf", errors)
            if "unique-identifier" not in content.attrib:
                errors.append("Contents/content.hpf missing unique-identifier")
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
        container = parsed.get("META-INF/container.xml")
        if container is not None:
            _require_root(container, OCF, "container", "META-INF/container.xml", errors)
            root_media = cast(
                list[str],
                container.xpath(
                    "//*[local-name()='rootfile'][@full-path='Contents/content.hpf']/@media-type"
                ),
            )
            if root_media != ["application/hwpml-package+xml"]:
                errors.append("container rootfile has invalid HWPX media type")
        header = parsed.get("Contents/header.xml")
        if header is not None:
            _require_root(header, HH, "head", "Contents/header.xml", errors)
            if not header.xpath("./*[local-name()='refList']"):
                errors.append("Contents/header.xml missing refList")
            for name in ("fontfaces", "charProperties", "paraProperties", "styles"):
                if not header.xpath(f".//*[local-name()='{name}']"):
                    errors.append(f"Contents/header.xml missing {name}")
        section = parsed.get("Contents/section0.xml")
        if section is not None:
            _require_root(section, HS, "sec", "Contents/section0.xml", errors)
            if not section.xpath("./hp:p/hp:run/hp:secPr", namespaces={"hp": HP}):
                errors.append("Contents/section0.xml missing section properties")
        version = parsed.get("version.xml")
        if version is not None:
            _require_root(version, HV, "HCFVersion", "version.xml", errors)
            if "major" not in version.attrib or "minor" not in version.attrib:
                errors.append("version.xml missing Hancom version attributes")
        settings = parsed.get("settings.xml")
        if settings is not None:
            _require_root(settings, HA, "HWPApplicationSetting", "settings.xml", errors)
    return ValidationResult(errors, warnings)


def _archive_safety_errors(archive: zipfile.ZipFile) -> list[str]:
    infos = archive.infolist()
    errors: list[str] = []
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        errors.append("ZIP package contains too many entries")
        return errors
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
        errors.append("ZIP package contains duplicate entry names")
    total = 0
    compressed_total = 0
    for info in infos:
        member = PurePosixPath(info.filename)
        if (
            "\\" in info.filename
            or member.is_absolute()
            or ".." in member.parts
            or any(":" in part for part in member.parts)
        ):
            errors.append(f"unsafe ZIP entry name: {info.filename}")
            continue
        if info.flag_bits & 0x1:
            errors.append(f"encrypted ZIP entry is forbidden: {info.filename}")
        if info.file_size > MAX_ENTRY_BYTES:
            errors.append(f"ZIP entry exceeds size limit: {info.filename}")
        total += info.file_size
        compressed_total += info.compress_size
        if info.compress_size == 0:
            if info.file_size > 0:
                errors.append(f"ZIP entry has invalid compressed size: {info.filename}")
        elif info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            errors.append(f"ZIP entry exceeds compression-ratio limit: {info.filename}")
    if total > MAX_PACKAGE_BYTES:
        errors.append("ZIP package exceeds uncompressed size limit")
    if compressed_total > MAX_COMPRESSED_PACKAGE_BYTES:
        errors.append("ZIP package exceeds compressed size limit")
    return errors


def _require_root(
    root: etree._Element,
    namespace: str,
    localname: str,
    path: str,
    errors: list[str],
) -> None:
    name = etree.QName(root)
    if name.namespace != namespace or name.localname != localname:
        errors.append(
            f"invalid root element in {path}: expected {{{namespace}}}{localname}, got {root.tag}"
        )
