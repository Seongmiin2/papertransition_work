from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

import pymupdf

_ALLOWED_HOSTS = {
    "codeload.github.com",
    "download.hancom.com",
    "raw.githubusercontent.com",
    "tech.hancom.com",
}
_SOURCE_TYPES = {"html", "pdf", "repository_zip"}
_ZIP_TEXT_SUFFIXES = {".cpp", ".h", ".hpp", ".json", ".md", ".txt", ".xml", ".xsd"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True, slots=True)
class KnowledgeSource:
    id: str
    title: str
    source_type: str
    revision: str
    url: str
    local_filename: str
    sha256: str
    bytes: int
    tags: tuple[str, ...]


def load_source_manifest(path: Path) -> tuple[KnowledgeSource, ...]:
    """Load the pinned official-source catalog and reject unsafe entries."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Hancom source manifest must use schema_version 1.0")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Hancom source manifest must contain sources")

    result: list[KnowledgeSource] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_sources, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"source {index} must be an object")  # noqa: TRY004
        source = _parse_source(raw, index)
        if source.id in ids:
            raise ValueError(f"duplicate source id: {source.id}")
        ids.add(source.id)
        result.append(source)
    return tuple(result)


def verify_source_files(sources: tuple[KnowledgeSource, ...], raw_dir: Path) -> None:
    """Verify size, digest and container signature before text extraction."""
    root = raw_dir.resolve()
    for source in sources:
        path = (root / source.local_filename).resolve()
        if path.parent != root:
            raise ValueError(f"source escapes raw directory: {source.local_filename}")
        if not path.is_file():
            raise FileNotFoundError(f"missing Hancom source: {path}")
        if path.stat().st_size != source.bytes:
            raise ValueError(f"size mismatch for {source.id}")
        if _file_sha256(path) != source.sha256:
            raise ValueError(f"SHA-256 mismatch for {source.id}")
        _verify_container(path, source.source_type, source.id)


def build_knowledge_corpus(
    manifest_path: Path,
    raw_dir: Path,
    output_dir: Path,
    *,
    max_chars: int = 1600,
    overlap: int = 160,
) -> dict[str, Any]:
    """Build deterministic JSONL chunks for Model B retrieval.

    The generated records are reference context, not supervised fine-tuning targets.
    """
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    sources = load_source_manifest(manifest_path)
    verify_source_files(sources, raw_dir)
    root = raw_dir.resolve()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped_sections: Counter[str] = Counter()

    for source in sources:
        path = root / source.local_filename
        for section, text in _iter_source_sections(path, source.source_type):
            chunks = _chunk_text(text, max_chars=max_chars, overlap=overlap)
            if not chunks:
                skipped_sections[source.id] += 1
                continue
            for part, chunk in enumerate(chunks, start=1):
                digest = hashlib.sha256(
                    f"{source.id}\0{section}\0{part}\0{chunk}".encode()
                ).hexdigest()
                records.append(
                    {
                        "id": f"{source.id}:{digest[:20]}",
                        "source_id": source.id,
                        "title": source.title,
                        "section": f"{section}#chunk-{part}",
                        "text": chunk,
                        "tags": list(source.tags),
                        "source_url": source.url,
                        "source_sha256": source.sha256,
                    }
                )
                counts[source.id] += 1

    records.sort(key=lambda item: (item["source_id"], item["section"], item["id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "chunks.jsonl"
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    _atomic_write(corpus_path, encoded)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": hashlib.sha256(encoded).hexdigest(),
        "sources": len(sources),
        "chunks": len(records),
        "chunks_by_source": dict(sorted(counts.items())),
        "skipped_sections_by_source": dict(sorted(skipped_sections.items())),
        "usage": "retrieval_context_for_document_author_model_b",
        "supervised_targets": False,
    }
    _atomic_write(
        output_dir / "build_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    return report


def _parse_source(raw: dict[str, Any], index: int) -> KnowledgeSource:
    required = {
        "id",
        "title",
        "source_type",
        "revision",
        "url",
        "local_filename",
        "sha256",
        "bytes",
        "authority",
        "license",
        "tags",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"source {index} missing fields: {', '.join(missing)}")
    if raw["authority"] != "Hancom Inc.":
        raise ValueError(f"source {index} is not marked as Hancom Inc. authority")
    source_type = _required_string(raw["source_type"], "source_type", index)
    if source_type not in _SOURCE_TYPES:
        raise ValueError(f"source {index} has unsupported source_type: {source_type}")
    url = _required_string(raw["url"], "url", index)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"source {index} URL is not on the official allowlist")
    filename = _required_string(raw["local_filename"], "local_filename", index)
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"source {index} local_filename must be a plain file name")
    digest = _required_string(raw["sha256"], "sha256", index).lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"source {index} sha256 must contain 64 hexadecimal characters")
    byte_count = raw["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise ValueError(f"source {index} bytes must be a positive integer")
    tags = raw["tags"]
    if not isinstance(tags, list) or not tags:
        raise ValueError(f"source {index} tags must be a non-empty list")
    normalized_tags = tuple(_required_string(tag, "tags", index) for tag in tags)
    if len({tag.casefold() for tag in normalized_tags}) != len(normalized_tags):
        raise ValueError(f"source {index} tags contain duplicates")
    license_info = raw["license"]
    if not isinstance(license_info, dict) or not license_info.get("redistribution"):
        raise ValueError(f"source {index} must document redistribution terms")
    return KnowledgeSource(
        id=_required_string(raw["id"], "id", index),
        title=_required_string(raw["title"], "title", index),
        source_type=source_type,
        revision=_required_string(raw["revision"], "revision", index),
        url=url,
        local_filename=filename,
        sha256=digest,
        bytes=byte_count,
        tags=normalized_tags,
    )


def _required_string(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source {index} {field} must be a non-empty string")
    return value.strip()


def _verify_container(path: Path, source_type: str, source_id: str) -> None:
    if source_type == "pdf":
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"invalid PDF signature for {source_id}")
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            if document.page_count == 0:
                raise ValueError(f"empty PDF for {source_id}")
        return
    if source_type == "html":
        if "<html" not in path.read_text(encoding="utf-8", errors="ignore").casefold():
            raise ValueError(f"invalid HTML for {source_id}")
        return
    try:
        with ZipFile(path) as archive:
            if not archive.infolist() or archive.testzip() is not None:
                raise ValueError(f"invalid ZIP contents for {source_id}")
    except BadZipFile as exc:
        raise ValueError(f"invalid ZIP for {source_id}") from exc


def _iter_source_sections(path: Path, source_type: str) -> Iterator[tuple[str, str]]:
    if source_type == "pdf":
        with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
            for index, page in enumerate(document, start=1):
                yield f"page-{index}", page.get_text("text")
        return
    if source_type == "html":
        parser = _VisibleTextParser()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        yield "web-page", parser.text()
        return
    with ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename.casefold()):
            if info.is_dir() or Path(info.filename).suffix.casefold() not in _ZIP_TEXT_SUFFIXES:
                continue
            if info.file_size > 2_000_000:
                continue
            raw = archive.read(info)
            if b"\x00" in raw:
                continue
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp949", errors="replace")
            yield info.filename, text


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1
        elif self._hidden_depth == 0 and tag in {"article", "br", "div", "h1", "h2", "h3", "li", "p", "section", "td", "th", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif self._hidden_depth == 0 and tag in {"article", "div", "h1", "h2", "h3", "li", "p", "section", "td", "th", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _chunk_text(text: str, *, max_chars: int, overlap: int) -> tuple[str, ...]:
    normalized_lines = [_SPACE.sub(" ", line).strip() for line in text.replace("\r", "").split("\n")]
    normalized = "\n".join(line for line in normalized_lines if line).strip()
    if len(normalized) < 40:
        return ()
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        proposed = min(len(normalized), start + max_chars)
        end = proposed
        if proposed < len(normalized):
            boundary = max(normalized.rfind("\n", start + max_chars // 2, proposed), normalized.rfind(" ", start + max_chars // 2, proposed))
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return tuple(chunks)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(path)
