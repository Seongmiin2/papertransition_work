from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pymupdf
import pytest

from scan2hwpx.knowledge.build import build_knowledge_corpus, load_source_manifest
from scan2hwpx.knowledge.hancom import load_chunks, retrieve_chunks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, identifier: str, source_type: str, url: str, tags: list[str]) -> dict:
    return {
        "id": identifier,
        "title": identifier,
        "source_type": source_type,
        "revision": "test-revision",
        "url": url,
        "local_filename": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "authority": "Hancom Inc.",
        "license": {"redistribution": "test only"},
        "tags": tags,
    }


def _write_manifest(path: Path, sources: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": "1.0", "sources": sources}),
        encoding="utf-8",
    )


def test_builds_loadable_deterministic_corpus_from_html_and_repository(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    article = raw / "article.html"
    article.write_text(
        "<html><body><article><h1>HWPX 구조</h1><p>header.xml의 스타일은 "
        "section.xml의 문단에서 참조합니다.</p></article><script>ignore me</script></body></html>",
        encoding="utf-8",
    )
    repository = raw / "model.zip"
    with ZipFile(repository, "w") as archive:
        archive.writestr(
            "official/OWPML/Paragraph.h",
            "Paragraph elements reference paragraph and character properties in header.xml.",
        )
        archive.writestr("official/binary.bin", b"\x00\x01")

    manifest = tmp_path / "sources.json"
    _write_manifest(
        manifest,
        [
            _source(
                article,
                "format",
                "html",
                "https://tech.hancom.com/hwpxformat/",
                ["hwpx", "style"],
            ),
            _source(
                repository,
                "model",
                "repository_zip",
                "https://codeload.github.com/hancom-io/hwpx-owpml-model/zip/test",
                ["owpml", "element"],
            ),
        ],
    )

    first = build_knowledge_corpus(manifest, raw, tmp_path / "first", max_chars=300)
    second = build_knowledge_corpus(manifest, raw, tmp_path / "second", max_chars=300)
    chunks = load_chunks(tmp_path / "first" / "chunks.jsonl")

    assert first["sources"] == 2
    assert first["chunks"] == 2
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert (
        retrieve_chunks("header.xml 스타일", tags=("style",), chunks=chunks)[0].source_id
        == "format"
    )
    assert "ignore me" not in " ".join(chunk.text for chunk in chunks)


def test_extracts_pdf_pages_and_rejects_changed_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "automation.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "HwpAutomation document save and export reference")
    document.save(pdf)
    document.close()
    source = _source(
        pdf,
        "automation",
        "pdf",
        "https://raw.githubusercontent.com/hancom-io/devcenter-archive/test/automation.pdf",
        ["automation"],
    )
    manifest = tmp_path / "sources.json"
    _write_manifest(manifest, [source])

    report = build_knowledge_corpus(manifest, raw, tmp_path / "processed")

    assert report["chunks_by_source"] == {"automation": 1}
    pdf.write_bytes(pdf.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        build_knowledge_corpus(manifest, raw, tmp_path / "changed")


def test_manifest_rejects_unofficial_url_and_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.json"
    source = {
        "id": "bad",
        "title": "bad",
        "source_type": "html",
        "revision": "test",
        "url": "https://example.com/spec",
        "local_filename": "../spec.html",
        "sha256": "a" * 64,
        "bytes": 1,
        "authority": "Hancom Inc.",
        "license": {"redistribution": "test"},
        "tags": ["hwpx"],
    }
    _write_manifest(manifest, [source])

    with pytest.raises(ValueError, match="official allowlist|plain file name"):
        load_source_manifest(manifest)
