from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ai.datasets.build_hwp_line_dataset as dataset_module
from ai.datasets.build_hwp_line_dataset import (
    _cached_pdf_matches_source,
    _load_split_manifest,
    _split_documents,
    _write_pdf_provenance,
)


def _source(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_train_document_does_not_move_when_documents_are_added(
    tmp_path: Path,
) -> None:
    initial = [
        _source(tmp_path / f"initial-{index}.hwp", f"document-{index}".encode())
        for index in range(30)
    ]
    before = _split_documents(initial, 20260826)
    prior_train = next(path for path in initial if before[path].split == "train")
    added = [
        _source(tmp_path / f"added-{index}.hwp", f"new-document-{index}".encode())
        for index in range(10)
    ]

    after = _split_documents([*initial, *added], 20260826)

    assert after[prior_train].split == "train"
    assert {path: after[path] for path in initial} == before


def test_identical_lineage_cannot_cross_splits(tmp_path: Path) -> None:
    first = _source(tmp_path / "first.hwp", b"same-document-bytes")
    copied = _source(tmp_path / "copied.hwp", b"same-document-bytes")

    assignments = _split_documents([first, copied], 20260826)

    assert assignments[first].source_sha256 == assignments[copied].source_sha256
    assert assignments[first].split == assignments[copied].split


def test_explicit_template_family_stays_in_one_frozen_split(tmp_path: Path) -> None:
    first = _source(tmp_path / "first.hwp", b"first")
    second = _source(tmp_path / "second.hwp", b"second")
    unrelated = _source(tmp_path / "unrelated.hwp", b"unrelated")
    manifest = tmp_path / "split-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "hwp-ocr-split-map/1.0",
                "documents": [
                    {
                        "sha256": _sha256(first),
                        "template_family": "Publisher-A",
                        "split": "train",
                    },
                    {
                        "source_sha256": _sha256(second),
                        "template_family": "publisher-a",
                    },
                    {
                        "sha256": _sha256(unrelated),
                        "template_family": "publisher-b",
                        "split": "test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assignments = _split_documents(
        [first, second, unrelated],
        20260826,
        frozen_splits=_load_split_manifest(manifest),
    )

    assert assignments[first].split == "train"
    assert assignments[first].source == "frozen_document"
    assert assignments[second].split == "train"
    assert assignments[second].source == "frozen_template_family"
    assert assignments[unrelated].split == "test"


def test_split_manifest_rejects_template_family_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "split-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "sha256": "1" * 64,
                        "template_family": "Publisher-A",
                        "split": "train",
                    },
                    {
                        "sha256": "2" * 64,
                        "template_family": "publisher-a",
                        "split": "test",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="template family .* multiple splits"):
        _load_split_manifest(manifest)


def test_split_manifest_rejects_format_control_family_alias(tmp_path: Path) -> None:
    manifest = tmp_path / "split-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "sha256": "1" * 64,
                        "template_family": "publisher\u200b-a",
                        "split": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid template_family"):
        _load_split_manifest(manifest)


def test_cached_pdf_requires_full_source_provenance_and_exact_pdf_digest(
    tmp_path: Path,
) -> None:
    source_sha256 = hashlib.sha256(b"source").hexdigest()
    pdf = tmp_path / f"{source_sha256}.pdf"
    provenance = tmp_path / f"{source_sha256}.provenance.json"
    pdf.write_bytes(b"%PDF-test")

    assert not _cached_pdf_matches_source(pdf, provenance, source_sha256)
    _write_pdf_provenance(provenance, source_sha256, _sha256(pdf))
    assert _cached_pdf_matches_source(pdf, provenance, source_sha256)

    pdf.write_bytes(b"%PDF-tampered")
    assert not _cached_pdf_matches_source(pdf, provenance, source_sha256)


def test_split_manifest_size_limit_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "split-manifest.json"
    manifest.write_text('{"documents": []}', encoding="utf-8")
    monkeypatch.setattr(dataset_module, "_MAX_SPLIT_MANIFEST_BYTES", 4)

    with pytest.raises(ValueError, match="exceeds the size limit"):
        _load_split_manifest(manifest)
