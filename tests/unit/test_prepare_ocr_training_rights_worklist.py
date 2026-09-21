from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import ai.datasets.prepare_ocr_training_rights_worklist as worklist_module
from ai.datasets.prepare_ocr_training_rights_worklist import (
    WORKLIST_SCHEMA,
    prepare_ocr_training_rights_worklist,
)
from scan2hwpx.evaluation.ocr_training_rights import (
    OcrTrainingRightsError,
    load_ocr_training_rights_manifest,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fixture(
    root: Path,
    *,
    refs: tuple[str, ...] = ("grade1/alpha.hwp", "grade2/beta.hwp"),
) -> tuple[Path, Path, list[dict[str, Any]]]:
    source_root = root / "source"
    source_root.mkdir()
    documents: list[dict[str, Any]] = []
    splits = ("train", "test")
    for index, source_ref in enumerate(refs):
        payload = f"source-{index}".encode()
        if _is_safe_fixture_ref(source_ref):
            source_path = source_root.joinpath(*PurePosixPath(source_ref).parts)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(payload)
        source_sha256 = _sha256(payload)
        documents.append(
            {
                "source_sha256": source_sha256,
                "source_ref": source_ref,
                "split": splits[index % len(splits)],
                "page_count": index + 1,
                "hwpx": {
                    "path": f"{source_sha256}/source.hwpx",
                    "sha256": _sha256(f"hwpx-{index}".encode()),
                },
                "pdf": {
                    "path": f"{source_sha256}/source.pdf",
                    "sha256": _sha256(f"pdf-{index}".encode()),
                },
            }
        )
    split_counts = {
        split: sum(document["split"] == split for document in documents)
        for split in ("train", "validation", "test")
    }
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "hwp-hwpx-projection-pairs/1.0",
                "artifact_role": "paired_source_for_projection",
                "research_only": True,
                "rights_manifest": {"status": "missing"},
                "source_dataset_report_sha256": "1" * 64,
                "source_inventory_sha256": "2" * 64,
                "document_count": len(documents),
                "splits": split_counts,
                "producer": {"name": "test", "version": "1.0"},
                "documents": documents,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest, source_root, documents


def _is_safe_fixture_ref(source_ref: str) -> bool:
    ref = PurePosixPath(source_ref)
    return (
        not ref.is_absolute()
        and ref.as_posix() == source_ref
        and ":" not in source_ref
        and "\\" not in source_ref
        and ".." not in ref.parts
        and all(ord(character) >= 32 for character in source_ref)
    )


def test_builds_non_authorizing_human_worklist_bound_to_manifest(tmp_path: Path) -> None:
    manifest, source_root, source_documents = _write_fixture(tmp_path)
    manifest_sha256 = _sha256(manifest.read_bytes())
    output = tmp_path / "rights-worklist.json"

    result = prepare_ocr_training_rights_worklist(manifest, source_root, output)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["schema_version"] == WORKLIST_SCHEMA
    assert result["artifact_role"] == "human_rights_review_worklist_only"
    assert result["training_authorized"] is False
    assert result["candidate_split_is_informational"] is True
    assert result["source_manifest_sha256"] == manifest_sha256
    assert result["document_count"] == 2
    expected_by_sha = {item["source_sha256"]: item for item in source_documents}
    for item in result["documents"]:
        source = expected_by_sha[item["source_sha256"]]
        assert item["source_ref"] == source["source_ref"]
        assert item["page_count"] == source["page_count"]
        assert item["candidate_split"] == source["split"]
        for field in item["needs_human_input"]:
            assert item[field] is None
        assert set(item["needs_human_input"]) == {
            "template_family",
            "final_split",
            "rights_basis",
            "rights_evidence_artifact_ref",
            "rights_evidence_sha256",
            "attestor_id",
            "attested_at",
        }
        assert not PurePosixPath(item["source_ref"]).is_absolute()
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".rights-worklist.json.*.partial"))


def test_worklist_cannot_validate_as_training_rights_manifest(tmp_path: Path) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)
    output = tmp_path / "rights-worklist.json"
    prepare_ocr_training_rights_worklist(manifest, source_root, output)

    with pytest.raises(OcrTrainingRightsError, match="rights manifest is invalid"):
        load_ocr_training_rights_manifest(output)


def test_rejects_duplicate_json_keys_before_creating_output(tmp_path: Path) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)
    raw = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        raw.replace('"research_only": true', '"research_only": true, "research_only": true'),
        encoding="utf-8",
    )
    output = tmp_path / "rights-worklist.json"

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        prepare_ocr_training_rights_worklist(manifest, source_root, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "source_ref",
    (
        "../escape.hwp",
        "/absolute.hwp",
        "C:/absolute.hwp",
        "grade\\file.hwp",
        "grade//file.hwp",
        "grade/not-hwp.txt",
        "grade/unsafe\u0001.hwp",
    ),
)
def test_rejects_unsafe_source_refs(tmp_path: Path, source_ref: str) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path, refs=(source_ref,))

    with pytest.raises(ValueError, match="unsafe source_ref"):
        prepare_ocr_training_rights_worklist(
            manifest,
            source_root,
            tmp_path / "rights-worklist.json",
        )


@pytest.mark.parametrize("failure", ("missing", "hash_mismatch"))
def test_rejects_missing_or_hash_mismatched_listed_source(
    tmp_path: Path,
    failure: str,
) -> None:
    manifest, source_root, documents = _write_fixture(tmp_path, refs=("grade/doc.hwp",))
    source_path = source_root / "grade/doc.hwp"
    if failure == "missing":
        source_path.unlink()
        expected = "listed source is missing"
    else:
        source_path.write_bytes(b"changed")
        expected = "SHA-256 mismatch"

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        prepare_ocr_training_rights_worklist(
            manifest,
            source_root,
            tmp_path / "rights-worklist.json",
        )

    assert documents[0]["source_ref"] == "grade/doc.hwp"


def test_rejects_symlinked_source(tmp_path: Path) -> None:
    manifest, source_root, documents = _write_fixture(tmp_path, refs=("grade/doc.hwp",))
    source_path = source_root / "grade/doc.hwp"
    source_path.unlink()
    target = tmp_path / "outside.hwp"
    target.write_bytes(b"source-0")
    try:
        source_path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    assert _sha256(target.read_bytes()) == documents[0]["source_sha256"]

    with pytest.raises(ValueError, match="must not contain symlinks"):
        prepare_ocr_training_rights_worklist(
            manifest,
            source_root,
            tmp_path / "rights-worklist.json",
        )


def test_rejects_duplicate_source_identity(tmp_path: Path) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["documents"][1]["source_sha256"] = payload["documents"][0]["source_sha256"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source SHA-256"):
        prepare_ocr_training_rights_worklist(
            manifest,
            source_root,
            tmp_path / "rights-worklist.json",
        )


def test_manifest_read_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)
    monkeypatch.setattr(worklist_module, "_MAX_MANIFEST_BYTES", 32)

    with pytest.raises(ValueError, match="exceeds the size limit"):
        prepare_ocr_training_rights_worklist(
            manifest,
            source_root,
            tmp_path / "rights-worklist.json",
        )


def test_output_is_create_only(tmp_path: Path) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)
    output = tmp_path / "rights-worklist.json"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="output already exists"):
        prepare_ocr_training_rights_worklist(manifest, source_root, output)

    assert output.read_bytes() == b"keep me"


def test_failed_atomic_publish_leaves_no_output_or_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)
    output = tmp_path / "rights-worklist.json"

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(worklist_module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected link failure"):
        prepare_ocr_training_rights_worklist(manifest, source_root, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".rights-worklist.json.*.partial"))


def test_output_inside_source_root_is_rejected(tmp_path: Path) -> None:
    manifest, source_root, _ = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="must not overlap source inputs"):
        prepare_ocr_training_rights_worklist(
            manifest,
            source_root,
            source_root / "rights-worklist.json",
        )
