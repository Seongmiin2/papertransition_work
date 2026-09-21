from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import scan2hwpx.evaluation.dataset as dataset_module
from scan2hwpx.evaluation.dataset import (
    GoldenDatasetDocument,
    GoldenDatasetManifest,
    load_golden_dataset_manifest,
    summarize_golden_dataset_manifest,
    validate_golden_dataset_manifest,
)

EXAMPLE_PATH = Path("ai/datasets/configs/golden_manifest.example.json")


def _document(
    identifier: str,
    source_digit: str,
    *,
    template_family: str,
    split: str,
    capture: str = "real_scan",
    page_count: int = 10,
    verified: bool = True,
    lineage_id: str | None = None,
    usage_rights_verified: bool = True,
) -> dict[str, object]:
    return {
        "id": identifier,
        "lineage_id": lineage_id or identifier,
        "source_sha256": source_digit * 64,
        "template_family": template_family,
        "split": split,
        "capture": capture,
        "page_count": page_count,
        "annotation_artifact_ref": f"artifact://golden-v1/annotations/{identifier}.json",
        "annotation_sha256": source_digit * 64,
        "verified": verified,
        "usage_rights_verified": usage_rights_verified,
        "verification_attestation": {
            "reviewer_id": "reviewer-001",
            "verified_at": "2026-09-09T09:00:00+09:00",
            "verification_artifact_ref": (
                f"artifact://golden-v1/verifications/{identifier}.json"
            ),
            "verification_artifact_sha256": hashlib.sha256(
                f"verification:{identifier}".encode()
            ).hexdigest(),
        },
    }


def _payload(documents: list[dict[str, object]], *, example_only: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "example_only": example_only,
        "documents": documents,
    }


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_example_manifest_is_valid_but_blocked_as_production_evidence() -> None:
    with pytest.raises(ValueError, match="example-only"):
        load_golden_dataset_manifest(EXAMPLE_PATH)

    manifest = load_golden_dataset_manifest(EXAMPLE_PATH, allow_example_only=True)
    evidence = summarize_golden_dataset_manifest(manifest)

    assert manifest.example_only is True
    assert evidence.document_level_split is True
    assert evidence.template_family_holdout is True
    assert evidence.real_scan_test_documents == 1
    assert evidence.real_scan_test_pages == 8


def test_load_validate_and_summary_produce_release_dataset_evidence(tmp_path: Path) -> None:
    payload = _payload(
        [
            _document("train-1", "1", template_family="family-a", split="train"),
            _document(
                "test-1", "2", template_family="family-b", split="test", page_count=300
            ),
            _document(
                "test-2", "3", template_family="family-c", split="test", page_count=220
            ),
            _document(
                "digital-test",
                "4",
                template_family="family-d",
                split="test",
                capture="born_digital",
                page_count=100,
            ),
        ]
    )

    manifest = load_golden_dataset_manifest(_write_manifest(tmp_path, payload))
    evidence = validate_golden_dataset_manifest(manifest)

    assert evidence.document_level_split is True
    assert evidence.template_family_holdout is True
    assert evidence.real_scan_test_documents == 2
    assert evidence.real_scan_test_pages == 520


@pytest.mark.parametrize("duplicate_field", ["id", "source_sha256"])
def test_rejects_duplicate_document_id_or_source_sha(
    tmp_path: Path, duplicate_field: str
) -> None:
    first = _document("doc-1", "1", template_family="family-a", split="train")
    second = _document("doc-2", "2", template_family="family-b", split="validation")
    second[duplicate_field] = first[duplicate_field]

    with pytest.raises(ValidationError, match="duplicate"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([first, second])))


@pytest.mark.parametrize(
    "duplicate_field", ["annotation_artifact_ref", "annotation_sha256"]
)
def test_rejects_reused_annotation_artifact(
    tmp_path: Path, duplicate_field: str
) -> None:
    first = _document("doc-1", "1", template_family="family-a", split="train")
    second = _document("doc-2", "2", template_family="family-b", split="validation")
    second[duplicate_field] = first[duplicate_field]

    with pytest.raises(ValidationError, match="duplicate annotation"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([first, second])))


@pytest.mark.parametrize(
    "duplicate_field",
    ["verification_artifact_ref", "verification_artifact_sha256"],
)
def test_rejects_reused_verification_artifact(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    first = _document("doc-1", "1", template_family="family-a", split="train")
    second = _document("doc-2", "2", template_family="family-b", split="validation")
    first_attestation = first["verification_attestation"]
    second_attestation = second["verification_attestation"]
    assert isinstance(first_attestation, dict)
    assert isinstance(second_attestation, dict)
    second_attestation[duplicate_field] = first_attestation[duplicate_field]

    with pytest.raises(ValidationError, match="duplicate verification"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([first, second])))


def test_requires_structured_verification_attestation(tmp_path: Path) -> None:
    document = _document("doc-1", "1", template_family="family-a", split="test")
    document.pop("verification_attestation")

    with pytest.raises(ValidationError, match="verification_attestation"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([document])))


def test_verification_attestation_requires_timezone_and_valid_evidence(
    tmp_path: Path,
) -> None:
    document = _document("doc-1", "1", template_family="family-a", split="test")
    attestation = document["verification_attestation"]
    assert isinstance(attestation, dict)
    attestation["verified_at"] = "2026-09-09T09:00:00"

    with pytest.raises(ValidationError, match="timezone"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([document])))

    for field, value in (
        ("reviewer_id", "reviewer/private"),
        ("verification_artifact_ref", "C:/private/verification.json"),
        ("verification_artifact_sha256", "not-a-sha"),
    ):
        invalid = _document("doc-1", "1", template_family="family-a", split="test")
        invalid_attestation = invalid["verification_attestation"]
        assert isinstance(invalid_attestation, dict)
        invalid_attestation[field] = value
        with pytest.raises(ValidationError):
            load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([invalid])))


def test_rejects_unverified_document(tmp_path: Path) -> None:
    document = _document(
        "doc-1", "1", template_family="family-a", split="test", verified=False
    )

    with pytest.raises(ValidationError, match="unverified documents"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([document])))


def test_rejects_document_without_verified_usage_rights(tmp_path: Path) -> None:
    document = _document(
        "doc-1",
        "1",
        template_family="family-a",
        split="test",
        usage_rights_verified=False,
    )

    with pytest.raises(ValidationError, match="usage rights not verified"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([document])))


def test_rejects_lineage_leakage_across_splits(tmp_path: Path) -> None:
    documents = [
        _document(
            "original",
            "1",
            template_family="family-a",
            split="train",
            lineage_id="source-1",
        ),
        _document(
            "augmentation",
            "2",
            template_family="family-b",
            split="test",
            lineage_id="SOURCE-1",
        ),
    ]

    with pytest.raises(ValidationError, match="lineage leakage across splits"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload(documents)))


def test_allows_multiple_lineage_variants_in_same_split(tmp_path: Path) -> None:
    documents = [
        _document(
            "original",
            "1",
            template_family="family-a",
            split="train",
            lineage_id="source-1",
        ),
        _document(
            "augmentation",
            "2",
            template_family="family-b",
            split="train",
            lineage_id="SOURCE-1",
        ),
    ]

    manifest = load_golden_dataset_manifest(_write_manifest(tmp_path, _payload(documents)))

    assert summarize_golden_dataset_manifest(manifest).document_level_split is True


def test_summary_uses_lineage_instead_of_source_hash_for_document_split() -> None:
    documents = tuple(
        GoldenDatasetDocument.model_validate(document)
        for document in [
            _document(
                "original",
                "1",
                template_family="family-a",
                split="train",
                lineage_id="source-1",
            ),
            _document(
                "augmentation",
                "2",
                template_family="family-b",
                split="test",
                lineage_id="source-1",
            ),
        ]
    )
    unchecked_manifest = GoldenDatasetManifest.model_construct(
        schema_version="1.0",
        example_only=False,
        documents=documents,
    )

    evidence = summarize_golden_dataset_manifest(unchecked_manifest)

    assert evidence.document_level_split is False


@pytest.mark.parametrize("development_split", ["train", "validation"])
def test_rejects_template_family_leakage_into_test(
    tmp_path: Path, development_split: str
) -> None:
    documents = [
        _document("development", "1", template_family="family-a", split=development_split),
        _document("test", "2", template_family="FAMILY-A", split="test"),
    ]

    with pytest.raises(ValidationError, match="template family leakage"):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload(documents)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha256", "not-a-sha"),
        ("annotation_sha256", "not-a-sha"),
        ("lineage_id", "not/a/logical-id"),
        ("annotation_artifact_ref", "C:/private/source.json"),
        ("annotation_artifact_ref", "artifact://source/../private.json"),
        ("annotation_artifact_ref", "artifact://source./annotation.json"),
        ("page_count", 0),
        ("split", "holdout"),
        ("capture", "photocopied"),
    ],
)
def test_rejects_invalid_document_fields(tmp_path: Path, field: str, value: object) -> None:
    document = _document("doc-1", "1", template_family="family-a", split="test")
    document[field] = value

    with pytest.raises(ValidationError):
        load_golden_dataset_manifest(_write_manifest(tmp_path, _payload([document])))


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _payload(
        [_document("doc-1", "1", template_family="family-a", split="test")]
    )
    invalid = copy.deepcopy(payload)
    invalid["documents"][0]["source_text"] = "원문이 manifest에 들어가면 안 됩니다."

    with pytest.raises(ValidationError):
        load_golden_dataset_manifest(_write_manifest(tmp_path, invalid))


def test_validate_requires_a_manifest_instance() -> None:
    with pytest.raises(TypeError, match="GoldenDatasetManifest"):
        validate_golden_dataset_manifest({})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="GoldenDatasetManifest"):
        summarize_golden_dataset_manifest({})  # type: ignore[arg-type]


def test_model_rejects_empty_document_list() -> None:
    with pytest.raises(ValidationError):
        GoldenDatasetManifest.model_validate(
            {"schema_version": "1.0", "example_only": False, "documents": []}
        )


def test_loader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        b'{"schema_version":"1.0","example_only":false,'
        b'"example_only":false,"documents":[]}'
    )
    with pytest.raises(ValueError, match="duplicate object key"):
        load_golden_dataset_manifest(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(
        b'{"schema_version":"1.0","example_only":false,'
        b'"documents":[],"unknown":NaN}'
    )
    with pytest.raises(ValueError, match="non-finite number"):
        load_golden_dataset_manifest(nonfinite)

    overflowing_float = tmp_path / "overflowing-float.json"
    overflowing_float.write_bytes(
        b'{"schema_version":"1.0","example_only":false,'
        b'"documents":[],"unknown":1e9999}'
    )
    with pytest.raises(ValueError, match="non-finite number"):
        load_golden_dataset_manifest(overflowing_float)

    oversized_integer = tmp_path / "oversized-integer.json"
    oversized_integer.write_bytes(
        b'{"schema_version":"1.0","example_only":false,'
        b'"documents":[],"unknown":9223372036854775808}'
    )
    with pytest.raises(ValueError, match="signed 64-bit"):
        load_golden_dataset_manifest(oversized_integer)


def test_loader_rejects_deep_or_oversized_json_before_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 33 + "0" + "]" * 33, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting limit"):
        load_golden_dataset_manifest(deep)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{}")
    monkeypatch.setattr(dataset_module, "_MAX_MANIFEST_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        load_golden_dataset_manifest(oversized)


def test_loader_bounds_json_node_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "many-nodes.json"
    path.write_bytes(
        b'{"schema_version":"1.0","example_only":false,"documents":[]}'
    )
    monkeypatch.setattr(dataset_module, "_MAX_JSON_NODES", 3)

    with pytest.raises(ValueError, match="JSON node limit"):
        load_golden_dataset_manifest(path)


def test_loader_rejects_symlinked_manifest(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "manifest.json"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="regular non-symlink"):
        load_golden_dataset_manifest(link)


def test_manifest_document_and_page_counts_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document("doc-1", "1", template_family="family-a", split="test")
    path = _write_manifest(tmp_path, _payload([document]))

    monkeypatch.setattr(dataset_module, "_MAX_DOCUMENTS", 0)
    with pytest.raises(ValidationError, match="document count exceeds"):
        load_golden_dataset_manifest(path)

    monkeypatch.setattr(dataset_module, "_MAX_DOCUMENTS", 10)
    monkeypatch.setattr(dataset_module, "_MAX_TOTAL_PAGES", 1)
    with pytest.raises(ValidationError, match="total page count exceeds"):
        load_golden_dataset_manifest(path)


def test_validate_rechecks_bypassed_model_construction() -> None:
    document = GoldenDatasetDocument.model_validate(
        _document("doc-1", "1", template_family="family-a", split="test")
    )
    bypassed = GoldenDatasetManifest.model_construct(
        schema_version="1.0",
        example_only=False,
        documents=(document, document),
    )

    with pytest.raises(ValidationError, match="duplicate document ids"):
        validate_golden_dataset_manifest(bypassed)
