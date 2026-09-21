from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import scan2hwpx.evaluation.artifacts as artifacts_module
from scan2hwpx.evaluation.artifacts import (
    ArtifactVerificationError,
    resolve_workspace_artifact,
    verify_golden_dataset_artifacts,
)
from scan2hwpx.evaluation.dataset import GoldenDatasetManifest
from scan2hwpx.evaluation.safe_artifact_io import read_bounded_regular_file

SOURCE_SHA256 = "1" * 64
VERIFIED_AT = "2026-09-09T09:00:00+09:00"
ANNOTATION_REF = "artifact://golden-v1/annotations/doc-1.json"
ATTESTATION_REF = "artifact://golden-v1/verifications/doc-1.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_artifact(root: Path, artifact_ref: str, payload: bytes) -> Path:
    relative = artifact_ref.removeprefix("artifact://").split("/")
    path = root.joinpath(*relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _attestation_payload(
    expected_annotation_sha256: str,
    **overrides: object,
) -> bytes:
    record: dict[str, object] = {
        "schema_version": "1.0",
        "document_id": "doc-1",
        "source_sha256": SOURCE_SHA256,
        "annotation_sha256": expected_annotation_sha256,
        "reviewer_id": "reviewer-001",
        "verified_at": VERIFIED_AT,
        "decision": "approved",
        "usage_rights_verified": True,
    }
    record.update(overrides)
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _manifest(
    *,
    annotation_ref: str = ANNOTATION_REF,
    annotation_sha256: str,
    attestation_ref: str = ATTESTATION_REF,
    attestation_sha256: str,
) -> GoldenDatasetManifest:
    return GoldenDatasetManifest.model_validate(
        {
            "schema_version": "1.0",
            "example_only": False,
            "documents": (
                {
                    "id": "doc-1",
                    "lineage_id": "lineage-1",
                    "source_sha256": SOURCE_SHA256,
                    "template_family": "family-a",
                    "split": "test",
                    "capture": "real_scan",
                    "page_count": 1,
                    "annotation_artifact_ref": annotation_ref,
                    "annotation_sha256": annotation_sha256,
                    "verified": True,
                    "usage_rights_verified": True,
                    "verification_attestation": {
                        "reviewer_id": "reviewer-001",
                        "verified_at": VERIFIED_AT,
                        "verification_artifact_ref": attestation_ref,
                        "verification_artifact_sha256": attestation_sha256,
                    },
                },
            ),
        }
    )


def _prepare_valid_workspace(
    tmp_path: Path,
) -> tuple[Path, GoldenDatasetManifest, bytes, bytes]:
    root = tmp_path / "workspace"
    root.mkdir()
    annotation = b'{"content_ir": "verified evaluator owns this schema"}'
    annotation_sha256 = _sha256(annotation)
    attestation = _attestation_payload(annotation_sha256)
    _write_artifact(root, ANNOTATION_REF, annotation)
    _write_artifact(root, ATTESTATION_REF, attestation)
    manifest = _manifest(
        annotation_sha256=annotation_sha256,
        attestation_sha256=_sha256(attestation),
    )
    return root, manifest, annotation, attestation


def test_verifies_annotation_and_attestation_bytes_for_each_document(
    tmp_path: Path,
) -> None:
    root, manifest, annotation, attestation = _prepare_valid_workspace(tmp_path)

    result = verify_golden_dataset_artifacts(manifest, workspace_root=root)

    assert len(result) == 1
    assert result[0].document_id == "doc-1"
    assert result[0].annotation.payload == annotation
    assert result[0].annotation.sha256 == _sha256(annotation)
    assert result[0].annotation.size_bytes == len(annotation)
    assert result[0].annotation.resolved_path == (
        root / "golden-v1" / "annotations" / "doc-1.json"
    ).resolve()
    assert result[0].verification_attestation.payload == attestation
    assert result[0].verification_attestation.sha256 == _sha256(attestation)
    assert result[0].reviewer_identity_assurance == "unsigned_manifest_consistency_only"
    assert result[0].usage_rights_assurance == "unsigned_manifest_consistency_only"


def test_rejects_example_only_manifest_as_production_evidence(tmp_path: Path) -> None:
    root, manifest, _, _ = _prepare_valid_workspace(tmp_path)
    example_manifest = manifest.model_copy(update={"example_only": True})

    with pytest.raises(
        ArtifactVerificationError,
        match="example-only manifest cannot be used as production evidence",
    ):
        verify_golden_dataset_artifacts(example_manifest, workspace_root=root)


@pytest.mark.parametrize(
    "artifact_ref",
    [
        "artifact://golden-v1/../secret.json",
        "artifact://golden-v1/%2e%2e/secret.json",
        "artifact://golden-v1\\..\\secret.json",
        "C:/outside/secret.json",
    ],
)
def test_rejects_traversal_and_non_artifact_paths(
    tmp_path: Path,
    artifact_ref: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(ArtifactVerificationError, match="invalid artifact reference"):
        resolve_workspace_artifact(root, artifact_ref)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    link = root / "golden-v1" / "escape.json"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ArtifactVerificationError, match="escapes workspace root"):
        resolve_workspace_artifact(root, "artifact://golden-v1/escape.json")


def test_rejects_missing_artifact(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(ArtifactVerificationError, match="artifact is missing"):
        resolve_workspace_artifact(root, ANNOTATION_REF)


@pytest.mark.parametrize("mismatch_role", ["annotation", "attestation"])
def test_rejects_digest_mismatch(tmp_path: Path, mismatch_role: str) -> None:
    root, manifest, _, _ = _prepare_valid_workspace(tmp_path)
    document = manifest.documents[0]
    annotation_sha256 = document.annotation_sha256
    attestation_sha256 = (
        document.verification_attestation.verification_artifact_sha256
    )
    if mismatch_role == "annotation":
        annotation_sha256 = "0" * 64
    else:
        attestation_sha256 = "0" * 64
    mismatched = _manifest(
        annotation_sha256=annotation_sha256,
        attestation_sha256=attestation_sha256,
    )

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        verify_golden_dataset_artifacts(mismatched, workspace_root=root)


def test_rejects_same_logical_artifact_bound_to_two_roles(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    annotation_sha256 = _sha256(b"placeholder")
    shared_payload = _attestation_payload(annotation_sha256)
    shared_sha256 = _sha256(shared_payload)
    _write_artifact(root, ANNOTATION_REF, shared_payload)
    manifest = _manifest(
        annotation_ref=ANNOTATION_REF,
        annotation_sha256=shared_sha256,
        attestation_ref=ANNOTATION_REF,
        attestation_sha256=shared_sha256,
    )

    with pytest.raises(ArtifactVerificationError, match="duplicate artifact binding"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_rejects_distinct_refs_bound_to_same_physical_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    placeholder_sha256 = _sha256(b"placeholder")
    shared_payload = _attestation_payload(placeholder_sha256)
    shared_sha256 = _sha256(shared_payload)
    annotation_path = _write_artifact(root, ANNOTATION_REF, shared_payload)
    attestation_path = root.joinpath(
        *ATTESTATION_REF.removeprefix("artifact://").split("/")
    )
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(annotation_path, attestation_path)
    except OSError as exc:
        pytest.skip(f"hard-link creation is unavailable: {exc}")
    if annotation_path.stat().st_ino == 0:
        pytest.skip("filesystem does not expose stable file identities")
    manifest = _manifest(
        annotation_sha256=shared_sha256,
        attestation_sha256=shared_sha256,
    )

    with pytest.raises(ArtifactVerificationError, match="duplicate artifact binding"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("document_id", "other-document"),
        ("source_sha256", "2" * 64),
        ("annotation_sha256", "3" * 64),
        ("reviewer_id", "other-reviewer"),
        ("verified_at", "2026-09-10T09:00:00+09:00"),
    ],
)
def test_rejects_attestation_that_disagrees_with_manifest(
    tmp_path: Path,
    field_name: str,
    wrong_value: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    annotation = b"annotation"
    annotation_sha256 = _sha256(annotation)
    attestation = _attestation_payload(
        annotation_sha256,
        **{field_name: wrong_value},
    )
    _write_artifact(root, ANNOTATION_REF, annotation)
    _write_artifact(root, ATTESTATION_REF, attestation)
    manifest = _manifest(
        annotation_sha256=annotation_sha256,
        attestation_sha256=_sha256(attestation),
    )

    with pytest.raises(ArtifactVerificationError, match="disagrees with manifest"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"schema_version": "2.0"},
        {"decision": "rejected"},
        {"usage_rights_verified": False},
        {"verified_at": "2026-09-09T09:00:00"},
        {"unexpected": "field"},
    ],
)
def test_rejects_attestation_without_strict_approval_evidence(
    tmp_path: Path,
    invalid_fields: dict[str, object],
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    annotation = b"annotation"
    annotation_sha256 = _sha256(annotation)
    attestation = _attestation_payload(annotation_sha256, **invalid_fields)
    _write_artifact(root, ANNOTATION_REF, annotation)
    _write_artifact(root, ATTESTATION_REF, attestation)
    manifest = _manifest(
        annotation_sha256=annotation_sha256,
        attestation_sha256=_sha256(attestation),
    )

    with pytest.raises(ArtifactVerificationError, match="not valid strict JSON"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_rejects_checksum_valid_non_json_attestation(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    annotation = b"annotation"
    attestation = b"approved by someone"
    _write_artifact(root, ANNOTATION_REF, annotation)
    _write_artifact(root, ATTESTATION_REF, attestation)
    manifest = _manifest(
        annotation_sha256=_sha256(annotation),
        attestation_sha256=_sha256(attestation),
    )

    with pytest.raises(ArtifactVerificationError, match="not valid strict JSON"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "deep"])
def test_rejects_ambiguous_or_resource_risky_attestation_json(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    annotation = b"annotation"
    annotation_sha256 = _sha256(annotation)
    valid = _attestation_payload(annotation_sha256).decode("utf-8")
    if kind == "duplicate":
        attestation = valid.replace(
            '"schema_version": "1.0"',
            '"schema_version": "1.0", "schema_version": "1.0"',
            1,
        ).encode("utf-8")
    elif kind == "nonfinite":
        attestation = (valid[:-1] + ', "confidence": NaN}').encode("utf-8")
    else:
        nested = "[" * 17 + "0" + "]" * 17
        attestation = (valid[:-1] + f', "nested": {nested}}}').encode("utf-8")
    _write_artifact(root, ANNOTATION_REF, annotation)
    _write_artifact(root, ATTESTATION_REF, attestation)
    manifest = _manifest(
        annotation_sha256=annotation_sha256,
        attestation_sha256=_sha256(attestation),
    )

    with pytest.raises(ArtifactVerificationError, match="not valid strict JSON"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_bounds_each_artifact_and_total_retained_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, annotation, attestation = _prepare_valid_workspace(tmp_path)
    monkeypatch.setattr(artifacts_module, "_MAX_ANNOTATION_BYTES", len(annotation) - 1)
    with pytest.raises(ArtifactVerificationError, match="artifact exceeds size limit"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)

    monkeypatch.setattr(artifacts_module, "_MAX_ANNOTATION_BYTES", len(annotation))
    monkeypatch.setattr(
        artifacts_module,
        "_MAX_TOTAL_VERIFIED_ARTIFACT_BYTES",
        len(annotation) + len(attestation) - 1,
    )
    with pytest.raises(ArtifactVerificationError, match="total byte limit"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_bounds_attestation_json_node_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, _, _ = _prepare_valid_workspace(tmp_path)
    monkeypatch.setattr(artifacts_module, "_MAX_ATTESTATION_JSON_NODES", 1)

    with pytest.raises(ArtifactVerificationError, match="not valid strict JSON"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_rejects_symlinked_workspace_root_and_intermediate_directory(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-workspace"
    real_root.mkdir()
    payload = b"artifact"
    real_artifacts = real_root / "real-artifacts"
    real_artifacts.mkdir()
    (real_artifacts / "doc.json").write_bytes(payload)

    linked_root = tmp_path / "linked-workspace"
    linked_parent = real_root / "golden-v1"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
        linked_parent.symlink_to(real_artifacts, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    with pytest.raises(ArtifactVerificationError, match="workspace root.*non-reparse"):
        resolve_workspace_artifact(linked_root, "artifact://real-artifacts/doc.json")
    with pytest.raises(ArtifactVerificationError, match="symlink/reparse"):
        resolve_workspace_artifact(real_root, "artifact://golden-v1/doc.json")


def test_rejects_path_replacement_after_stable_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, _, _ = _prepare_valid_workspace(tmp_path)
    original_read = read_bounded_regular_file
    replaced = False

    def replace_after_read(path: Path, *, max_bytes: int, label: str) -> bytes:
        nonlocal replaced
        payload = original_read(path, max_bytes=max_bytes, label=label)
        if not replaced and path.name == "doc-1.json" and path.parent.name == "annotations":
            replacement = path.with_name("replacement.json")
            replacement.write_bytes(payload)
            os.replace(replacement, path)
            replaced = True
        return payload

    monkeypatch.setattr(
        artifacts_module,
        "_read_bounded_regular_file",
        replace_after_read,
    )

    with pytest.raises(ArtifactVerificationError, match="changed while it was read"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_rejects_external_hardlink_alias(tmp_path: Path) -> None:
    root, manifest, _, _ = _prepare_valid_workspace(tmp_path)
    annotation_path = root.joinpath(*ANNOTATION_REF.removeprefix("artifact://").split("/"))
    outside_alias = tmp_path / "outside-alias.json"
    try:
        os.link(annotation_path, outside_alias)
    except OSError as exc:
        pytest.skip(f"hard-link creation is unavailable: {exc}")
    if annotation_path.stat().st_ino == 0:
        pytest.skip("filesystem does not expose stable file identities")

    with pytest.raises(ArtifactVerificationError, match="hard-link alias"):
        verify_golden_dataset_artifacts(manifest, workspace_root=root)


def test_verified_payload_is_an_immutable_snapshot(tmp_path: Path) -> None:
    root, manifest, annotation, _ = _prepare_valid_workspace(tmp_path)
    result = verify_golden_dataset_artifacts(manifest, workspace_root=root)
    annotation_path = root.joinpath(*ANNOTATION_REF.removeprefix("artifact://").split("/"))

    annotation_path.write_bytes(b"later mutation")

    assert result[0].annotation.payload == annotation
    assert result[0].annotation.sha256 == _sha256(annotation)
