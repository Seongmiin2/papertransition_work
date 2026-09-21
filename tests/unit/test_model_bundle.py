from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scan2hwpx.evaluation.model_bundle import load_model_bundle_manifest

EXAMPLE_PATH = Path("ai/evaluation/model_bundle.example.json")


def _payload() -> dict[str, Any]:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "model-bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_models(payload: dict[str, Any]) -> None:
    payload["model_a"] = {
        "status": "candidate",
        "model_id": "document-understanding-model-a-v1",
        "revision": "candidate-v1",
        "artifact_ref": "artifact://models/model-a/document-understanding-v1",
        "artifact_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    }
    payload["model_b"] = {
        "status": "candidate",
        "model_id": "document-author-model-b-v1",
        "revision": "candidate-v1",
        "artifact_ref": "artifact://models/model-b/document-author-v1",
        "artifact_sha256": "7777777777777777777777777777777777777777777777777777777777777777",
    }


def test_example_candidate_records_both_placeholders_and_is_incomplete() -> None:
    bundle = load_model_bundle_manifest(EXAMPLE_PATH)

    assert bundle.status == "candidate"
    assert bundle.model_a.status == "placeholder"
    assert bundle.model_a.artifact_ref is None
    assert bundle.model_b.status == "placeholder"
    assert bundle.model_b.artifact_ref is None
    assert bundle.components_complete is False
    assert bundle.contract_versions.content_ir == "content-ir/1.0"


def test_non_placeholder_components_make_candidate_complete(tmp_path: Path) -> None:
    payload = _payload()
    _complete_models(payload)

    bundle = load_model_bundle_manifest(_write_manifest(tmp_path, payload))

    assert bundle.components_complete is True


def test_deployed_bundle_rejects_placeholder_component(tmp_path: Path) -> None:
    payload = _payload()
    payload["status"] = "deployed"

    with pytest.raises(ValidationError, match="deployed bundle cannot contain placeholder"):
        load_model_bundle_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("component_status", ["candidate", "deployed"])
def test_non_placeholder_component_requires_artifact_and_sha(
    tmp_path: Path, component_status: str
) -> None:
    payload = _payload()
    payload["model_b"]["status"] = component_status

    with pytest.raises(ValidationError, match="requires artifact ref and sha256"):
        load_model_bundle_manifest(_write_manifest(tmp_path, payload))


def test_placeholder_component_cannot_claim_artifact(tmp_path: Path) -> None:
    payload = _payload()
    payload["model_b"]["artifact_ref"] = "artifact://models/model-b/not-trained"
    payload["model_b"]["artifact_sha256"] = "7" * 64

    with pytest.raises(ValidationError, match="must not claim an artifact"):
        load_model_bundle_manifest(_write_manifest(tmp_path, payload))


def test_rejects_duplicate_model_ids_and_artifact_refs(tmp_path: Path) -> None:
    duplicate_id = _payload()
    duplicate_id["model_b"]["model_id"] = duplicate_id["model_a"]["model_id"]
    with pytest.raises(ValidationError, match="duplicate model_id"):
        load_model_bundle_manifest(_write_manifest(tmp_path, duplicate_id))

    duplicate_ref = _payload()
    duplicate_ref["capability_profile_ref"] = duplicate_ref["dataset_manifest_ref"]
    with pytest.raises(ValidationError, match="duplicate artifact refs"):
        load_model_bundle_manifest(_write_manifest(tmp_path, duplicate_ref))


@pytest.mark.parametrize(
    "field_path",
    [
        ("model_a", "artifact_sha256"),
        ("dataset_manifest_sha256",),
        ("capability_profile_sha256",),
        ("hancom_knowledge_manifest_sha256",),
        ("hancom_knowledge_corpus_sha256",),
        ("compiler_sha256",),
    ],
)
def test_rejects_invalid_sha256(tmp_path: Path, field_path: tuple[str, ...]) -> None:
    payload = _payload()
    target: dict[str, Any] = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = "not-a-sha"

    with pytest.raises(ValidationError):
        load_model_bundle_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    "invalid_ref",
    [
        "C:/private/model.bin",
        "file:///private/model.bin",
        "https://example.com/model.bin?token=secret",
        "artifact://models/../secret.bin",
    ],
)
def test_rejects_local_external_or_traversing_refs(tmp_path: Path, invalid_ref: str) -> None:
    payload = _payload()
    payload["dataset_manifest_ref"] = invalid_ref

    with pytest.raises(ValidationError):
        load_model_bundle_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize("location", ["bundle", "component", "contracts"])
def test_rejects_extra_or_secret_fields(tmp_path: Path, location: str) -> None:
    payload = copy.deepcopy(_payload())
    if location == "bundle":
        payload["api_key"] = "secret"
    elif location == "component":
        payload["model_a"]["token"] = "secret"
    else:
        payload["contract_versions"]["legacy_ir"] = "legacy/1.0"

    with pytest.raises(ValidationError):
        load_model_bundle_manifest(_write_manifest(tmp_path, payload))


def test_created_at_requires_iso_datetime_with_timezone(tmp_path: Path) -> None:
    invalid_iso = _payload()
    invalid_iso["created_at"] = "not-a-date"
    with pytest.raises(ValidationError):
        load_model_bundle_manifest(_write_manifest(tmp_path, invalid_iso))

    missing_timezone = _payload()
    missing_timezone["created_at"] = "2026-09-09T00:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        load_model_bundle_manifest(_write_manifest(tmp_path, missing_timezone))
