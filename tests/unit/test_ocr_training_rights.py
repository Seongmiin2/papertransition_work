from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import scan2hwpx.evaluation.ocr_training_rights as rights_module
from scan2hwpx.evaluation import (
    AuthenticatedAttestorContext,
    OcrTrainingRightsError,
    load_ocr_training_rights_manifest,
    verify_ocr_training_dataset_rights,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ocr_training_module() -> ModuleType:
    path = Path("ai/modeling/train_korean_ocr.py")
    spec = importlib.util.spec_from_file_location("korean_ocr_training_entrypoint", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _document(digit: str, split: str, family: str) -> dict[str, Any]:
    evidence_payload = f"rights-evidence-{digit}".encode()
    return {
        "source_sha256": digit * 64,
        "split": split,
        "template_family": family,
        "usage_scope": "ocr_model_training_and_evaluation",
        "rights_verified": True,
        "attestation": {
            "attestor_id": "rights-reviewer-01",
            "attested_at": "2026-09-10T09:00:00+09:00",
            "basis": "licensed",
            "evidence_artifact_ref": f"artifact://rights/{digit}",
            "evidence_sha256": hashlib.sha256(evidence_payload).hexdigest(),
        },
    }


def _write_fixture(
    tmp_path: Path,
    *,
    example_only: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest_documents = [
        _document("1", "train", "publisher-a-v1"),
        _document("2", "validation", "publisher-b-v1"),
        _document("3", "test", "publisher-c-v1"),
    ]
    evidence_root = tmp_path / "evidence" / "rights"
    evidence_root.mkdir(parents=True)
    for digit in ("1", "2", "3"):
        (evidence_root / digit).write_bytes(f"rights-evidence-{digit}".encode())

    image_refs: dict[str, str] = {}
    for digit, split in (("1", "train"), ("2", "validation"), ("3", "test")):
        image_ref = f"images/{split}/{digit * 64}/{digit * 24}.png"
        image_path = dataset.joinpath(*image_ref.split("/"))
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(f"image-{split}".encode())
        image_refs[split] = image_ref
    artifacts = {
        "train_label": dataset / "train.txt",
        "validation_label": dataset / "validation.txt",
        "test_label": dataset / "test.txt",
        "character_dict": dataset / "korean_exam_dict.txt",
    }
    artifacts["train_label"].write_text(f"{image_refs['train']}\t가\n", encoding="utf-8")
    artifacts["validation_label"].write_text(
        f"{image_refs['validation']}\t나\n", encoding="utf-8"
    )
    artifacts["test_label"].write_text(f"{image_refs['test']}\t다\n", encoding="utf-8")
    artifacts["character_dict"].write_text("가\n나\n다\n", encoding="utf-8")

    manifest_payload: dict[str, Any] = {
        "schema_version": "ocr-training-rights/1.0",
        "example_only": example_only,
        "documents": manifest_documents,
    }
    manifest_path = tmp_path / "rights.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8"
    )

    report_payload = {
        "schema_version": "1.1",
        "split_policy": {
            "frozen_manifest_sha256": _sha256(manifest_path),
            "template_family_contract": "explicit_manifest_only",
        },
        "documents": [
            {
                "sha256": document["source_sha256"],
                "split": document["split"],
                "template_family": document["template_family"],
            }
            for document in manifest_payload["documents"]
        ],
        "artifacts": {
            name: {"path": path.name, "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "image_inventory": _image_inventory(dataset, image_refs),
    }
    (dataset / "dataset_report.json").write_text(
        json.dumps(report_payload, ensure_ascii=False), encoding="utf-8"
    )
    return dataset, manifest_path, report_payload


def _verify(dataset: Path, manifest: Path) -> None:
    verify_ocr_training_dataset_rights(
        dataset,
        manifest,
        train_label=dataset / "train.txt",
        validation_label=dataset / "validation.txt",
        character_dict=dataset / "korean_exam_dict.txt",
        evidence_root=manifest.parent / "evidence",
        attestor_context=_context(dataset),
    )


def _image_inventory(dataset: Path, image_refs: dict[str, str]) -> dict[str, Any]:
    digest = hashlib.sha256(b"ocr-training-images/v1\0")
    for split in ("train", "validation", "test"):
        image_ref = image_refs[split]
        image_sha256 = _sha256(dataset.joinpath(*image_ref.split("/")))
        digest.update(split.encode("ascii"))
        digest.update(b"\0")
        digest.update(image_ref.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(image_sha256))
    return {
        "digest_definition": "split-label-ref-source-lineage-and-image-bytes/v1",
        "image_count": len(image_refs),
        "sha256": digest.hexdigest(),
    }


def _context(dataset: Path) -> AuthenticatedAttestorContext:
    return AuthenticatedAttestorContext(
        frozenset({"rights-reviewer-01", "trusted-dataset-builder"}),
        trusted_dataset_report_sha256=_sha256(dataset / "dataset_report.json"),
        derivation_attestor_id="trusted-dataset-builder",
    )


def test_verifies_manifest_inventory_and_registered_artifacts(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)

    evidence = verify_ocr_training_dataset_rights(
        dataset,
        manifest,
        train_label=dataset / "train.txt",
        validation_label=dataset / "validation.txt",
        character_dict=dataset / "korean_exam_dict.txt",
        evidence_root=manifest.parent / "evidence",
        attestor_context=_context(dataset),
    )

    assert evidence.rights_manifest_sha256 == _sha256(manifest)
    assert evidence.dataset_report_sha256 == _sha256(dataset / "dataset_report.json")
    assert evidence.document_count == 3


def test_example_only_manifest_never_authorizes_training(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path, example_only=True)

    preview = load_ocr_training_rights_manifest(manifest, allow_example_only=True)
    assert preview.example_only is True
    with pytest.raises(OcrTrainingRightsError, match="example-only"):
        _verify(dataset, manifest)


def test_repository_example_is_valid_but_non_authorizing() -> None:
    example = (
        Path(__file__).resolve().parents[2]
        / "ai"
        / "datasets"
        / "configs"
        / "ocr_training_rights.example.json"
    )

    manifest = load_ocr_training_rights_manifest(example, allow_example_only=True)
    assert manifest.example_only is True
    with pytest.raises(OcrTrainingRightsError, match="example-only"):
        load_ocr_training_rights_manifest(example)


def test_rejects_template_family_leakage_across_splits(tmp_path: Path) -> None:
    _, manifest, _ = _write_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["documents"][1]["template_family"] = "publisher-a-v1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OcrTrainingRightsError, match="manifest is invalid"):
        load_ocr_training_rights_manifest(manifest)


def test_rejects_format_control_template_family_alias(tmp_path: Path) -> None:
    _, manifest, _ = _write_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["documents"][1]["template_family"] = "publisher\u200b-a-v1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OcrTrainingRightsError, match="manifest is invalid"):
        load_ocr_training_rights_manifest(manifest)


def test_rejects_tampered_registered_label(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    (dataset / "train.txt").write_text("images/train/a.png\t변조\n", encoding="utf-8")

    with pytest.raises(OcrTrainingRightsError, match="train_label digest mismatch"):
        _verify(dataset, manifest)


def test_rejects_report_inventory_not_matching_rights_manifest(tmp_path: Path) -> None:
    dataset, manifest, report = _write_fixture(tmp_path)
    report["documents"][2]["template_family"] = "different-family"
    (dataset / "dataset_report.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(OcrTrainingRightsError, match="inventory does not match"):
        _verify(dataset, manifest)


def test_rejects_unregistered_label_path(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    alternate = tmp_path / "alternate-train.txt"
    alternate.write_text((dataset / "train.txt").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(OcrTrainingRightsError, match="registered artifact"):
        verify_ocr_training_dataset_rights(
            dataset,
            manifest,
            train_label=alternate,
            validation_label=dataset / "validation.txt",
            character_dict=dataset / "korean_exam_dict.txt",
            evidence_root=manifest.parent / "evidence",
            attestor_context=_context(dataset),
        )


def test_requires_authenticated_attestor_and_trusted_derivation_context(
    tmp_path: Path,
) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)

    with pytest.raises(OcrTrainingRightsError, match="authenticated external context"):
        verify_ocr_training_dataset_rights(
            dataset,
            manifest,
            train_label=dataset / "train.txt",
            validation_label=dataset / "validation.txt",
            character_dict=dataset / "korean_exam_dict.txt",
            evidence_root=tmp_path / "evidence",
            attestor_context=None,
        )

    wrong_context = AuthenticatedAttestorContext(
        frozenset({"different-reviewer", "trusted-dataset-builder"}),
        trusted_dataset_report_sha256=_sha256(dataset / "dataset_report.json"),
        derivation_attestor_id="trusted-dataset-builder",
    )
    with pytest.raises(OcrTrainingRightsError, match="not authenticated"):
        verify_ocr_training_dataset_rights(
            dataset,
            manifest,
            train_label=dataset / "train.txt",
            validation_label=dataset / "validation.txt",
            character_dict=dataset / "korean_exam_dict.txt",
            evidence_root=tmp_path / "evidence",
            attestor_context=wrong_context,
        )

    same_role_context = AuthenticatedAttestorContext(
        frozenset({"rights-reviewer-01"}),
        trusted_dataset_report_sha256=_sha256(dataset / "dataset_report.json"),
        derivation_attestor_id="rights-reviewer-01",
    )
    with pytest.raises(OcrTrainingRightsError, match="independently authenticated"):
        verify_ocr_training_dataset_rights(
            dataset,
            manifest,
            train_label=dataset / "train.txt",
            validation_label=dataset / "validation.txt",
            character_dict=dataset / "korean_exam_dict.txt",
            evidence_root=tmp_path / "evidence",
            attestor_context=same_role_context,
        )

    untrusted_report_context = AuthenticatedAttestorContext(
        frozenset({"rights-reviewer-01", "trusted-dataset-builder"}),
        trusted_dataset_report_sha256="0" * 64,
        derivation_attestor_id="trusted-dataset-builder",
    )
    with pytest.raises(OcrTrainingRightsError, match="report was not trusted"):
        verify_ocr_training_dataset_rights(
            dataset,
            manifest,
            train_label=dataset / "train.txt",
            validation_label=dataset / "validation.txt",
            character_dict=dataset / "korean_exam_dict.txt",
            evidence_root=tmp_path / "evidence",
            attestor_context=untrusted_report_context,
        )


def test_authenticated_context_is_runtime_immutable_and_exact() -> None:
    digest = "1" * 64
    with pytest.raises(ValueError, match="non-empty frozenset"):
        AuthenticatedAttestorContext(  # type: ignore[arg-type]
            {"rights-reviewer", "derivation-reviewer"},
            trusted_dataset_report_sha256=digest,
            derivation_attestor_id="derivation-reviewer",
        )
    with pytest.raises(ValueError, match="invalid assurance"):
        AuthenticatedAttestorContext(
            frozenset({"rights-reviewer", "derivation-reviewer"}),
            trusted_dataset_report_sha256=digest,
            derivation_attestor_id="derivation-reviewer",
            assurance="local-claim",  # type: ignore[arg-type]
        )


def test_rejects_tampered_rights_evidence_artifact(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    (tmp_path / "evidence" / "rights" / "1").write_bytes(b"tampered evidence")

    with pytest.raises(OcrTrainingRightsError, match="evidence artifact digest mismatch"):
        _verify(dataset, manifest)


def test_rejects_symlinked_rights_evidence_directory(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    rights_directory = tmp_path / "evidence" / "rights"
    actual_directory = tmp_path / "actual-rights"
    rights_directory.rename(actual_directory)
    try:
        rights_directory.symlink_to(actual_directory, target_is_directory=True)
    except OSError as exc:
        actual_directory.rename(rights_directory)
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    with pytest.raises(OcrTrainingRightsError, match="non-symlink directories"):
        _verify(dataset, manifest)


def test_rejects_tampered_training_image_bytes(tmp_path: Path) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    train_ref = (dataset / "train.txt").read_text(encoding="utf-8").split("\t", 1)[0]
    dataset.joinpath(*train_ref.split("/")).write_bytes(b"tampered image")

    with pytest.raises(OcrTrainingRightsError, match="image inventory digest mismatch"):
        _verify(dataset, manifest)


def test_manifest_size_limit_is_enforced_before_unbounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest, _ = _write_fixture(tmp_path)
    monkeypatch.setattr(rights_module, "_MAX_MANIFEST_BYTES", 32)

    with pytest.raises(OcrTrainingRightsError, match="exceeds the size limit"):
        load_ocr_training_rights_manifest(manifest)


def test_ocr_entrypoint_blocks_download_and_output_without_authenticated_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    paddle_repo = tmp_path / "PaddleOCR"
    tools = paddle_repo / "tools"
    tools.mkdir(parents=True)
    for name in ("train.py", "eval.py", "export_model.py"):
        (tools / name).write_text("# test fixture\n", encoding="utf-8")
    config = tmp_path / "rec.yml"
    config.write_text("Global: {}\n", encoding="utf-8")
    pretrained = tmp_path / "download-must-not-start" / "base.pdparams"
    output = tmp_path / "output-must-not-start" / "run"
    training = _load_ocr_training_module()

    def forbidden_download(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"download must not start: {args}, {kwargs}")

    monkeypatch.setattr(training.urllib.request, "urlretrieve", forbidden_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path("ai/modeling/train_korean_ocr.py")),
            "--paddleocr-repo",
            str(paddle_repo),
            "--dataset",
            str(dataset),
            "--rights-manifest",
            str(manifest),
            "--rights-evidence-root",
            str(tmp_path / "evidence"),
            "--output",
            str(output),
            "--pretrained",
            str(pretrained),
            "--config",
            str(config),
            "--download-pretrained",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        training.main()

    assert raised.value.code == 2
    assert "rights gate failed" in capsys.readouterr().err
    assert not pretrained.parent.exists()
    assert not output.parent.exists()


def test_ocr_entrypoint_rejects_before_any_path_resolution_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training = _load_ocr_training_module()

    def forbidden_path_resolution(*args: object, **kwargs: object) -> Path:
        raise AssertionError(f"path I/O must not start: {args}, {kwargs}")

    monkeypatch.setattr(training, "_project_path", forbidden_path_resolution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path("ai/modeling/train_korean_ocr.py")),
            "--paddleocr-repo",
            str(tmp_path / "missing-paddle"),
            "--rights-manifest",
            str(tmp_path / "missing-rights.json"),
            "--rights-evidence-root",
            str(tmp_path / "missing-evidence"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        training.main()

    assert raised.value.code == 2
    assert "authenticated external context" in capsys.readouterr().err


def test_ocr_entrypoint_uses_verified_snapshot_and_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, manifest, _ = _write_fixture(tmp_path)
    paddle_repo = tmp_path / "PaddleOCR"
    tools = paddle_repo / "tools"
    tools.mkdir(parents=True)
    for name in ("train.py", "eval.py", "export_model.py"):
        (tools / name).write_text("# test fixture\n", encoding="utf-8")
    config = tmp_path / "rec.yml"
    config.write_text("Global: {}\n", encoding="utf-8")
    pretrained = tmp_path / "base.pdparams"
    pretrained.write_bytes(b"pretrained")
    output = tmp_path / "published-model"
    training = _load_ocr_training_module()
    invoked: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        invoked.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="test-commit\n", stderr="")
        save_override = next(
            value for value in command if value.startswith("Global.save_model_dir=")
        )
        save_dir = Path(save_override.split("=", 1)[1])
        assert save_dir != output
        assert save_dir.name.endswith(".partial")
        data_override = next(
            value for value in command if value.startswith("Train.dataset.data_dir=")
        )
        training_input = Path(data_override.split("=", 1)[1])
        assert training_input.parent == save_dir
        assert training_input.name == ".verified-input"
        assert training_input.is_dir()
        (save_dir / "mock-checkpoint").write_bytes(b"weights")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path("ai/modeling/train_korean_ocr.py")),
            "--paddleocr-repo",
            str(paddle_repo),
            "--dataset",
            str(dataset),
            "--rights-manifest",
            str(manifest),
            "--rights-evidence-root",
            str(tmp_path / "evidence"),
            "--output",
            str(output),
            "--pretrained",
            str(pretrained),
            "--config",
            str(config),
            "--cpu",
            "--no-export",
        ],
    )

    assert training.main(attestor_context=_context(dataset)) == 0
    assert output.is_dir()
    assert not (output / ".verified-input").exists()
    assert (output / "mock-checkpoint").read_bytes() == b"weights"
    report = json.loads((output / "training_run.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.1"
    assert report["rights_identity_assurance"] == "authenticated_external_context"
    assert report["derivation_assurance"] == (
        "authenticated_external_report_attestation"
    )
    assert report["derivation_verification_scope"] == (
        "report_digest_and_artifact_integrity_only"
    )
    assert len(invoked) == 3
    assert not list(tmp_path.glob(".published-model.*.partial"))


def test_snapshot_copy_rejects_windows_separator_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    for name in (
        "dataset_report.json",
        "validation.txt",
        "test.txt",
        "korean_exam_dict.txt",
    ):
        (source / name).write_text("{}\n", encoding="utf-8")
    digest = "1" * 64
    (source / "train.txt").write_text(
        f"images/train/{digest}/..\\..\\..\\..\\outside.png\t가\n",
        encoding="utf-8",
    )
    destination = tmp_path / ".ocr.partial" / ".verified-input"
    destination.parent.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"must-not-copy")
    training = _load_ocr_training_module()

    with pytest.raises(RuntimeError, match="lineage ref"):
        training._copy_verified_dataset(source, destination)

    assert outside.read_bytes() == b"must-not-copy"


def test_snapshot_copy_rejects_cross_split_identical_image_bytes(
    tmp_path: Path,
) -> None:
    dataset, _, _ = _write_fixture(tmp_path)
    train_ref = (dataset / "train.txt").read_text(encoding="utf-8").split("\t", 1)[0]
    validation_ref = (
        (dataset / "validation.txt").read_text(encoding="utf-8").split("\t", 1)[0]
    )
    shared_bytes = dataset.joinpath(*train_ref.split("/")).read_bytes()
    dataset.joinpath(*validation_ref.split("/")).write_bytes(shared_bytes)
    destination = tmp_path / ".ocr.partial" / ".verified-input"
    destination.parent.mkdir()
    training = _load_ocr_training_module()

    with pytest.raises(RuntimeError, match="cross-split image byte leakage"):
        training._copy_verified_dataset(dataset, destination)


def test_pretrained_download_never_replaces_a_racing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _load_ocr_training_module()
    pretrained = tmp_path / "pretrained.pdparams"

    def racing_download(url: str, staging_name: str | Path) -> tuple[str, None]:
        assert url == training.PRETRAINED_URL
        Path(staging_name).write_bytes(b"downloaded")
        pretrained.write_bytes(b"racer")
        return str(staging_name), None

    monkeypatch.setattr(training.urllib.request, "urlretrieve", racing_download)

    with pytest.raises(FileExistsError, match="appeared"):
        training._download_pretrained(pretrained)

    assert pretrained.read_bytes() == b"racer"
    assert not list(tmp_path.glob(".pretrained.pdparams.*.partial"))
    marker = training._pretrained_transaction_marker(pretrained)
    assert marker.is_file()
    with pytest.raises(RuntimeError, match="quarantined"):
        training._reject_incomplete_pretrained_download(pretrained)


def test_pretrained_download_is_size_checked_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _load_ocr_training_module()
    pretrained = tmp_path / "pretrained.pdparams"

    def oversized_download(url: str, staging_name: str | Path) -> tuple[str, None]:
        assert url == training.PRETRAINED_URL
        Path(staging_name).write_bytes(b"oversized")
        return str(staging_name), None

    monkeypatch.setattr(training.urllib.request, "urlretrieve", oversized_download)
    monkeypatch.setattr(training, "_MAX_MODEL_ARTIFACT_BYTES", 3)

    with pytest.raises(ValueError, match="size limit"):
        training._download_pretrained(pretrained)

    assert not pretrained.exists()
    assert not list(tmp_path.glob(".pretrained.pdparams.*.partial"))
    marker = training._pretrained_transaction_marker(pretrained)
    assert marker.is_file()
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload["status"] == "downloading"


def test_pretrained_download_cleanup_never_unlinks_replaced_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _load_ocr_training_module()
    pretrained = tmp_path / "pretrained.pdparams"
    replaced_staging: list[Path] = []

    def replacing_download(url: str, staging_name: str | Path) -> tuple[str, None]:
        assert url == training.PRETRAINED_URL
        staging = Path(staging_name)
        staging.unlink()
        staging.write_bytes(b"racer")
        replaced_staging.append(staging)
        raise OSError("simulated download failure")

    monkeypatch.setattr(training.urllib.request, "urlretrieve", replacing_download)

    with pytest.raises(OSError, match="simulated download failure"):
        training._download_pretrained(pretrained)

    assert len(replaced_staging) == 1
    assert replaced_staging[0].read_bytes() == b"racer"
    assert training._pretrained_transaction_marker(pretrained).is_file()
