from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

import ai.datasets.build_model_a_line_risk_dataset as risk_module
from ai.datasets.build_model_a_line_risk_dataset import (
    RECORD_FIELDS,
    build_model_a_line_risk_dataset,
)
from scan2hwpx.evaluation import AuthenticatedAttestorContext


class FakeResult:
    def __init__(self, text: str, confidence: float) -> None:
        self.json = {"res": {"rec_text": text, "rec_score": confidence}}


class FakePredictor:
    def __init__(self, predictions: dict[str, tuple[str, float]]) -> None:
        self.predictions = predictions
        self.calls: list[tuple[list[str], int]] = []
        self.closed = False

    def predict(self, input: list[str], *, batch_size: int) -> Iterable[object]:
        self.calls.append((input, batch_size))
        return [FakeResult(*self.predictions[Path(path).name]) for path in input]

    def close(self) -> None:
        self.closed = True


def _write_dataset(
    root: Path,
    rows: dict[str, list[tuple[str, str, bytes]]],
) -> Path:
    root.mkdir()
    for split, split_rows in rows.items():
        lines: list[str] = []
        for image_ref, target, payload in split_rows:
            lineage_ref = _lineage_ref(split, Path(image_ref).name)
            image = root.joinpath(*lineage_ref.split("/"))
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(payload)
            lines.append(f"{lineage_ref}\t{target}")
        (root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _minimal_rows() -> dict[str, list[tuple[str, str, bytes]]]:
    return {
        "train": [("images/train/a.png", "TRAIN_ONLY", b"train")],
        "validation": [("images/validation/b.png", "VALIDATION_ONLY", b"validation")],
        "test": [("images/test/c.png", "TEST_ONLY", b"test")],
    }


def _source_sha256(split: str) -> str:
    return hashlib.sha256(f"{split}-source".encode()).hexdigest()


def _lineage_ref(split: str, filename: str) -> str:
    return f"images/{split}/{_source_sha256(split)}/{filename}"


def _write_model(root: Path) -> Path:
    root.mkdir()
    (root / "inference.pdiparams").write_bytes(b"model weights")
    (root / "inference.yml").write_text("model: fake\n", encoding="utf-8")
    return root


def _write_rights_fixture(
    dataset: Path,
    path: Path,
    *,
    example_only: bool = False,
) -> Path:
    character_dict = dataset / "korean_exam_dict.txt"
    character_dict.write_text("가\n나\n다\n", encoding="utf-8")
    documents = []
    for index, split in enumerate(risk_module.SPLITS, start=1):
        digest = _source_sha256(split)
        evidence_path = path.parent / "evidence" / "test-rights" / split
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_bytes(f"{split}-evidence".encode())
        documents.append(
            {
                "source_sha256": digest,
                "split": split,
                "template_family": f"family-{split}",
                "usage_scope": "ocr_model_training_and_evaluation",
                "rights_verified": True,
                "attestation": {
                    "attestor_id": "test-rights-reviewer",
                    "attested_at": "2026-09-10T09:00:00+09:00",
                    "basis": "licensed",
                    "evidence_artifact_ref": f"artifact://test-rights/{split}",
                    "evidence_sha256": hashlib.sha256(
                        f"{split}-evidence".encode()
                    ).hexdigest(),
                },
            }
        )
    manifest = {
        "schema_version": "ocr-training-rights/1.0",
        "example_only": example_only,
        "documents": documents,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    artifacts = {
        "train_label": dataset / "train.txt",
        "validation_label": dataset / "validation.txt",
        "test_label": dataset / "test.txt",
        "character_dict": character_dict,
    }
    report = {
        "schema_version": "1.1",
        "split_policy": {
            "frozen_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "template_family_contract": "explicit_manifest_only",
        },
        "documents": [
            {
                "sha256": document["source_sha256"],
                "split": document["split"],
                "template_family": document["template_family"],
            }
            for document in documents
        ],
        "artifacts": {
            name: {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            for name, artifact in artifacts.items()
        },
        "image_inventory": _image_inventory(dataset),
    }
    (dataset / "dataset_report.json").write_text(json.dumps(report), encoding="utf-8")
    return path


def _image_inventory(dataset: Path) -> dict[str, Any]:
    digest = hashlib.sha256(b"ocr-training-images/v1\0")
    image_count = 0
    try:
        for split in risk_module.SPLITS:
            for line in (dataset / f"{split}.txt").read_text(encoding="utf-8").splitlines():
                image_ref, _ = line.split("\t")
                image_sha256 = hashlib.sha256(
                    dataset.joinpath(*image_ref.split("/")).read_bytes()
                ).hexdigest()
                digest.update(split.encode("ascii"))
                digest.update(b"\0")
                digest.update(image_ref.encode("utf-8"))
                digest.update(b"\0")
                digest.update(bytes.fromhex(image_sha256))
                image_count += 1
    except (OSError, ValueError):
        return {
            "digest_definition": "split-label-ref-source-lineage-and-image-bytes/v1",
            "image_count": 0,
            "sha256": "0" * 64,
        }
    return {
        "digest_definition": "split-label-ref-source-lineage-and-image-bytes/v1",
        "image_count": image_count,
        "sha256": digest.hexdigest(),
    }


def _authorization(dataset: Path, manifest: Path) -> dict[str, Any]:
    return {
        "rights_evidence_root": manifest.parent / "evidence",
        "attestor_context": AuthenticatedAttestorContext(
            frozenset({"test-rights-reviewer", "trusted-dataset-builder"}),
            trusted_dataset_report_sha256=hashlib.sha256(
                (dataset / "dataset_report.json").read_bytes()
            ).hexdigest(),
            derivation_attestor_id="trusted-dataset-builder",
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builder_filters_cross_split_leaks_and_writes_metrics(tmp_path: Path) -> None:
    shared_image = b"same image bytes in train and validation"
    dataset = _write_dataset(
        tmp_path / "dataset",
        {
            "train": [
                ("images/train/final.png", "PRIVATE_TARGET_TRAIN_42", b"train-final"),
                ("images/train/image-leak.png", "IMAGE_LEAK_TRAIN", shared_image),
                ("images/train/pair-leak.png", "\u00e9_PAIR_TARGET_SECRET", b"pair-train"),
            ],
            "validation": [
                (
                    "images/validation/final.png",
                    "PRIVATE_TARGET_VALIDATION_42",
                    b"validation-final",
                ),
                ("images/validation/image-leak.png", "IMAGE_LEAK_VALIDATION", shared_image),
                (
                    "images/validation/pair-leak.png",
                    "e\u0301_PAIR_TARGET_SECRET",
                    b"pair-validation",
                ),
            ],
            "test": [("images/test/final.png", "XY", b"test-final")],
        },
    )
    model = _write_model(tmp_path / "model")
    rights_manifest = _write_rights_fixture(dataset, tmp_path / "rights.json")
    predictor = FakePredictor(
        {
            "final.png": ("", 0.1),
            "pair-leak.png": ("\u00e9_PAIR_OBSERVED_SECRET", 0.8),
        }
    )

    def factory(model_dir: Path, device: str) -> FakePredictor:
        assert model_dir == model.resolve()
        assert device == "gpu:0"
        return predictor

    # Each split uses a final.png name, so resolve predictions by full parent below.
    def predict(input: list[str], *, batch_size: int) -> Iterable[object]:
        predictor.calls.append((input, batch_size))
        values = {
            "train/final.png": ("PRIVATE_TARGET_TRAIN_42", 0.99),
            "train/pair-leak.png": ("\u00e9_PAIR_OBSERVED_SECRET", 0.8),
            "validation/final.png": ("PRIVATE_TARGET_VALIDATION_4X", 0.75),
            "validation/pair-leak.png": ("e\u0301_PAIR_OBSERVED_SECRET", 0.85),
            "test/final.png": ("", 0.2),
        }
        return [
            FakeResult(*values[Path(path).parts[-3] + "/" + Path(path).name])
            for path in input
        ]

    predictor.predict = predict  # type: ignore[method-assign]
    output = tmp_path / "risk"

    report = build_model_a_line_risk_dataset(
        dataset,
        model,
        output,
        rights_manifest=rights_manifest,
        **_authorization(dataset, rights_manifest),
        device="gpu:0",
        batch_size=2,
        predictor_factory=factory,
    )

    assert predictor.closed
    assert all("image-leak.png" not in path for call, _ in predictor.calls for path in call)
    assert [len(call) for call, _ in predictor.calls] == [2, 2, 1]
    assert report["record_fields"] == list(RECORD_FIELDS)
    assert report["filters"] == {
        "cross_split_image_sha256": {
            "fingerprint_count": 1,
            "removed_by_split": {"train": 1, "validation": 1, "test": 0},
        },
        "cross_split_observed_target_pair": {
            "fingerprint_definition": "nfc-observed-text-and-nfc-target-text/v1",
            "fingerprint_count": 1,
            "removed_by_split": {"train": 1, "validation": 1, "test": 0},
        },
        "final_cross_split_image_sha256_count": 0,
        "final_cross_split_observed_target_pair_count": 0,
    }
    assert report["splits"]["train"]["source_count"] == 3
    assert report["splits"]["train"]["after_image_filter_count"] == 2
    assert report["splits"]["train"]["final_count"] == 1
    assert report["splits"]["train"]["exact_match_rate"] == 1.0
    assert report["splits"]["validation"]["cer"] == pytest.approx(1 / 28)
    assert report["splits"]["test"]["cer"] == 1.0
    assert len(report["source_dataset_sha256"]) == 64
    assert len(report["derived_dataset_sha256"]) == 64
    assert len(report["model_sha256"]) == 64
    assert report["schema_version"] == "model-a-line-risk-dataset/1.2"
    assert report["source_rights"]["validation"] == "verified"
    assert report["source_rights"]["rights_manifest_sha256"] == hashlib.sha256(
        rights_manifest.read_bytes()
    ).hexdigest()

    expected_refs = {
        "train": _lineage_ref("train", "final.png"),
        "validation": _lineage_ref("validation", "final.png"),
        "test": _lineage_ref("test", "final.png"),
    }
    for split in risk_module.SPLITS:
        rows = _read_jsonl(output / f"{split}.jsonl")
        assert len(rows) == 1
        assert set(rows[0]) == set(RECORD_FIELDS)
        assert rows[0]["image_ref"] == expected_refs[split]
        image = dataset.joinpath(*expected_refs[split].split("/"))
        assert rows[0]["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    serialized_report = (output / "report.json").read_text(encoding="utf-8")
    assert "PRIVATE_TARGET" not in serialized_report
    assert "PAIR_TARGET_SECRET" not in serialized_report
    assert str(tmp_path) not in serialized_report
    assert not list(tmp_path.glob(".risk.*.partial"))


@pytest.mark.parametrize(
    ("train_payload", "message"),
    [
        ("missing-tab\n", "malformed"),
        (f"{_lineage_ref('train', 'a.png')}\tTRAIN_ONLY\textra\n", "malformed"),
        (f"{_lineage_ref('train', 'a.png')}\t\n", "empty target"),
        (f"{_lineage_ref('train', 'a.png')}\tTRAIN_ONLY\n\n", "malformed"),
        ("../outside.png\tTRAIN_ONLY\n", "unsafe lineage ref"),
        (
            (
                f"{_lineage_ref('train', 'a.png')}\tTRAIN_ONLY\n"
                f"{_lineage_ref('train', 'a.png')}\tSECOND\n"
            ),
            "duplicate image ref",
        ),
    ],
)
def test_builder_rejects_malformed_or_unsafe_rows_before_inference(
    tmp_path: Path,
    train_payload: str,
    message: str,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset", _minimal_rows())
    (dataset / "train.txt").write_text(train_payload, encoding="utf-8")
    model = _write_model(tmp_path / "model")
    rights_manifest = _write_rights_fixture(dataset, tmp_path / "rights.json")
    output = tmp_path / "risk"

    def forbidden_factory(model_dir: Path, device: str) -> FakePredictor:
        raise AssertionError(f"predictor must not be created: {model_dir}, {device}")

    with pytest.raises(ValueError, match=message):
        build_model_a_line_risk_dataset(
            dataset,
            model,
            output,
            rights_manifest=rights_manifest,
            **_authorization(dataset, rights_manifest),
            device="cpu",
            batch_size=2,
            predictor_factory=forbidden_factory,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".risk.*.partial"))


def test_partial_output_write_failure_is_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset", _minimal_rows())
    model = _write_model(tmp_path / "model")
    rights_manifest = _write_rights_fixture(dataset, tmp_path / "rights.json")
    predictor = FakePredictor(
        {
            "a.png": ("TRAIN_ONLY", 0.9),
            "b.png": ("VALIDATION_ONLY", 0.9),
            "c.png": ("TEST_ONLY", 0.9),
        }
    )
    original_write = risk_module._write_jsonl

    def fail_validation(path: Path, samples: tuple[risk_module.PredictedSample, ...]) -> None:
        if path.name == "validation.jsonl":
            raise OSError("simulated partial output failure")
        original_write(path, samples)

    monkeypatch.setattr(risk_module, "_write_jsonl", fail_validation)
    output = tmp_path / "risk"

    with pytest.raises(OSError, match="simulated partial output failure"):
        build_model_a_line_risk_dataset(
            dataset,
            model,
            output,
            rights_manifest=rights_manifest,
            **_authorization(dataset, rights_manifest),
            device="cpu",
            batch_size=2,
            predictor_factory=lambda _model, _device: predictor,
        )

    assert predictor.closed
    assert not output.exists()
    assert not list(tmp_path.glob(".risk.*.partial"))


def test_predictor_result_count_mismatch_blocks_publication(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", _minimal_rows())
    model = _write_model(tmp_path / "model")
    rights_manifest = _write_rights_fixture(dataset, tmp_path / "rights.json")

    class ShortPredictor(FakePredictor):
        def predict(self, input: list[str], *, batch_size: int) -> Iterable[object]:
            return []

    predictor = ShortPredictor({})
    output = tmp_path / "risk"

    with pytest.raises(RuntimeError, match="returned 0 results for 1 train inputs"):
        build_model_a_line_risk_dataset(
            dataset,
            model,
            output,
            rights_manifest=rights_manifest,
            **_authorization(dataset, rights_manifest),
            device="cpu",
            batch_size=2,
            predictor_factory=lambda _model, _device: predictor,
        )

    assert predictor.closed
    assert not output.exists()
    assert not list(tmp_path.glob(".risk.*.partial"))


def test_rights_gate_rejects_before_predictor_or_output(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", _minimal_rows())
    model = _write_model(tmp_path / "model")
    rights_manifest = _write_rights_fixture(
        dataset,
        tmp_path / "rights.json",
        example_only=True,
    )
    output = tmp_path / "new-parent" / "risk"

    def forbidden_factory(model_dir: Path, device: str) -> FakePredictor:
        raise AssertionError(f"predictor must not be created: {model_dir}, {device}")

    with pytest.raises(ValueError, match="example-only"):
        build_model_a_line_risk_dataset(
            dataset,
            model,
            output,
            rights_manifest=rights_manifest,
            **_authorization(dataset, rights_manifest),
            device="cpu",
            batch_size=2,
            predictor_factory=forbidden_factory,
        )

    assert not output.parent.exists()


def test_artifact_hashing_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"1234")
    monkeypatch.setattr(risk_module, "_MAX_HASHED_ARTIFACT_BYTES", 3)

    with pytest.raises(ValueError, match="size limit"):
        risk_module._sha256(artifact)
