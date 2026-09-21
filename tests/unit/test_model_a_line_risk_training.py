from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import ai.datasets.build_model_a_line_risk_dataset as DATASET_BUILDER
from scan2hwpx.evaluation.ocr_training_rights import OcrTrainingRightsEvidence


def _load_training_module() -> ModuleType:
    path = Path("ai/modeling/train_model_a_line_risk.py")
    spec = importlib.util.spec_from_file_location("model_a_line_risk_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINING = _load_training_module()


def _rights_evidence() -> OcrTrainingRightsEvidence:
    return OcrTrainingRightsEvidence(
        rights_manifest_sha256=hashlib.sha256(b"rights-manifest").hexdigest(),
        dataset_report_sha256=hashlib.sha256(b"source-report").hexdigest(),
        document_count=3,
        identity_assurance="authenticated_external_context",
        evidence_artifacts_sha256=hashlib.sha256(b"evidence-artifacts").hexdigest(),
        training_images_sha256=hashlib.sha256(b"training-images").hexdigest(),
        derivation_assurance="authenticated_external_report_attestation",
        derivation_verification_scope="report_digest_and_artifact_integrity_only",
        derivation_attestor_id="trusted-dataset-builder",
    )


def _source_dataset(source_samples: dict[str, tuple[Any, ...]]) -> Any:
    return DATASET_BUILDER.SourceDataset(
        samples=source_samples,
        split_tsv_sha256={
            split: hashlib.sha256(f"{split}-tsv".encode()).hexdigest()
            for split in TRAINING.SPLITS
        },
        split_images_sha256={
            split: hashlib.sha256(f"{split}-images".encode()).hexdigest()
            for split in TRAINING.SPLITS
        },
        sha256=hashlib.sha256(b"source-dataset").hexdigest(),
    )


def _write_builder_dataset(root: Path, *, leak: str | None = None) -> Path:
    root.mkdir()
    source_samples: dict[str, tuple[Any, ...]] = {}
    predictions: dict[str, tuple[Any, ...]] = {}
    train_first_sha = hashlib.sha256(b"train-0").hexdigest()
    train_first_pair = ("train-correct", "train-correct")
    for split in TRAINING.SPLITS:
        split_sources: list[Any] = []
        split_predictions: list[Any] = []
        for index in range(2):
            target = f"{split}-correct" if index == 0 else f"{split}-target"
            observed = target if index == 0 else f"{split}-observed"
            image_sha256 = hashlib.sha256(f"{split}-{index}".encode()).hexdigest()
            if split == "validation" and index == 0 and leak == "image":
                image_sha256 = train_first_sha
            if split == "validation" and index == 0 and leak == "pair":
                observed, target = train_first_pair
            source = DATASET_BUILDER.SourceSample(
                split=split,
                image_ref=f"images/{split}/{index}.png",
                image_path=root / f"unused-{split}-{index}.png",
                image_sha256=image_sha256,
                target_text=target,
            )
            split_sources.append(source)
            split_predictions.append(
                DATASET_BUILDER.PredictedSample(
                    source=source,
                    observed_text=observed,
                    confidence=0.9 if index == 0 else 0.4,
                    is_error=observed != target,
                )
            )
        source_samples[split] = tuple(split_sources)
        predictions[split] = tuple(split_predictions)

    output_hashes: dict[str, str] = {}
    for split in TRAINING.SPLITS:
        output_path = root / f"{split}.jsonl"
        DATASET_BUILDER._write_jsonl(output_path, predictions[split])
        output_hashes[split] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    source_dataset = _source_dataset(source_samples)
    zeros = {split: 0 for split in TRAINING.SPLITS}
    report = DATASET_BUILDER._build_report(
        source_dataset,
        _rights_evidence(),
        hashlib.sha256(b"ocr-model").hexdigest(),
        "cpu",
        2,
        source_samples,
        predictions,
        set(),
        set(),
        zeros,
        zeros,
        output_hashes,
    )
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def _load_and_validate_builder_dataset(root: Path) -> dict[str, Any]:
    split_payloads = {
        split: (root / f"{split}.jsonl").read_bytes() for split in TRAINING.SPLITS
    }
    rows = {
        split: TRAINING._load_rows(root / f"{split}.jsonl", split_payloads[split])
        for split in TRAINING.SPLITS
    }
    split_sha256 = {
        split: hashlib.sha256(split_payloads[split]).hexdigest() for split in TRAINING.SPLITS
    }
    return TRAINING._validate_dataset_report(
        root / "report.json",
        (root / "report.json").read_bytes(),
        rows,
        split_sha256,
    )


def _load_fixture_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: TRAINING._load_rows(
            root / f"{split}.jsonl",
            (root / f"{split}.jsonl").read_bytes(),
        )
        for split in TRAINING.SPLITS
    }


def _source_dataset_for_rows(
    root: Path,
    rows: dict[str, list[dict[str, Any]]],
) -> Any:
    samples = {
        split: tuple(
            DATASET_BUILDER.SourceSample(
                split=split,
                image_ref=row["image_ref"],
                image_path=root / "unused" / row["image_ref"],
                image_sha256=row["image_sha256"],
                target_text=row["target_text"],
            )
            for row in rows[split]
        )
        for split in TRAINING.SPLITS
    }
    return _source_dataset(samples)


def test_builder_output_contract_is_accepted(tmp_path: Path) -> None:
    root = _write_builder_dataset(tmp_path / "risk")

    report = _load_and_validate_builder_dataset(root)

    assert report["record_fields"] == list(DATASET_BUILDER.RECORD_FIELDS)
    assert report["source_rights"]["validation"] == "verified"


def test_training_snapshot_is_bounded_and_regular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "risk.jsonl"
    source.write_bytes(b"1234")
    monkeypatch.setattr(TRAINING, "_MAX_SNAPSHOT_BYTES", 3)

    with pytest.raises(ValueError, match="size limit"):
        TRAINING._snapshot(source)


def test_legacy_risk_dataset_contract_cannot_authorize_new_weights(tmp_path: Path) -> None:
    root = _write_builder_dataset(tmp_path / "risk")
    payload = json.loads((root / "report.json").read_text(encoding="utf-8"))
    payload["schema_version"] = "model-a-line-risk-dataset/1.0"
    (root / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        _load_and_validate_builder_dataset(root)


def test_source_authorization_is_recomputed_not_accepted_from_report_alone(
    tmp_path: Path,
) -> None:
    root = _write_builder_dataset(tmp_path / "risk")
    report = _load_and_validate_builder_dataset(root)
    rows = _load_fixture_rows(root)
    source_dataset = _source_dataset_for_rows(root, rows)

    TRAINING._validate_source_authorization(
        report,
        rows,
        source_dataset,
        _rights_evidence(),
    )

    changed_source = DATASET_BUILDER.SourceDataset(
        samples=source_dataset.samples,
        split_tsv_sha256=source_dataset.split_tsv_sha256,
        split_images_sha256=source_dataset.split_images_sha256,
        sha256=hashlib.sha256(b"different-source").hexdigest(),
    )
    with pytest.raises(ValueError, match="not bound to the supplied source dataset"):
        TRAINING._validate_source_authorization(
            report,
            rows,
            changed_source,
            _rights_evidence(),
        )

    changed_evidence = OcrTrainingRightsEvidence(
        rights_manifest_sha256=hashlib.sha256(b"different-rights").hexdigest(),
        dataset_report_sha256=_rights_evidence().dataset_report_sha256,
        document_count=3,
        identity_assurance="authenticated_external_context",
        evidence_artifacts_sha256=_rights_evidence().evidence_artifacts_sha256,
        training_images_sha256=_rights_evidence().training_images_sha256,
        derivation_assurance="authenticated_external_report_attestation",
        derivation_verification_scope="report_digest_and_artifact_integrity_only",
        derivation_attestor_id="trusted-dataset-builder",
    )
    with pytest.raises(ValueError, match="source rights binding mismatch"):
        TRAINING._validate_source_authorization(
            report,
            rows,
            source_dataset,
            changed_evidence,
        )


def test_derived_rows_must_be_members_of_the_rights_verified_source(
    tmp_path: Path,
) -> None:
    root = _write_builder_dataset(tmp_path / "risk")
    report = _load_and_validate_builder_dataset(root)
    rows = _load_fixture_rows(root)
    source_dataset = _source_dataset_for_rows(root, rows)
    tampered_rows = {split: [dict(row) for row in rows[split]] for split in TRAINING.SPLITS}
    tampered_rows["validation"][0]["target_text"] = "UNAUTHORIZED_TARGET"

    with pytest.raises(ValueError, match="row target text mismatch"):
        TRAINING._validate_source_authorization(
            report,
            tampered_rows,
            source_dataset,
            _rights_evidence(),
        )


def test_risk_entrypoint_blocks_output_before_rights_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    risk_dataset = _write_builder_dataset(tmp_path / "risk")
    source_dataset = tmp_path / "source"
    source_dataset.mkdir()
    character_dict = tmp_path / "characters.txt"
    character_dict.write_text("가\n나\n", encoding="utf-8")
    rights_manifest = (
        Path(__file__).resolve().parents[2]
        / "ai"
        / "datasets"
        / "configs"
        / "ocr_training_rights.example.json"
    )
    output = tmp_path / "output-must-not-start" / "risk-model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path("ai/modeling/train_model_a_line_risk.py")),
            "--dataset",
            str(risk_dataset),
            "--source-dataset",
            str(source_dataset),
            "--character-dict",
            str(character_dict),
            "--rights-manifest",
            str(rights_manifest),
            "--rights-evidence-root",
            str(tmp_path / "rights-evidence"),
            "--out",
            str(output),
            "--train-split",
            "validation",
            "--validation-split",
            "test",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        TRAINING.main()

    assert raised.value.code == 2
    assert "rights gate failed" in capsys.readouterr().err
    assert not output.parent.exists()


def test_risk_entrypoint_rejects_before_any_path_resolution_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_path_resolution(*args: object, **kwargs: object) -> Path:
        raise AssertionError(f"path I/O must not start: {args}, {kwargs}")

    monkeypatch.setattr(TRAINING, "_resolve", forbidden_path_resolution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path("ai/modeling/train_model_a_line_risk.py")),
            "--dataset",
            str(tmp_path / "missing-risk"),
            "--source-dataset",
            str(tmp_path / "missing-source"),
            "--character-dict",
            str(tmp_path / "missing-dict"),
            "--rights-manifest",
            str(tmp_path / "missing-rights"),
            "--rights-evidence-root",
            str(tmp_path / "missing-evidence"),
            "--out",
            str(tmp_path / "missing-output"),
            "--train-split",
            "validation",
            "--validation-split",
            "test",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        TRAINING.main()

    assert raised.value.code == 2
    assert "authenticated external context" in capsys.readouterr().err


def test_report_jsonl_digest_is_recomputed(tmp_path: Path) -> None:
    root = _write_builder_dataset(tmp_path / "risk")
    train_path = root / "train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8").replace("images/train/0.png", "images/train/x.png"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSONL sha256 mismatch"):
        _load_and_validate_builder_dataset(root)


@pytest.mark.parametrize("leak", ["image", "pair"])
def test_report_zero_leak_claim_is_independently_recomputed(
    tmp_path: Path, leak: str
) -> None:
    root = _write_builder_dataset(tmp_path / "risk", leak=leak)

    with pytest.raises(ValueError, match="cross-split"):
        _load_and_validate_builder_dataset(root)


def test_split_roles_must_be_distinct_from_each_other_and_upstream() -> None:
    TRAINING._validate_split_selection("validation", "test", "train")

    with pytest.raises(ValueError, match="must differ"):
        TRAINING._validate_split_selection("validation", "validation", "train")
    with pytest.raises(ValueError, match="overlaps"):
        TRAINING._validate_split_selection("train", "test", "train")
    with pytest.raises(ValueError, match="fixed to train"):
        TRAINING._validate_split_selection("train", "test", "validation")


def test_metrics_and_recall_threshold_use_only_supplied_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    assert TRAINING.roc_auc(labels, scores) == pytest.approx(0.75)
    assert TRAINING.average_precision(labels, scores) == pytest.approx(5 / 6)

    threshold = TRAINING.select_threshold(labels, scores, target_recall=1.0)
    metrics = TRAINING.evaluate_risk(labels, scores, threshold)
    assert threshold == pytest.approx(0.35)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(2 / 3)


def test_recall_threshold_prefers_the_highest_precision_candidate() -> None:
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.2, 0.9, 0.7, 0.6])

    threshold = TRAINING.select_threshold(labels, scores, target_recall=0.5)

    assert threshold == pytest.approx(0.9)


def test_average_precision_groups_tied_scores_and_is_permutation_invariant() -> None:
    scores = np.array([0.5, 0.5])

    first = TRAINING.average_precision(np.array([1, 0]), scores)
    reversed_rows = TRAINING.average_precision(np.array([0, 1]), scores)

    assert first == pytest.approx(0.5)
    assert reversed_rows == pytest.approx(first)


def test_encoded_empty_observation_uses_unknown_token() -> None:
    rows: list[dict[str, Any]] = [
        {
            "image_ref": "images/train/a.png",
            "observed_text": "",
            "confidence": 0.0,
            "target_text": "정답",
            "is_error": True,
        }
    ]

    tokens, numeric, labels = TRAINING._encode_rows(
        rows, ["<pad>", "<unk>", "정", "답"], max_length=4
    )

    assert tokens.tolist() == [[1, 0, 0, 0]]
    assert numeric.shape == (1, 6)
    assert labels.tolist() == [1.0]


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        (np.array([]), np.array([])),
        (np.array([0, 2]), np.array([0.1, 0.2])),
        (np.array([0, 1]), np.array([0.1, np.nan])),
    ],
)
def test_metric_inputs_are_rejected(labels: np.ndarray, scores: np.ndarray) -> None:
    with pytest.raises(ValueError):
        TRAINING.roc_auc(labels, scores)
