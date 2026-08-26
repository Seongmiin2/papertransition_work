from __future__ import annotations

import json
from pathlib import Path

from scan2hwpx.training.corpus import classify_role, prepare_training_corpus, split_for_document


def test_classify_role_keeps_answers_out_of_source_pool() -> None:
    assert classify_role(Path("학교 시험.pdf")) == "source_pdf"
    assert classify_role(Path("학교 시험 답지.pdf")) == "answer_key"
    assert classify_role(Path("학교 시험 정답.hwp")) == "answer_key"
    assert classify_role(Path("학교 시험.hwp")) == "editable_reference"


def test_split_is_document_stable_and_non_targets_have_no_split() -> None:
    digest = "a" * 64

    assert split_for_document(digest) == split_for_document(digest)
    assert split_for_document(digest) in {"train", "validation", "test"}
    assert split_for_document(digest, eligible=False) is None


def test_missing_declared_pair_is_reported_without_aborting_corpus(tmp_path: Path) -> None:
    pair_config = tmp_path / "pairs.json"
    pair_config.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "id": "missing-gold",
                        "source_pdf": "missing.pdf",
                        "transcript_pdf": "missing-transcript.pdf",
                        "editable_target": "missing.docx",
                        "split": "test",
                        "visually_verified": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = prepare_training_corpus(
        [],
        tmp_path / "corpus",
        pair_config=pair_config,
        project_root=tmp_path,
        copy_files=False,
    )

    assert report["summary"]["gold_pairs"] == 0
    assert report["summary"]["failures"] == 1
    assert report["declared_pairs"][0]["status"] == "invalid"
    assert "FileNotFoundError" in report["declared_pairs"][0]["error"]
    assert (tmp_path / "corpus" / "corpus_report.json").is_file()
