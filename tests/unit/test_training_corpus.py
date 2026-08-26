from __future__ import annotations

from pathlib import Path

from scan2hwpx.training.corpus import classify_role, split_for_document


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
