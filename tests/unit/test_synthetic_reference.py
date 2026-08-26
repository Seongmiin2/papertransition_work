from __future__ import annotations

from pathlib import Path

from scan2hwpx.reference.synthetic import (
    split_for_document,
    split_training_text,
    write_training_dictionary,
)


def test_split_training_text_wraps_long_lines_without_losing_content() -> None:
    source = "다음 글을 읽고 물음에 답하시오. 우리말 문장을 정확하게 인식해야 합니다."

    lines = split_training_text(source, maximum=20)

    assert all(2 <= len(line) <= 20 for line in lines)
    assert "".join(lines).replace(" ", "") == source.replace(" ", "")


def test_document_split_is_stable() -> None:
    assert split_for_document("same-document") == split_for_document("same-document")
    assert split_for_document("same-document") in {"train", "validation", "test"}


def test_training_dictionary_preserves_model_order_and_appends_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text(
        "PostProcess:\n  character_dict:\n    - 가\n    - 나\n", encoding="utf-8"
    )
    output = tmp_path / "dict.txt"

    report = write_training_dictionary({"가", "다", " "}, output, base_model_dir=model_dir)

    assert output.read_text(encoding="utf-8").splitlines() == ["가", "나", "다"]
    assert report["appended_characters"] == 1
