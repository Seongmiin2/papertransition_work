from __future__ import annotations

from scan2hwpx.ir.models import AnnotationState, BBox, Block, BlockKind
from scan2hwpx.training.ocr_adaptation import (
    accept_silver_label,
    write_dictionary_compatible_labels,
)


def _block(text: str = "국어 시험", confidence: float = 0.99) -> Block:
    return Block(
        id="b1",
        kind=BlockKind.UNKNOWN,
        bbox=BBox(pixel=(1, 2, 100, 20), normalized=(0.01, 0.01, 0.5, 0.1)),
        reading_order=0,
        text=text,
        confidence=confidence,
        source_provider="test",
        source_payload_ref="fixture#/line/1",
        annotation_state=AnnotationState.PRINTED,
    )


def test_silver_filter_only_accepts_high_confidence_language_lines() -> None:
    assert accept_silver_label(_block(), 0.95, minimum_confidence=0.96, minimum_quality=0.88) == (
        True,
        "accepted",
    )
    assert accept_silver_label(
        _block(confidence=0.7), 0.95, minimum_confidence=0.96, minimum_quality=0.88
    ) == (False, "confidence")
    assert accept_silver_label(
        _block(text="--"), 0.95, minimum_confidence=0.96, minimum_quality=0.88
    ) == (False, "no_alphanumeric")


def test_dictionary_compatibility_filters_unknown_characters(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "chars.txt").write_text("국\n어\n", encoding="utf-8")
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.txt").write_text(
            "images/a.png\t국어\nimages/b.png\t국㉠\n", encoding="utf-8"
        )

    report = write_dictionary_compatible_labels(tmp_path, tmp_path / "chars.txt")

    assert report["splits"]["train"]["accepted"] == 1
    assert report["splits"]["train"]["rejected"] == 1
    assert (tmp_path / "train.official.txt").read_text(encoding="utf-8").endswith("국어\n")
