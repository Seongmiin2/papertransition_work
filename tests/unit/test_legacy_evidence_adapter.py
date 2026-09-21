from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scan2hwpx.contracts import EvidenceSourceKind
from scan2hwpx.contracts.legacy import evidence_from_document
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider

FIXTURE = Path("tests/fixtures/ocr_page_1.json")


def test_legacy_document_maps_every_block_with_stable_evidence() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)

    first = evidence_from_document(document, {1: "blob://pages/source-page-1"})
    second = evidence_from_document(document, {1: "blob://pages/source-page-1"})

    assert first == second
    assert first.source_document_sha256 == document.source_hash
    assert len(first.pages) == len(document.pages)
    page = first.pages[0]
    assert len(page.observations) == len(document.pages[0].blocks)
    image_source = next(source for source in first.sources if source.id == page.image_source_ref)
    assert image_source.artifact_ref == "blob://pages/source-page-1"

    source_by_id = {source.id: source for source in first.sources}
    for block_index, (observation, block) in enumerate(
        zip(page.observations, document.pages[0].blocks, strict=True)
    ):
        assert observation.bbox.pixel == block.bbox.pixel
        assert observation.bbox.normalized == block.bbox.normalized
        assert observation.confidence == block.confidence
        assert observation.ocr_candidates[0].text == block.text
        assert observation.ocr_candidates[0].provider == block.source_provider
        candidate_source = source_by_id[observation.ocr_candidates[0].source_ref]
        assert candidate_source.artifact_ref == (
            f"artifact://{document.source_hash}/legacy-ocr/pages/1/blocks/{block_index}"
        )
        assert "per-candidate provider artifacts unavailable" in candidate_source.producer
        assert set(observation.source_refs) <= source_by_id.keys()

    original_source = next(
        source
        for source in first.sources
        if source.kind == EvidenceSourceKind.ORIGINAL_DOCUMENT
    )
    assert original_source.artifact_ref == f"artifact://{document.source_hash}/source"
    assert original_source.sha256 == document.source_hash
    assert all(
        block.source_payload_ref not in first.model_dump_json()
        for block in document.pages[0].blocks
    )


def test_legacy_ocr_candidates_preserve_engine_text_and_confidence() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    block = document.pages[0].blocks[0]
    block.style["ocr_candidates"] = [
        {
            "engine": "paddleocr",
            "text": block.text,
            "confidence": 0.81,
            "quality": 0.72,
        },
        {
            "engine": "windows-ocr-ko",
            "text": "3학년 국어 기말 고사",
            "confidence": 0.92,
            "quality": 0.88,
        },
    ]

    evidence = evidence_from_document(document)
    candidates = evidence.pages[0].observations[0].ocr_candidates

    assert [(item.text, item.confidence, item.quality, item.selected) for item in candidates] == [
        (block.text, 0.81, 0.72, True),
        ("3학년 국어 기말 고사", 0.92, 0.88, False),
    ]
    source_by_id = {source.id: source for source in evidence.sources}
    assert [item.provider for item in candidates] == [
        "paddleocr",
        "windows-ocr-ko",
    ]
    assert len({item.source_ref for item in candidates}) == 1
    shared_source = source_by_id[candidates[0].source_ref]
    assert "per-candidate provider artifacts unavailable" in shared_source.producer
    assert shared_source.artifact_ref.startswith("artifact://")
    assert block.source_payload_ref not in shared_source.artifact_ref


def test_unmarked_candidates_add_selected_block_fallback_when_text_is_missing() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    block = document.pages[0].blocks[0]
    block.style["ocr_candidates"] = [
        {
            "engine": "other-ocr",
            "text": "후보에 원문이 없음",
            "confidence": 0.9,
            "quality": 0.8,
        }
    ]

    evidence = evidence_from_document(document)
    candidates = evidence.pages[0].observations[0].ocr_candidates

    assert [candidate.text for candidate in candidates] == ["후보에 원문이 없음", block.text]
    assert [candidate.selected for candidate in candidates] == [False, True]
    assert candidates[-1].quality is None


def test_metadata_paths_are_never_copied_into_evidence() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    local_path = "C:/Users/Alice/student-name.pdf"
    document.metadata["source"] = local_path
    document.metadata["fixture"] = "https://private.example/student-name.pdf"

    evidence = evidence_from_document(document)
    encoded = evidence.model_dump_json()

    assert local_path not in encoded
    assert "private.example" not in encoded


@pytest.mark.parametrize(
    "unsafe_ref",
    [
        "C:/private/page-1.png",
        "https://private.example/page-1.png",
        "artifact://document/../private/page-1.png",
    ],
)
def test_page_image_ref_rejects_local_external_or_traversal_paths(
    unsafe_ref: str,
) -> None:
    document = FixtureOcrProvider().convert(FIXTURE)

    with pytest.raises(ValidationError):
        evidence_from_document(document, {1: unsafe_ref})


def test_page_image_sha_binds_source_bytes_and_artifact_ref() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    page_sha256 = "a" * 64

    evidence = evidence_from_document(
        document,
        page_image_sha256={1: page_sha256},
    )
    page_source = next(
        source for source in evidence.sources if source.kind == EvidenceSourceKind.PAGE_IMAGE
    )

    assert page_source.artifact_ref == f"artifact://{page_sha256}/page-image"
    assert page_source.sha256 == page_sha256


def test_page_image_falls_back_to_deterministic_logical_artifact_ref() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)

    evidence = evidence_from_document(document)
    page_source = next(
        source for source in evidence.sources if source.kind == EvidenceSourceKind.PAGE_IMAGE
    )

    assert page_source.artifact_ref == f"artifact://{document.source_hash}/pages/1/image"
