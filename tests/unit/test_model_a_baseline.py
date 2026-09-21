from __future__ import annotations

from pathlib import Path

from scan2hwpx.contracts import EvidenceIR, TextContentNode, contract_sha256
from scan2hwpx.contracts.legacy import evidence_from_document
from scan2hwpx.model_a import RuleBasedModelABaseline
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider

FIXTURE = Path("tests/fixtures/ocr_page_1.json")


def test_baseline_builds_stable_content_and_reading_order() -> None:
    evidence = evidence_from_document(FixtureOcrProvider().convert(FIXTURE))
    baseline = RuleBasedModelABaseline()

    first = baseline.analyze(evidence)
    second = baseline.analyze(evidence)

    assert first == second
    assert first.evidence_ir_sha256 == contract_sha256(evidence)
    assert len(first.nodes) == len(evidence.pages[0].observations)
    assert first.reading_order == tuple(node.id for node in first.nodes)
    assert [node.role.value for node in first.nodes[1:3]] == ["question", "choice"]
    assert first.nodes[3].needs_review is True
    first.assert_evidence_integrity(evidence)


def test_baseline_uses_selected_candidate_and_flags_disagreement() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    block = document.pages[0].blocks[0]
    block.style["ocr_candidates"] = [
        {
            "engine": "paddleocr",
            "text": block.text,
            "confidence": 0.81,
            "quality": 0.88,
            "selected": True,
        },
        {
            "engine": "windows-ocr-ko",
            "text": "다른 후보",
            "confidence": 0.95,
            "quality": 0.96,
            "selected": False,
        },
    ]

    content = RuleBasedModelABaseline().analyze(evidence_from_document(document))
    node = content.nodes[0]

    assert isinstance(node, TextContentNode)
    assert node.text == block.text
    assert node.confidence == 0.81
    assert node.needs_review is True


def test_baseline_uses_highest_confidence_when_no_candidate_is_selected() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    block = document.pages[0].blocks[0]
    block.style["ocr_candidates"] = [
        {"engine": "first", "text": "낮은 후보", "confidence": 0.7, "selected": True},
        {"engine": "second", "text": "높은 후보", "confidence": 0.95},
    ]
    payload = evidence_from_document(document).model_dump(mode="json")
    for candidate in payload["pages"][0]["observations"][0]["ocr_candidates"]:
        candidate["selected"] = False
    evidence = EvidenceIR.model_validate(payload)

    content = RuleBasedModelABaseline().analyze(evidence)
    node = content.nodes[0]

    assert isinstance(node, TextContentNode)
    assert node.text == "높은 후보"
    assert node.confidence == 0.95
    assert node.needs_review is True
