from __future__ import annotations

from scan2hwpx.blueprint.model_a import classify_role
from scan2hwpx.contracts.models import (
    ContentIR,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    ObservationKind,
    OcrCandidate,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.ir.models import BlockKind

LOW_CONFIDENCE_THRESHOLD = 0.75

_CONTENT_ROLES = {
    BlockKind.QUESTION: ContentRole.QUESTION,
    BlockKind.CHOICE: ContentRole.CHOICE,
    BlockKind.INSTRUCTION: ContentRole.INSTRUCTION,
    BlockKind.PASSAGE: ContentRole.PASSAGE,
}


class RuleBasedModelABaseline:
    """Deterministic benchmark baseline, not a trained Model A."""

    model_id = "rule-based-model-a-baseline/1.0"

    def analyze(self, evidence: EvidenceIR) -> ContentIR:
        nodes: list[TextContentNode] = []
        reading_order: list[str] = []
        question_started = False

        for page in sorted(evidence.pages, key=lambda item: item.page_no):
            for observation in page.observations:
                if observation.kind != ObservationKind.TEXT_LINE:
                    continue
                candidate = _selected_candidate(observation)
                block_kind = classify_role(
                    candidate.text,
                    question_started=question_started,
                )
                if block_kind == BlockKind.QUESTION:
                    question_started = True
                node_id = f"{evidence.id}:content:{observation.id}"
                nodes.append(
                    TextContentNode(
                        id=node_id,
                        role=_CONTENT_ROLES.get(block_kind, ContentRole.OTHER),
                        text=candidate.text,
                        evidence_refs=(observation.id,),
                        confidence=candidate.confidence,
                        needs_review=_needs_review(observation, candidate),
                    )
                )
                reading_order.append(node_id)

        if not nodes:
            raise ValueError("EvidenceIR contains no text_line observations")
        return ContentIR(
            id=f"{evidence.id}:content",
            evidence_ir_id=evidence.id,
            evidence_ir_sha256=contract_sha256(evidence),
            nodes=tuple(nodes),
            reading_order=tuple(reading_order),
        )


def _selected_candidate(observation: EvidenceObservation) -> OcrCandidate:
    selected = next(
        (candidate for candidate in observation.ocr_candidates if candidate.selected),
        None,
    )
    if selected is not None:
        return selected
    return max(observation.ocr_candidates, key=lambda candidate: candidate.confidence)


def _needs_review(
    observation: EvidenceObservation,
    selected: OcrCandidate,
) -> bool:
    disagreement = len({candidate.text for candidate in observation.ocr_candidates}) > 1
    low_quality = selected.quality is not None and selected.quality < LOW_CONFIDENCE_THRESHOLD
    return disagreement or selected.confidence < LOW_CONFIDENCE_THRESHOLD or low_quality
