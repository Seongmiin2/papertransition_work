from __future__ import annotations

import hashlib
import json

import pytest

from scan2hwpx.contracts import (
    BBox,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
    contract_sha256,
)
from scan2hwpx.model_a.classification_blocks import (
    CLASSIFICATION_GROUPER_SHA256,
    ModelAClassificationBlockFormation,
    ModelAClassificationGroupingArtifact,
    ModelAClassificationLayoutZone,
    ModelAClassificationPlan,
    ModelAClassificationRoleSource,
    build_model_a_classification_plan,
    resolve_model_a_classification_roles,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bbox(
    x0: float,
    y0: float,
    *,
    x1: float | None = None,
    y1: float | None = None,
) -> BBox:
    normalized_x1 = x0 + 0.2 if x1 is None else x1
    normalized_y1 = y0 + 0.01 if y1 is None else y1
    return BBox(
        pixel=(x0 * 100, y0 * 100, normalized_x1 * 100, normalized_y1 * 100),
        normalized=(x0, y0, normalized_x1, normalized_y1),
    )


def _text(
    observation_id: str,
    text: str,
    x0: float,
    y0: float,
    *,
    x1: float | None = None,
    y1: float | None = None,
) -> EvidenceObservation:
    return EvidenceObservation(
        id=observation_id,
        kind=ObservationKind.TEXT_LINE,
        bbox=_bbox(x0, y0, x1=x1, y1=y1),
        confidence=0.9,
        source_refs=("page-image", "ocr-source"),
        ocr_candidates=(
            OcrCandidate(
                text=text,
                provider="fixture-ocr",
                confidence=0.9,
                source_ref="ocr-source",
                selected=True,
            ),
        ),
    )


def _image(observation_id: str, x0: float, y0: float) -> EvidenceObservation:
    return EvidenceObservation(
        id=observation_id,
        kind=ObservationKind.IMAGE,
        bbox=_bbox(x0, y0),
        confidence=0.8,
        source_refs=("page-image",),
    )


def _evidence(*observations: EvidenceObservation) -> EvidenceIR:
    return EvidenceIR(
        id="evidence-fixture",
        source_document_sha256="a" * 64,
        sources=(
            EvidenceSource(
                id="page-image",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://page-image",
                producer="fixture-renderer",
                sha256=_sha256(b"page"),
            ),
            EvidenceSource(
                id="ocr-source",
                kind=EvidenceSourceKind.OCR_PROVIDER,
                artifact_ref="artifact://ocr-source",
                producer="fixture-ocr",
                sha256=_sha256(b"ocr"),
            ),
        ),
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=100,
                height=100,
                image_source_ref="page-image",
                observations=observations,
            ),
        ),
    )


def _mixed_document_evidence() -> EvidenceIR:
    return _evidence(
        _text("footer", "- 1 -", 0.48, 0.94, x1=0.52),
        _text("question", "1. 충분히 긴 질문 문장입니다.", 0.10, 0.10),
        _text("question-wrap", "질문의 둘째 줄입니다.", 0.11, 0.115),
        _text("score", "[3점]", 0.11, 0.13),
        _text("choice-1", "① 첫 번째 선택지", 0.11, 0.16),
        _text("choice-1-wrap", "선택지의 아주 먼 줄", 0.13, 0.31),
        _text("choice-1-speaker", "고운: 선택지 안의 발화", 0.11, 0.45),
        _text("choice-2", "② 두 번째 선택지", 0.11, 0.48),
        _text("choice-2-wrap", "두 번째 선택지의 둘째 줄", 0.13, 0.50),
        _text("bogi", "----------------<보 기>----------------", 0.10, 0.54),
        _text("range", "[3~4] 다음을 읽고 답하시오.", 0.10, 0.58),
        _text("range-wrap", "지시문의 둘째 줄입니다.", 0.12, 0.595),
        _text("generic", "새로운 일반 문단입니다.", 0.30, 0.64),
    )


def test_plan_partitions_semantic_blocks_and_preserves_canonical_lines() -> None:
    evidence = _mixed_document_evidence()

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert plan.evidence_ir_sha256 == contract_sha256(evidence)
    assert plan.grouping.grouper_sha256 == CLASSIFICATION_GROUPER_SHA256
    assert plan.ordered_text_observation_ids == (
        "question",
        "question-wrap",
        "score",
        "choice-1",
        "choice-1-wrap",
        "choice-1-speaker",
        "choice-2",
        "choice-2-wrap",
        "bogi",
        "range",
        "range-wrap",
        "generic",
        "footer",
    )
    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("question", "question-wrap", "score"),
        ("choice-1", "choice-1-wrap", "choice-1-speaker"),
        ("choice-2", "choice-2-wrap"),
        ("bogi",),
        ("range",),
        ("range-wrap",),
        ("generic",),
        ("footer",),
    )
    assert tuple(block.formation for block in plan.blocks) == (
        ModelAClassificationBlockFormation.NUMBERED_QUESTION,
        ModelAClassificationBlockFormation.CIRCLED_CHOICE,
        ModelAClassificationBlockFormation.CIRCLED_CHOICE,
        ModelAClassificationBlockFormation.BOGI_SEPARATOR,
        ModelAClassificationBlockFormation.RANGE_INSTRUCTION,
        ModelAClassificationBlockFormation.GENERIC_GEOMETRY,
        ModelAClassificationBlockFormation.GENERIC_GEOMETRY,
        ModelAClassificationBlockFormation.BARE_PAGE_NUMBER_FOOTER,
    )
    assert tuple(block.forced_role for block in plan.blocks) == (
        "question",
        "choice",
        "choice",
        "caption",
        "instruction",
        None,
        None,
        "footer",
    )
    assert plan.blocks[-1].layout_zone == ModelAClassificationLayoutZone.BOTTOM_MARGIN
    assert plan == build_model_a_classification_plan(evidence, target_page_id="page-1")
    assert contract_sha256(plan) == contract_sha256(
        build_model_a_classification_plan(evidence, target_page_id="page-1")
    )


def _two_generic_lines(
    *,
    delta_x: float,
    delta_y: float,
    with_image: bool = False,
) -> EvidenceIR:
    first = _text("first", "첫 번째 일반 줄", 0.10, 0.19, y1=0.20)
    second = _text(
        "second",
        "두 번째 일반 줄",
        0.10 + delta_x,
        0.20 + delta_y,
        y1=0.21 + delta_y,
    )
    if with_image:
        return _evidence(first, _image("between-image", 0.30, 0.20), second)
    return _evidence(first, second)


@pytest.mark.parametrize(
    ("delta_x", "delta_y"),
    (
        (-0.005, 0.0),
        (0.05, 0.0),
        (0.0, -0.005),
        (0.0, 0.028),
    ),
)
def test_generic_geometry_inclusive_boundaries_merge(
    delta_x: float,
    delta_y: float,
) -> None:
    plan = build_model_a_classification_plan(
        _two_generic_lines(delta_x=delta_x, delta_y=delta_y),
        target_page_id="page-1",
    )

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("first", "second"),
    )


@pytest.mark.parametrize(
    ("delta_x", "delta_y"),
    (
        (-0.005_1, 0.0),
        (0.050_1, 0.0),
        (0.0, -0.005_1),
        (0.0, 0.028_1),
    ),
)
def test_generic_geometry_outside_boundaries_split(
    delta_x: float,
    delta_y: float,
) -> None:
    plan = build_model_a_classification_plan(
        _two_generic_lines(delta_x=delta_x, delta_y=delta_y),
        target_page_id="page-1",
    )

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("first",),
        ("second",),
    )


def test_image_content_is_a_hard_grouping_boundary() -> None:
    plan = build_model_a_classification_plan(
        _two_generic_lines(delta_x=0.0, delta_y=0.0, with_image=True),
        target_page_id="page-1",
    )

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("first",),
        ("second",),
    )


@pytest.mark.parametrize("image_y", (0.05, 0.94))
def test_margin_image_does_not_split_body_block(image_y: float) -> None:
    first = _text("first", "first body line", 0.10, 0.19, y1=0.20)
    second = _text("second", "second body line", 0.10, 0.20, y1=0.21)
    evidence = _evidence(first, _image("margin-image", 0.30, image_y), second)

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("first", "second"),
    )


def _spatial_choice_evidence() -> EvidenceIR:
    return _evidence(
        _text("question", "3. 충분히 긴 오른쪽 질문입니다.", 0.52, 0.35, x1=0.90),
        _text("score", "[5점]", 0.53, 0.37, x1=0.58),
        _text("marker-1", "①", 0.53, 0.44, x1=0.55, y1=0.45),
        _text("choice-1-a", "첫 선택지의 위쪽 본문", 0.56, 0.405, x1=0.90, y1=0.415),
        _text("choice-1-b", "첫 선택지의 아래쪽 본문", 0.56, 0.46, x1=0.90, y1=0.47),
        _text("marker-2", "②", 0.53, 0.53, x1=0.55, y1=0.54),
        _text("choice-2-a", "둘째 선택지의 위쪽 본문", 0.56, 0.50, x1=0.90, y1=0.51),
        _text("choice-2-b", "둘째 선택지의 아래쪽 본문", 0.56, 0.56, x1=0.90, y1=0.57),
        _text("marker-3", "③", 0.53, 0.63, x1=0.55, y1=0.64),
        _text("choice-3-a", "셋째 선택지의 위쪽 본문", 0.56, 0.59, x1=0.90, y1=0.60),
        _text("choice-3-b", "셋째 선택지의 아래쪽 본문", 0.56, 0.67, x1=0.90, y1=0.68),
    )


def test_standalone_choice_markers_use_spatial_bands_despite_y_inversion() -> None:
    plan = build_model_a_classification_plan(
        _spatial_choice_evidence(),
        target_page_id="page-1",
    )

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("question", "score"),
        ("marker-1", "choice-1-a", "choice-1-b"),
        ("marker-2", "choice-2-a", "choice-2-b"),
        ("marker-3", "choice-3-a", "choice-3-b"),
    )
    assert tuple(block.formation for block in plan.blocks[1:]) == (
        ModelAClassificationBlockFormation.CIRCLED_CHOICE_SPATIAL_BAND,
        ModelAClassificationBlockFormation.CIRCLED_CHOICE_SPATIAL_BAND,
        ModelAClassificationBlockFormation.CIRCLED_CHOICE_SPATIAL_BAND,
    )
    assert all(block.layout_zone == "right_body" for block in plan.blocks)


@pytest.mark.parametrize(
    "range_text",
    (
        "[10~12] 다음을 읽고 물음에 답하시오.",
        "[10-12] 다음을 읽고 물음에 답하시오.",
    ),
)
def test_range_instruction_is_singleton_and_does_not_consume_passage(
    range_text: str,
) -> None:
    evidence = _evidence(
        _text("range", range_text, 0.52, 0.84, x1=0.90),
        _text("passage-head", "(가)", 0.53, 0.86, x1=0.57),
        _text("poem-1", "저기 사라진 별의 자리", 0.53, 0.875, x1=0.80),
        _text("poem-2", "한동안은 꺼내 볼 수 있을 거야", 0.53, 0.89, x1=0.85),
    )

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert plan.blocks[0].ordered_text_observation_ids == ("range",)
    assert plan.blocks[0].forced_role == "instruction"
    assert plan.blocks[1].ordered_text_observation_ids == (
        "passage-head",
        "poem-1",
        "poem-2",
    )
    assert plan.blocks[1].forced_role is None


def test_question_anchor_rejects_answer_key_numbers_without_a_stem() -> None:
    evidence = _evidence(
        _text("empty-number", "1.", 0.10, 0.10),
        _text("number-and-answer", "11. ②", 0.10, 0.20),
        _text("real-question", "1. 질문 문장입니까?", 0.10, 0.30),
    )

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert tuple(block.formation for block in plan.blocks) == (
        ModelAClassificationBlockFormation.GENERIC_GEOMETRY,
        ModelAClassificationBlockFormation.GENERIC_GEOMETRY,
        ModelAClassificationBlockFormation.NUMBERED_QUESTION,
    )
    assert tuple(block.forced_role for block in plan.blocks) == (
        None,
        None,
        "question",
    )


def test_numbered_pronunciation_rules_without_question_evidence_are_generic() -> None:
    evidence = _evidence(
        _text(
            "rule-1",
            "1. ‘ㅎ(ㄶ, ㅀ)’ 뒤에 ‘ㄱ, ㄷ, ㅈ’이 결합되는 경우에는, 뒤",
            0.519,
            0.10,
        ),
        _text(
            "rule-1-wrap",
            "음절 첫소리와 합쳐져 [ㅋ, ㅌ, ㅊ]으로 발음한다.",
            0.544,
            0.115,
        ),
        _text(
            "rule-2",
            "2. ‘ㅎ(ㄶ, ㅀ)’ 뒤에 ‘ㅅ’이 결합되는 경우에는 ‘ㅅ’을 [ㅆ]으",
            0.519,
            0.20,
        ),
        _text("rule-2-wrap", "로 발음한다.", 0.544, 0.215),
        _text(
            "rule-3",
            "3. ‘ㅎ’ 뒤에 ‘ㄴ’이 결합되는 경우에는, [ㄴ]으로 발음한다.",
            0.519,
            0.30,
        ),
        _text(
            "rule-4",
            "4. ‘ㅎ(ㄶ, ㅀ)’ 뒤에 모음으로 시작된 어미나 접미사가 결합",
            0.095,
            0.40,
        ),
        _text(
            "rule-4-wrap",
            "되는 경우에는, ‘ㅎ’을 발음하지 않는다.",
            0.12,
            0.415,
        ),
    )

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("rule-1", "rule-1-wrap"),
        ("rule-2", "rule-2-wrap"),
        ("rule-3",),
        ("rule-4", "rule-4-wrap"),
    )
    assert all(
        block.formation == ModelAClassificationBlockFormation.GENERIC_GEOMETRY
        and block.forced_role is None
        for block in plan.blocks
    )


@pytest.mark.parametrize("directive", ("고르시오", "쓰시오", "답하시오", "설명하시오"))
def test_wrapped_question_directive_forces_question(directive: str) -> None:
    evidence = _evidence(
        _text("question", "19. 다음 자료를 읽고", 0.10, 0.10),
        _text("question-wrap", f"적절한 답을 {directive}.", 0.12, 0.115),
    )

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        ("question", "question-wrap"),
    )
    assert plan.blocks[0].formation == ModelAClassificationBlockFormation.NUMBERED_QUESTION
    assert plan.blocks[0].forced_role == "question"


def test_answer_key_pattern_forms_one_forced_table_until_footer() -> None:
    evidence = _evidence(
        _text("answer-header", "[정답] ① ② ③ ④ ⑤", 0.10, 0.10),
        _text("answer-choice", "①", 0.12, 0.20),
        _text("answer-number", "1.", 0.12, 0.30),
        _text("answer-pair", "2. ③", 0.12, 0.40),
        _text("answer-pair-2", "11. ②", 0.12, 0.50),
        _text("footer", "- 8 -", 0.48, 0.94, x1=0.52),
    )

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert tuple(block.ordered_text_observation_ids for block in plan.blocks) == (
        (
            "answer-header",
            "answer-choice",
            "answer-number",
            "answer-pair",
            "answer-pair-2",
        ),
        ("footer",),
    )
    assert plan.blocks[0].formation == ModelAClassificationBlockFormation.ANSWER_KEY_TABLE
    assert plan.blocks[0].forced_role == "table"


def test_answer_key_pattern_falls_back_when_a_real_question_is_present() -> None:
    evidence = _evidence(
        _text("answer-header", "[정답]", 0.10, 0.10),
        _text("answer-choice", "①", 0.12, 0.20),
        _text("real-question", "1. 질문 문장입니까?", 0.12, 0.30),
        _text("answer-pair", "2. ③", 0.12, 0.40),
        _text("answer-pair-2", "11. ②", 0.12, 0.50),
        _text("footer", "- 8 -", 0.48, 0.94, x1=0.52),
    )

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")

    assert all(
        block.formation != ModelAClassificationBlockFormation.ANSWER_KEY_TABLE
        for block in plan.blocks
    )
    real_question_block = next(
        block for block in plan.blocks if "real-question" in block.ordered_text_observation_ids
    )
    assert real_question_block.formation == ModelAClassificationBlockFormation.NUMBERED_QUESTION
    assert real_question_block.forced_role == "question"


def test_forced_roles_override_model_roles_and_expand_to_each_line() -> None:
    plan = build_model_a_classification_plan(
        _mixed_document_evidence(),
        target_page_id="page-1",
    )

    role_plan = resolve_model_a_classification_roles(
        plan,
        ("passage",) * len(plan.blocks),
    )

    assert role_plan.classification_plan_sha256 == contract_sha256(plan)
    assert tuple(resolution.role_source for resolution in role_plan.block_resolutions) == (
        ModelAClassificationRoleSource.DETERMINISTIC,
        ModelAClassificationRoleSource.DETERMINISTIC,
        ModelAClassificationRoleSource.DETERMINISTIC,
        ModelAClassificationRoleSource.DETERMINISTIC,
        ModelAClassificationRoleSource.DETERMINISTIC,
        ModelAClassificationRoleSource.MODEL,
        ModelAClassificationRoleSource.MODEL,
        ModelAClassificationRoleSource.DETERMINISTIC,
    )
    assert tuple(resolution.conflict for resolution in role_plan.block_resolutions) == (
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
    )
    assert role_plan.expanded_line_roles == (
        "question",
        "question",
        "question",
        "choice",
        "choice",
        "choice",
        "choice",
        "choice",
        "caption",
        "instruction",
        "passage",
        "passage",
        "footer",
    )


def test_role_resolution_requires_exact_block_count() -> None:
    plan = build_model_a_classification_plan(
        _mixed_document_evidence(),
        target_page_id="page-1",
    )

    with pytest.raises(ValueError, match="must match classification block count"):
        resolve_model_a_classification_roles(plan, ("passage",))


def test_plan_contract_rejects_noncontiguous_forged_membership() -> None:
    plan = build_model_a_classification_plan(
        _mixed_document_evidence(),
        target_page_id="page-1",
    )
    payload = plan.model_dump(mode="json")
    payload["blocks"][1]["global_start_index"] += 1

    with pytest.raises(ValueError, match="index range must match|ranges must be contiguous"):
        ModelAClassificationPlan.model_validate_json(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            strict=True,
        )


def test_grouping_artifact_rejects_unknown_policy_digest() -> None:
    payload = ModelAClassificationGroupingArtifact().model_dump(mode="json")
    payload["grouper_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="not the deployed policy"):
        ModelAClassificationGroupingArtifact.model_validate_json(
            json.dumps(payload).encode("utf-8"),
            strict=True,
        )


def test_empty_text_page_has_an_empty_reproducible_partition() -> None:
    evidence = _evidence(_image("only-image", 0.2, 0.2))

    plan = build_model_a_classification_plan(evidence, target_page_id="page-1")
    role_plan = resolve_model_a_classification_roles(plan, ())

    assert plan.ordered_text_observation_ids == ()
    assert plan.blocks == ()
    assert role_plan.block_resolutions == ()
    assert role_plan.expanded_line_roles == ()
