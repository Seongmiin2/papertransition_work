from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from scan2hwpx.blueprint.model_a import match_choice, match_question_number, match_score
from scan2hwpx.contracts import (
    EvidenceIR,
    EvidenceObservation,
    ObservationKind,
    contract_sha256,
)
from scan2hwpx.contracts.models import StrictContractModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedText = Annotated[str, Field(min_length=1, max_length=1_024)]
ModelATextRole = Literal[
    "title",
    "instruction",
    "passage",
    "question",
    "choice",
    "table",
    "caption",
    "header",
    "footer",
    "other",
]

CLASSIFICATION_GROUPER_VERSION: Literal["model-a-classification-grouper/1.1"] = (
    "model-a-classification-grouper/1.1"
)
CLASSIFICATION_PLAN_VERSION: Literal["model-a-classification-plan/1.0"] = (
    "model-a-classification-plan/1.0"
)
CLASSIFICATION_ROLE_PLAN_VERSION: Literal["model-a-classification-role-plan/1.0"] = (
    "model-a-classification-role-plan/1.0"
)
_MAX_TEXT_OBSERVATIONS = 20_000
_DX_MIN = -0.005
_DX_MAX = 0.05
_DY_MIN = -0.005
_DY_MAX = 0.028
_FLOAT_COMPARISON_TOLERANCE = 1e-12
_RANGE_RE = re.compile(r"^\s*\[?\d{1,3}\s*[~∼～-]\s*\d{1,3}\]?\s*")
_QUESTION_PREFIX_RE = re.compile(r"^\s*\d{1,3}[.)]\s*")
_QUESTION_DIRECTIVE_ENDINGS = (
    "고르시오",
    "쓰시오",
    "답하시오",
    "설명하시오",
)
_ROLE_VALUES: tuple[ModelATextRole, ...] = (
    "title",
    "instruction",
    "passage",
    "question",
    "choice",
    "table",
    "caption",
    "header",
    "footer",
    "other",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ModelAClassificationLayoutZone(StrEnum):
    TOP_MARGIN = "top_margin"
    LEFT_BODY = "left_body"
    RIGHT_BODY = "right_body"
    SPANNING_BODY = "spanning_body"
    BOTTOM_MARGIN = "bottom_margin"


class ModelAClassificationBlockFormation(StrEnum):
    GENERIC_GEOMETRY = "generic_geometry"
    ANSWER_KEY_TABLE = "answer_key_table"
    NUMBERED_QUESTION = "numbered_question"
    STANDALONE_SCORE = "standalone_score"
    CIRCLED_CHOICE = "circled_choice"
    CIRCLED_CHOICE_SPATIAL_BAND = "circled_choice_spatial_band"
    BOGI_SEPARATOR = "bogi_separator"
    RANGE_INSTRUCTION = "range_instruction"
    BARE_PAGE_NUMBER_FOOTER = "bare_page_number_footer"


class ModelAClassificationRoleSource(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"


_FORCED_ROLE_BY_FORMATION: dict[
    ModelAClassificationBlockFormation,
    ModelATextRole | None,
] = {
    ModelAClassificationBlockFormation.GENERIC_GEOMETRY: None,
    ModelAClassificationBlockFormation.ANSWER_KEY_TABLE: "table",
    ModelAClassificationBlockFormation.NUMBERED_QUESTION: "question",
    ModelAClassificationBlockFormation.STANDALONE_SCORE: "question",
    ModelAClassificationBlockFormation.CIRCLED_CHOICE: "choice",
    ModelAClassificationBlockFormation.CIRCLED_CHOICE_SPATIAL_BAND: "choice",
    ModelAClassificationBlockFormation.BOGI_SEPARATOR: "caption",
    ModelAClassificationBlockFormation.RANGE_INSTRUCTION: "instruction",
    ModelAClassificationBlockFormation.BARE_PAGE_NUMBER_FOOTER: "footer",
}

_GROUPING_POLICY = {
    "anchor_detectors": {
        "bare_page_number_footer": "hyphen_bounded_digits_and_y0_gte_0.92",
        "bogi_separator": "hyphen_bounded_compact_text_containing_bogi",
        "circled_choice": "scan2hwpx.blueprint.model_a.match_choice",
        "numbered_question": {
            "base": "scan2hwpx.blueprint.model_a.match_question_number",
            "prefix_regex": _QUESTION_PREFIX_RE.pattern,
            "reject": ["empty_remainder", "circled_choice_only_remainder"],
            "required_block_evidence_any": {
                "ascii_question_mark": True,
                "directive_endings": _QUESTION_DIRECTIVE_ENDINGS,
                "score": "scan2hwpx.blueprint.model_a.match_score",
            },
        },
        "range_instruction_regex": _RANGE_RE.pattern,
        "score": "scan2hwpx.blueprint.model_a.match_score",
    },
    "anchor_priority": [
        "answer_key_table",
        "numbered_question",
        "circled_choice",
        "score",
        "bogi_separator",
        "range_instruction",
        "bare_page_number_footer",
    ],
    "canonical_order": (
        "all_content_top_margin_yx_then_evidence_tuple_body_then_"
        "all_content_bottom_margin_yx"
    ),
    "answer_key_table": {
        "header_compact_text": "[정답] optionally followed by circled choices only",
        "minimum_answer_lines": 4,
        "requires_bare_page_number_footer": True,
        "row_patterns": ["number_only", "number_plus_circled_choice", "circled_choice_only"],
    },
    "choice_continuation": "same_column_until_next_structural_anchor",
    "comparison_abs_tolerance": _FLOAT_COMPARISON_TOLERANCE,
    "forced_roles": {
        formation.value: role
        for formation, role in _FORCED_ROLE_BY_FORMATION.items()
        if role is not None
    },
    "generic_delta_x_inclusive": [_DX_MIN, _DX_MAX],
    "generic_delta_y_inclusive": [_DY_MIN, _DY_MAX],
    "generic_delta_y_definition": "next_y0_minus_current_y1",
    "grouper_id": "scan2hwpx.model_a.classification_blocks",
    "grouper_version": CLASSIFICATION_GROUPER_VERSION,
    "image_boundary": "never_merge_across_image_content_observation",
    "question_continuation": "same_flow_wrapping_lines_plus_terminal_score",
    "range_instruction": "singleton",
    "speaker_policy": "not_an_anchor",
    "standalone_choice_spatial_band": (
        "canonical_slice_between_adjacent_right_column_markers_with_midpoint_bands"
    ),
}
CLASSIFICATION_GROUPER_POLICY_BYTES = _canonical_json_bytes(_GROUPING_POLICY)
CLASSIFICATION_GROUPER_SHA256 = _sha256(CLASSIFICATION_GROUPER_POLICY_BYTES)


class _StrictModel(StrictContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ModelAClassificationGroupingArtifact(_StrictModel):
    grouper_id: Literal["scan2hwpx.model_a.classification_blocks"] = (
        "scan2hwpx.model_a.classification_blocks"
    )
    grouper_version: Literal["model-a-classification-grouper/1.1"] = CLASSIFICATION_GROUPER_VERSION
    grouper_sha256: Sha256 = CLASSIFICATION_GROUPER_SHA256

    @model_validator(mode="after")
    def validate_known_policy(self) -> Self:
        if self.grouper_sha256 != CLASSIFICATION_GROUPER_SHA256:
            raise ValueError("classification grouper digest is not the deployed policy")
        return self


class ModelAClassificationBlock(_StrictModel):
    block_index: int = Field(ge=0, lt=_MAX_TEXT_OBSERVATIONS)
    global_start_index: int = Field(ge=0, lt=_MAX_TEXT_OBSERVATIONS)
    global_end_index_exclusive: int = Field(gt=0, le=_MAX_TEXT_OBSERVATIONS)
    ordered_text_observation_ids: tuple[BoundedText, ...] = Field(
        min_length=1,
        max_length=_MAX_TEXT_OBSERVATIONS,
    )
    layout_zone: ModelAClassificationLayoutZone
    formation: ModelAClassificationBlockFormation
    forced_role: ModelATextRole | None

    @model_validator(mode="after")
    def validate_block(self) -> Self:
        if self.global_end_index_exclusive - self.global_start_index != len(
            self.ordered_text_observation_ids
        ):
            raise ValueError("classification block index range must match its line ids")
        if len(self.ordered_text_observation_ids) != len(set(self.ordered_text_observation_ids)):
            raise ValueError("classification block line ids must be unique")
        expected_role = _FORCED_ROLE_BY_FORMATION[self.formation]
        if self.forced_role != expected_role:
            raise ValueError("classification block forced role does not match its formation")
        return self


class ModelAClassificationPlan(_StrictModel):
    schema_version: Literal["model-a-classification-plan/1.0"] = CLASSIFICATION_PLAN_VERSION
    evidence_ir_id: BoundedText
    evidence_ir_sha256: Sha256
    page_id: BoundedText
    grouping: ModelAClassificationGroupingArtifact
    ordered_text_observation_ids: tuple[BoundedText, ...] = Field(max_length=_MAX_TEXT_OBSERVATIONS)
    blocks: tuple[ModelAClassificationBlock, ...] = Field(max_length=_MAX_TEXT_OBSERVATIONS)

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        expected_indexes = tuple(range(len(self.blocks)))
        if tuple(block.block_index for block in self.blocks) != expected_indexes:
            raise ValueError("classification block indexes must be canonical and contiguous")
        expected_start = 0
        flattened_ids: list[str] = []
        for block in self.blocks:
            if block.global_start_index != expected_start:
                raise ValueError("classification block ranges must be contiguous")
            expected_start = block.global_end_index_exclusive
            flattened_ids.extend(block.ordered_text_observation_ids)
        if tuple(flattened_ids) != self.ordered_text_observation_ids:
            raise ValueError("classification blocks must exactly partition canonical text ids")
        if len(flattened_ids) != len(set(flattened_ids)):
            raise ValueError("classification plan must not repeat text observations")
        return self


class ModelAClassificationBlockResolution(_StrictModel):
    block_index: int = Field(ge=0, lt=_MAX_TEXT_OBSERVATIONS)
    ordered_text_observation_ids: tuple[BoundedText, ...] = Field(
        min_length=1,
        max_length=_MAX_TEXT_OBSERVATIONS,
    )
    model_role: ModelATextRole
    resolved_role: ModelATextRole
    role_source: ModelAClassificationRoleSource
    conflict: bool

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        if self.role_source == ModelAClassificationRoleSource.MODEL:
            if self.resolved_role != self.model_role or self.conflict:
                raise ValueError("model-sourced block must preserve its model role")
        elif self.conflict != (self.model_role != self.resolved_role):
            raise ValueError("deterministic block conflict must match its role transition")
        return self


class ModelAClassificationRolePlan(_StrictModel):
    schema_version: Literal["model-a-classification-role-plan/1.0"] = (
        CLASSIFICATION_ROLE_PLAN_VERSION
    )
    classification_plan_sha256: Sha256
    block_resolutions: tuple[ModelAClassificationBlockResolution, ...] = Field(
        max_length=_MAX_TEXT_OBSERVATIONS
    )
    expanded_line_roles: tuple[ModelATextRole, ...] = Field(max_length=_MAX_TEXT_OBSERVATIONS)

    @model_validator(mode="after")
    def validate_expansion(self) -> Self:
        if tuple(resolution.block_index for resolution in self.block_resolutions) != tuple(
            range(len(self.block_resolutions))
        ):
            raise ValueError("block resolutions must be canonical and contiguous")
        expected_roles = tuple(
            resolution.resolved_role
            for resolution in self.block_resolutions
            for _ in resolution.ordered_text_observation_ids
        )
        if self.expanded_line_roles != expected_roles:
            raise ValueError("expanded line roles must repeat each resolved block role")
        return self


class _Anchor(StrEnum):
    NONE = "none"
    NUMBERED_QUESTION = "numbered_question"
    CIRCLED_CHOICE = "circled_choice"
    SCORE = "score"
    BOGI_SEPARATOR = "bogi_separator"
    RANGE_INSTRUCTION = "range_instruction"
    BARE_PAGE_NUMBER_FOOTER = "bare_page_number_footer"


@dataclass(frozen=True, slots=True)
class _CanonicalLine:
    observation: EvidenceObservation
    global_index: int
    zone: ModelAClassificationLayoutZone
    anchor: _Anchor
    standalone_choice_marker: bool
    image_boundary_before: bool


def model_a_classification_grouping_artifact() -> ModelAClassificationGroupingArtifact:
    return ModelAClassificationGroupingArtifact()


def build_model_a_classification_plan(
    evidence_ir: EvidenceIR,
    *,
    target_page_id: str,
) -> ModelAClassificationPlan:
    """Build a deterministic, lossless partition of one page's text observations."""

    if type(evidence_ir) is not EvidenceIR:
        raise TypeError("evidence_ir must be EvidenceIR")
    if not isinstance(target_page_id, str) or not target_page_id:
        raise TypeError("target_page_id must be a nonempty string")
    try:
        page = next(page for page in evidence_ir.pages if page.id == target_page_id)
    except StopIteration as exc:
        raise ValueError("target page is not present in EvidenceIR") from exc

    content = _ordered_content_observations(page.observations)
    lines = _canonical_lines(content)
    if len(lines) > _MAX_TEXT_OBSERVATIONS:
        raise ValueError("page text observation count exceeds classification limit")
    spatial_choice_ends = _standalone_choice_spatial_ends(lines)
    answer_key_table_ends = _answer_key_table_ends(lines)
    blocks = _partition_lines(
        lines,
        spatial_choice_ends=spatial_choice_ends,
        answer_key_table_ends=answer_key_table_ends,
    )
    return ModelAClassificationPlan(
        evidence_ir_id=evidence_ir.id,
        evidence_ir_sha256=contract_sha256(evidence_ir),
        page_id=page.id,
        grouping=model_a_classification_grouping_artifact(),
        ordered_text_observation_ids=tuple(line.observation.id for line in lines),
        blocks=blocks,
    )


def resolve_model_a_classification_roles(
    plan: ModelAClassificationPlan,
    model_roles: Sequence[ModelATextRole],
) -> ModelAClassificationRolePlan:
    """Resolve forced block roles and expand every block role back to its lines."""

    if type(plan) is not ModelAClassificationPlan:
        raise TypeError("plan must be ModelAClassificationPlan")
    roles = tuple(model_roles)
    if len(roles) != len(plan.blocks):
        raise ValueError("model role count must match classification block count")
    if any(role not in _ROLE_VALUES for role in roles):
        raise ValueError("model roles contain an unsupported role")

    resolutions: list[ModelAClassificationBlockResolution] = []
    for block, model_role in zip(plan.blocks, roles, strict=True):
        resolved_role = block.forced_role or model_role
        source = (
            ModelAClassificationRoleSource.DETERMINISTIC
            if block.forced_role is not None
            else ModelAClassificationRoleSource.MODEL
        )
        resolutions.append(
            ModelAClassificationBlockResolution(
                block_index=block.block_index,
                ordered_text_observation_ids=block.ordered_text_observation_ids,
                model_role=model_role,
                resolved_role=resolved_role,
                role_source=source,
                conflict=model_role != resolved_role,
            )
        )
    return ModelAClassificationRolePlan(
        classification_plan_sha256=contract_sha256(plan),
        block_resolutions=tuple(resolutions),
        expanded_line_roles=tuple(
            resolution.resolved_role
            for resolution in resolutions
            for _ in resolution.ordered_text_observation_ids
        ),
    )


def _partition_lines(
    lines: tuple[_CanonicalLine, ...],
    *,
    spatial_choice_ends: dict[int, int],
    answer_key_table_ends: dict[int, int],
) -> tuple[ModelAClassificationBlock, ...]:
    blocks: list[ModelAClassificationBlock] = []
    start = 0
    while start < len(lines):
        line = lines[start]
        anchor = line.anchor
        answer_key_end = answer_key_table_ends.get(start)
        if answer_key_end is not None:
            end = answer_key_end
            formation = ModelAClassificationBlockFormation.ANSWER_KEY_TABLE
        elif anchor == _Anchor.NUMBERED_QUESTION:
            end = _question_block_end(lines, start)
            formation = (
                ModelAClassificationBlockFormation.NUMBERED_QUESTION
                if _question_block_has_forcing_evidence(lines, start, end)
                else ModelAClassificationBlockFormation.GENERIC_GEOMETRY
            )
        elif anchor == _Anchor.SCORE:
            end = start + 1
            formation = ModelAClassificationBlockFormation.STANDALONE_SCORE
        elif anchor == _Anchor.CIRCLED_CHOICE:
            spatial_end = spatial_choice_ends.get(start)
            if spatial_end is not None:
                end = spatial_end
                formation = ModelAClassificationBlockFormation.CIRCLED_CHOICE_SPATIAL_BAND
            elif line.standalone_choice_marker:
                end = start + 1
                formation = ModelAClassificationBlockFormation.CIRCLED_CHOICE
            else:
                end = _choice_block_end(lines, start)
                formation = ModelAClassificationBlockFormation.CIRCLED_CHOICE
        elif anchor == _Anchor.BOGI_SEPARATOR:
            end = start + 1
            formation = ModelAClassificationBlockFormation.BOGI_SEPARATOR
        elif anchor == _Anchor.RANGE_INSTRUCTION:
            end = start + 1
            formation = ModelAClassificationBlockFormation.RANGE_INSTRUCTION
        elif anchor == _Anchor.BARE_PAGE_NUMBER_FOOTER:
            end = start + 1
            formation = ModelAClassificationBlockFormation.BARE_PAGE_NUMBER_FOOTER
        else:
            end = _same_flow_block_end(lines, start)
            formation = ModelAClassificationBlockFormation.GENERIC_GEOMETRY
        blocks.append(_make_block(lines, start, end, len(blocks), formation))
        start = end
    return tuple(blocks)


def _make_block(
    lines: tuple[_CanonicalLine, ...],
    start: int,
    end: int,
    block_index: int,
    formation: ModelAClassificationBlockFormation,
) -> ModelAClassificationBlock:
    return ModelAClassificationBlock(
        block_index=block_index,
        global_start_index=start,
        global_end_index_exclusive=end,
        ordered_text_observation_ids=tuple(line.observation.id for line in lines[start:end]),
        layout_zone=lines[start].zone,
        formation=formation,
        forced_role=_FORCED_ROLE_BY_FORMATION[formation],
    )


def _question_block_end(lines: tuple[_CanonicalLine, ...], start: int) -> int:
    end = start + 1
    current = lines[start]
    while end < len(lines):
        candidate = lines[end]
        if candidate.image_boundary_before or candidate.zone != current.zone:
            break
        if candidate.anchor == _Anchor.SCORE:
            if _same_flow(current, candidate):
                end += 1
            break
        if candidate.anchor != _Anchor.NONE or not _same_flow(current, candidate):
            break
        current = candidate
        end += 1
    return end


def _question_block_has_forcing_evidence(
    lines: tuple[_CanonicalLine, ...],
    start: int,
    end: int,
) -> bool:
    for line in lines[start:end]:
        text = _preferred_ocr_text(line.observation)
        if "?" in text or match_score(text) is not None:
            return True
        compact = _compact_text(text).rstrip(".!")
        if compact.endswith(_QUESTION_DIRECTIVE_ENDINGS):
            return True
    return False


def _choice_block_end(lines: tuple[_CanonicalLine, ...], start: int) -> int:
    end = start + 1
    zone = lines[start].zone
    while end < len(lines):
        candidate = lines[end]
        if candidate.image_boundary_before or candidate.zone != zone:
            break
        if candidate.anchor != _Anchor.NONE:
            break
        end += 1
    return end


def _same_flow_block_end(lines: tuple[_CanonicalLine, ...], start: int) -> int:
    end = start + 1
    current = lines[start]
    while end < len(lines):
        candidate = lines[end]
        if candidate.image_boundary_before or candidate.anchor != _Anchor.NONE:
            break
        if not _same_flow(current, candidate):
            break
        current = candidate
        end += 1
    return end


def _same_flow(current: _CanonicalLine, candidate: _CanonicalLine) -> bool:
    if current.zone != candidate.zone:
        return False
    current_bbox = current.observation.bbox.normalized
    candidate_bbox = candidate.observation.bbox.normalized
    delta_x = candidate_bbox[0] - current_bbox[0]
    delta_y = candidate_bbox[1] - current_bbox[3]
    return _within(delta_x, _DX_MIN, _DX_MAX) and _within(delta_y, _DY_MIN, _DY_MAX)


def _within(value: float, minimum: float, maximum: float) -> bool:
    above_minimum = value > minimum or math.isclose(
        value,
        minimum,
        rel_tol=0.0,
        abs_tol=_FLOAT_COMPARISON_TOLERANCE,
    )
    below_maximum = value < maximum or math.isclose(
        value,
        maximum,
        rel_tol=0.0,
        abs_tol=_FLOAT_COMPARISON_TOLERANCE,
    )
    return above_minimum and below_maximum


def _standalone_choice_spatial_ends(
    lines: tuple[_CanonicalLine, ...],
) -> dict[int, int]:
    spatial_ends: dict[int, int] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.zone != ModelAClassificationLayoutZone.RIGHT_BODY or _spatial_section_breaker(line):
            index += 1
            continue
        section_start = index
        index += 1
        while index < len(lines):
            candidate = lines[index]
            if (
                candidate.image_boundary_before
                or candidate.zone != ModelAClassificationLayoutZone.RIGHT_BODY
                or _spatial_section_breaker(candidate)
            ):
                break
            index += 1
        section_end = index
        markers = tuple(
            marker_index
            for marker_index in range(section_start, section_end)
            if lines[marker_index].anchor == _Anchor.CIRCLED_CHOICE
            and lines[marker_index].standalone_choice_marker
        )
        if len(markers) < 2:
            continue
        centers = tuple(_vertical_center(lines[marker].observation) for marker in markers)
        if any(right <= left for left, right in pairwise(centers)):
            continue
        for marker_position, marker_index in enumerate(markers):
            lower, upper = _choice_band(centers, marker_position)
            slice_end = (
                markers[marker_position + 1] if marker_position + 1 < len(markers) else section_end
            )
            candidates = lines[marker_index + 1 : slice_end]
            if candidates and all(
                _vertical_span_intersects(candidate.observation, lower, upper)
                for candidate in candidates
            ):
                spatial_ends[marker_index] = slice_end
    return spatial_ends


def _answer_key_table_ends(lines: tuple[_CanonicalLine, ...]) -> dict[int, int]:
    table_ends: dict[int, int] = {}
    for start, line in enumerate(lines):
        if not _is_answer_key_header(_preferred_ocr_text(line.observation)):
            continue
        end = start + 1
        while end < len(lines) and lines[end].anchor != _Anchor.BARE_PAGE_NUMBER_FOOTER:
            if lines[end].image_boundary_before or not _is_answer_key_row(
                _preferred_ocr_text(lines[end].observation)
            ):
                break
            end += 1
        if (
            end < len(lines)
            and lines[end].anchor == _Anchor.BARE_PAGE_NUMBER_FOOTER
            and end - start - 1 >= 4
        ):
            table_ends[start] = end
    return table_ends


def _is_answer_key_header(text: str) -> bool:
    compact = _compact_text(text)
    prefix = "[정답]"
    if not compact.startswith(prefix):
        return False
    remainder = compact[len(prefix) :]
    if not remainder:
        return True
    choice_count = 0
    while remainder:
        choice = match_choice(remainder)
        if choice is None or not remainder.startswith(choice):
            return False
        choice_count += 1
        remainder = remainder[len(choice) :]
    return choice_count > 0


def _is_answer_key_row(text: str) -> bool:
    compact = _compact_text(text)
    if _choice_has_no_remainder(compact):
        return True
    prefix = _QUESTION_PREFIX_RE.match(compact)
    if prefix is None:
        return False
    remainder = compact[prefix.end() :]
    return not remainder or _choice_has_no_remainder(remainder)


def _spatial_section_breaker(line: _CanonicalLine) -> bool:
    if line.anchor in {
        _Anchor.NUMBERED_QUESTION,
        _Anchor.SCORE,
        _Anchor.BOGI_SEPARATOR,
        _Anchor.RANGE_INSTRUCTION,
        _Anchor.BARE_PAGE_NUMBER_FOOTER,
    }:
        return True
    return line.anchor == _Anchor.CIRCLED_CHOICE and not line.standalone_choice_marker


def _choice_band(centers: tuple[float, ...], position: int) -> tuple[float, float]:
    center = centers[position]
    if position == 0:
        lower = center - (centers[1] - center) / 2
    else:
        lower = (centers[position - 1] + center) / 2
    if position == len(centers) - 1:
        upper = center + (center - centers[position - 1]) / 2
    else:
        upper = (center + centers[position + 1]) / 2
    return max(0.0, lower), min(1.0, upper)


def _vertical_center(observation: EvidenceObservation) -> float:
    _, y0, _, y1 = observation.bbox.normalized
    return (y0 + y1) / 2


def _vertical_span_intersects(
    observation: EvidenceObservation,
    lower: float,
    upper: float,
) -> bool:
    _, y0, _, y1 = observation.bbox.normalized
    return _within(y1, lower, 1.0) and _within(y0, 0.0, upper)


def _canonical_lines(
    content: tuple[EvidenceObservation, ...],
) -> tuple[_CanonicalLine, ...]:
    lines: list[_CanonicalLine] = []
    image_since_text = False
    saw_text = False
    for observation in content:
        if observation.kind == ObservationKind.IMAGE:
            if saw_text:
                image_since_text = True
            continue
        text = _preferred_ocr_text(observation)
        lines.append(
            _CanonicalLine(
                observation=observation,
                global_index=len(lines),
                zone=_layout_zone(observation),
                anchor=_anchor(observation, text),
                standalone_choice_marker=_choice_has_no_remainder(text),
                image_boundary_before=image_since_text,
            )
        )
        saw_text = True
        image_since_text = False
    return tuple(lines)


def _anchor(observation: EvidenceObservation, text: str) -> _Anchor:
    compact = _compact_text(text)
    if _is_semantic_question_start(text):
        return _Anchor.NUMBERED_QUESTION
    if match_choice(text) is not None:
        return _Anchor.CIRCLED_CHOICE
    if match_score(text) is not None:
        return _Anchor.SCORE
    if compact.startswith("-") and "보기" in compact and compact.endswith("-"):
        return _Anchor.BOGI_SEPARATOR
    if _RANGE_RE.match(text):
        return _Anchor.RANGE_INSTRUCTION
    if (
        observation.bbox.normalized[1] >= 0.92
        and compact.startswith("-")
        and compact.endswith("-")
        and compact.replace("-", "").isdigit()
    ):
        return _Anchor.BARE_PAGE_NUMBER_FOOTER
    return _Anchor.NONE


def _is_semantic_question_start(text: str) -> bool:
    if match_question_number(text) is None:
        return False
    prefix = _QUESTION_PREFIX_RE.match(text)
    if prefix is None:
        return False
    remainder = text[prefix.end() :].strip()
    return bool(remainder) and not _choice_has_no_remainder(remainder)


def _choice_has_no_remainder(text: str) -> bool:
    choice = match_choice(text)
    if choice is None:
        return False
    position = text.find(choice)
    return position >= 0 and not text[position + len(choice) :].strip()


def _compact_text(text: str) -> str:
    return "".join(text.strip().split())


def _layout_zone(observation: EvidenceObservation) -> ModelAClassificationLayoutZone:
    x0, y0, x1, _ = observation.bbox.normalized
    if y0 < 0.08:
        return ModelAClassificationLayoutZone.TOP_MARGIN
    if y0 >= 0.92:
        return ModelAClassificationLayoutZone.BOTTOM_MARGIN
    if x0 < 0.2 and x1 > 0.8:
        return ModelAClassificationLayoutZone.SPANNING_BODY
    if (x0 + x1) / 2 < 0.5:
        return ModelAClassificationLayoutZone.LEFT_BODY
    return ModelAClassificationLayoutZone.RIGHT_BODY


def _ordered_content_observations(
    observations: Sequence[EvidenceObservation],
) -> tuple[EvidenceObservation, ...]:
    content = tuple(
        observation
        for observation in observations
        if observation.kind in {ObservationKind.TEXT_LINE, ObservationKind.IMAGE}
    )
    top_margin: list[tuple[float, float, int, EvidenceObservation]] = []
    body: list[EvidenceObservation] = []
    bottom_margin: list[tuple[float, float, int, EvidenceObservation]] = []
    for position, observation in enumerate(content):
        x0, y0, _, _ = observation.bbox.normalized
        if y0 < 0.08:
            top_margin.append((y0, x0, position, observation))
        elif y0 >= 0.92:
            bottom_margin.append((y0, x0, position, observation))
        else:
            body.append(observation)
    return (
        tuple(item[3] for item in sorted(top_margin))
        + tuple(body)
        + tuple(item[3] for item in sorted(bottom_margin))
    )


def _preferred_ocr_text(observation: EvidenceObservation) -> str:
    selected = next(
        (candidate for candidate in observation.ocr_candidates if candidate.selected),
        None,
    )
    if selected is not None:
        return selected.text
    return max(
        observation.ocr_candidates,
        key=lambda candidate: candidate.confidence,
    ).text
