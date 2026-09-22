from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

import scan2hwpx.evaluation as evaluation_api
import scan2hwpx.evaluation.model_b_plan_review as plan_review_module
from scan2hwpx.contracts import ContentIR, ContentPlanItem, HwpDocumentPlan, contract_sha256
from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    CandidateToModelBHandoff,
    CompletedCandidateReviewBinding,
    GroundingSidecarHandoffBinding,
    VerifiedCandidateToModelBHandoff,
    model_b_plan_review_draft_from_verified_handoff,
)
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBOfficialSpecRefEvidence,
    ModelBPlanGroundingEvidence,
    ModelBPlanReviewCompletionError,
    ModelBPlanReviewCompletionRequest,
    ModelBPlanReviewDraft,
    ModelBPlanReviewDraftView,
    ModelBPlanReviewPatchError,
    ModelBPlanReviewPatchRequest,
    ModelBPlanReviewVerificationError,
    apply_model_b_plan_review_patch,
    apply_persisted_model_b_plan_review_patch,
    build_model_b_plan_review_draft_view,
    complete_model_b_plan_review_draft,
    complete_persisted_model_b_plan_review_draft,
    load_model_b_plan_review_completion_request,
    load_model_b_plan_review_draft,
    load_model_b_plan_review_patch_request,
    model_b_plan_review_candidate_contract_sha256,
    reopen_model_b_plan_review_draft,
    reopen_model_b_plan_review_draft_view,
    start_model_b_plan_review_draft,
    start_persisted_model_b_plan_review_draft,
    verify_model_b_plan_review_draft,
)


def _grounding_evidence(
    refs: tuple[str, ...] = ("official-spec-a", "official-spec-b"),
) -> ModelBPlanGroundingEvidence:
    return ModelBPlanGroundingEvidence(
        knowledge_corpus_sha256="2" * 64,
        retrieval_artifact_sha256="3" * 64,
        capability_profile_id="exam2hwpx-authoring-v1",
        capability_profile_sha256="4" * 64,
        official_spec_refs=tuple(
            ModelBOfficialSpecRefEvidence(
                official_spec_ref=ref,
                source_document_sha256=f"{index + 5:x}" * 64,
                retrieved_chunk_sha256=f"{index + 7:x}" * 64,
            )
            for index, ref in enumerate(refs)
        ),
    )


def _content() -> ContentIR:
    return ContentIR.model_validate(
        {
            "id": "content-1",
            "evidence_ir_id": "evidence-1",
            "evidence_ir_sha256": "1" * 64,
            "revision": 3,
            "nodes": [
                {
                    "id": "text-1",
                    "kind": "text",
                    "role": "question",
                    "text": "본문은 Model B가 바꾸면 안 된다.",
                    "evidence_refs": ["obs-text"],
                    "confidence": 1.0,
                },
                {
                    "id": "table-1",
                    "kind": "table",
                    "rows": 1,
                    "columns": 1,
                    "cells": [
                        {
                            "row": 0,
                            "column": 0,
                            "text": "고정 셀",
                            "evidence_refs": ["obs-table"],
                        }
                    ],
                    "evidence_refs": ["obs-table"],
                    "confidence": 1.0,
                },
            ],
            "reading_order": ["text-1", "table-1"],
        }
    )


def _plan(content: ContentIR) -> HwpDocumentPlan:
    return HwpDocumentPlan.model_validate(
        {
            "id": "plan-1",
            "content_ir_id": content.id,
            "content_ir_revision": content.revision,
            "content_ir_sha256": contract_sha256(content),
            "capability_profile_id": "exam2hwpx-authoring-v1",
            "design_profile_id": "review-candidate-v1",
            "official_spec_refs": ["official-spec-a"],
            "page_layout": {
                "width_mm": 210.0,
                "height_mm": 297.0,
                "margin_top_mm": 15.0,
                "margin_right_mm": 15.0,
                "margin_bottom_mm": 15.0,
                "margin_left_mm": 15.0,
            },
            "styles": [
                {
                    "id": "style-body",
                    "semantic_role": "question",
                    "font_family": "함초롬바탕",
                    "font_size_pt": 10.0,
                }
            ],
            "flow": [
                {
                    "id": "flow-text",
                    "kind": "content",
                    "render_as": "paragraph",
                    "content_ref": "text-1",
                    "style_ref": "style-body",
                },
                {"id": "break-1", "kind": "page_break"},
                {
                    "id": "flow-table",
                    "kind": "content",
                    "render_as": "table",
                    "content_ref": "table-1",
                },
            ],
        }
    )


def _start() -> ModelBPlanReviewDraft:
    content = _content()
    return start_model_b_plan_review_draft(
        content,
        _plan(content),
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        grounding_evidence=_grounding_evidence(),
        updated_at=datetime(2026, 9, 10, 16, 0, tzinfo=UTC),
    )


def _request(payload: dict[str, object]) -> ModelBPlanReviewPatchRequest:
    value = dict(payload)
    content = _content()
    value["expected_source_candidate_contract_sha256"] = (
        model_b_plan_review_candidate_contract_sha256(
            content,
            _plan(content),
            document_id="document-1",
        )
    )
    value["expected_grounding_evidence_contract_sha256"] = contract_sha256(
        _grounding_evidence()
    )
    value["expected_reviewed_hwp_document_plan_contract_sha256"] = contract_sha256(
        _plan(content)
    )
    return ModelBPlanReviewPatchRequest.model_validate_json(
        json.dumps(value, ensure_ascii=False),
        strict=True,
    )


def _completion_request(
    draft: ModelBPlanReviewDraft,
    **overrides: object,
) -> ModelBPlanReviewCompletionRequest:
    value: dict[str, object] = {
        "expected_draft_revision": draft.draft_revision,
        "expected_source_candidate_contract_sha256": (
            draft.source_candidate_contract_sha256
        ),
        "expected_grounding_evidence_contract_sha256": (
            draft.grounding_evidence_contract_sha256
        ),
        "expected_reviewed_hwp_document_plan_contract_sha256": (
            draft.reviewed_hwp_document_plan_contract_sha256
        ),
    }
    value.update(overrides)
    return ModelBPlanReviewCompletionRequest.model_validate(value, strict=True)


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    content = _content()
    content_path = tmp_path / "content-ir.json"
    plan_path = tmp_path / "base-plan.json"
    grounding_path = tmp_path / "grounding-evidence.json"
    paths_and_contracts = (
        (content_path, content),
        (plan_path, _plan(content)),
        (grounding_path, _grounding_evidence()),
    )
    for path, contract in paths_and_contracts:
        path.write_text(
            json.dumps(
                contract.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return content_path, plan_path, grounding_path


def test_starts_noneligible_unverified_plan_review() -> None:
    draft = _start()

    assert draft.draft_revision == 1
    assert draft.status == "in_progress"
    assert draft.content_ir_contract_sha256 == contract_sha256(draft.content_ir)
    assert draft.reviewed_hwp_document_plan == draft.base_hwp_document_plan
    assert draft.grounding_evidence_contract_sha256 == contract_sha256(
        draft.grounding_evidence
    )
    assert draft.grounding_evidence.knowledge_corpus_sha256 == "2" * 64
    assert draft.grounding_evidence.retrieval_artifact_sha256 == "3" * 64
    assert draft.grounding_evidence.capability_profile_sha256 == "4" * 64
    assert draft.golden_eligible is False
    assert draft.training_eligible is False
    assert draft.release_eligible is False
    assert draft.rights_status == "unverified"
    assert draft.identity_assurance == "self_asserted_untrusted"


def test_applies_all_design_patch_surfaces_as_new_immutable_revision() -> None:
    current = _start()
    before = current.model_dump(mode="json")
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_page_layout",
                    "page_layout": {
                        "width_mm": 210.0,
                        "height_mm": 297.0,
                        "margin_top_mm": 20.0,
                        "margin_right_mm": 18.0,
                        "margin_bottom_mm": 20.0,
                        "margin_left_mm": 18.0,
                        "columns": 2,
                        "column_gap_mm": 8.0,
                    },
                },
                {
                    "op": "replace_styles",
                    "styles": [
                        {
                            "id": "style-reviewed",
                            "semantic_role": "question",
                            "font_family": "함초롬돋움",
                            "font_size_pt": 11.0,
                            "bold": True,
                            "italic": False,
                            "alignment": "justify",
                            "line_spacing": 1.6,
                        }
                    ],
                },
                {
                    "op": "replace_flow",
                    "flow": [
                        {
                            "id": "flow-table",
                            "kind": "content",
                            "render_as": "table",
                            "style_ref": None,
                            "layout": {
                                "width_fraction": 0.8,
                                "column_span": 2,
                                "keep_with_next": False,
                            },
                        },
                        {"id": "break-reviewed", "kind": "column_break"},
                        {
                            "id": "flow-text",
                            "kind": "content",
                            "render_as": "paragraph",
                            "style_ref": "style-reviewed",
                            "layout": {
                                "width_fraction": 1.0,
                                "column_span": 1,
                                "keep_with_next": True,
                            },
                        },
                    ],
                },
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-b"],
                },
            ],
        }
    )

    patched = apply_model_b_plan_review_patch(
        current,
        request,
        updated_at=datetime(2026, 9, 10, 17, 0, tzinfo=UTC),
    )

    assert patched.draft_revision == 2
    assert patched.updated_at == "2026-09-10T17:00:00+00:00"
    assert patched.content_ir.model_dump(mode="json") == current.content_ir.model_dump(mode="json")
    assert patched.content_ir_contract_sha256 == current.content_ir_contract_sha256
    assert patched.base_hwp_document_plan == current.base_hwp_document_plan
    assert patched.grounding_evidence == current.grounding_evidence
    assert (
        patched.grounding_evidence_contract_sha256
        == current.grounding_evidence_contract_sha256
    )
    assert current.model_dump(mode="json") == before
    plan = patched.reviewed_hwp_document_plan
    assert plan.page_layout.columns == 2
    assert [style.id for style in plan.styles] == ["style-reviewed"]
    assert [item.kind for item in plan.flow] == ["content", "column_break", "content"]
    refs = {
        item.id: item.content_ref for item in plan.flow if isinstance(item, ContentPlanItem)
    }
    assert refs == {"flow-text": "text-1", "flow-table": "table-1"}
    assert plan.official_spec_refs == ("official-spec-b",)
    assert patched.training_eligible is False
    assert patched.rights_status == "unverified"


def test_rejects_stale_revision_without_mutating_current() -> None:
    current = _start()
    before = current.model_dump(mode="json")
    request = _request(
        {
            "expected_draft_revision": 2,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="stale Model B plan review"):
        apply_model_b_plan_review_patch(current, request)

    assert current.model_dump(mode="json") == before


def test_patch_rejects_same_revision_from_a_different_reviewed_plan_branch() -> None:
    current = _start()
    branch_a = apply_model_b_plan_review_patch(
        current,
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "set_page_layout",
                        "page_layout": {
                            "width_mm": 210.0,
                            "height_mm": 297.0,
                            "margin_top_mm": 20.0,
                            "margin_right_mm": 15.0,
                            "margin_bottom_mm": 15.0,
                            "margin_left_mm": 15.0,
                        },
                    }
                ],
            }
        ),
    )
    branch_b = apply_model_b_plan_review_patch(
        current,
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "set_official_spec_refs",
                        "official_spec_refs": ["official-spec-b"],
                    }
                ],
            }
        ),
    )
    request_payload = _request(
        {
            "expected_draft_revision": 2,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    ).model_dump(mode="json")
    request_payload["expected_reviewed_hwp_document_plan_contract_sha256"] = (
        branch_a.reviewed_hwp_document_plan_contract_sha256
    )
    request = ModelBPlanReviewPatchRequest.model_validate_json(
        json.dumps(request_payload),
        strict=True,
    )

    assert branch_a.draft_revision == branch_b.draft_revision == 2
    assert (
        branch_a.reviewed_hwp_document_plan_contract_sha256
        != branch_b.reviewed_hwp_document_plan_contract_sha256
    )
    with pytest.raises(ModelBPlanReviewPatchError, match="current reviewed plan binding changed"):
        apply_model_b_plan_review_patch(branch_b, request)


def test_patch_v13_requires_current_reviewed_plan_digest() -> None:
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    )
    assert request.schema_version == "model-b-plan-review-patch/1.3"
    payload = request.model_dump(mode="json")
    payload.pop("expected_reviewed_hwp_document_plan_contract_sha256")

    with pytest.raises(
        ValidationError,
        match="expected_reviewed_hwp_document_plan_contract_sha256",
    ):
        ModelBPlanReviewPatchRequest.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "set_json_pointer", "pointer": "/content_ir/nodes/0/text", "value": "x"},
        {"op": "set_eligibility", "training_eligible": True},
        {"op": "set_rights_status", "rights_status": "verified"},
        {"op": "write_file", "path": "draft.json"},
    ],
)
def test_patch_contract_exposes_no_arbitrary_or_promotion_operations(
    operation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _request({"expected_draft_revision": 1, "operations": [operation]})


def test_replace_flow_never_accepts_content_ref() -> None:
    with pytest.raises(ValidationError, match="content_ref"):
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "replace_flow",
                        "flow": [
                            {
                                "id": "flow-text",
                                "kind": "content",
                                "render_as": "paragraph",
                                "content_ref": "table-1",
                            }
                        ],
                    }
                ],
            }
        )


def test_replace_flow_requires_exact_content_item_inventory() -> None:
    current = _start()
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "replace_flow",
                    "flow": [
                        {
                            "id": "flow-text",
                            "kind": "content",
                            "render_as": "paragraph",
                            "style_ref": "style-body",
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="preserve every content flow item"):
        apply_model_b_plan_review_patch(current, request)


def test_render_as_must_still_match_immutable_content_kind() -> None:
    current = _start()
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "replace_flow",
                    "flow": [
                        {
                            "id": "flow-text",
                            "kind": "content",
                            "render_as": "table",
                            "style_ref": "style-body",
                        },
                        {
                            "id": "flow-table",
                            "kind": "content",
                            "render_as": "table",
                        },
                    ],
                }
            ],
        }
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="kind/render_as mismatch"):
        apply_model_b_plan_review_patch(current, request)


def test_official_spec_refs_must_be_grounded() -> None:
    current = _start()
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["not-retrieved"],
                }
            ],
        }
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="ungrounded official spec refs"):
        apply_model_b_plan_review_patch(current, request)


def test_patch_request_requires_exact_grounding_evidence_binding() -> None:
    current = _start()
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    )
    payload = current.model_dump(mode="json")
    changed_evidence_payload = dict(payload["grounding_evidence"])
    changed_evidence_payload["retrieval_artifact_sha256"] = "9" * 64
    changed_evidence = ModelBPlanGroundingEvidence.model_validate_json(
        json.dumps(changed_evidence_payload),
        strict=True,
    )
    payload["grounding_evidence"] = changed_evidence.model_dump(mode="json")
    payload["grounding_evidence_contract_sha256"] = contract_sha256(changed_evidence)
    changed_current = ModelBPlanReviewDraft.model_validate_json(
        json.dumps(payload),
        strict=True,
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="grounding evidence binding changed"):
        apply_model_b_plan_review_patch(changed_current, request)


def test_patch_request_requires_exact_source_candidate_binding() -> None:
    current = _start()
    payload = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    ).model_dump(mode="json")
    payload["expected_source_candidate_contract_sha256"] = "f" * 64
    request = ModelBPlanReviewPatchRequest.model_validate_json(
        json.dumps(payload),
        strict=True,
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="source candidate binding changed"):
        apply_model_b_plan_review_patch(current, request)


def test_missing_or_tampered_grounding_binding_is_rejected() -> None:
    current = _start()
    request_payload = {
        "expected_draft_revision": 1,
        "operations": [
            {
                "op": "set_official_spec_refs",
                "official_spec_refs": ["official-spec-a"],
            }
        ],
    }
    with pytest.raises(ValidationError, match="expected_grounding"):
        ModelBPlanReviewPatchRequest.model_validate(request_payload, strict=True)

    draft_payload = current.model_dump(mode="json")
    draft_payload.pop("grounding_evidence")
    with pytest.raises(ValidationError, match="grounding_evidence"):
        ModelBPlanReviewDraft.model_validate_json(json.dumps(draft_payload), strict=True)

    draft_payload = current.model_dump(mode="json")
    draft_payload["grounding_evidence_contract_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="grounding evidence contract digest mismatch"):
        ModelBPlanReviewDraft.model_validate_json(json.dumps(draft_payload), strict=True)


def test_grounding_evidence_requires_typed_ref_digests_and_matching_capability() -> None:
    payload = _grounding_evidence().model_dump(mode="json")
    del payload["official_spec_refs"][0]["retrieved_chunk_sha256"]
    with pytest.raises(ValidationError, match="retrieved_chunk_sha256"):
        ModelBPlanGroundingEvidence.model_validate_json(json.dumps(payload), strict=True)

    draft_payload = _start().model_dump(mode="json")
    evidence_payload = dict(draft_payload["grounding_evidence"])
    evidence_payload["capability_profile_id"] = "different-profile"
    evidence = ModelBPlanGroundingEvidence.model_validate_json(
        json.dumps(evidence_payload),
        strict=True,
    )
    draft_payload["grounding_evidence"] = evidence.model_dump(mode="json")
    draft_payload["grounding_evidence_contract_sha256"] = contract_sha256(evidence)
    with pytest.raises(ValidationError, match="capability profile"):
        ModelBPlanReviewDraft.model_validate_json(json.dumps(draft_payload), strict=True)


def test_patch_request_bounds_operations_ids_and_duplicate_surfaces() -> None:
    page_operation = {
        "op": "set_page_layout",
        "page_layout": {
            "width_mm": 210.0,
            "height_mm": 297.0,
            "margin_top_mm": 15.0,
            "margin_right_mm": 15.0,
            "margin_bottom_mm": 15.0,
            "margin_left_mm": 15.0,
        },
    }
    with pytest.raises(ValidationError):
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [page_operation] * 5,
            }
        )
    with pytest.raises(ValidationError, match="duplicate Model B plan patch operation"):
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [page_operation, page_operation],
            }
        )
    with pytest.raises(ValidationError):
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "replace_flow",
                        "flow": [{"id": "x" * 1_025, "kind": "page_break"}],
                    }
                ],
            }
        )


def test_patch_request_bounds_total_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_review_module, "_MAX_PATCH_TEXT_CHARS", 1)

    with pytest.raises(ValidationError, match="total size limit"):
        _request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "set_official_spec_refs",
                        "official_spec_refs": ["official-spec-a"],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("golden_eligible", True),
        ("training_eligible", True),
        ("release_eligible", True),
        ("rights_status", "verified"),
        ("identity_assurance", "trusted"),
    ],
)
def test_draft_cannot_claim_eligibility_rights_or_identity(
    field: str,
    value: object,
) -> None:
    payload = _start().model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError):
        ModelBPlanReviewDraft.model_validate_json(json.dumps(payload), strict=True)


def test_start_rejects_ungrounded_base_plan_and_naive_timestamp() -> None:
    content = _content()
    plan = _plan(content)
    with pytest.raises(ModelBPlanReviewPatchError, match="not valid"):
        start_model_b_plan_review_draft(
            content,
            plan,
            document_id="document-1",
            reviewer_label="reviewer-local-1",
            grounding_evidence=_grounding_evidence(("official-spec-b",)),
        )

    with pytest.raises(ValueError, match="timezone"):
        start_model_b_plan_review_draft(
            content,
            plan,
            document_id="document-1",
            reviewer_label="reviewer-local-1",
            grounding_evidence=_grounding_evidence(("official-spec-a",)),
            updated_at=datetime(2026, 9, 10, 16, 0, tzinfo=UTC).replace(tzinfo=None),
        )


def test_persisted_workflow_starts_reopens_views_and_patches_immutable_revisions(
    tmp_path: Path,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_1_path = tmp_path / "review-r1.json"
    revision_2_path = tmp_path / "review-r2.json"
    source_bytes = tuple(
        path.read_bytes() for path in (content_path, plan_path, grounding_path)
    )

    revision_1 = start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=revision_1_path,
        updated_at=datetime(2026, 9, 10, 16, 0, tzinfo=UTC),
    )
    reopened = reopen_model_b_plan_review_draft(
        revision_1_path,
        content_path,
        plan_path,
        grounding_path,
    )
    view = build_model_b_plan_review_draft_view(reopened)
    safely_reopened_view = reopen_model_b_plan_review_draft_view(
        revision_1_path,
        content_path,
        plan_path,
        grounding_path,
    )
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-b"],
                }
            ],
        }
    )
    revision_2 = apply_persisted_model_b_plan_review_patch(
        revision_1_path,
        content_path,
        plan_path,
        grounding_path,
        request,
        revision_2_path,
        updated_at=datetime(2026, 9, 10, 17, 0, tzinfo=UTC),
    )

    assert load_model_b_plan_review_draft(revision_1_path) == revision_1
    assert reopened == revision_1
    assert isinstance(view, ModelBPlanReviewDraftView)
    assert safely_reopened_view == view
    assert view.available_official_spec_refs == (
        "official-spec-a",
        "official-spec-b",
    )
    assert revision_2.draft_revision == 2
    assert revision_2.reviewed_hwp_document_plan.official_spec_refs == (
        "official-spec-b",
    )
    assert revision_2.source_candidate_contract_sha256 == (
        revision_1.source_candidate_contract_sha256
    )
    assert revision_2.grounding_evidence_contract_sha256 == (
        revision_1.grounding_evidence_contract_sha256
    )
    assert revision_1_path.read_bytes() != revision_2_path.read_bytes()
    assert tuple(
        path.read_bytes() for path in (content_path, plan_path, grounding_path)
    ) == source_bytes
    assert revision_2.training_eligible is False
    assert revision_2.golden_eligible is False
    assert revision_2.release_eligible is False
    assert revision_2.rights_status == "unverified"


def test_reopen_rejects_changed_candidate_or_grounding_contract(tmp_path: Path) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_path = tmp_path / "review-r1.json"
    start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=revision_path,
    )

    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_payload["page_layout"]["margin_top_mm"] = 21.0
    changed_plan_path = tmp_path / "changed-plan.json"
    changed_plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    with pytest.raises(ModelBPlanReviewVerificationError, match="source candidate"):
        reopen_model_b_plan_review_draft(
            revision_path,
            content_path,
            changed_plan_path,
            grounding_path,
        )

    changed_grounding_payload = _grounding_evidence().model_dump(mode="json")
    changed_grounding_payload["knowledge_corpus_sha256"] = "a" * 64
    changed_grounding_path = tmp_path / "changed-grounding.json"
    changed_grounding_path.write_text(
        json.dumps(changed_grounding_payload),
        encoding="utf-8",
    )
    with pytest.raises(ModelBPlanReviewVerificationError, match="grounding evidence"):
        reopen_model_b_plan_review_draft(
            revision_path,
            content_path,
            plan_path,
            changed_grounding_path,
        )


def test_persisted_patch_rejects_stale_revision_without_creating_output(
    tmp_path: Path,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_path = tmp_path / "review-r1.json"
    output_path = tmp_path / "review-r2.json"
    start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=revision_path,
    )
    request = _request(
        {
            "expected_draft_revision": 2,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    )

    with pytest.raises(ModelBPlanReviewPatchError, match="stale"):
        apply_persisted_model_b_plan_review_patch(
            revision_path,
            content_path,
            plan_path,
            grounding_path,
            request,
            output_path,
        )

    assert not output_path.exists()


def test_persisted_start_never_overwrites_an_existing_revision(tmp_path: Path) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    output_path = tmp_path / "review-r1.json"
    output_path.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        start_persisted_model_b_plan_review_draft(
            content_path,
            plan_path,
            grounding_path,
            document_id="document-1",
            reviewer_label="reviewer-local-1",
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "keep"


def test_persisted_start_does_not_touch_recreated_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    output_path = tmp_path / "review-r1.json"
    real_publish = plan_review_module.publish_file_create_only
    recreated: Path | None = None

    def publish_then_recreate(source: Path, destination: Path) -> None:
        nonlocal recreated
        real_publish(source, destination)
        source.mkdir()
        (source / "keep.txt").write_text("keep", encoding="utf-8")
        recreated = source

    monkeypatch.setattr(
        plan_review_module,
        "publish_file_create_only",
        publish_then_recreate,
    )

    draft = start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=output_path,
    )

    assert load_model_b_plan_review_draft(output_path) == draft
    assert recreated is not None
    assert (recreated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_persisted_start_preflights_draft_size_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    output_path = tmp_path / "review-r1.json"
    monkeypatch.setattr(plan_review_module, "_MAX_REVIEW_DRAFT_BYTES", 1)

    with pytest.raises(ModelBPlanReviewVerificationError, match="exceeds size limit"):
        start_persisted_model_b_plan_review_draft(
            content_path,
            plan_path,
            grounding_path,
            document_id="document-1",
            reviewer_label="reviewer-local-1",
            output_path=output_path,
        )

    assert not output_path.exists()


def test_bounded_stream_rejects_limit_plus_one_byte() -> None:
    with pytest.raises(ModelBPlanReviewVerificationError, match="exceeds size limit"):
        plan_review_module._read_stream_bounded(
            BytesIO(b"x" * 9),
            "source",
            max_bytes=8,
        )


def test_source_file_replacement_between_lstat_and_open_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    replacement_path = tmp_path / "replacement-content.json"
    replacement_path.write_bytes(content_path.read_bytes())
    original_path = tmp_path / "original-content.json"
    real_open = plan_review_module.os.open
    swapped = False

    def swapping_open(path: Path, flags: int) -> int:
        nonlocal swapped
        if not swapped and Path(path) == content_path:
            content_path.replace(original_path)
            replacement_path.replace(content_path)
            swapped = True
        return real_open(path, flags)

    monkeypatch.setattr(plan_review_module.os, "open", swapping_open)

    with pytest.raises(ModelBPlanReviewVerificationError, match="changed while it was read"):
        start_persisted_model_b_plan_review_draft(
            content_path,
            plan_path,
            grounding_path,
            document_id="document-1",
            reviewer_label="reviewer-local-1",
            output_path=tmp_path / "review-r1.json",
        )


def test_symlink_source_is_rejected(tmp_path: Path) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    symlink_path = tmp_path / "content-link.json"
    try:
        symlink_path.symlink_to(content_path)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ModelBPlanReviewVerificationError, match="regular non-symlink"):
        start_persisted_model_b_plan_review_draft(
            symlink_path,
            plan_path,
            grounding_path,
            document_id="document-1",
            reviewer_label="reviewer-local-1",
            output_path=tmp_path / "review-r1.json",
        )


def test_patch_cli_bounds_stdin_before_json_validation() -> None:
    with pytest.raises(ModelBPlanReviewPatchError, match="patch request exceeds size limit"):
        load_model_b_plan_review_patch_request(BytesIO(b"x" * 9), max_bytes=8)


def test_public_api_exports_model_b_verifier_and_verified_view_facade() -> None:
    assert evaluation_api.verify_model_b_plan_review_draft is verify_model_b_plan_review_draft
    assert (
        evaluation_api.reopen_model_b_plan_review_draft_view
        is reopen_model_b_plan_review_draft_view
    )


def test_model_b_plan_review_cli_round_trip(tmp_path: Path) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_1_path = tmp_path / "cli-r1.json"
    revision_2_path = tmp_path / "cli-r2.json"
    project_root = Path(__file__).resolve().parents[2]

    def run_cli(command: str, *arguments: str, stdin: bytes | None = None) -> bytes:
        script = project_root / "ai" / "datasets" / "model_b_plan_review_draft.py"
        completed = subprocess.run(
            [sys.executable, str(script), command, *arguments],
            cwd=project_root,
            input=stdin,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        return completed.stdout

    start_payload = json.loads(
        run_cli(
            "start",
            str(content_path),
            str(plan_path),
            str(grounding_path),
            "document-1",
            "--reviewer-label",
            "reviewer-local-1",
            "--out",
            str(revision_1_path),
        )
    )
    assert start_payload["draft_revision"] == 1
    assert start_payload["training_eligible"] is False
    assert start_payload["rights_status"] == "unverified"

    verified_payload = json.loads(
        run_cli(
            "verify",
            str(revision_1_path),
            str(content_path),
            str(plan_path),
            str(grounding_path),
        )
    )
    assert verified_payload["source_candidate_contract_sha256"] == (
        start_payload["source_candidate_contract_sha256"]
    )

    view_payload = json.loads(
        run_cli(
            "view",
            str(revision_1_path),
            str(content_path),
            str(plan_path),
            str(grounding_path),
        )
    )
    assert view_payload["schema_version"] == "model-b-plan-review-view/1.0"
    assert view_payload["available_official_spec_refs"] == [
        "official-spec-a",
        "official-spec-b",
    ]

    current = load_model_b_plan_review_draft(revision_1_path)
    request = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-b"],
                }
            ],
        }
    )
    assert request.expected_source_candidate_contract_sha256 == (
        current.source_candidate_contract_sha256
    )
    patch_payload = json.loads(
        run_cli(
            "patch",
            str(revision_1_path),
            str(content_path),
            str(plan_path),
            str(grounding_path),
            "--out",
            str(revision_2_path),
            stdin=request.model_dump_json().encode("utf-8"),
        )
    )
    assert patch_payload["draft_revision"] == 2
    assert load_model_b_plan_review_draft(revision_1_path).draft_revision == 1
    assert load_model_b_plan_review_draft(revision_2_path).draft_revision == 2


def test_completion_seals_only_status_revision_and_time_without_promotion() -> None:
    current = _start()
    before = current.model_dump(mode="python")
    completed = complete_model_b_plan_review_draft(
        current,
        _completion_request(current),
        updated_at=datetime(2026, 9, 10, 18, 0, tzinfo=UTC),
    )
    after = completed.model_dump(mode="python")

    assert current.model_dump(mode="python") == before
    assert completed.schema_version == "model-b-plan-review-draft/1.2"
    assert completed.status == "complete"
    assert completed.draft_revision == 2
    assert completed.updated_at == "2026-09-10T18:00:00+00:00"
    for field_name in before.keys() - {"status", "draft_revision", "updated_at"}:
        assert after[field_name] == before[field_name]
    assert completed.identity_assurance == "self_asserted_untrusted"
    assert completed.rights_status == "unverified"
    assert completed.golden_eligible is False
    assert completed.training_eligible is False
    assert completed.release_eligible is False

    view = build_model_b_plan_review_draft_view(completed)
    assert view.schema_version == "model-b-plan-review-view/1.0"
    assert view.status == "complete"
    assert view.identity_assurance == "self_asserted_untrusted"
    assert view.golden_eligible is False


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"expected_draft_revision": 2}, "stale"),
        (
            {"expected_source_candidate_contract_sha256": "0" * 64},
            "source candidate binding changed",
        ),
        (
            {"expected_grounding_evidence_contract_sha256": "0" * 64},
            "grounding evidence binding changed",
        ),
        (
            {"expected_reviewed_hwp_document_plan_contract_sha256": "0" * 64},
            "current reviewed plan binding changed",
        ),
    ),
)
def test_completion_rejects_stale_or_rebound_request(
    override: dict[str, object],
    message: str,
) -> None:
    current = _start()

    with pytest.raises(ModelBPlanReviewCompletionError, match=message):
        complete_model_b_plan_review_draft(
            current,
            _completion_request(current, **override),
        )


def test_completed_revision_rejects_recompletion_and_patch() -> None:
    current = _start()
    completed = complete_model_b_plan_review_draft(
        current,
        _completion_request(current),
    )

    with pytest.raises(ModelBPlanReviewCompletionError, match="already complete"):
        complete_model_b_plan_review_draft(
            completed,
            _completion_request(completed),
        )

    patch_request = ModelBPlanReviewPatchRequest(
        expected_draft_revision=completed.draft_revision,
        expected_source_candidate_contract_sha256=(
            completed.source_candidate_contract_sha256
        ),
        expected_grounding_evidence_contract_sha256=(
            completed.grounding_evidence_contract_sha256
        ),
        expected_reviewed_hwp_document_plan_contract_sha256=(
            completed.reviewed_hwp_document_plan_contract_sha256
        ),
        operations=(
            {
                "op": "set_official_spec_refs",
                "official_spec_refs": ("official-spec-a",),
            },
        ),
    )
    with pytest.raises(ModelBPlanReviewPatchError, match="cannot be patched"):
        apply_model_b_plan_review_patch(completed, patch_request)


def test_completion_request_is_strict_and_bounded() -> None:
    current = _start()
    valid = _completion_request(current).model_dump(mode="json")

    assert (
        ModelBPlanReviewCompletionRequest.model_validate(valid, strict=True)
        .schema_version
        == "model-b-plan-review-completion/1.0"
    )
    with pytest.raises(ValidationError):
        ModelBPlanReviewCompletionRequest.model_validate(
            {**valid, "unexpected": True},
            strict=True,
        )
    with pytest.raises(ValidationError):
        ModelBPlanReviewCompletionRequest.model_validate(
            {**valid, "expected_draft_revision": 1.0},
            strict=True,
        )
    with pytest.raises(
        ModelBPlanReviewCompletionError,
        match="completion request exceeds size limit",
    ):
        load_model_b_plan_review_completion_request(BytesIO(b"x" * 9), max_bytes=8)


def test_request_loaders_reject_trailing_json_after_a_short_read() -> None:
    class ShortReadBytesIO(BytesIO):
        def __init__(self, payload: bytes, first_chunk_size: int) -> None:
            super().__init__(payload)
            self._first_chunk_size = first_chunk_size
            self._is_first_read = True

        def read(self, size: int = -1) -> bytes:
            if self._is_first_read:
                self._is_first_read = False
                size = self._first_chunk_size
            return super().read(size)

    current = _start()
    completion_payload = _completion_request(current).model_dump_json().encode("utf-8")
    completion_stream = ShortReadBytesIO(
        completion_payload + b"{}",
        len(completion_payload),
    )
    with pytest.raises(ModelBPlanReviewCompletionError, match="not valid strict JSON"):
        load_model_b_plan_review_completion_request(completion_stream)

    patch_payload = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    ).model_dump_json().encode("utf-8")
    patch_stream = ShortReadBytesIO(patch_payload + b"{}", len(patch_payload))
    with pytest.raises(ModelBPlanReviewPatchError, match="not valid strict JSON"):
        load_model_b_plan_review_patch_request(patch_stream)


@pytest.mark.parametrize("invalid_token", ["NaN", "Infinity", "-Infinity"])
def test_request_loaders_reject_nonfinite_json_numbers(invalid_token: str) -> None:
    current = _start()
    completion_payload = _completion_request(current).model_dump_json()
    invalid = completion_payload.replace(
        '"expected_draft_revision":1',
        f'"expected_draft_revision":{invalid_token}',
        1,
    ).encode("utf-8")

    with pytest.raises(ModelBPlanReviewCompletionError, match="not valid strict JSON"):
        load_model_b_plan_review_completion_request(BytesIO(invalid))


def test_request_and_draft_loaders_reject_duplicate_json_keys(tmp_path: Path) -> None:
    current = _start()
    completion_payload = _completion_request(current).model_dump_json()
    duplicate_completion = completion_payload.replace(
        '"expected_draft_revision":1',
        '"expected_draft_revision":999,"expected_draft_revision":1',
        1,
    ).encode("utf-8")
    with pytest.raises(ModelBPlanReviewCompletionError, match="not valid strict JSON"):
        load_model_b_plan_review_completion_request(BytesIO(duplicate_completion))

    patch_payload = _request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_official_spec_refs",
                    "official_spec_refs": ["official-spec-a"],
                }
            ],
        }
    ).model_dump_json()
    duplicate_patch = patch_payload.replace(
        '"expected_draft_revision":1',
        '"expected_draft_revision":999,"expected_draft_revision":1',
        1,
    ).encode("utf-8")
    with pytest.raises(ModelBPlanReviewPatchError, match="not valid strict JSON"):
        load_model_b_plan_review_patch_request(BytesIO(duplicate_patch))

    draft_payload = current.model_dump_json(indent=2).encode("utf-8")
    duplicate_draft = draft_payload.replace(
        b'"draft_revision": 1',
        b'"draft_revision": 999,\n  "draft_revision": 1',
        1,
    )
    draft_path = tmp_path / "duplicate-draft.json"
    draft_path.write_bytes(duplicate_draft)
    with pytest.raises(ModelBPlanReviewVerificationError, match="not valid strict JSON"):
        load_model_b_plan_review_draft(draft_path)


def test_persisted_completion_is_create_only_and_rebinds_sources(
    tmp_path: Path,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_1_path = tmp_path / "review-r1.json"
    revision_2_path = tmp_path / "review-r2.json"
    current = start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=revision_1_path,
        updated_at=datetime(2026, 9, 10, 16, 0, tzinfo=UTC),
    )
    revision_1_bytes = revision_1_path.read_bytes()
    source_bytes = tuple(
        path.read_bytes() for path in (content_path, plan_path, grounding_path)
    )

    completed = complete_persisted_model_b_plan_review_draft(
        revision_1_path,
        content_path,
        plan_path,
        grounding_path,
        _completion_request(current),
        revision_2_path,
        updated_at=datetime(2026, 9, 10, 18, 0, tzinfo=UTC),
    )

    assert revision_1_path.read_bytes() == revision_1_bytes
    assert load_model_b_plan_review_draft(revision_1_path).status == "in_progress"
    assert load_model_b_plan_review_draft(revision_2_path) == completed
    assert completed.status == "complete"
    assert completed.draft_revision == 2
    assert tuple(
        path.read_bytes() for path in (content_path, plan_path, grounding_path)
    ) == source_bytes

    occupied = tmp_path / "occupied-r2.json"
    occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        complete_persisted_model_b_plan_review_draft(
            revision_1_path,
            content_path,
            plan_path,
            grounding_path,
            _completion_request(current),
            occupied,
        )
    assert occupied.read_text(encoding="utf-8") == "keep"


def test_persisted_completion_rejects_external_rebinding_without_output(
    tmp_path: Path,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_1_path = tmp_path / "review-r1.json"
    output_path = tmp_path / "review-r2.json"
    current = start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=revision_1_path,
    )
    changed_grounding_value = _grounding_evidence().model_dump(mode="json")
    changed_grounding_value["retrieval_artifact_sha256"] = "0" * 64
    changed_grounding_path = tmp_path / "changed-grounding.json"
    changed_grounding_path.write_text(
        json.dumps(changed_grounding_value),
        encoding="utf-8",
    )

    with pytest.raises(ModelBPlanReviewVerificationError, match="grounding evidence"):
        complete_persisted_model_b_plan_review_draft(
            revision_1_path,
            content_path,
            plan_path,
            changed_grounding_path,
            _completion_request(current),
            output_path,
        )

    assert not output_path.exists()


def test_model_b_completion_cli_uses_bounded_request_and_keeps_noneligible(
    tmp_path: Path,
) -> None:
    content_path, plan_path, grounding_path = _write_sources(tmp_path)
    revision_1_path = tmp_path / "cli-r1.json"
    revision_2_path = tmp_path / "cli-r2.json"
    current = start_persisted_model_b_plan_review_draft(
        content_path,
        plan_path,
        grounding_path,
        document_id="document-1",
        reviewer_label="reviewer-local-1",
        output_path=revision_1_path,
    )
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "ai"
                / "datasets"
                / "model_b_plan_review_draft.py"
            ),
            "complete",
            str(revision_1_path),
            str(content_path),
            str(plan_path),
            str(grounding_path),
            "--out",
            str(revision_2_path),
        ],
        cwd=project_root,
        input=_completion_request(current).model_dump_json().encode("utf-8"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["draft_revision"] == 2
    assert payload["identity_assurance"] == "self_asserted_untrusted"
    assert payload["rights_status"] == "unverified"
    assert payload["golden_eligible"] is False
    assert payload["training_eligible"] is False
    assert payload["release_eligible"] is False


def test_completion_accepts_draft_extracted_from_verified_handoff_envelope() -> None:
    draft = _start()
    lineage_id = "a" * 64
    candidate_manifest_sha256 = "b" * 64
    envelope = CandidateToModelBHandoff(
        created_at=draft.updated_at,
        document_id=draft.document_id,
        lineage_id=lineage_id,
        completed_candidate_review=CompletedCandidateReviewBinding(
            artifact_sha256="c" * 64,
            draft_revision=3,
            candidate_manifest_sha256=candidate_manifest_sha256,
            document_id=draft.document_id,
            lineage_id=lineage_id,
            base_content_ir={
                "path": "content-ir.json",
                "sha256": "d" * 64,
                "contract_sha256": "e" * 64,
            },
            base_hwp_document_plan={
                "path": "plan.json",
                "sha256": "f" * 64,
                "contract_sha256": "1" * 64,
            },
            reviewed_content_ir_contract_sha256=(
                draft.content_ir_contract_sha256
            ),
            reviewed_hwp_document_plan_contract_sha256=(
                draft.base_hwp_document_plan_contract_sha256
            ),
        ),
        grounding_sidecar=GroundingSidecarHandoffBinding(
            manifest_sha256="2" * 64,
            candidate_manifest_sha256=candidate_manifest_sha256,
            document_id=draft.document_id,
            lineage_id=lineage_id,
            knowledge_corpus_sha256=(
                draft.grounding_evidence.knowledge_corpus_sha256
            ),
            capability_profile_id=(
                draft.grounding_evidence.capability_profile_id
            ),
            capability_profile_sha256=(
                draft.grounding_evidence.capability_profile_sha256
            ),
            retrieval_artifact_sha256=(
                draft.grounding_evidence.retrieval_artifact_sha256
            ),
            grounding_evidence_contract_sha256=(
                draft.grounding_evidence_contract_sha256
            ),
        ),
        model_b_plan_review_draft=draft,
    )
    verified = VerifiedCandidateToModelBHandoff(
        envelope=envelope,
        artifact_sha256="3" * 64,
    )
    extracted = model_b_plan_review_draft_from_verified_handoff(verified)

    completed = complete_model_b_plan_review_draft(
        extracted,
        _completion_request(extracted),
    )

    assert completed.status == "complete"
    assert completed.draft_revision == 2
    assert completed.content_ir == draft.content_ir
    assert completed.reviewed_hwp_document_plan == draft.reviewed_hwp_document_plan
    assert completed.grounding_evidence == draft.grounding_evidence
    assert completed.golden_eligible is False
