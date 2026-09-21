from __future__ import annotations

import binascii
import hashlib
import json
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import scan2hwpx.evaluation.review_draft as review_draft_module
from scan2hwpx.contracts import (
    ContentIR,
    ContentPlanItem,
    EvidenceIR,
    HwpDocumentPlan,
    ImageContentNode,
    contract_sha256,
)
from scan2hwpx.evaluation.review_draft import (
    CandidateArtifactBinding,
    CandidateContractArtifactBinding,
    CandidateReviewCompletionRequest,
    CandidateReviewDraft,
    CandidateReviewDraftView,
    CandidateReviewPatchRequest,
    IssueDisposition,
    ReviewDraftCompletionError,
    ReviewDraftPatchError,
    ReviewDraftVerificationError,
    SetTextPatchOperation,
    VerifiedCandidateCompileInputs,
    apply_candidate_review_patch,
    build_candidate_review_draft_view,
    complete_candidate_review_draft,
    load_candidate_review_draft,
    materialize_verified_candidate_compile_inputs,
    start_candidate_review_draft,
    verify_candidate_review_draft,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _png(pixel: bytes) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00" + pixel))
        + chunk(b"IEND", b"")
    )


_PAGE_IMAGE_PNG = _png(b"\x10\x20\x30\xff")
_CROP_A_PNG = _png(b"\x40\x50\x60\xff")
_CROP_B_PNG = _png(b"\x70\x80\x90\xff")
_UNREGISTERED_CROP_PNG = _png(b"\xa0\xb0\xc0\xff")


def _contract_bytes(value: EvidenceIR | ContentIR | HwpDocumentPlan) -> bytes:
    return _json_bytes(value.model_dump(mode="json"))


def _draft_from_json(value: object) -> CandidateReviewDraft:
    return CandidateReviewDraft.model_validate_json(_json_bytes(value), strict=True)


def _patch_request(value: object) -> CandidateReviewPatchRequest:
    return CandidateReviewPatchRequest.model_validate_json(
        _json_bytes(value),
        strict=True,
    )


def _completion_request(value: object) -> CandidateReviewCompletionRequest:
    return CandidateReviewCompletionRequest.model_validate_json(
        _json_bytes(value),
        strict=True,
    )


def _view_from_json(value: object) -> CandidateReviewDraftView:
    return CandidateReviewDraftView.model_validate_json(
        _json_bytes(value),
        strict=True,
    )


def _evidence(pdf_sha256: str, page_sha256: str) -> EvidenceIR:
    return EvidenceIR.model_validate(
        {
            "id": "evidence-1",
            "source_document_sha256": pdf_sha256,
            "sources": [
                {
                    "id": "input-pdf",
                    "kind": "original_document",
                    "artifact_ref": f"blob://pdf-{pdf_sha256}/source.pdf",
                    "producer": "fixture",
                    "sha256": pdf_sha256,
                },
                {
                    "id": "page-1",
                    "kind": "page_image",
                    "artifact_ref": f"blob://pdf-{pdf_sha256}/page-0001.png",
                    "producer": "fixture",
                    "sha256": page_sha256,
                },
                {
                    "id": "crop-a",
                    "kind": "crop",
                    "artifact_ref": f"blob://pdf-{pdf_sha256}/crop-a.png",
                    "producer": "fixture",
                    "sha256": _sha256(_CROP_A_PNG),
                },
                {
                    "id": "crop-b",
                    "kind": "crop",
                    "artifact_ref": f"blob://pdf-{pdf_sha256}/crop-b.png",
                    "producer": "fixture",
                    "sha256": _sha256(_CROP_B_PNG),
                },
                {
                    "id": "crop-unregistered",
                    "kind": "crop",
                    "artifact_ref": f"blob://pdf-{pdf_sha256}/crop-unregistered.png",
                    "producer": "fixture",
                    "sha256": _sha256(_UNREGISTERED_CROP_PNG),
                },
            ],
            "pages": [
                {
                    "id": "evidence-page-1",
                    "page_no": 1,
                    "width": 100.0,
                    "height": 100.0,
                    "image_source_ref": "page-1",
                    "observations": [
                        {
                            "id": "obs-text",
                            "kind": "text_line",
                            "bbox": {
                                "pixel": [1.0, 1.0, 90.0, 10.0],
                                "normalized": [0.01, 0.01, 0.9, 0.1],
                            },
                            "confidence": 0.0,
                            "source_refs": ["page-1"],
                            "ocr_candidates": [
                                {
                                    "text": "초안 문장",
                                    "provider": "fixture",
                                    "confidence": 0.0,
                                    "source_ref": "page-1",
                                    "selected": True,
                                }
                            ],
                        },
                        {
                            "id": "obs-table",
                            "kind": "table_grid",
                            "bbox": {
                                "pixel": [1.0, 20.0, 90.0, 60.0],
                                "normalized": [0.01, 0.2, 0.9, 0.6],
                            },
                            "confidence": 0.0,
                            "source_refs": ["page-1"],
                        },
                        {
                            "id": "obs-image",
                            "kind": "image",
                            "bbox": {
                                "pixel": [1.0, 65.0, 50.0, 95.0],
                                "normalized": [0.01, 0.65, 0.5, 0.95],
                            },
                            "confidence": 0.0,
                            "source_refs": ["crop-a"],
                        },
                        {
                            "id": "obs-image-b",
                            "kind": "image",
                            "bbox": {
                                "pixel": [51.0, 65.0, 90.0, 95.0],
                                "normalized": [0.51, 0.65, 0.9, 0.95],
                            },
                            "confidence": 1.0,
                            "source_refs": ["crop-b"],
                        },
                        {
                            "id": "obs-image-unregistered",
                            "kind": "image",
                            "bbox": {
                                "pixel": [51.0, 65.0, 90.0, 95.0],
                                "normalized": [0.51, 0.65, 0.9, 0.95],
                            },
                            "confidence": 1.0,
                            "source_refs": ["crop-unregistered"],
                        },
                        {
                            "id": "obs-page-region",
                            "kind": "region",
                            "bbox": {
                                "pixel": [1.0, 65.0, 90.0, 95.0],
                                "normalized": [0.01, 0.65, 0.9, 0.95],
                            },
                            "confidence": 0.0,
                            "source_refs": ["page-1"],
                        },
                    ],
                }
            ],
        }
    )


def _base_content(evidence: EvidenceIR) -> ContentIR:
    return ContentIR.model_validate(
        {
            "id": "content-1",
            "evidence_ir_id": evidence.id,
            "evidence_ir_sha256": contract_sha256(evidence),
            "revision": 1,
            "nodes": [
                {
                    "id": "text-1",
                    "kind": "text",
                    "role": "other",
                    "text": "초안 문장",
                    "evidence_refs": ["obs-text"],
                    "confidence": 0.0,
                    "needs_review": True,
                },
                {
                    "id": "table-1",
                    "kind": "table",
                    "rows": 2,
                    "columns": 2,
                    "cells": [
                        {
                            "row": row,
                            "column": column,
                            "text": f"{row},{column}",
                            "evidence_refs": ["obs-table"],
                        }
                        for row in range(2)
                        for column in range(2)
                    ],
                    "evidence_refs": ["obs-table"],
                    "confidence": 0.0,
                    "needs_review": True,
                },
                {
                    "id": "image-1",
                    "kind": "image",
                    "asset_ref": "page-1",
                    "evidence_refs": ["obs-page-region"],
                    "confidence": 0.0,
                    "needs_review": True,
                },
                {
                    "id": "formula-1",
                    "kind": "formula",
                    "expression": "x+1",
                    "format": "latex",
                    "evidence_refs": ["obs-text"],
                    "confidence": 0.0,
                    "needs_review": True,
                },
            ],
            "reading_order": ["text-1", "table-1", "image-1", "formula-1"],
        }
    )


def _image_only_content(evidence: EvidenceIR) -> ContentIR:
    value = _base_content(evidence).model_dump(mode="json")
    image = next(node for node in value["nodes"] if node["id"] == "image-1")
    value["nodes"] = [image]
    value["reading_order"] = ["image-1"]
    return ContentIR.model_validate(value)


def _multi_image_content(evidence: EvidenceIR) -> ContentIR:
    value = _base_content(evidence).model_dump(mode="json")
    image = next(node for node in value["nodes"] if node["id"] == "image-1")
    image_position = value["nodes"].index(image)
    image_2 = dict(image)
    image_2["id"] = "image-2"
    image_3 = dict(image)
    image_3["id"] = "image-3"
    value["nodes"][image_position : image_position + 1] = [
        image_2,
        image,
        image_3,
    ]
    value["reading_order"][image_position : image_position + 1] = [
        "image-2",
        "image-1",
        "image-3",
    ]
    return ContentIR.model_validate(value)


def _plan(content: ContentIR) -> HwpDocumentPlan:
    render_as = {
        "text": "paragraph",
        "table": "table",
        "image": "image",
        "formula": "formula",
    }
    return HwpDocumentPlan.model_validate(
        {
            "id": "plan-1",
            "content_ir_id": content.id,
            "content_ir_revision": content.revision,
            "content_ir_sha256": contract_sha256(content),
            "capability_profile_id": "exam2hwpx-authoring-v1",
            "design_profile_id": "source-projection-candidate-v1",
            "official_spec_refs": ["official-fixture"],
            "page_layout": {
                "width_mm": 210.0,
                "height_mm": 297.0,
                "margin_top_mm": 15.0,
                "margin_right_mm": 15.0,
                "margin_bottom_mm": 15.0,
                "margin_left_mm": 15.0,
                "columns": 1,
                "column_gap_mm": 0.0,
            },
            "styles": [],
            "flow": [
                {
                    "id": f"flow-{index}",
                    "kind": "content",
                    "render_as": render_as[node.kind],
                    "content_ref": node.id,
                }
                for index, node in enumerate(content.nodes, start=1)
            ],
        }
    )


def _reviewed_content(base: ContentIR) -> ContentIR:
    value = base.model_dump(mode="json")
    value["revision"] = 2
    for node in value["nodes"]:
        node["needs_review"] = False
        if node["id"] == "text-1":
            node["text"] = "사람이 교정한 문장"
            node["role"] = "instruction"
        if node["id"] == "table-1":
            for cell in node["cells"]:
                cell["text"] = "교정"
        image_grounding = {
            "image-1": ("crop-a", "obs-image"),
            "image-2": ("crop-b", "obs-image-b"),
            "image-3": ("crop-a", "obs-image"),
        }.get(node["id"])
        if image_grounding is not None:
            node["asset_ref"], observation_ref = image_grounding
            node["evidence_refs"] = [observation_ref]
    return ContentIR.model_validate(value)


def _write_binding(root: Path, relative: str, payload: bytes) -> dict[str, str]:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": _sha256(payload)}


def _write_contract_binding(
    root: Path,
    relative: str,
    contract: EvidenceIR | ContentIR | HwpDocumentPlan,
) -> dict[str, str]:
    binding = _write_binding(root, relative, _contract_bytes(contract))
    binding["contract_sha256"] = contract_sha256(contract)
    return binding


@dataclass(frozen=True)
class ReviewFixture:
    root: Path
    manifest_path: Path
    draft: CandidateReviewDraft
    evidence: EvidenceIR
    base_content: ContentIR
    base_plan: HwpDocumentPlan


def _fixture(
    tmp_path: Path,
    *,
    image_only: bool = False,
    multi_image: bool = False,
) -> ReviewFixture:
    root = tmp_path / "candidates"
    root.mkdir(parents=True)
    lineage = _sha256(b"original-hwp")
    document_id = f"candidate-{lineage[:24]}"
    pdf_sha256 = _sha256(b"paired-pdf")
    hwpx_sha256 = _sha256(b"paired-hwpx")
    page_payload = _PAGE_IMAGE_PNG
    evidence = _evidence(pdf_sha256, _sha256(page_payload))
    if image_only:
        base_content = _image_only_content(evidence)
    elif multi_image:
        base_content = _multi_image_content(evidence)
    else:
        base_content = _base_content(evidence)
    base_plan = _plan(base_content)
    reviewed_content = _reviewed_content(base_content)
    reviewed_plan = _plan(reviewed_content)

    prefix = f"{lineage}/"
    projection = _write_binding(root, prefix + "projection_source.json", b"{}")
    evidence_binding = _write_contract_binding(root, prefix + "evidence_ir.json", evidence)
    content_binding = _write_contract_binding(
        root, prefix + "content_ir.candidate.json", base_content
    )
    plan_binding = _write_contract_binding(
        root, prefix + "hwp_document_plan.candidate.json", base_plan
    )
    report = _write_binding(root, prefix + "report.json", b"{}")
    page = _write_binding(root, prefix + "pages/page-0001.png", page_payload)
    crop_a = _write_binding(root, prefix + "assets/crop-a.png", _CROP_A_PNG)
    crop_b = _write_binding(root, prefix + "assets/crop-b.png", _CROP_B_PNG)
    issue_codes = [
        "human_review_required",
        "page_assignment_from_pdf_alignment",
        "rights_manifest_missing",
    ]
    manifest: dict[str, Any] = {
        "schema_version": "hwp-human-review-candidates/1.0",
        "artifact_role": "human_review_candidate",
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "rights_manifest": {"status": "missing"},
        "source_bundle": {
            "schema_version": "hwp-hwpx-projection-pairs/1.0",
            "manifest_sha256": "1" * 64,
            "source_dataset_report_sha256": "2" * 64,
            "source_inventory_sha256": "3" * 64,
            "producer": {"name": "fixture", "version": "1"},
        },
        "producer": {
            "name": "scan2hwpx-hwpx-projection-candidate-builder",
            "version": "1.1",
            "pymupdf_version": "1.26.0",
            "dpi": 300,
            "pdf_text_order": "native",
        },
        "capability_profile": {
            "id": "exam2hwpx-authoring-v1",
            "sha256": "4" * 64,
        },
        "knowledge_corpus_sha256": "5" * 64,
        "contract_versions": {
            "projection_source": "hwpx-projection-source/1.0",
            "evidence_ir": "evidence-ir/1.0",
            "content_ir": "content-ir/1.0",
            "hwp_document_plan": "hwp-document-plan/1.0",
        },
        "document_count": 1,
        "page_count": 1,
        "splits": {
            "train": {"documents": 1, "pages": 1},
            "validation": {"documents": 0, "pages": 0},
            "test": {"documents": 0, "pages": 0},
        },
        "issue_counts": {code: 1 for code in issue_codes},
        "documents": [
            {
                "document_id": document_id,
                "lineage_id": lineage,
                "split": "train",
                "page_count": 1,
                "status": "needs_human_review",
                "source_artifacts": {
                    "hwp": {"sha256": lineage},
                    "hwpx": {"sha256": hwpx_sha256},
                    "pdf": {"sha256": pdf_sha256},
                },
                "artifacts": {
                    "projection_source": projection,
                    "evidence_ir": evidence_binding,
                    "content_ir_candidate": content_binding,
                    "hwp_document_plan_candidate": plan_binding,
                    "report": report,
                    "page_images": [page],
                    "image_crops": [crop_a, crop_b],
                },
                "issue_codes": issue_codes,
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_payload = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_payload)
    draft = CandidateReviewDraft(
        status="complete",
        draft_revision=1,
        reviewer_label="reviewer-local-1",
        updated_at="2026-09-10T13:30:00+09:00",
        candidate_manifest_sha256=_sha256(manifest_payload),
        document_id=document_id,
        lineage_id=lineage,
        source_pdf_sha256=pdf_sha256,
        evidence_ir=CandidateContractArtifactBinding.model_validate(evidence_binding),
        base_content_ir=CandidateContractArtifactBinding.model_validate(content_binding),
        base_hwp_document_plan=CandidateContractArtifactBinding.model_validate(plan_binding),
        reviewed_content_ir=reviewed_content,
        reviewed_content_ir_contract_sha256=contract_sha256(reviewed_content),
        reviewed_hwp_document_plan=reviewed_plan,
        reviewed_hwp_document_plan_contract_sha256=contract_sha256(reviewed_plan),
        issue_dispositions=tuple(
            IssueDisposition(issue_code=code, disposition="resolved") for code in issue_codes
        ),
    )
    return ReviewFixture(
        root=root,
        manifest_path=manifest_path,
        draft=draft,
        evidence=evidence,
        base_content=base_content,
        base_plan=base_plan,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _replace_reviewed_content(
    fixture: ReviewFixture,
    content: ContentIR,
) -> CandidateReviewDraft:
    plan = _plan(content)
    payload = fixture.draft.model_dump(mode="json")
    payload["reviewed_content_ir"] = content.model_dump(mode="json")
    payload["reviewed_content_ir_contract_sha256"] = contract_sha256(content)
    payload["reviewed_hwp_document_plan"] = plan.model_dump(mode="json")
    payload["reviewed_hwp_document_plan_contract_sha256"] = contract_sha256(plan)
    return _draft_from_json(payload)


def _in_progress_draft(fixture: ReviewFixture) -> CandidateReviewDraft:
    payload = fixture.draft.model_dump(mode="json")
    payload["status"] = "in_progress"
    return _draft_from_json(payload)


def _complete_draft_without_image(fixture: ReviewFixture) -> CandidateReviewDraft:
    content_value = fixture.draft.reviewed_content_ir.model_dump(mode="json")
    content_value["nodes"] = [
        node for node in content_value["nodes"] if node["id"] != "image-1"
    ]
    content_value["reading_order"] = [
        node_id
        for node_id in content_value["reading_order"]
        if node_id != "image-1"
    ]
    content = ContentIR.model_validate(content_value)

    plan_value = fixture.draft.reviewed_hwp_document_plan.model_dump(mode="json")
    plan_value["content_ir_sha256"] = contract_sha256(content)
    plan_value["flow"] = [
        item
        for item in plan_value["flow"]
        if item.get("content_ref") != "image-1"
    ]
    plan = HwpDocumentPlan.model_validate(plan_value)

    draft_value = fixture.draft.model_dump(mode="json")
    draft_value["reviewed_content_ir"] = content.model_dump(mode="json")
    draft_value["reviewed_content_ir_contract_sha256"] = contract_sha256(content)
    draft_value["reviewed_hwp_document_plan"] = plan.model_dump(mode="json")
    draft_value["reviewed_hwp_document_plan_contract_sha256"] = contract_sha256(plan)
    return _draft_from_json(draft_value)


def test_verifies_complete_draft_without_writes_or_eligibility(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = _tree_hashes(fixture.root)

    result = verify_candidate_review_draft(
        fixture.draft,
        candidate_root=fixture.root,
    )

    assert result.status == "complete"
    assert result.golden_eligible is False
    assert result.training_eligible is False
    assert result.release_eligible is False
    assert result.rights_status == "unverified"
    assert result.identity_assurance == "self_asserted_untrusted"
    assert _tree_hashes(fixture.root) == before
    assert not list(fixture.root.rglob("*golden*"))
    assert not list(fixture.root.rglob("*attestation*"))


def test_materializes_verified_compile_inputs_with_canonical_unique_assets(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, multi_image=True)
    before = _tree_hashes(fixture.root)
    image_refs_in_node_order = tuple(
        node.asset_ref
        for node in fixture.draft.reviewed_content_ir.nodes
        if isinstance(node, ImageContentNode)
    )
    assert image_refs_in_node_order == ("crop-b", "crop-a", "crop-a")

    result = materialize_verified_candidate_compile_inputs(
        fixture.draft,
        candidate_root=fixture.root,
    )

    assert isinstance(result, VerifiedCandidateCompileInputs)
    assert result.candidate_review == fixture.draft
    assert result.candidate_review is not fixture.draft
    assert result.evidence_ir == fixture.evidence
    assert result.image_assets is not None
    assert tuple(asset.asset_ref for asset in result.image_assets.assets) == (
        "crop-a",
        "crop-b",
    )
    assert {
        asset.asset_ref: asset.payload for asset in result.image_assets.assets
    } == {
        "crop-a": _CROP_A_PNG,
        "crop-b": _CROP_B_PNG,
    }
    assert result.image_assets.content_ir_contract_sha256 == contract_sha256(
        result.candidate_review.reviewed_content_ir
    )
    assert result.candidate_review.golden_eligible is False
    assert result.candidate_review.training_eligible is False
    assert result.candidate_review.release_eligible is False
    assert result.candidate_review.rights_status == "unverified"
    assert _tree_hashes(fixture.root) == before


def test_materializes_none_for_complete_review_without_images(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft = _complete_draft_without_image(fixture)
    before = _tree_hashes(fixture.root)

    result = materialize_verified_candidate_compile_inputs(
        draft,
        candidate_root=fixture.root,
    )

    assert result.candidate_review == draft
    assert result.evidence_ir == fixture.evidence
    assert result.image_assets is None
    assert _tree_hashes(fixture.root) == before


def test_materializer_rejects_in_progress_review_before_candidate_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, bytes]:
        raise AssertionError("candidate artifacts must not be read")

    monkeypatch.setattr(
        review_draft_module,
        "_read_verified_candidate_artifacts",
        fail_if_called,
    )
    with pytest.raises(ReviewDraftVerificationError, match="must be complete"):
        materialize_verified_candidate_compile_inputs(
            _in_progress_draft(fixture),
            candidate_root=fixture.root,
        )


def test_materializer_preflights_asset_count_before_candidate_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(review_draft_module, "MAX_IMAGE_ASSETS", 0)

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, bytes]:
        raise AssertionError("candidate artifacts must not be read")

    monkeypatch.setattr(
        review_draft_module,
        "_read_verified_candidate_artifacts",
        fail_if_called,
    )
    with pytest.raises(ReviewDraftVerificationError, match="asset count"):
        materialize_verified_candidate_compile_inputs(
            fixture.draft,
            candidate_root=fixture.root,
        )


def test_materializer_preflights_asset_bytes_before_retaining_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    original_reader = review_draft_module._read_and_verify_registered_artifact
    retained_image_reads = 0

    def observed_reader(
        root: Path,
        binding: CandidateArtifactBinding,
        *,
        max_bytes: int,
        retain_payload: bool,
    ) -> tuple[bytes | None, int]:
        nonlocal retained_image_reads
        if binding.path.endswith("/assets/crop-a.png") and retain_payload:
            retained_image_reads += 1
        return original_reader(
            root,
            binding,
            max_bytes=max_bytes,
            retain_payload=retain_payload,
        )

    monkeypatch.setattr(
        review_draft_module,
        "_read_and_verify_registered_artifact",
        observed_reader,
    )
    monkeypatch.setattr(
        review_draft_module,
        "MAX_IMAGE_ASSET_BYTES",
        len(_CROP_A_PNG) - 1,
    )

    with pytest.raises(ReviewDraftVerificationError, match="per-asset byte limit"):
        materialize_verified_candidate_compile_inputs(
            fixture.draft,
            candidate_root=fixture.root,
        )

    assert retained_image_reads == 0


def test_materializer_rejects_tampered_registered_image_payload(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    image_path = fixture.root.joinpath(
        *next(
            binding["path"]
            for binding in json.loads(fixture.manifest_path.read_text(encoding="utf-8"))[
                "documents"
            ][0]["artifacts"]["image_crops"]
            if binding["path"].endswith("/assets/crop-a.png")
        ).split("/")
    )
    image_path.write_bytes(_CROP_B_PNG)

    with pytest.raises(ReviewDraftVerificationError, match="SHA-256 mismatch"):
        materialize_verified_candidate_compile_inputs(
            fixture.draft,
            candidate_root=fixture.root,
        )


def test_materializer_rejects_symlinked_registered_image_payload(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    image_path = fixture.root / fixture.draft.lineage_id / "assets" / "crop-a.png"
    outside = tmp_path / "outside-crop-a.png"
    outside.write_bytes(image_path.read_bytes())
    image_path.unlink()
    try:
        image_path.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ReviewDraftVerificationError, match="symlink"):
        materialize_verified_candidate_compile_inputs(
            fixture.draft,
            candidate_root=fixture.root,
        )


def test_loads_strict_review_draft_json(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-draft.json"
    draft_path.write_bytes(_json_bytes(fixture.draft.model_dump(mode="json")))

    loaded = load_candidate_review_draft(draft_path)

    assert loaded == fixture.draft
    invalid = fixture.draft.model_dump(mode="json")
    invalid["unexpected"] = True
    draft_path.write_bytes(_json_bytes(invalid))
    with pytest.raises(ReviewDraftVerificationError, match="strict JSON"):
        load_candidate_review_draft(draft_path)


def test_starts_verified_in_progress_draft_outside_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "review-workspace" / "draft.json"
    before = _tree_hashes(fixture.root)
    updated_at = datetime(2026, 9, 10, 14, 0, tzinfo=UTC)

    draft = start_candidate_review_draft(
        fixture.root,
        fixture.draft.document_id,
        "reviewer-local-2",
        output,
        updated_at=updated_at,
    )

    assert load_candidate_review_draft(output) == draft
    assert draft.status == "in_progress"
    assert draft.draft_revision == 1
    assert draft.reviewer_label == "reviewer-local-2"
    assert draft.updated_at == "2026-09-10T14:00:00+00:00"
    assert draft.reviewed_content_ir.revision == fixture.base_content.revision + 1
    assert all(item.disposition == "pending" for item in draft.issue_dispositions)
    assert {item.issue_code for item in draft.issue_dispositions} == {
        item.issue_code for item in fixture.draft.issue_dispositions
    }
    content_copy = draft.reviewed_content_ir.model_dump(mode="json")
    content_copy["revision"] = fixture.base_content.revision
    assert content_copy == fixture.base_content.model_dump(mode="json")
    plan_copy = draft.reviewed_hwp_document_plan.model_dump(mode="json")
    plan_copy.update(
        {
            "content_ir_id": fixture.base_content.id,
            "content_ir_revision": fixture.base_content.revision,
            "content_ir_sha256": contract_sha256(fixture.base_content),
        }
    )
    assert plan_copy == fixture.base_plan.model_dump(mode="json")
    assert draft.golden_eligible is False
    assert draft.training_eligible is False
    assert draft.release_eligible is False
    assert draft.rights_status == "unverified"
    assert draft.identity_assurance == "self_asserted_untrusted"
    assert _tree_hashes(fixture.root) == before
    assert not list(output.parent.glob("*.partial"))


def test_start_preflights_bounded_view_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "oversized-view.json"
    monkeypatch.setattr(review_draft_module, "_MAX_VIEW_TOTAL_CHARS", 10)

    with pytest.raises(ValidationError, match="too much text"):
        start_candidate_review_draft(
            fixture.root,
            fixture.draft.document_id,
            "reviewer-local-2",
            output,
        )

    assert not output.exists()


def test_start_rejects_candidate_internal_or_existing_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    internal = fixture.root / "review-draft.json"
    with pytest.raises(ValueError, match="outside candidate root"):
        start_candidate_review_draft(
            fixture.root,
            fixture.draft.document_id,
            "reviewer-local-2",
            internal,
        )
    assert not internal.exists()

    output = tmp_path / "review-draft.json"
    output.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError, match="already exists"):
        start_candidate_review_draft(
            fixture.root,
            fixture.draft.document_id,
            "reviewer-local-2",
            output,
        )
    assert output.read_bytes() == b"keep-me"


def test_start_rejects_unknown_document_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "unknown.json"

    with pytest.raises(ReviewDraftVerificationError, match="not uniquely registered"):
        start_candidate_review_draft(
            fixture.root,
            "candidate-not-registered",
            "reviewer-local-2",
            output,
        )

    assert not output.exists()


def test_start_rejects_mismatched_selected_snapshot_without_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "mismatched-snapshot.json"

    with pytest.raises(ReviewDraftVerificationError, match="selected snapshot"):
        start_candidate_review_draft(
            fixture.root,
            fixture.draft.document_id,
            "reviewer-local-2",
            output,
            expected_manifest_sha256="f" * 64,
            expected_lineage_id=fixture.draft.lineage_id,
        )

    assert not output.exists()


def test_start_checks_selected_snapshot_before_bulk_artifact_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "mismatched-snapshot.json"

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, bytes]:
        raise AssertionError("artifact reads must happen after the snapshot check")

    monkeypatch.setattr(
        review_draft_module,
        "_read_verified_candidate_artifacts",
        fail_if_called,
    )
    with pytest.raises(ReviewDraftVerificationError, match="selected snapshot"):
        start_candidate_review_draft(
            fixture.root,
            fixture.draft.document_id,
            "reviewer-local-2",
            output,
            expected_manifest_sha256="f" * 64,
            expected_lineage_id=fixture.draft.lineage_id,
        )

    assert not output.exists()


def test_start_review_draft_cli_reports_only_noneligible_metadata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "cli-review" / "draft.json"
    before = _tree_hashes(fixture.root)
    result = subprocess.run(
        [
            sys.executable,
            "ai/datasets/start_candidate_review_draft.py",
            str(fixture.root),
            fixture.draft.document_id,
            "--reviewer-label",
            "reviewer-cli",
            "--expected-manifest-sha256",
            fixture.draft.candidate_manifest_sha256,
            "--expected-lineage-id",
            fixture.draft.lineage_id,
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=Path.cwd(),
    )
    report = json.loads(result.stdout)
    saved = load_candidate_review_draft(output)
    assert report["document_id"] == saved.document_id
    assert report["candidate_manifest_sha256"] == saved.candidate_manifest_sha256
    assert report["lineage_id"] == saved.lineage_id
    assert report["status"] == "in_progress"
    assert report["identity_assurance"] == "self_asserted_untrusted"
    assert report["rights_status"] == "unverified"
    assert report["golden_eligible"] is False
    assert report["training_eligible"] is False
    assert report["release_eligible"] is False
    assert _tree_hashes(fixture.root) == before
    assert "reviewed_content_ir" not in report


def test_verify_review_draft_cli_reports_only_noneligible_metadata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-draft.json"
    draft_path.write_bytes(_json_bytes(fixture.draft.model_dump(mode="json")))
    before = _tree_hashes(fixture.root)

    result = subprocess.run(
        [
            sys.executable,
            "ai/datasets/verify_candidate_review_draft.py",
            str(fixture.root),
            str(draft_path),
            "--expected-manifest-sha256",
            fixture.draft.candidate_manifest_sha256,
            "--expected-lineage-id",
            fixture.draft.lineage_id,
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=Path.cwd(),
    )
    report = json.loads(result.stdout)

    assert report == {
        "candidate_manifest_sha256": fixture.draft.candidate_manifest_sha256,
        "document_id": fixture.draft.document_id,
        "draft_revision": fixture.draft.draft_revision,
        "golden_eligible": False,
        "identity_assurance": "self_asserted_untrusted",
        "lineage_id": fixture.draft.lineage_id,
        "release_eligible": False,
        "rights_status": "unverified",
        "schema_version": "candidate-review-draft/1.0",
        "status": fixture.draft.status,
        "training_eligible": False,
    }
    assert str(fixture.root) not in result.stdout
    assert str(draft_path) not in result.stdout
    assert "reviewed_content_ir" not in report
    assert _tree_hashes(fixture.root) == before


def test_builds_strict_sanitized_review_draft_view(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = verify_candidate_review_draft(
        fixture.draft,
        candidate_root=fixture.root,
    )

    view = build_candidate_review_draft_view(verified)
    payload = view.model_dump(mode="json")

    assert set(payload) == {
        "document_id",
        "draft_revision",
        "golden_eligible",
        "identity_assurance",
        "issue_dispositions",
        "nodes",
        "release_eligible",
        "reviewed_content_ir_contract_sha256",
        "rights_status",
        "schema_version",
        "status",
        "training_eligible",
        "updated_at",
    }
    assert payload["schema_version"] == "candidate-review-view/1.0"
    assert payload["reviewed_content_ir_contract_sha256"] == (
        fixture.draft.reviewed_content_ir_contract_sha256
    )
    assert payload["golden_eligible"] is False
    assert payload["training_eligible"] is False
    assert payload["release_eligible"] is False
    assert payload["rights_status"] == "unverified"
    assert payload["identity_assurance"] == "self_asserted_untrusted"

    nodes = {node["id"]: node for node in payload["nodes"]}
    assert set(nodes["text-1"]) == {"id", "kind", "text", "role", "needs_review"}
    assert set(nodes["table-1"]) == {"id", "kind", "needs_review", "cells"}
    assert all(
        set(cell) == {"row", "column", "row_span", "column_span", "text"}
        for cell in nodes["table-1"]["cells"]
    )
    assert set(nodes["image-1"]) == {"id", "kind", "needs_review"}
    assert set(nodes["formula-1"]) == {"id", "kind", "needs_review"}
    assert all(
        set(item) == {"issue_code", "disposition", "note"} for item in payload["issue_dispositions"]
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden_key in (
        "artifact_ref",
        "asset_ref",
        "base_content_ir",
        "base_hwp_document_plan",
        "candidate_manifest_sha256",
        "confidence",
        "evidence_ir",
        "evidence_refs",
        "expression",
        "format",
        "lineage_id",
        "path",
        "reviewed_hwp_document_plan",
        "reviewer_label",
        "source_pdf_sha256",
    ):
        assert f'"{forbidden_key}"' not in serialized
    assert str(fixture.root) not in serialized
    assert fixture.draft.candidate_manifest_sha256 not in serialized
    assert fixture.draft.lineage_id not in serialized
    assert fixture.draft.source_pdf_sha256 not in serialized


def test_view_cli_reverifies_candidate_and_emits_only_sanitized_json(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-draft.json"
    draft_path.write_bytes(_json_bytes(fixture.draft.model_dump(mode="json")))
    before = _tree_hashes(fixture.root)

    result = subprocess.run(
        [
            sys.executable,
            "ai/datasets/view_candidate_review_draft.py",
            str(fixture.root),
            str(draft_path),
            "--expected-manifest-sha256",
            fixture.draft.candidate_manifest_sha256,
            "--expected-lineage-id",
            fixture.draft.lineage_id,
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=Path.cwd(),
    )
    report = json.loads(result.stdout)
    expected = build_candidate_review_draft_view(fixture.draft).model_dump(mode="json")

    assert report == expected
    assert str(fixture.root) not in result.stdout
    assert str(draft_path) not in result.stdout
    assert fixture.draft.source_pdf_sha256 not in result.stdout
    assert "reviewed_hwp_document_plan" not in result.stdout
    assert "evidence_refs" not in result.stdout
    assert "confidence" not in result.stdout
    assert _tree_hashes(fixture.root) == before


def test_view_cli_rejects_candidate_change_before_stdout(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-draft.json"
    draft_path.write_bytes(_json_bytes(fixture.draft.model_dump(mode="json")))
    fixture.manifest_path.write_bytes(fixture.manifest_path.read_bytes() + b" ")
    before = _tree_hashes(fixture.root)

    result = subprocess.run(
        [
            sys.executable,
            "ai/datasets/view_candidate_review_draft.py",
            str(fixture.root),
            str(draft_path),
            "--expected-manifest-sha256",
            fixture.draft.candidate_manifest_sha256,
            "--expected-lineage-id",
            fixture.draft.lineage_id,
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        cwd=Path.cwd(),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert _tree_hashes(fixture.root) == before


def test_review_view_rejects_duplicate_node_ids_and_cell_keys(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = build_candidate_review_draft_view(fixture.draft).model_dump(mode="json")
    payload["nodes"][1]["id"] = payload["nodes"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate review view node ids"):
        _view_from_json(payload)

    payload = build_candidate_review_draft_view(fixture.draft).model_dump(mode="json")
    table = next(node for node in payload["nodes"] if node["kind"] == "table")
    for field in ("row", "column", "row_span", "column_span"):
        table["cells"][1][field] = table["cells"][0][field]
    with pytest.raises(ValidationError, match="duplicate review view table cell keys"):
        _view_from_json(payload)


def test_review_view_enforces_payload_size_and_strict_fields(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = build_candidate_review_draft_view(fixture.draft).model_dump(mode="json")
    text = next(node for node in payload["nodes"] if node["kind"] == "text")
    text["text"] = "x" * 1_000_001
    with pytest.raises(ValidationError, match="at most 1000000 characters"):
        _view_from_json(payload)

    payload = build_candidate_review_draft_view(fixture.draft).model_dump(mode="json")
    payload["nodes"] = [
        {"id": f"node-{index}", "kind": "image", "needs_review": False} for index in range(5_001)
    ]
    with pytest.raises(ValidationError, match="at most 5000 items"):
        _view_from_json(payload)

    payload = build_candidate_review_draft_view(fixture.draft).model_dump(mode="json")
    payload["nodes"][0]["confidence"] = 1.0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _view_from_json(payload)


def test_review_view_counts_utf16_code_units_like_electron(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = build_candidate_review_draft_view(_fixture(tmp_path).draft).model_dump(mode="json")
    text = next(node for node in payload["nodes"] if node["kind"] == "text")
    text["text"] = "😀"
    for node in payload["nodes"]:
        if node["kind"] == "table":
            for cell in node["cells"]:
                cell["text"] = ""
    for disposition in payload["issue_dispositions"]:
        disposition["note"] = ""

    monkeypatch.setattr(review_draft_module, "_MAX_VIEW_TOTAL_CHARS", 2)
    assert _view_from_json(payload).nodes

    text["text"] = "😀a"
    with pytest.raises(ValidationError, match="too much text"):
        _view_from_json(payload)


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
def test_review_view_policy_boundary_cannot_be_overridden(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = build_candidate_review_draft_view(fixture := _fixture(tmp_path).draft)
    value_payload = payload.model_dump(mode="json")
    value_payload[field] = value

    with pytest.raises(ValidationError):
        _view_from_json(value_payload)

    assert fixture.golden_eligible is False


def test_applies_all_five_typed_patch_operations_as_new_revision(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    original_draft_bytes = _json_bytes(current.model_dump(mode="json"))
    draft_path.write_bytes(original_draft_bytes)
    output = tmp_path / "revisions" / "review-r2.json"
    before = _tree_hashes(fixture.root)
    request = _patch_request(
        {
            "schema_version": "candidate-review-patch/1.1",
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_text",
                    "node_id": "text-1",
                    "text": "검수자가 교정한 문장",
                },
                {
                    "op": "set_role",
                    "node_id": "text-1",
                    "role": "question",
                },
                {
                    "op": "set_table_cell_text",
                    "node_id": "table-1",
                    "row": 0,
                    "column": 0,
                    "row_span": 1,
                    "column_span": 1,
                    "text": "교정된 셀",
                },
                {
                    "op": "set_needs_review",
                    "node_id": "image-1",
                    "needs_review": True,
                },
                {
                    "op": "set_issue_disposition",
                    "issue_code": "rights_manifest_missing",
                    "disposition": "accepted_limitation",
                    "note": "권리 확인은 별도 단계에서 수행",
                },
            ],
        }
    )

    patched = apply_candidate_review_patch(
        fixture.root,
        draft_path,
        request,
        output,
        updated_at=datetime(2026, 9, 10, 15, 0, tzinfo=UTC),
    )

    assert load_candidate_review_draft(output) == patched
    assert patched.status == "in_progress"
    assert patched.draft_revision == 2
    assert patched.updated_at == "2026-09-10T15:00:00+00:00"
    assert patched.reviewed_content_ir.revision == current.reviewed_content_ir.revision
    expected_content = current.reviewed_content_ir.model_dump(mode="json")
    expected_nodes = {node["id"]: node for node in expected_content["nodes"]}
    expected_nodes["text-1"]["text"] = "검수자가 교정한 문장"
    expected_nodes["text-1"]["role"] = "question"
    expected_nodes["table-1"]["cells"][0]["text"] = "교정된 셀"
    expected_nodes["image-1"]["needs_review"] = True
    assert patched.reviewed_content_ir.model_dump(mode="json") == expected_content
    expected_plan = current.reviewed_hwp_document_plan.model_dump(mode="json")
    expected_plan["content_ir_sha256"] = contract_sha256(patched.reviewed_content_ir)
    assert patched.reviewed_hwp_document_plan.model_dump(mode="json") == expected_plan
    dispositions = {item.issue_code: item for item in patched.issue_dispositions}
    assert dispositions["rights_manifest_missing"].disposition == "accepted_limitation"
    assert dispositions["rights_manifest_missing"].note == "권리 확인은 별도 단계에서 수행"
    assert patched.golden_eligible is False
    assert patched.training_eligible is False
    assert patched.release_eligible is False
    assert patched.rights_status == "unverified"
    assert patched.identity_assurance == "self_asserted_untrusted"
    assert draft_path.read_bytes() == original_draft_bytes
    assert _tree_hashes(fixture.root) == before
    assert not list(output.parent.glob("*.partial"))


def test_patch_sets_registered_exact_image_grounding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-r1.fallback.json"
    output = tmp_path / "review-r2.grounded.json"
    before = _tree_hashes(fixture.root)
    current = start_candidate_review_draft(
        fixture.root,
        fixture.draft.document_id,
        "reviewer-local-2",
        draft_path,
    )
    original = draft_path.read_bytes()
    current_image = next(
        node for node in current.reviewed_content_ir.nodes if node.id == "image-1"
    )
    assert isinstance(current_image, ImageContentNode)
    assert current_image.asset_ref == "page-1"
    assert current_image.evidence_refs == ("obs-page-region",)

    patched = apply_candidate_review_patch(
        fixture.root,
        draft_path,
        _patch_request(
            {
                "schema_version": "candidate-review-patch/1.1",
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "set_image_grounding",
                        "node_id": "image-1",
                        "asset_ref": "crop-a",
                        "observation_ref": "obs-image",
                    }
                ],
            }
        ),
        output,
    )

    image = next(
        node for node in patched.reviewed_content_ir.nodes if node.id == "image-1"
    )
    assert isinstance(image, ImageContentNode)
    assert image.asset_ref == "crop-a"
    assert image.evidence_refs == ("obs-image",)
    assert patched.reviewed_hwp_document_plan.content_ir_sha256 == contract_sha256(
        patched.reviewed_content_ir
    )
    assert verify_candidate_review_draft(patched, candidate_root=fixture.root) == patched
    assert patched.golden_eligible is False
    assert patched.training_eligible is False
    assert patched.release_eligible is False
    assert patched.rights_status == "unverified"
    assert draft_path.read_bytes() == original
    assert _tree_hashes(fixture.root) == before


def test_drops_fallback_image_and_completes_verified_revision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-r1.fallback.json"
    patched_path = tmp_path / "review-r2.dropped.json"
    completed_path = tmp_path / "review-r3.complete.json"
    before = _tree_hashes(fixture.root)
    current = start_candidate_review_draft(
        fixture.root,
        fixture.draft.document_id,
        "reviewer-local-2",
        draft_path,
    )
    original = draft_path.read_bytes()
    operations: list[dict[str, object]] = [
        {"op": "drop_image_node", "node_id": "image-1"},
    ]
    for node in current.reviewed_content_ir.nodes:
        if node.id != "image-1":
            operations.append(
                {
                    "op": "set_needs_review",
                    "node_id": node.id,
                    "needs_review": False,
                }
            )
    for disposition in current.issue_dispositions:
        operations.append(
            {
                "op": "set_issue_disposition",
                "issue_code": disposition.issue_code,
                "disposition": "resolved",
            }
        )

    patched = apply_candidate_review_patch(
        fixture.root,
        draft_path,
        _patch_request(
            {
                "schema_version": "candidate-review-patch/1.1",
                "expected_draft_revision": 1,
                "operations": operations,
            }
        ),
        patched_path,
    )

    assert tuple(node.id for node in patched.reviewed_content_ir.nodes) == (
        "text-1",
        "table-1",
        "formula-1",
    )
    assert patched.reviewed_content_ir.reading_order == (
        "text-1",
        "table-1",
        "formula-1",
    )
    assert tuple(
        item.content_ref
        for item in patched.reviewed_hwp_document_plan.flow
        if isinstance(item, ContentPlanItem)
    ) == ("text-1", "table-1", "formula-1")
    assert patched.reviewed_hwp_document_plan.content_ir_sha256 == contract_sha256(
        patched.reviewed_content_ir
    )

    completed = complete_candidate_review_draft(
        fixture.root,
        patched_path,
        _completion_request({"expected_draft_revision": 2}),
        completed_path,
    )

    assert completed.status == "complete"
    assert completed.draft_revision == 3
    assert verify_candidate_review_draft(completed, candidate_root=fixture.root) == completed
    assert completed.golden_eligible is False
    assert completed.training_eligible is False
    assert completed.release_eligible is False
    assert completed.rights_status == "unverified"
    assert draft_path.read_bytes() == original
    assert _tree_hashes(fixture.root) == before


def test_complete_rejects_unchanged_region_fallback_image(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-r1.fallback.json"
    ready_path = tmp_path / "review-r2.ready.json"
    complete_path = tmp_path / "review-r3.complete.json"
    current = start_candidate_review_draft(
        fixture.root,
        fixture.draft.document_id,
        "reviewer-local-2",
        draft_path,
    )
    operations: list[dict[str, object]] = []
    for node in current.reviewed_content_ir.nodes:
        operations.append(
            {
                "op": "set_needs_review",
                "node_id": node.id,
                "needs_review": False,
            }
        )
    for disposition in current.issue_dispositions:
        operations.append(
            {
                "op": "set_issue_disposition",
                "issue_code": disposition.issue_code,
                "disposition": "resolved",
            }
        )
    ready = apply_candidate_review_patch(
        fixture.root,
        draft_path,
        _patch_request(
            {
                "expected_draft_revision": 1,
                "operations": operations,
            }
        ),
        ready_path,
    )
    assert verify_candidate_review_draft(ready, candidate_root=fixture.root) == ready

    with pytest.raises(
        ReviewDraftVerificationError,
        match="directly grounded by exactly one IMAGE",
    ):
        complete_candidate_review_draft(
            fixture.root,
            ready_path,
            _completion_request({"expected_draft_revision": 2}),
            complete_path,
        )

    assert not complete_path.exists()


def test_drop_image_rejects_final_content_node(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, image_only=True)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.json"
    before = _tree_hashes(fixture.root)

    with pytest.raises(ReviewDraftPatchError, match="final content node"):
        apply_candidate_review_patch(
            fixture.root,
            draft_path,
            _patch_request(
                {
                    "expected_draft_revision": 1,
                    "operations": [
                        {"op": "drop_image_node", "node_id": "image-1"}
                    ],
                }
            ),
            output,
        )

    assert not output.exists()
    assert _tree_hashes(fixture.root) == before


def test_patch_cli_reads_strict_request_from_stdin_and_reports_metadata_only(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.json"
    before = _tree_hashes(fixture.root)
    request = {
        "schema_version": "candidate-review-patch/1.1",
        "expected_draft_revision": 1,
        "operations": [
            {
                "op": "set_text",
                "node_id": "text-1",
                "text": "renderer-private-text",
            }
        ],
    }

    result = subprocess.run(
        [
            sys.executable,
            "ai/datasets/patch_candidate_review_draft.py",
            str(fixture.root),
            str(draft_path),
            "--expected-manifest-sha256",
            current.candidate_manifest_sha256,
            "--expected-lineage-id",
            current.lineage_id,
            "--out",
            str(output),
        ],
        input=json.dumps(request),
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=Path.cwd(),
    )
    report = json.loads(result.stdout)
    saved = load_candidate_review_draft(output)

    assert report == {
        "candidate_manifest_sha256": current.candidate_manifest_sha256,
        "document_id": current.document_id,
        "draft_revision": 2,
        "golden_eligible": False,
        "identity_assurance": "self_asserted_untrusted",
        "lineage_id": current.lineage_id,
        "release_eligible": False,
        "rights_status": "unverified",
        "schema_version": "candidate-review-draft/1.0",
        "status": "in_progress",
        "training_eligible": False,
    }
    assert str(fixture.root) not in result.stdout
    assert str(draft_path) not in result.stdout
    assert str(output) not in result.stdout
    assert "renderer-private-text" not in result.stdout
    assert saved.draft_revision == 2
    assert _tree_hashes(fixture.root) == before


def test_completes_ready_draft_as_new_immutable_revision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    original = _json_bytes(current.model_dump(mode="json"))
    draft_path.write_bytes(original)
    output = tmp_path / "revisions" / "review-r2.complete.json"
    before = _tree_hashes(fixture.root)

    completed = complete_candidate_review_draft(
        fixture.root,
        draft_path,
        _completion_request(
            {
                "schema_version": "candidate-review-completion/1.0",
                "expected_draft_revision": 1,
            }
        ),
        output,
        updated_at=datetime(2026, 9, 10, 16, 0, tzinfo=UTC),
    )

    assert load_candidate_review_draft(output) == completed
    assert completed.status == "complete"
    assert completed.draft_revision == 2
    assert completed.updated_at == "2026-09-10T16:00:00+00:00"
    assert completed.reviewed_content_ir == current.reviewed_content_ir
    assert completed.reviewed_content_ir_contract_sha256 == (
        current.reviewed_content_ir_contract_sha256
    )
    assert completed.reviewed_hwp_document_plan == current.reviewed_hwp_document_plan
    assert completed.reviewed_hwp_document_plan_contract_sha256 == (
        current.reviewed_hwp_document_plan_contract_sha256
    )
    assert completed.issue_dispositions == current.issue_dispositions
    assert completed.rights_status == "unverified"
    assert completed.golden_eligible is False
    assert completed.training_eligible is False
    assert completed.release_eligible is False
    assert draft_path.read_bytes() == original
    assert _tree_hashes(fixture.root) == before
    assert not list(output.parent.glob("*.partial"))


def test_completion_cli_reads_strict_request_and_reports_metadata_only(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.complete.json"
    before = _tree_hashes(fixture.root)

    result = subprocess.run(
        [
            sys.executable,
            "ai/datasets/complete_candidate_review_draft.py",
            str(fixture.root),
            str(draft_path),
            "--expected-manifest-sha256",
            current.candidate_manifest_sha256,
            "--expected-lineage-id",
            current.lineage_id,
            "--out",
            str(output),
        ],
        input=json.dumps({"expected_draft_revision": 1}),
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=Path.cwd(),
    )
    report = json.loads(result.stdout)
    saved = load_candidate_review_draft(output)

    assert report == {
        "candidate_manifest_sha256": current.candidate_manifest_sha256,
        "document_id": current.document_id,
        "draft_revision": 2,
        "golden_eligible": False,
        "identity_assurance": "self_asserted_untrusted",
        "lineage_id": current.lineage_id,
        "release_eligible": False,
        "rights_status": "unverified",
        "schema_version": "candidate-review-draft/1.0",
        "status": "complete",
        "training_eligible": False,
    }
    assert saved.status == "complete"
    assert saved.reviewed_content_ir == current.reviewed_content_ir
    assert str(fixture.root) not in result.stdout
    assert str(draft_path) not in result.stdout
    assert str(output) not in result.stdout
    assert _tree_hashes(fixture.root) == before


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_draft_revision": "1"},
        {"expected_draft_revision": True},
        {"expected_draft_revision": 0},
        {"expected_draft_revision": 1_000_001},
        {"expected_draft_revision": 1, "unexpected": True},
        {
            "schema_version": "candidate-review-completion/2.0",
            "expected_draft_revision": 1,
        },
    ],
)
def test_completion_request_is_strict(payload: object) -> None:
    with pytest.raises(ValidationError):
        _completion_request(payload)


def test_completion_rejects_stale_or_already_complete_without_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    in_progress = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(in_progress.model_dump(mode="json")))
    stale_output = tmp_path / "review-r2.stale.json"

    with pytest.raises(ReviewDraftCompletionError, match="stale review draft revision"):
        complete_candidate_review_draft(
            fixture.root,
            draft_path,
            _completion_request({"expected_draft_revision": 2}),
            stale_output,
        )
    assert not stale_output.exists()

    draft_path.write_bytes(_json_bytes(fixture.draft.model_dump(mode="json")))
    complete_output = tmp_path / "review-r2.already-complete.json"
    with pytest.raises(ReviewDraftCompletionError, match="already complete"):
        complete_candidate_review_draft(
            fixture.root,
            draft_path,
            _completion_request({"expected_draft_revision": 1}),
            complete_output,
        )
    assert not complete_output.exists()


def test_completion_rejects_pending_issue_or_review_node_without_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    request = _completion_request({"expected_draft_revision": 1})

    pending_value = current.model_dump(mode="json")
    pending_value["issue_dispositions"][0]["disposition"] = "pending"
    pending_path = tmp_path / "review-r1.pending.json"
    pending_path.write_bytes(_json_bytes(pending_value))
    pending_output = tmp_path / "review-r2.pending.json"
    with pytest.raises(ReviewDraftVerificationError):
        complete_candidate_review_draft(
            fixture.root,
            pending_path,
            request,
            pending_output,
        )
    assert not pending_output.exists()

    review_value = current.model_dump(mode="json")
    review_value["reviewed_content_ir"]["nodes"][0]["needs_review"] = True
    reviewed_content = ContentIR.model_validate(review_value["reviewed_content_ir"])
    review_value["reviewed_content_ir_contract_sha256"] = contract_sha256(reviewed_content)
    review_value["reviewed_hwp_document_plan"] = _plan(reviewed_content).model_dump(
        mode="json"
    )
    review_value["reviewed_hwp_document_plan_contract_sha256"] = contract_sha256(
        _plan(reviewed_content)
    )
    review_path = tmp_path / "review-r1.node.json"
    review_path.write_bytes(_json_bytes(review_value))
    review_output = tmp_path / "review-r2.node.json"
    with pytest.raises(ReviewDraftVerificationError, match="still need review"):
        complete_candidate_review_draft(
            fixture.root,
            review_path,
            request,
            review_output,
        )
    assert not review_output.exists()


def test_completion_preflights_view_and_enforces_new_external_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    request = _completion_request({"expected_draft_revision": 1})

    internal = fixture.root / "review-r2.complete.json"
    with pytest.raises(ValueError, match="outside candidate root"):
        complete_candidate_review_draft(fixture.root, draft_path, request, internal)
    assert not internal.exists()

    existing = tmp_path / "existing.complete.json"
    existing.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError, match="already exists"):
        complete_candidate_review_draft(fixture.root, draft_path, request, existing)
    assert existing.read_bytes() == b"keep-me"

    output = tmp_path / "review-r2.too-large.json"
    monkeypatch.setattr(review_draft_module, "_MAX_VIEW_TOTAL_CHARS", 10)
    with pytest.raises(ValidationError, match="too much text"):
        complete_candidate_review_draft(fixture.root, draft_path, request, output)
    assert not output.exists()


def test_patch_rejects_stale_revision_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.json"
    before = _tree_hashes(fixture.root)
    request = _patch_request(
        {
            "expected_draft_revision": 2,
            "operations": [
                {
                    "op": "drop_image_node",
                    "node_id": "image-1",
                }
            ],
        }
    )

    with pytest.raises(ReviewDraftPatchError, match="stale review draft revision"):
        apply_candidate_review_patch(fixture.root, draft_path, request, output)

    assert not output.exists()
    assert _tree_hashes(fixture.root) == before


def test_patch_rejects_completed_draft_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft_path = tmp_path / "review-complete.json"
    original = _json_bytes(fixture.draft.model_dump(mode="json"))
    draft_path.write_bytes(original)
    output = tmp_path / "review-r2.json"
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_needs_review",
                    "node_id": "text-1",
                    "needs_review": True,
                }
            ],
        }
    )

    with pytest.raises(ReviewDraftPatchError, match="completed"):
        apply_candidate_review_patch(fixture.root, draft_path, request, output)

    assert draft_path.read_bytes() == original
    assert not output.exists()


def test_patch_preflights_bounded_view_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.json"
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_needs_review",
                    "node_id": "text-1",
                    "needs_review": False,
                }
            ],
        }
    )
    monkeypatch.setattr(review_draft_module, "_MAX_VIEW_TOTAL_CHARS", 10)

    with pytest.raises(ValidationError, match="too much text"):
        apply_candidate_review_patch(fixture.root, draft_path, request, output)

    assert not output.exists()


def test_patch_rejects_mismatched_selected_snapshot_without_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.json"
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_needs_review",
                    "node_id": "text-1",
                    "needs_review": True,
                }
            ],
        }
    )

    with pytest.raises(ReviewDraftVerificationError, match="selected snapshot"):
        apply_candidate_review_patch(
            fixture.root,
            draft_path,
            request,
            output,
            expected_manifest_sha256="f" * 64,
            expected_lineage_id=current.lineage_id,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            {
                "op": "set_needs_review",
                "node_id": "missing-node",
                "needs_review": False,
            },
            "node does not exist",
        ),
        (
            {
                "op": "set_role",
                "node_id": "image-1",
                "role": "caption",
            },
            "requires a text node",
        ),
        (
            {
                "op": "set_image_grounding",
                "node_id": "text-1",
                "asset_ref": "crop-a",
                "observation_ref": "obs-image",
            },
            "requires an image node",
        ),
        (
            {
                "op": "drop_image_node",
                "node_id": "text-1",
            },
            "requires an image node",
        ),
        (
            {
                "op": "drop_image_node",
                "node_id": "missing-node",
            },
            "node does not exist",
        ),
        (
            {
                "op": "set_image_grounding",
                "node_id": "image-1",
                "asset_ref": "page-1",
                "observation_ref": "obs-page-region",
            },
            "directly grounded by exactly one IMAGE",
        ),
        (
            {
                "op": "set_image_grounding",
                "node_id": "image-1",
                "asset_ref": "crop-b",
                "observation_ref": "obs-image",
            },
            "directly grounded by exactly one IMAGE",
        ),
        (
            {
                "op": "set_image_grounding",
                "node_id": "image-1",
                "asset_ref": "crop-unregistered",
                "observation_ref": "obs-image-unregistered",
            },
            "source SHA is not registered",
        ),
        (
            {
                "op": "set_table_cell_text",
                "node_id": "table-1",
                "row": 0,
                "column": 0,
                "row_span": 2,
                "column_span": 1,
                "text": "wrong key",
            },
            "cell key does not exist exactly",
        ),
        (
            {
                "op": "set_issue_disposition",
                "issue_code": "not_registered",
                "disposition": "resolved",
            },
            "unknown issue dispositions",
        ),
    ],
)
def test_patch_rejects_invalid_target_without_output(
    tmp_path: Path,
    operation: dict[str, object],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    original = _json_bytes(current.model_dump(mode="json"))
    draft_path.write_bytes(original)
    output = tmp_path / "review-r2.json"
    before = _tree_hashes(fixture.root)
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [operation],
        }
    )

    with pytest.raises(ValueError, match=message):
        apply_candidate_review_patch(fixture.root, draft_path, request, output)

    assert not output.exists()
    assert draft_path.read_bytes() == original
    assert _tree_hashes(fixture.root) == before


def test_patch_rejects_candidate_internal_or_existing_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_needs_review",
                    "node_id": "text-1",
                    "needs_review": True,
                }
            ],
        }
    )
    before = _tree_hashes(fixture.root)

    internal = fixture.root / "review-r2.json"
    with pytest.raises(ValueError, match="outside candidate root"):
        apply_candidate_review_patch(fixture.root, draft_path, request, internal)
    assert not internal.exists()

    existing = tmp_path / "existing-r2.json"
    existing.write_bytes(b"keep-me")
    with pytest.raises(FileExistsError, match="already exists"):
        apply_candidate_review_patch(fixture.root, draft_path, request, existing)
    assert existing.read_bytes() == b"keep-me"
    assert _tree_hashes(fixture.root) == before


def test_patch_requires_timezone_aware_timestamp_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    current = _in_progress_draft(fixture)
    draft_path = tmp_path / "review-r1.json"
    draft_path.write_bytes(_json_bytes(current.model_dump(mode="json")))
    output = tmp_path / "review-r2.json"
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_needs_review",
                    "node_id": "text-1",
                    "needs_review": True,
                }
            ],
        }
    )

    naive_timestamp = datetime(2026, 9, 10, 15, 0, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone"):
        apply_candidate_review_patch(
            fixture.root,
            draft_path,
            request,
            output,
            updated_at=naive_timestamp,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "forbidden_op",
    [
        "set_confidence",
        "set_evidence_refs",
        "set_reading_order",
        "set_table_topology",
        "set_formula",
        "set_image_asset",
        "set_plan",
        "set_eligibility",
    ],
)
def test_patch_contract_exposes_no_arbitrary_mutation_operation(
    forbidden_op: str,
) -> None:
    with pytest.raises(ValidationError):
        _patch_request(
            {
                "expected_draft_revision": 1,
                "operations": [{"op": forbidden_op}],
            }
        )


def test_patch_request_rejects_version_1_0_boundary() -> None:
    with pytest.raises(ValidationError):
        _patch_request(
            {
                "schema_version": "candidate-review-patch/1.0",
                "expected_draft_revision": 1,
                "operations": [
                    {
                        "op": "set_image_grounding",
                        "node_id": "image-1",
                        "asset_ref": "crop-a",
                        "observation_ref": "obs-image",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "expected_draft_revision": "1",
            "operations": [{"op": "set_text", "node_id": "text-1", "text": "valid"}],
        },
        {
            "expected_draft_revision": 1,
            "operations": [{"op": "set_text", "node_id": "text-1", "text": ""}],
        },
        {
            "expected_draft_revision": 1,
            "operations": [{"op": "set_role", "node_id": "text-1", "role": "table"}],
        },
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_issue_disposition",
                    "issue_code": "rights_manifest_missing",
                    "disposition": "accepted_limitation",
                    "note": "   ",
                }
            ],
        },
        {
            "expected_draft_revision": 1,
            "operations": [
                {
                    "op": "set_needs_review",
                    "node_id": "text-1",
                    "needs_review": True,
                    "confidence": 1.0,
                }
            ],
        },
        {"expected_draft_revision": 1, "operations": []},
    ],
)
def test_patch_request_is_strict(payload: object) -> None:
    with pytest.raises(ValidationError):
        _patch_request(payload)


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "set_text", "node_id": "n" * 1_025, "text": "valid"},
        {"op": "set_text", "node_id": "text-1", "text": "x" * 1_000_001},
        {
            "op": "set_issue_disposition",
            "issue_code": "i" * 257,
            "disposition": "resolved",
        },
    ],
)
def test_patch_request_bounds_individual_fields(operation: object) -> None:
    with pytest.raises(ValidationError):
        _patch_request({"expected_draft_revision": 1, "operations": [operation]})


def test_patch_request_bounds_operation_count() -> None:
    operation = {
        "op": "set_needs_review",
        "node_id": "text-1",
        "needs_review": True,
    }

    with pytest.raises(ValidationError):
        _patch_request(
            {
                "expected_draft_revision": 1,
                "operations": [operation] * 1_001,
            }
        )


def test_patch_request_bounds_total_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review_draft_module, "_MAX_VIEW_TOTAL_CHARS", 15)

    with pytest.raises(ValidationError, match="total size limit"):
        _patch_request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {"op": "set_text", "node_id": "text-1", "text": "1234567890"}
                ],
            }
        )


def test_patch_request_counts_utf16_code_units_like_electron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at_field_limit = "😀" * 500_000
    request = _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [{"op": "set_text", "node_id": "n", "text": at_field_limit}],
        }
    )
    operation = request.operations[0]
    assert isinstance(operation, SetTextPatchOperation)
    assert operation.text == at_field_limit

    with pytest.raises(ValidationError, match="UTF-16"):
        _patch_request(
            {
                "expected_draft_revision": 1,
                "operations": [
                    {"op": "set_text", "node_id": "n", "text": at_field_limit + "a"}
                ],
            }
        )

    monkeypatch.setattr(review_draft_module, "_MAX_VIEW_TOTAL_CHARS", 3)
    _patch_request(
        {
            "expected_draft_revision": 1,
            "operations": [{"op": "set_text", "node_id": "n", "text": "😀"}],
        }
    )
    with pytest.raises(ValidationError, match="total size limit"):
        _patch_request(
            {
                "expected_draft_revision": 1,
                "operations": [{"op": "set_text", "node_id": "n", "text": "😀a"}],
            }
        )


def test_utf16_counter_rejects_unpaired_surrogates() -> None:
    for value in ("\ud800", "\udc00"):
        with pytest.raises(ValueError, match="unpaired UTF-16 surrogates"):
            review_draft_module._utf16_code_units(value)

    assert review_draft_module._utf16_code_units("😀") == 2


def test_patch_request_bounds_normalized_utf8_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "expected_draft_revision": 1,
        "operations": [{"op": "set_text", "node_id": "text-1", "text": "검수😀"}],
    }
    request_bytes = len(_patch_request(payload).model_dump_json().encode("utf-8"))

    monkeypatch.setattr(review_draft_module, "_MAX_PATCH_REQUEST_BYTES", request_bytes)
    _patch_request(payload)

    monkeypatch.setattr(review_draft_module, "_MAX_PATCH_REQUEST_BYTES", request_bytes - 1)
    with pytest.raises(ValidationError, match="serialized size limit"):
        _patch_request(payload)


@pytest.mark.parametrize("field", ["row", "column", "row_span", "column_span"])
def test_patch_and_view_table_coordinates_use_js_safe_integer_boundary(
    tmp_path: Path,
    field: str,
) -> None:
    operation = {
        "op": "set_table_cell_text",
        "node_id": "table-1",
        "row": 0,
        "column": 0,
        "row_span": 1,
        "column_span": 1,
        "text": "",
    }
    operation[field] = (1 << 53) - 1
    _patch_request({"expected_draft_revision": 1, "operations": [operation]})

    operation[field] = 1 << 53
    with pytest.raises(ValidationError, match="less than or equal to"):
        _patch_request({"expected_draft_revision": 1, "operations": [operation]})

    payload = build_candidate_review_draft_view(_fixture(tmp_path).draft).model_dump(mode="json")
    table = next(node for node in payload["nodes"] if node["kind"] == "table")
    table["cells"][0][field] = (1 << 53) - 1
    _view_from_json(payload)

    table["cells"][0][field] = 1 << 53
    with pytest.raises(ValidationError, match="less than or equal to"):
        _view_from_json(payload)


def test_patch_request_revision_matches_electron_upper_bound() -> None:
    payload = {
        "expected_draft_revision": 1_000_000,
        "operations": [
            {"op": "set_needs_review", "node_id": "text-1", "needs_review": True}
        ],
    }
    _patch_request(payload)

    payload["expected_draft_revision"] = 1_000_001
    with pytest.raises(ValidationError, match="less than or equal to 1000000"):
        _patch_request(payload)


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
def test_policy_boundary_cannot_be_overridden(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _fixture(tmp_path).draft.model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError):
        _draft_from_json(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("draft_revision", 0, "greater than or equal to 1"),
        ("reviewer_label", "reviewer/private", "String should match pattern"),
        ("updated_at", "2026-09-10T13:30:00", "timezone"),
        ("updated_at", "not-a-timestamp", "ISO-8601"),
    ],
)
def test_rejects_invalid_autosave_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _fixture(tmp_path).draft.model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        _draft_from_json(payload)


def test_complete_requires_every_registered_issue_disposition(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture.draft.model_dump(mode="json")
    payload["issue_dispositions"] = payload["issue_dispositions"][:-1]
    incomplete = _draft_from_json(payload)

    with pytest.raises(ReviewDraftVerificationError, match="missing issue dispositions"):
        verify_candidate_review_draft(incomplete, candidate_root=fixture.root)

    payload["status"] = "in_progress"
    in_progress = _draft_from_json(payload)
    assert (
        verify_candidate_review_draft(in_progress, candidate_root=fixture.root).status
        == "in_progress"
    )


def test_complete_rejects_pending_disposition_and_review_nodes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    pending = fixture.draft.model_dump(mode="json")
    pending["issue_dispositions"][0]["disposition"] = "pending"
    with pytest.raises(ValidationError, match="pending"):
        _draft_from_json(pending)

    content_value = fixture.draft.reviewed_content_ir.model_dump(mode="json")
    content_value["nodes"][0]["needs_review"] = True
    content = ContentIR.model_validate(content_value)
    draft = _replace_reviewed_content(fixture, content)
    with pytest.raises(ReviewDraftVerificationError, match="still need review"):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


def test_candidate_manifest_requires_one_registered_image_per_page(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["artifacts"]["page_images"] = []
    fixture.manifest_path.write_bytes(_json_bytes(manifest))
    output = tmp_path / "review.json"

    with pytest.raises(ReviewDraftVerificationError, match="manifest.*strict JSON"):
        start_candidate_review_draft(
            fixture.root,
            fixture.draft.document_id,
            "reviewer-local-2",
            output,
        )

    assert not output.exists()


def test_candidate_verification_bounds_manifest_and_binary_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(review_draft_module, "_MAX_CANDIDATE_MANIFEST_BYTES", 1)
    with pytest.raises(ReviewDraftVerificationError, match="manifest exceeds"):
        verify_candidate_review_draft(fixture.draft, candidate_root=fixture.root)

    monkeypatch.setattr(
        review_draft_module,
        "_MAX_CANDIDATE_MANIFEST_BYTES",
        16 * 1024 * 1024,
    )
    monkeypatch.setattr(review_draft_module, "_MAX_BINARY_ARTIFACT_BYTES", 1)
    with pytest.raises(ReviewDraftVerificationError, match="artifact exceeds"):
        verify_candidate_review_draft(fixture.draft, candidate_root=fixture.root)


def test_candidate_verification_retains_only_contract_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    observed: dict[str, bool] = {}
    original = review_draft_module._read_and_verify_registered_artifact

    def record_retention(
        root: Path,
        binding: CandidateArtifactBinding,
        *,
        max_bytes: int,
        retain_payload: bool,
    ) -> tuple[bytes | None, int]:
        observed[binding.path] = retain_payload
        return original(
            root,
            binding,
            max_bytes=max_bytes,
            retain_payload=retain_payload,
        )

    monkeypatch.setattr(
        review_draft_module,
        "_read_and_verify_registered_artifact",
        record_retention,
    )
    verify_candidate_review_draft(fixture.draft, candidate_root=fixture.root)

    contract_paths = {
        fixture.draft.evidence_ir.path,
        fixture.draft.base_content_ir.path,
        fixture.draft.base_hwp_document_plan.path,
    }
    assert {path for path, retained in observed.items() if retained} == contract_paths
    assert all(
        not observed[binding["path"]]
        for binding in json.loads(
            fixture.manifest_path.read_text(encoding="utf-8")
        )["documents"][0]["artifacts"]["page_images"]
    )


def test_rejects_manifest_or_registered_artifact_digest_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.manifest_path.write_bytes(fixture.manifest_path.read_bytes() + b" ")
    with pytest.raises(ReviewDraftVerificationError, match="manifest SHA-256"):
        verify_candidate_review_draft(fixture.draft, candidate_root=fixture.root)

    fixture = _fixture(tmp_path / "artifact-case")
    content_path = fixture.root.joinpath(*fixture.draft.base_content_ir.path.split("/"))
    content_path.write_bytes(content_path.read_bytes() + b" ")
    with pytest.raises(ReviewDraftVerificationError, match="artifact SHA-256"):
        verify_candidate_review_draft(fixture.draft, candidate_root=fixture.root)


def test_rejects_unknown_candidate_manifest_fields(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_payload = _json_bytes(manifest)
    fixture.manifest_path.write_bytes(manifest_payload)
    draft_payload = fixture.draft.model_dump(mode="json")
    draft_payload["candidate_manifest_sha256"] = _sha256(manifest_payload)
    draft = _draft_from_json(draft_payload)

    with pytest.raises(ReviewDraftVerificationError, match="manifest.*strict JSON"):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


def test_rejects_schema_invalid_registered_content_even_when_rehashed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    content_binding = manifest["documents"][0]["artifacts"]["content_ir_candidate"]
    content_path = fixture.root.joinpath(*content_binding["path"].split("/"))
    content_payload = json.loads(content_path.read_text(encoding="utf-8"))
    content_payload["unexpected"] = True
    content_bytes = _json_bytes(content_payload)
    content_path.write_bytes(content_bytes)
    content_binding["sha256"] = _sha256(content_bytes)
    manifest_payload = _json_bytes(manifest)
    fixture.manifest_path.write_bytes(manifest_payload)
    draft_payload = fixture.draft.model_dump(mode="json")
    draft_payload["candidate_manifest_sha256"] = _sha256(manifest_payload)
    draft_payload["base_content_ir"]["sha256"] = _sha256(content_bytes)
    draft = _draft_from_json(draft_payload)

    with pytest.raises(ReviewDraftVerificationError, match="base ContentIR.*strict JSON"):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("revision", "revision must increment"),
        ("node_id", "node ids must be preserved"),
        ("node_kind", "node kind must be preserved"),
        ("evidence_ref", "evidence refs must be preserved"),
        ("table_cell_evidence", "table cell evidence refs must be preserved"),
        ("table_topology", "table topology must be preserved"),
        ("reading_order", "reading order and metadata must be preserved"),
        ("confidence", "only change typed review fields"),
        ("formula_expression", "only change typed review fields"),
    ],
)
def test_rejects_forbidden_base_to_reviewed_changes(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.draft.reviewed_content_ir.model_dump(mode="json")
    nodes = {node["id"]: node for node in value["nodes"]}
    if mutation == "revision":
        value["revision"] = 3
    elif mutation == "node_id":
        nodes["text-1"]["id"] = "text-renamed"
        value["reading_order"][0] = "text-renamed"
    elif mutation == "node_kind":
        nodes["text-1"].clear()
        nodes["text-1"].update(
            {
                "id": "text-1",
                "kind": "formula",
                "expression": "x+1",
                "format": "latex",
                "evidence_refs": ["obs-text"],
                "confidence": 1.0,
                "needs_review": False,
            }
        )
    elif mutation == "evidence_ref":
        nodes["text-1"]["evidence_refs"] = ["obs-image"]
    elif mutation == "table_cell_evidence":
        nodes["table-1"]["cells"][0]["evidence_refs"] = ["obs-image"]
    elif mutation == "table_topology":
        table = nodes["table-1"]
        table["rows"] = 1
        table["columns"] = 4
        for column, cell in enumerate(table["cells"]):
            cell.update({"row": 0, "column": column, "row_span": 1, "column_span": 1})
    elif mutation == "reading_order":
        value["reading_order"] = list(reversed(value["reading_order"]))
    elif mutation == "confidence":
        nodes["text-1"]["confidence"] = 0.5
    else:
        nodes["formula-1"]["expression"] = "x+2"
    reviewed = ContentIR.model_validate(value)
    draft = _replace_reviewed_content(fixture, reviewed)

    with pytest.raises(ReviewDraftVerificationError, match=message):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


def test_accepts_exact_image_regrounding_in_reviewed_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.draft.reviewed_content_ir.model_dump(mode="json")
    image = next(node for node in value["nodes"] if node["id"] == "image-1")
    assert image["asset_ref"] == "crop-a"
    image["asset_ref"] = "crop-b"
    image["evidence_refs"] = ["obs-image-b"]
    reviewed = ContentIR.model_validate(value)
    draft = _replace_reviewed_content(fixture, reviewed)

    assert verify_candidate_review_draft(draft, candidate_root=fixture.root) == draft


def test_rejects_image_asset_unrelated_to_selected_observation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.draft.reviewed_content_ir.model_dump(mode="json")
    image = next(node for node in value["nodes"] if node["id"] == "image-1")
    image["asset_ref"] = "crop-b"
    reviewed = ContentIR.model_validate(value)
    draft = _replace_reviewed_content(fixture, reviewed)

    with pytest.raises(
        ReviewDraftVerificationError,
        match="directly grounded by exactly one IMAGE",
    ):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


def test_rejects_image_regrounding_to_text_observation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.draft.reviewed_content_ir.model_dump(mode="json")
    image = next(node for node in value["nodes"] if node["id"] == "image-1")
    image["asset_ref"] = "crop-a"
    image["evidence_refs"] = ["obs-text"]
    reviewed = ContentIR.model_validate(value)
    draft = _replace_reviewed_content(fixture, reviewed)

    with pytest.raises(
        ReviewDraftVerificationError,
        match="directly grounded by exactly one IMAGE",
    ):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


def test_rejects_reviewed_plan_design_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.draft.reviewed_hwp_document_plan.model_dump(mode="json")
    value["page_layout"]["margin_top_mm"] = 20.0
    reviewed_plan = HwpDocumentPlan.model_validate(value)
    draft_value = fixture.draft.model_dump(mode="json")
    draft_value["reviewed_hwp_document_plan"] = reviewed_plan.model_dump(mode="json")
    draft_value["reviewed_hwp_document_plan_contract_sha256"] = contract_sha256(reviewed_plan)
    draft = _draft_from_json(draft_value)

    with pytest.raises(ReviewDraftVerificationError, match="may only change"):
        verify_candidate_review_draft(draft, candidate_root=fixture.root)


def test_rejects_symlinked_registered_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence_path = fixture.root.joinpath(*fixture.draft.evidence_ir.path.split("/"))
    outside = tmp_path / "outside-evidence.json"
    outside.write_bytes(evidence_path.read_bytes())
    evidence_path.unlink()
    try:
        evidence_path.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ReviewDraftVerificationError, match="symlink"):
        verify_candidate_review_draft(fixture.draft, candidate_root=fixture.root)
