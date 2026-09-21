from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import scan2hwpx.evaluation.candidate_to_model_b_handoff as handoff_module
from scan2hwpx.contracts import (
    ContentIR,
    EvidenceIR,
    HwpDocumentPlan,
    contract_sha256,
)
from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    CandidateToModelBHandoffError,
    model_b_plan_review_draft_from_verified_handoff,
    reopen_candidate_to_model_b_handoff,
    start_candidate_to_model_b_handoff,
)
from scan2hwpx.evaluation.model_b_grounding_sidecars import (
    build_model_b_grounding_sidecars,
)
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBPlanReviewPatchRequest,
    apply_model_b_plan_review_patch,
)
from scan2hwpx.evaluation.review_draft import (
    CandidateContractArtifactBinding,
    CandidateReviewDraft,
    IssueDisposition,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _artifact_binding(root: Path, relative: str, payload: bytes) -> dict[str, str]:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": _sha256(payload)}


def _contract_binding(
    root: Path,
    relative: str,
    value: EvidenceIR | ContentIR | HwpDocumentPlan,
) -> dict[str, str]:
    binding = _artifact_binding(
        root,
        relative,
        _json_bytes(value.model_dump(mode="json")),
    )
    binding["contract_sha256"] = contract_sha256(value)
    return binding


def _evidence(pdf_sha256: str, page_sha256: str) -> EvidenceIR:
    return EvidenceIR.model_validate(
        {
            "id": "evidence-1",
            "source_document_sha256": pdf_sha256,
            "sources": [
                {
                    "id": "input-pdf",
                    "kind": "original_document",
                    "artifact_ref": "blob://fixture/source.pdf",
                    "producer": "fixture",
                    "sha256": pdf_sha256,
                },
                {
                    "id": "page-1",
                    "kind": "page_image",
                    "artifact_ref": "blob://fixture/page-0001.png",
                    "producer": "fixture",
                    "sha256": page_sha256,
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
                            "confidence": 0.5,
                            "source_refs": ["page-1"],
                            "ocr_candidates": [
                                {
                                    "text": "OCR 초안",
                                    "provider": "fixture",
                                    "confidence": 0.5,
                                    "source_ref": "page-1",
                                    "selected": True,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def _content(evidence: EvidenceIR, *, reviewed: bool) -> ContentIR:
    return ContentIR.model_validate(
        {
            "id": "content-1",
            "evidence_ir_id": evidence.id,
            "evidence_ir_sha256": contract_sha256(evidence),
            "revision": 2 if reviewed else 1,
            "nodes": [
                {
                    "id": "text-1",
                    "kind": "text",
                    "role": "question",
                    "text": "사람이 교정한 본문" if reviewed else "OCR 초안",
                    "evidence_refs": ["obs-text"],
                    "confidence": 0.5,
                    "needs_review": not reviewed,
                }
            ],
            "reading_order": ["text-1"],
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
            "design_profile_id": "fixture-v1",
            "official_spec_refs": ["official-hwpx-fixture"],
            "page_layout": {
                "width_mm": 210.0,
                "height_mm": 297.0,
                "margin_top_mm": 15.0,
                "margin_right_mm": 15.0,
                "margin_bottom_mm": 15.0,
                "margin_left_mm": 15.0,
            },
            "flow": [
                {
                    "id": "flow-1",
                    "kind": "content",
                    "render_as": "paragraph",
                    "content_ref": "text-1",
                }
            ],
        }
    )


@dataclass(frozen=True)
class _Fixture:
    candidate_root: Path
    review_path: Path
    sidecar_root: Path
    corpus_path: Path
    capability_path: Path
    review: CandidateReviewDraft
    base_content: ContentIR


def _fixture(tmp_path: Path, *, name: str = "fixture") -> _Fixture:
    workspace = tmp_path / name
    capability_value = {
        "schema_version": "1.0",
        "profile_id": "exam2hwpx-authoring-v1",
        "target_format": "hwpx",
        "knowledge_manifest": "sources.json",
        "content_policy": "immutable",
        "layout_policy": "clean_reauthor",
        "planner_may_emit": ["paragraph", "table"],
        "planner_must_reference_content": True,
        "compiler_owns": ["xml", "relationships"],
        "planner_forbidden": ["raw_xml", "local_file_path", "external_url"],
        "requires_review": ["generated_body_text"],
        "retrieval_routes": {
            "package": ["hwpx", "package"],
            "validation": ["hwpx", "validation"],
        },
    }
    capability_path = workspace / "capability.json"
    capability_payload = _write_json(capability_path, capability_value)
    chunk = {
        "id": "official-hwpx-fixture",
        "source_id": "hancom-hwpx-format",
        "title": "HWPX package paragraph table validation",
        "section": "manifest header style reference",
        "text": "HWPX package manifest header section paragraph table style validation DVC.",
        "tags": [
            "hwpx",
            "package",
            "paragraph",
            "table",
            "style-reference",
            "validation",
        ],
        "source_url": "https://tech.hancom.com/hwpxformat/",
        "source_sha256": "f" * 64,
    }
    corpus_path = workspace / "chunks.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_payload = (
        json.dumps(chunk, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    corpus_path.write_bytes(corpus_payload)

    candidate_root = workspace / "candidates"
    candidate_root.mkdir()
    lineage = _sha256(f"{name}-source".encode())
    document_id = f"candidate-{lineage[:24]}"
    pdf_sha256 = _sha256(f"{name}-pdf".encode())
    page_payload = b"fixture-page"
    evidence = _evidence(pdf_sha256, _sha256(page_payload))
    base_content = _content(evidence, reviewed=False)
    reviewed_content = _content(evidence, reviewed=True)
    base_plan = _plan(base_content)
    reviewed_plan = _plan(reviewed_content)
    prefix = f"{lineage}/"
    projection = _artifact_binding(
        candidate_root,
        prefix + "projection_source.json",
        b"{}\n",
    )
    evidence_binding = _contract_binding(
        candidate_root,
        prefix + "evidence_ir.json",
        evidence,
    )
    content_binding = _contract_binding(
        candidate_root,
        prefix + "content_ir.candidate.json",
        base_content,
    )
    plan_binding = _contract_binding(
        candidate_root,
        prefix + "hwp_document_plan.candidate.json",
        base_plan,
    )
    report = _artifact_binding(candidate_root, prefix + "report.json", b"{}\n")
    page = _artifact_binding(
        candidate_root,
        prefix + "pages/page-0001.png",
        page_payload,
    )
    issue_code = "human_review_required"
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
            "name": "fixture-candidate-builder",
            "version": "1",
            "pymupdf_version": "1.26",
            "dpi": 300,
            "pdf_text_order": "native",
        },
        "capability_profile": {
            "id": "exam2hwpx-authoring-v1",
            "sha256": _sha256(capability_payload),
        },
        "knowledge_corpus_sha256": _sha256(corpus_payload),
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
        "issue_counts": {issue_code: 1},
        "documents": [
            {
                "document_id": document_id,
                "lineage_id": lineage,
                "split": "train",
                "page_count": 1,
                "status": "needs_human_review",
                "source_artifacts": {
                    "hwp": {"sha256": lineage},
                    "hwpx": {"sha256": _sha256(f"{name}-hwpx".encode())},
                    "pdf": {"sha256": pdf_sha256},
                },
                "artifacts": {
                    "projection_source": projection,
                    "evidence_ir": evidence_binding,
                    "content_ir_candidate": content_binding,
                    "hwp_document_plan_candidate": plan_binding,
                    "report": report,
                    "page_images": [page],
                    "image_crops": [],
                },
                "issue_codes": [issue_code],
            }
        ],
    }
    manifest_payload = _write_json(candidate_root / "manifest.json", manifest)
    review = CandidateReviewDraft(
        status="complete",
        draft_revision=3,
        reviewer_label="candidate-reviewer",
        updated_at="2026-09-10T10:00:00+09:00",
        candidate_manifest_sha256=_sha256(manifest_payload),
        document_id=document_id,
        lineage_id=lineage,
        source_pdf_sha256=pdf_sha256,
        evidence_ir=CandidateContractArtifactBinding.model_validate(evidence_binding),
        base_content_ir=CandidateContractArtifactBinding.model_validate(content_binding),
        base_hwp_document_plan=CandidateContractArtifactBinding.model_validate(
            plan_binding
        ),
        reviewed_content_ir=reviewed_content,
        reviewed_content_ir_contract_sha256=contract_sha256(reviewed_content),
        reviewed_hwp_document_plan=reviewed_plan,
        reviewed_hwp_document_plan_contract_sha256=contract_sha256(reviewed_plan),
        issue_dispositions=(
            IssueDisposition(issue_code=issue_code, disposition="resolved"),
        ),
    )
    review_path = workspace / "candidate-review-r3.json"
    review_path.write_bytes(_json_bytes(review.model_dump(mode="json")))
    sidecar_root = tmp_path / f"{name}-sidecars"
    build_model_b_grounding_sidecars(
        candidate_root,
        corpus_path,
        capability_path,
        sidecar_root,
        max_workers=1,
    )
    return _Fixture(
        candidate_root=candidate_root,
        review_path=review_path,
        sidecar_root=sidecar_root,
        corpus_path=corpus_path,
        capability_path=capability_path,
        review=review,
        base_content=base_content,
    )


def _start(fixture: _Fixture, output: Path):
    return start_candidate_to_model_b_handoff(
        fixture.candidate_root,
        fixture.review_path,
        fixture.sidecar_root,
        fixture.corpus_path,
        fixture.capability_path,
        reviewer_label="model-b-reviewer",
        output_path=output,
        updated_at=datetime(2026, 9, 10, 2, 0, tzinfo=UTC),
    )


def test_handoff_uses_completed_reviewed_content_and_is_reopenable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"
    verified = _start(fixture, output)
    envelope = verified.envelope
    model_b = model_b_plan_review_draft_from_verified_handoff(verified)

    assert output.is_file()
    assert verified.artifact_sha256 == _sha256(output.read_bytes())
    assert model_b.content_ir == fixture.review.reviewed_content_ir
    assert model_b.content_ir != fixture.base_content
    assert model_b.base_hwp_document_plan == fixture.review.reviewed_hwp_document_plan
    assert envelope.completed_candidate_review.artifact_sha256 == _sha256(
        fixture.review_path.read_bytes()
    )
    assert envelope.completed_candidate_review.draft_revision == 3
    assert envelope.completed_candidate_review.candidate_manifest_sha256 == (
        fixture.review.candidate_manifest_sha256
    )
    assert envelope.grounding_sidecar.manifest_sha256 == _sha256(
        (fixture.sidecar_root / "manifest.json").read_bytes()
    )
    assert envelope.golden_eligible is False
    assert envelope.training_eligible is False
    assert envelope.release_eligible is False
    assert envelope.rights_status == "unverified"
    assert envelope.identity_assurance == "self_asserted_untrusted"

    reopened = reopen_candidate_to_model_b_handoff(
        output,
        fixture.candidate_root,
        fixture.review_path,
        fixture.sidecar_root,
        fixture.corpus_path,
        fixture.capability_path,
    )
    assert reopened.envelope == envelope
    assert reopened.artifact_sha256 == verified.artifact_sha256

    request = ModelBPlanReviewPatchRequest(
        expected_draft_revision=model_b.draft_revision,
        expected_source_candidate_contract_sha256=(
            model_b.source_candidate_contract_sha256
        ),
        expected_grounding_evidence_contract_sha256=(
            model_b.grounding_evidence_contract_sha256
        ),
        expected_reviewed_hwp_document_plan_contract_sha256=(
            model_b.reviewed_hwp_document_plan_contract_sha256
        ),
        operations=(
            {
                "op": "set_page_layout",
                "page_layout": {
                    **model_b.reviewed_hwp_document_plan.page_layout.model_dump(
                        mode="python"
                    ),
                    "margin_top_mm": 18.0,
                },
            },
        ),
    )
    patched = apply_model_b_plan_review_patch(model_b, request)
    assert patched.draft_revision == 2
    assert patched.content_ir == fixture.review.reviewed_content_ir


def test_rejects_incomplete_candidate_review_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.review.model_dump(mode="json")
    value["status"] = "in_progress"
    fixture.review_path.write_bytes(_json_bytes(value))
    output = tmp_path / "handoff.json"

    with pytest.raises(CandidateToModelBHandoffError, match="must be complete"):
        _start(fixture, output)

    assert not output.exists()


def test_rejects_sidecar_from_a_different_candidate(tmp_path: Path) -> None:
    first = _fixture(tmp_path, name="first")
    second = _fixture(tmp_path, name="second")
    output = tmp_path / "handoff.json"

    with pytest.raises(ValueError, match="candidate manifest binding"):
        start_candidate_to_model_b_handoff(
            first.candidate_root,
            first.review_path,
            second.sidecar_root,
            first.corpus_path,
            first.capability_path,
            reviewer_label="model-b-reviewer",
            output_path=output,
        )

    assert not output.exists()


def test_reopen_rejects_upstream_review_substitution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"
    _start(fixture, output)
    changed = fixture.review.model_dump(mode="json")
    changed["reviewer_label"] = "different-reviewer"
    fixture.review_path.write_bytes(_json_bytes(changed))

    with pytest.raises(CandidateToModelBHandoffError, match="provenance"):
        reopen_candidate_to_model_b_handoff(
            output,
            fixture.candidate_root,
            fixture.review_path,
            fixture.sidecar_root,
            fixture.corpus_path,
            fixture.capability_path,
        )


@pytest.mark.parametrize("tamper", ["completed", "edited_initial_plan"])
def test_reopen_rejects_noninitial_embedded_model_b_revision(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"
    _start(fixture, output)
    value = json.loads(output.read_text(encoding="utf-8"))
    draft = value["model_b_plan_review_draft"]
    if tamper == "completed":
        draft["status"] = "complete"
        draft["draft_revision"] = 2
    else:
        reviewed = draft["reviewed_hwp_document_plan"]
        reviewed["page_layout"]["margin_top_mm"] = 19.0
        reviewed_plan = HwpDocumentPlan.model_validate_json(
            json.dumps(reviewed, ensure_ascii=False),
            strict=True,
        )
        draft["reviewed_hwp_document_plan_contract_sha256"] = contract_sha256(
            reviewed_plan
        )
    output.write_bytes(_json_bytes(value))

    with pytest.raises(CandidateToModelBHandoffError, match="not valid strict JSON"):
        reopen_candidate_to_model_b_handoff(
            output,
            fixture.candidate_root,
            fixture.review_path,
            fixture.sidecar_root,
            fixture.corpus_path,
            fixture.capability_path,
        )


def test_detects_input_change_between_preflight_and_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"
    original = handoff_module._load_source_context
    calls = 0

    def mutating_load(*args: object, **kwargs: object):
        nonlocal calls
        context = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            changed = fixture.review.model_dump(mode="json")
            changed["reviewer_label"] = "swapped-reviewer"
            fixture.review_path.write_bytes(_json_bytes(changed))
        return context

    monkeypatch.setattr(handoff_module, "_load_source_context", mutating_load)

    with pytest.raises(CandidateToModelBHandoffError, match="inputs changed"):
        _start(fixture, output)

    assert not output.exists()


def test_create_only_rejects_existing_path_and_concurrent_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _start(fixture, existing)
    assert existing.read_text(encoding="utf-8") == "keep"

    raced = tmp_path / "raced.json"

    def racing_publish(source: Path, destination: Path) -> None:
        assert source.is_file()
        destination.write_text("racer", encoding="utf-8")
        raise FileExistsError("output appeared during build")

    monkeypatch.setattr(handoff_module, "publish_file_create_only", racing_publish)
    with pytest.raises(FileExistsError, match="already exists"):
        _start(fixture, raced)
    assert raced.read_text(encoding="utf-8") == "racer"
    assert not list(tmp_path.glob(".raced.json.*.partial"))


def test_handoff_does_not_report_success_when_staging_changes_during_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"

    def reject_changed_staging(source: Path, destination: Path) -> None:
        raise handoff_module.SafeArtifactIOError("staged artifact changed")

    monkeypatch.setattr(
        handoff_module,
        "publish_file_create_only",
        reject_changed_staging,
    )
    with pytest.raises(CandidateToModelBHandoffError, match="changed during publication"):
        _start(fixture, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".handoff.json.*.partial"))


def test_successful_handoff_does_not_touch_recreated_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"
    real_publish = handoff_module.publish_file_create_only
    recreated: Path | None = None

    def publish_then_recreate(source: Path, destination: Path) -> None:
        nonlocal recreated
        real_publish(source, destination)
        source.mkdir()
        (source / "keep.txt").write_text("keep", encoding="utf-8")
        recreated = source

    monkeypatch.setattr(
        handoff_module,
        "publish_file_create_only",
        publish_then_recreate,
    )

    verified = _start(fixture, output)

    assert verified.artifact_sha256 == _sha256(output.read_bytes())
    assert recreated is not None
    assert (recreated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_completed_review_descriptor_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    replacement = tmp_path / "replacement.json"
    changed = fixture.review.model_dump(mode="json")
    changed["reviewer_label"] = "replacement-reviewer"
    replacement.write_bytes(_json_bytes(changed))
    original_open = handoff_module.os.open
    swapped = False

    def swapping_open(path: object, flags: int) -> int:
        nonlocal swapped
        if not swapped and Path(os.fspath(path)) == fixture.review_path:
            swapped = True
            fixture.review_path.unlink()
            replacement.replace(fixture.review_path)
        return original_open(path, flags)

    monkeypatch.setattr(handoff_module.os, "open", swapping_open)
    with pytest.raises(CandidateToModelBHandoffError, match="changed while it was read"):
        _start(fixture, tmp_path / "handoff.json")


def test_start_handoff_cli_reports_only_noneligible_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff.json"
    command = [
        sys.executable,
        "ai/datasets/start_candidate_to_model_b_handoff.py",
        str(fixture.candidate_root),
        str(fixture.review_path),
        str(fixture.sidecar_root),
        str(fixture.corpus_path),
        str(fixture.capability_path),
        "--reviewer-label",
        "model-b-reviewer",
        "--out",
        str(output),
        "--expected-review-sha256",
        _sha256(fixture.review_path.read_bytes()),
        "--expected-review-revision",
        "3",
        "--expected-manifest-sha256",
        fixture.review.candidate_manifest_sha256,
        "--expected-lineage-id",
        fixture.review.lineage_id,
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(completed.stdout)

    assert value["artifact_sha256"] == _sha256(output.read_bytes())
    assert value["candidate_review_revision"] == 3
    assert value["model_b_draft_revision"] == 1
    assert value["golden_eligible"] is False
    assert value["training_eligible"] is False
    assert value["release_eligible"] is False
    assert value["rights_status"] == "unverified"
    assert value["identity_assurance"] == "self_asserted_untrusted"
    assert "model_b_plan_review_draft" not in value
