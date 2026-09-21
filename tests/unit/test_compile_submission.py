from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import pytest
from PIL import Image

import scan2hwpx.evaluation.compile_submission as submission
from scan2hwpx.contracts import (
    REQUIRED_DOCUMENT_QUALITY_CHECKS_V1,
    BBox,
    CheckDecision,
    ContentIR,
    ContentPlanItem,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    FormulaContentNode,
    FormulaFormat,
    HwpDocumentPlan,
    ImageContentNode,
    LayoutIntent,
    ObservationKind,
    PageLayoutIntent,
    QualityOutcome,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    CandidateToModelBHandoff,
    CompletedCandidateReviewBinding,
    GroundingSidecarHandoffBinding,
    VerifiedCandidateToModelBHandoff,
)
from scan2hwpx.evaluation.model_b_grounding_sidecars import retrieved_chunk_sha256
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBOfficialSpecRefEvidence,
    ModelBPlanGroundingEvidence,
    ModelBPlanReviewCompletionRequest,
    ModelBPlanReviewDraft,
    complete_model_b_plan_review_draft,
    start_model_b_plan_review_draft,
)
from scan2hwpx.evaluation.review_draft import (
    CandidateContractArtifactBinding,
    CandidateReviewDraft,
    VerifiedCandidateCompileInputs,
)
from scan2hwpx.hwpx.assets import ImageAssetBundle, PngImageAsset, build_image_asset_bundle
from scan2hwpx.hwpx.compiler import (
    IMAGE_DESIGN_PROFILE_ID,
    PlanCompilationError,
    PlanCompileResult,
    compile_plan_hwpx,
)
from scan2hwpx.hwpx.validate import validate_hwpx
from scan2hwpx.knowledge.hancom import HancomChunk
from scan2hwpx.model_b import (
    BuiltModelBInferenceRequest,
    BuiltModelBInferenceResult,
    ModelBModelArtifact,
    execute_model_b_plan,
    start_model_b_plan_review_from_inference,
    validate_model_b_inference_response,
)


@dataclass(frozen=True)
class _Fixture:
    candidate_root: Path
    candidate: CandidateReviewDraft
    evidence: EvidenceIR
    verified_handoff: VerifiedCandidateToModelBHandoff
    initial_model_b: ModelBPlanReviewDraft
    completed_model_b: ModelBPlanReviewDraft
    inference_result: BuiltModelBInferenceResult
    materialized: VerifiedCandidateCompileInputs


class _PlanProvider:
    def generate(self, request: BuiltModelBInferenceRequest, /) -> bytes:
        value = request.design_seed_plan.model_dump(mode="python")
        value["id"] = "model-b-generated-plan"
        for index, item in enumerate(value["flow"], start=1):
            item["id"] = f"model-b-flow-{index}"
        plan = HwpDocumentPlan.model_validate(value, strict=True)
        return json.dumps(
            plan.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")


def _capability_bytes() -> bytes:
    return json.dumps(
        {
            "schema_version": "1.0",
            "profile_id": "exam2hwpx-authoring-v1",
            "target_format": "hwpx",
            "knowledge_manifest": "sources.json",
            "content_policy": "immutable",
            "layout_policy": "clean_reauthor",
            "planner_may_emit": [
                "paragraph",
                "table",
                "image",
                "formula",
                "page_break",
                "column_break",
            ],
            "planner_must_reference_content": True,
            "compiler_owns": ["xml", "relationships"],
            "planner_forbidden": [
                "raw_xml",
                "local_file_path",
                "external_url",
                "macro",
                "script",
                "automation_command",
            ],
            "requires_review": ["generated_body_text"],
            "retrieval_routes": {"styles": ["hwpx", "paragraph"]},
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _handoff_sha256(envelope: CandidateToModelBHandoff) -> str:
    payload = (
        json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    content_kind: Literal["text", "image", "formula"] = "text",
) -> _Fixture:
    candidate_root = tmp_path / f"candidate-{content_kind}"
    candidate_root.mkdir()
    lineage_id = hashlib.sha256(f"lineage-{content_kind}".encode()).hexdigest()
    document_id = f"document-{content_kind}"
    pdf_sha256 = hashlib.sha256(f"pdf-{content_kind}".encode()).hexdigest()
    image_payload = _png()
    page_sha256 = hashlib.sha256(image_payload).hexdigest()
    observation_kind = ObservationKind.IMAGE if content_kind == "image" else ObservationKind.REGION
    evidence = EvidenceIR(
        id=f"evidence-{content_kind}",
        source_document_sha256=pdf_sha256,
        sources=(
            EvidenceSource(
                id="source-pdf",
                kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
                artifact_ref="blob://fixtures/source.pdf",
                producer="fixture",
                sha256=pdf_sha256,
            ),
            EvidenceSource(
                id="page-image",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="blob://fixtures/page.png",
                producer="fixture",
                sha256=page_sha256,
            ),
        ),
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=100.0,
                height=100.0,
                image_source_ref="page-image",
                observations=(
                    EvidenceObservation(
                        id="observation-1",
                        kind=observation_kind,
                        bbox=BBox(
                            pixel=(0.0, 0.0, 100.0, 100.0),
                            normalized=(0.0, 0.0, 1.0, 1.0),
                        ),
                        confidence=1.0,
                        source_refs=("page-image",),
                    ),
                ),
            ),
        ),
    )
    node: TextContentNode | ImageContentNode | FormulaContentNode
    if content_kind == "image":
        node = ImageContentNode(
            id="node-1",
            asset_ref="page-image",
            evidence_refs=("observation-1",),
            confidence=1.0,
        )
        render_as = "image"
        layout = LayoutIntent(width_fraction=0.5)
        design_profile_id = IMAGE_DESIGN_PROFILE_ID
    elif content_kind == "formula":
        node = FormulaContentNode(
            id="node-1",
            expression="x + 1",
            format=FormulaFormat.LATEX,
            evidence_refs=("observation-1",),
            confidence=1.0,
        )
        render_as = "formula"
        layout = LayoutIntent()
        design_profile_id = "scan2hwpx-canonical-text-v1"
    else:
        node = TextContentNode(
            id="node-1",
            role=ContentRole.QUESTION,
            text="검증된 본문",
            evidence_refs=("observation-1",),
            confidence=1.0,
        )
        render_as = "paragraph"
        layout = LayoutIntent()
        design_profile_id = "scan2hwpx-canonical-text-v1"
    content = ContentIR(
        id=f"content-{content_kind}",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        revision=2,
        nodes=(node,),
        reading_order=(node.id,),
    )
    plan = HwpDocumentPlan(
        id=f"plan-{content_kind}",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        capability_profile_id="exam2hwpx-authoring-v1",
        design_profile_id=design_profile_id,
        official_spec_refs=("official-spec-1",),
        page_layout=PageLayoutIntent(
            width_mm=210.0,
            height_mm=297.0,
            margin_top_mm=20.0,
            margin_right_mm=30.0,
            margin_bottom_mm=15.0,
            margin_left_mm=30.0,
            columns=2,
            column_gap_mm=8.0,
        ),
        flow=(
            ContentPlanItem(
                id="flow-1",
                render_as=render_as,
                content_ref=node.id,
                layout=layout,
            ),
        ),
    )
    binding_prefix = f"{lineage_id}/contracts"
    evidence_binding = CandidateContractArtifactBinding(
        path=f"{binding_prefix}/evidence.json",
        sha256="1" * 64,
        contract_sha256=contract_sha256(evidence),
    )
    content_binding = CandidateContractArtifactBinding(
        path=f"{binding_prefix}/content.json",
        sha256="2" * 64,
        contract_sha256=contract_sha256(content),
    )
    plan_binding = CandidateContractArtifactBinding(
        path=f"{binding_prefix}/plan.json",
        sha256="3" * 64,
        contract_sha256=contract_sha256(plan),
    )
    candidate = CandidateReviewDraft(
        status="complete",
        draft_revision=2,
        reviewer_label="reviewer",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        candidate_manifest_sha256="4" * 64,
        document_id=document_id,
        lineage_id=lineage_id,
        source_pdf_sha256=pdf_sha256,
        evidence_ir=evidence_binding,
        base_content_ir=content_binding,
        base_hwp_document_plan=plan_binding,
        reviewed_content_ir=content,
        reviewed_content_ir_contract_sha256=contract_sha256(content),
        reviewed_hwp_document_plan=plan,
        reviewed_hwp_document_plan_contract_sha256=contract_sha256(plan),
    )
    capability_bytes = _capability_bytes()
    chunk = HancomChunk(
        id="official-spec-1",
        source_id="hancom-hwpx-format",
        title="HWPX format",
        section="document layout",
        text="HWPX 문서 배치와 스타일 규격 설명",
        tags=("hwpx", "layout", "style-reference"),
        source_url="https://tech.hancom.com/hwpxformat/",
        source_sha256="8" * 64,
    )
    grounding = ModelBPlanGroundingEvidence(
        knowledge_corpus_sha256="5" * 64,
        retrieval_artifact_sha256="6" * 64,
        capability_profile_id=plan.capability_profile_id,
        capability_profile_sha256=hashlib.sha256(capability_bytes).hexdigest(),
        official_spec_refs=(
            ModelBOfficialSpecRefEvidence(
                official_spec_ref="official-spec-1",
                source_document_sha256=chunk.source_sha256,
                retrieved_chunk_sha256=retrieved_chunk_sha256(chunk),
            ),
        ),
    )
    started_at = datetime(2026, 1, 2, tzinfo=UTC)
    initial_model_b = start_model_b_plan_review_draft(
        content,
        plan,
        document_id=document_id,
        reviewer_label="model-b-reviewer",
        grounding_evidence=grounding,
        updated_at=started_at,
    )
    envelope = CandidateToModelBHandoff(
        created_at=initial_model_b.updated_at,
        document_id=document_id,
        lineage_id=lineage_id,
        completed_candidate_review=CompletedCandidateReviewBinding(
            artifact_sha256="a" * 64,
            draft_revision=candidate.draft_revision,
            candidate_manifest_sha256=candidate.candidate_manifest_sha256,
            document_id=document_id,
            lineage_id=lineage_id,
            base_content_ir=candidate.base_content_ir,
            base_hwp_document_plan=candidate.base_hwp_document_plan,
            reviewed_content_ir_contract_sha256=(candidate.reviewed_content_ir_contract_sha256),
            reviewed_hwp_document_plan_contract_sha256=(
                candidate.reviewed_hwp_document_plan_contract_sha256
            ),
        ),
        grounding_sidecar=GroundingSidecarHandoffBinding(
            manifest_sha256="b" * 64,
            candidate_manifest_sha256=candidate.candidate_manifest_sha256,
            document_id=document_id,
            lineage_id=lineage_id,
            knowledge_corpus_sha256=grounding.knowledge_corpus_sha256,
            capability_profile_id=grounding.capability_profile_id,
            capability_profile_sha256=grounding.capability_profile_sha256,
            retrieval_artifact_sha256=grounding.retrieval_artifact_sha256,
            grounding_evidence_contract_sha256=contract_sha256(grounding),
        ),
        model_b_plan_review_draft=initial_model_b,
    )
    verified_handoff = VerifiedCandidateToModelBHandoff(
        envelope=envelope,
        artifact_sha256=_handoff_sha256(envelope),
    )
    inference_result = execute_model_b_plan(
        verified_handoff,
        ModelBModelArtifact(
            model_id="fixture-model-b",
            model_revision="revision-1",
            artifact_sha256="9" * 64,
        ),
        capability_bytes,
        (chunk,),
        _PlanProvider(),
    )
    initial_model_b = start_model_b_plan_review_from_inference(
        inference_result,
        updated_at=started_at + timedelta(minutes=1),
    )
    completed_model_b = complete_model_b_plan_review_draft(
        initial_model_b,
        ModelBPlanReviewCompletionRequest(
            expected_draft_revision=initial_model_b.draft_revision,
            expected_source_candidate_contract_sha256=(
                initial_model_b.source_candidate_contract_sha256
            ),
            expected_grounding_evidence_contract_sha256=(
                initial_model_b.grounding_evidence_contract_sha256
            ),
            expected_reviewed_hwp_document_plan_contract_sha256=(
                initial_model_b.reviewed_hwp_document_plan_contract_sha256
            ),
        ),
        updated_at=started_at + timedelta(minutes=2),
    )
    image_assets = None
    if content_kind == "image":
        image_assets = build_image_asset_bundle(
            evidence,
            content,
            (
                PngImageAsset(
                    asset_ref="page-image",
                    media_type="image/png",
                    sha256=page_sha256,
                    payload=image_payload,
                ),
            ),
        )
    return _Fixture(
        candidate_root=candidate_root,
        candidate=candidate,
        evidence=evidence,
        verified_handoff=verified_handoff,
        initial_model_b=initial_model_b,
        completed_model_b=completed_model_b,
        inference_result=inference_result,
        materialized=VerifiedCandidateCompileInputs(
            candidate_review=candidate,
            evidence_ir=evidence,
            image_assets=image_assets,
        ),
    )


def _install_materializer(
    monkeypatch: pytest.MonkeyPatch,
    materialized: VerifiedCandidateCompileInputs,
) -> None:
    def materialize(
        draft: CandidateReviewDraft,
        *,
        candidate_root: Path | str,
    ) -> VerifiedCandidateCompileInputs:
        assert draft == materialized.candidate_review
        assert Path(candidate_root).is_dir()
        return materialized

    monkeypatch.setattr(
        submission,
        "materialize_verified_candidate_compile_inputs",
        materialize,
    )


@pytest.mark.parametrize("content_kind", ["text", "image"])
def test_compiles_reviewed_submission_with_review_only_quality_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_kind: Literal["text", "image"],
) -> None:
    fixture = _fixture(tmp_path, content_kind=content_kind)
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "published" / f"{content_kind}.hwpx"

    result = submission.compile_reviewed_submission(
        fixture.candidate,
        fixture.verified_handoff,
        fixture.completed_model_b,
        fixture.inference_result,
        candidate_root=fixture.candidate_root,
        output_path=output,
        compiled_artifact_ref=f"artifact://documents/{content_kind}/result.hwpx",
    )

    assert result.output_path == output.resolve()
    assert validate_hwpx(output).valid
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result.compile_result.artifact_sha256
    report = result.quality_report
    assert report.outcome == QualityOutcome.REVIEW
    assert report.compiled_artifact_sha256 == result.compile_result.artifact_sha256
    assert report.model_bundle_id == "unbundled-model-b-" + "9" * 64
    assert {check.id for check in report.checks} == REQUIRED_DOCUMENT_QUALITY_CHECKS_V1
    decisions = {check.id: check.decision for check in report.checks}
    assert {
        check_id for check_id, decision in decisions.items() if decision == CheckDecision.REVIEW
    } == {"visual-layout", "dvc-validation", "hancom-round-trip"}
    report.assert_contract_lineage(
        fixture.evidence,
        fixture.completed_model_b.content_ir,
        fixture.completed_model_b.reviewed_hwp_document_plan,
    )
    assert not list(output.parent.glob(f".{output.name}.*.partial*"))


def test_rejects_candidate_handoff_mismatch_before_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    value = fixture.candidate.model_dump(mode="python")
    value["document_id"] = "another-document"
    mismatched = CandidateReviewDraft.model_validate(value, strict=True)
    materialized = VerifiedCandidateCompileInputs(
        candidate_review=mismatched,
        evidence_ir=fixture.evidence,
        image_assets=None,
    )
    _install_materializer(monkeypatch, materialized)
    output = tmp_path / "mismatch.hwpx"

    with pytest.raises(submission.CompileSubmissionError, match="verified handoff"):
        submission.compile_reviewed_submission(
            mismatched,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/mismatch/result.hwpx",
        )

    assert not output.exists()


def test_requires_completed_model_b_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "in-progress.hwpx"

    with pytest.raises(submission.CompileSubmissionError, match="must be complete"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.initial_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/in-progress/result.hwpx",
        )

    assert not output.exists()


def test_rejects_forged_verified_handoff_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    forged = VerifiedCandidateToModelBHandoff(
        envelope=fixture.verified_handoff.envelope,
        artifact_sha256="d" * 64,
    )
    output = tmp_path / "forged.hwpx"

    with pytest.raises(submission.CompileSubmissionError, match="does not match envelope"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            forged,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/forged/result.hwpx",
        )

    assert not output.exists()


def test_formula_compile_failure_publishes_nothing_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, content_kind="formula")
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "published" / "formula.hwpx"

    with pytest.raises(PlanCompilationError, match="unsupported content kind"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/formula/result.hwpx",
        )

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.partial*"))


def test_rejects_compiler_result_with_another_lineage_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "published" / "wrong-lineage.hwpx"
    real_compile = compile_plan_hwpx

    def compile_with_wrong_lineage(
        content_ir: ContentIR,
        plan: HwpDocumentPlan,
        staged_output: Path,
        *,
        evidence_ir: EvidenceIR | None = None,
        image_assets: ImageAssetBundle | None = None,
    ) -> PlanCompileResult:
        result = real_compile(
            content_ir,
            plan,
            staged_output,
            evidence_ir=evidence_ir,
            image_assets=image_assets,
        )
        return replace(result, content_ir_sha256="0" * 64)

    monkeypatch.setattr(submission, "compile_plan_hwpx", compile_with_wrong_lineage)
    with pytest.raises(submission.CompileSubmissionError, match="ContentIR lineage"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/wrong-lineage/result.hwpx",
        )

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.partial*"))


def test_does_not_overwrite_existing_or_racing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    existing = tmp_path / "existing.hwpx"
    existing.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=existing,
            compiled_artifact_ref="artifact://documents/existing/result.hwpx",
        )
    assert existing.read_bytes() == b"keep"

    racing = tmp_path / "racing.hwpx"

    def race_publish(source: Path, destination: Path) -> None:
        assert source.is_file()
        destination.write_bytes(b"racer")
        raise FileExistsError("output appeared during build")

    monkeypatch.setattr(submission, "publish_file_create_only", race_publish)
    with pytest.raises(FileExistsError, match="appeared"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=racing,
            compiled_artifact_ref="artifact://documents/racing/result.hwpx",
        )
    assert racing.read_bytes() == b"racer"
    assert not list(tmp_path.glob(f".{racing.name}.*.partial*"))


def test_successful_compile_does_not_touch_recreated_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "result.hwpx"
    real_publish = submission.publish_file_create_only
    recreated: Path | None = None

    def publish_then_recreate(source: Path, destination: Path) -> None:
        nonlocal recreated
        real_publish(source, destination)
        source.mkdir()
        (source / "keep.txt").write_text("keep", encoding="utf-8")
        recreated = source

    monkeypatch.setattr(
        submission,
        "publish_file_create_only",
        publish_then_recreate,
    )

    result = submission.compile_reviewed_submission(
        fixture.candidate,
        fixture.verified_handoff,
        fixture.completed_model_b,
        fixture.inference_result,
        candidate_root=fixture.candidate_root,
        output_path=output,
        compiled_artifact_ref="artifact://documents/result/result.hwpx",
    )

    assert result.output_path == output.resolve()
    assert validate_hwpx(output).valid
    assert recreated is not None
    assert (recreated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_does_not_replace_existing_output_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    destination = tmp_path / "linked.hwpx"
    try:
        destination.symlink_to(tmp_path / "missing-target.hwpx")
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(FileExistsError, match="already exists"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=destination,
            compiled_artifact_ref="artifact://documents/linked/result.hwpx",
        )

    assert destination.is_symlink()


def test_rejects_candidate_internal_output_and_invalid_report_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)

    with pytest.raises(ValueError, match="outside the candidate root"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=fixture.candidate_root / "inside.hwpx",
            compiled_artifact_ref="artifact://documents/inside/result.hwpx",
        )

    invalid_report_output = tmp_path / "invalid-report.hwpx"
    with pytest.raises(submission.CompileSubmissionError, match="quality report"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=invalid_report_output,
            compiled_artifact_ref="not-a-logical-artifact-ref",
        )
    assert not invalid_report_output.exists()
    assert not list(tmp_path.glob(f".{invalid_report_output.name}.*.partial*"))


def test_rejects_inference_result_from_another_handoff_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, content_kind="text")
    other = _fixture(tmp_path, content_kind="image")
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "mixed-inference.hwpx"

    with pytest.raises(submission.CompileSubmissionError, match="another handoff"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            other.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/mixed/result.hwpx",
        )

    assert not output.exists()


def test_rejects_tampered_inference_result_and_persisted_binding_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    output = tmp_path / "tampered-inference.hwpx"
    object.__setattr__(fixture.inference_result, "result_sha256", "0" * 64)

    with pytest.raises(submission.CompileSubmissionError, match="not valid"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            fixture.inference_result,
            candidate_root=fixture.candidate_root,
            output_path=output,
            compiled_artifact_ref="artifact://documents/tampered/result.hwpx",
        )

    assert not output.exists()

    fresh = _fixture(tmp_path, content_kind="image")
    _install_materializer(monkeypatch, fresh.materialized)
    value = fresh.completed_model_b.model_dump(mode="python")
    value["model_b_inference_binding"]["result_sha256"] = "0" * 64
    tampered_review = ModelBPlanReviewDraft.model_validate(value, strict=True)
    binding_output = tmp_path / "tampered-binding.hwpx"
    with pytest.raises(submission.CompileSubmissionError, match="binding does not match"):
        submission.compile_reviewed_submission(
            fresh.candidate,
            fresh.verified_handoff,
            tampered_review,
            fresh.inference_result,
            candidate_root=fresh.candidate_root,
            output_path=binding_output,
            compiled_artifact_ref="artifact://documents/tampered-binding/result.hwpx",
        )

    assert not binding_output.exists()


def test_rejects_blocked_or_missing_inference_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _install_materializer(monkeypatch, fixture.materialized)
    blocked = validate_model_b_inference_response(
        fixture.inference_result.request,
        b"{}",
    )
    blocked_output = tmp_path / "blocked-inference.hwpx"

    with pytest.raises(submission.CompileSubmissionError, match="blocked"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            blocked,
            candidate_root=fixture.candidate_root,
            output_path=blocked_output,
            compiled_artifact_ref="artifact://documents/blocked/result.hwpx",
        )
    assert not blocked_output.exists()

    missing_output = tmp_path / "missing-inference.hwpx"
    with pytest.raises(submission.CompileSubmissionError, match="exact Model B"):
        submission.compile_reviewed_submission(
            fixture.candidate,
            fixture.verified_handoff,
            fixture.completed_model_b,
            cast(BuiltModelBInferenceResult, None),
            candidate_root=fixture.candidate_root,
            output_path=missing_output,
            compiled_artifact_ref="artifact://documents/missing/result.hwpx",
        )
    assert not missing_output.exists()
