from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scan2hwpx.contracts.models import ContentIR, HwpDocumentPlan, contract_sha256
from scan2hwpx.evaluation import model_b as model_b_module
from scan2hwpx.evaluation.model_b import (
    MODEL_B_METRIC_DEFINITION_IDS,
    MODEL_B_PLAN_METRIC_SET_ID,
    VERIFIED_MODEL_B_EVALUATOR_ARTIFACT_REF,
    VerifiedModelBDocumentArtifacts,
    VerifiedModelBEvaluationInputs,
    VerifiedModelBMeasurementReport,
    evaluate_verified_model_b_records,
)
from scan2hwpx.evaluation.verified import ArtifactPayload


def _sha(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _artifact(ref: str, payload: bytes) -> ArtifactPayload:
    return ArtifactPayload(artifact_ref=ref, sha256=_sha(payload), payload=payload)


def _content(document_id: str) -> tuple[ContentIR, bytes]:
    content = ContentIR.model_validate(
        {
            "schema_version": "content-ir/1.0",
            "id": f"content-{document_id}",
            "evidence_ir_id": f"evidence-{document_id}",
            "evidence_ir_sha256": _sha(f"evidence:{document_id}"),
            "revision": 2,
            "nodes": [
                {
                    "kind": "text",
                    "id": "q1",
                    "evidence_refs": ["obs-q1"],
                    "confidence": 0.99,
                    "role": "question",
                    "text": "1. 문제 [2점]",
                },
                {
                    "kind": "text",
                    "id": "c1",
                    "evidence_refs": ["obs-c1"],
                    "confidence": 0.98,
                    "role": "choice",
                    "text": "① 정답",
                },
            ],
            "reading_order": ["q1", "c1"],
        }
    )
    return content, content.model_dump_json().encode("utf-8")


def _plan_payload(
    document_id: str,
    content: ContentIR,
    *,
    gold: bool,
    mode: str = "exact",
) -> dict[str, Any]:
    content_sha256 = contract_sha256(content)
    if mode == "wrong_lineage":
        content_sha256 = "f" * 64
    refs = ["q1", "c1"]
    if mode == "reordered":
        refs.reverse()
    elif mode == "missing":
        refs.pop()
    flow: list[dict[str, Any]] = [
        {
            "id": f"item-{index}",
            "kind": "content",
            "render_as": "paragraph",
            "content_ref": content_ref,
        }
        for index, content_ref in enumerate(refs, start=1)
    ]
    if not gold:
        flow.insert(1, {"id": "page-break-1", "kind": "page_break"})
    return {
        "schema_version": "hwp-document-plan/1.0",
        "id": ("gold-" if gold else "prediction-") + document_id,
        "content_ir_id": content.id,
        "content_ir_revision": content.revision,
        "content_ir_sha256": content_sha256,
        "capability_profile_id": "hwpx-v1",
        "design_profile_id": "approved-layout" if gold else "new-layout",
        "official_spec_refs": ["hancom-hwpx-format:fixture"],
        "page_layout": {
            "width_mm": 210.0,
            "height_mm": 297.0,
            "margin_top_mm": 20.0 if gold else 18.0,
            "margin_right_mm": 20.0,
            "margin_bottom_mm": 20.0,
            "margin_left_mm": 20.0,
            "columns": 1,
            "column_gap_mm": 0.0,
        },
        "styles": [],
        "flow": flow,
    }


def _prediction_bytes(document_id: str, content: ContentIR, mode: str) -> bytes:
    if mode == "invalid_json":
        return b"{not-json"
    if mode == "excessive_nesting":
        return b"[" * 2_000 + b"]" * 2_000
    if mode == "excessive_integer":
        return b"1" * 5_000
    payload = _plan_payload(document_id, content, gold=False, mode=mode)
    if mode == "extra_content_field":
        payload["flow"][0]["text"] = "content must not be authored in a plan"
    prediction = _json_bytes(payload)
    if mode == "duplicate_key":
        return b'{"id":"shadow",' + prediction[1:]
    return prediction


def _bundle_bytes(
    dataset_ref: str,
    dataset_sha256: str,
    model_b_ref: str,
    model_b_sha256: str,
    *,
    placeholder_model_b: bool,
) -> bytes:
    model_b: dict[str, object]
    if placeholder_model_b:
        model_b = {
            "status": "placeholder",
            "model_id": "document-author-model-b-v1",
            "revision": "not-trained",
            "artifact_ref": None,
            "artifact_sha256": None,
        }
    else:
        model_b = {
            "status": "candidate",
            "model_id": "document-author-model-b-v1",
            "revision": "candidate-v1",
            "artifact_ref": model_b_ref,
            "artifact_sha256": model_b_sha256,
        }
    return _json_bytes(
        {
            "schema_version": "1.0",
            "bundle_id": "candidate-bundle-1",
            "status": "candidate",
            "created_at": "2026-09-10T09:00:00+09:00",
            "model_a": {
                "status": "placeholder",
                "model_id": "document-understanding-model-a-v1",
                "revision": "not-trained",
                "artifact_ref": None,
                "artifact_sha256": None,
            },
            "model_b": model_b,
            "dataset_manifest_ref": dataset_ref,
            "dataset_manifest_sha256": dataset_sha256,
            "release_gate_set_id": "initial_v1",
            "capability_profile_ref": "artifact://profiles/hwpx-v1.json",
            "capability_profile_sha256": "2" * 64,
            "hancom_knowledge_manifest_ref": "artifact://hancom/manifest-v1.json",
            "hancom_knowledge_manifest_sha256": "3" * 64,
            "hancom_knowledge_corpus_ref": "artifact://hancom/corpus-v1.jsonl",
            "hancom_knowledge_corpus_sha256": "4" * 64,
            "compiler_version": "compiler-v1",
            "compiler_sha256": "5" * 64,
            "prompt_version": "prompt-v1",
            "contract_versions": {
                "evidence_ir": "evidence-ir/1.0",
                "content_ir": "content-ir/1.0",
                "hwp_document_plan": "hwp-document-plan/1.0",
                "quality_report": "quality-report/1.0",
            },
        }
    )


def _make_inputs(
    document_ids: Sequence[str] = ("doc-001",),
    *,
    prediction_mode: str = "exact",
    golden_document_ids: Sequence[str] | None = None,
    prediction_document_ids: Sequence[str] | None = None,
    input_document_ids: Sequence[str] | None = None,
    prediction_digest_override: str | None = None,
    golden_content_digest_override: str | None = None,
    golden_content_contract_override: str | None = None,
    gold_plan_contract_override: str | None = None,
    plan_attestation_override: tuple[str, str] | None = None,
    metric_definitions: dict[str, str] | None = None,
    include_aggregate_measurements: bool = False,
    placeholder_model_b: bool = False,
    golden_example_only: bool = False,
) -> VerifiedModelBEvaluationInputs:
    raw_records: dict[
        str,
        tuple[
            str,
            str,
            ContentIR,
            ArtifactPayload,
            ArtifactPayload,
            ArtifactPayload,
            ArtifactPayload,
        ],
    ] = {}
    dataset_documents: list[dict[str, Any]] = []
    for index, document_id in enumerate(document_ids, start=1):
        lineage_id = f"lineage-{index:03d}"
        source_sha256 = _sha(f"source:{document_id}")
        content, content_bytes = _content(document_id)
        content_artifact = _artifact(
            f"artifact://golden/content/{document_id}.json",
            content_bytes,
        )
        content_attestation_ref = f"artifact://golden/content-verifications/{document_id}.json"
        manifest_content_sha256 = golden_content_digest_override or content_artifact.sha256
        content_attestation = _artifact(
            content_attestation_ref,
            _json_bytes(
                {
                    "schema_version": "1.0",
                    "document_id": document_id,
                    "source_sha256": source_sha256,
                    "annotation_sha256": manifest_content_sha256,
                    "reviewer_id": "content-reviewer",
                    "verified_at": "2026-09-09T09:00:00+09:00",
                    "decision": "approved",
                    "usage_rights_verified": True,
                }
            ),
        )
        gold_plan = HwpDocumentPlan.model_validate(_plan_payload(document_id, content, gold=True))
        gold_plan_artifact = _artifact(
            f"artifact://golden/plans/{document_id}.json",
            gold_plan.model_dump_json().encode("utf-8"),
        )
        prediction_artifact = _artifact(
            f"artifact://predictions/{document_id}.json",
            _prediction_bytes(document_id, content, prediction_mode),
        )
        raw_records[document_id] = (
            lineage_id,
            source_sha256,
            content,
            content_artifact,
            content_attestation,
            gold_plan_artifact,
            prediction_artifact,
        )
        dataset_documents.append(
            {
                "id": document_id,
                "lineage_id": lineage_id,
                "source_sha256": source_sha256,
                "template_family": f"held-out-{index:03d}",
                "split": "test",
                "capture": "real_scan",
                "page_count": 1,
                "annotation_artifact_ref": content_artifact.artifact_ref,
                "annotation_sha256": manifest_content_sha256,
                "verified": True,
                "usage_rights_verified": True,
                "verification_attestation": {
                    "reviewer_id": "content-reviewer",
                    "verified_at": "2026-09-09T09:00:00+09:00",
                    "verification_artifact_ref": content_attestation_ref,
                    "verification_artifact_sha256": content_attestation.sha256,
                },
            }
        )

    dataset_artifact = _artifact(
        "artifact://golden/dataset-manifest.json",
        _json_bytes(
            {
                "schema_version": "1.0",
                "example_only": False,
                "documents": dataset_documents,
            }
        ),
    )
    model_b_artifact = _artifact(
        "artifact://models/model-b-v1.bin",
        b"model-b-weights-fixture",
    )
    bundle_artifact = _artifact(
        "artifact://models/candidate-bundle-1.json",
        _bundle_bytes(
            dataset_artifact.artifact_ref,
            dataset_artifact.sha256,
            model_b_artifact.artifact_ref,
            model_b_artifact.sha256,
            placeholder_model_b=placeholder_model_b,
        ),
    )

    plan_attestations: dict[str, ArtifactPayload] = {}
    golden_bindings: list[dict[str, Any]] = []
    selected_golden = set(golden_document_ids or document_ids)
    for document_id in document_ids:
        (
            lineage_id,
            source_sha256,
            content,
            content_artifact,
            content_attestation,
            gold_plan_artifact,
            _,
        ) = raw_records[document_id]
        gold_plan = HwpDocumentPlan.model_validate_json(gold_plan_artifact.payload)
        content_contract_sha256 = golden_content_contract_override or contract_sha256(content)
        gold_plan_contract_sha256 = gold_plan_contract_override or contract_sha256(gold_plan)
        plan_attestation_payload: dict[str, Any] = {
            "schema_version": "model-b-plan-verification/1.0",
            "document_id": document_id,
            "lineage_id": lineage_id,
            "source_document_sha256": source_sha256,
            "dataset_manifest_sha256": dataset_artifact.sha256,
            "content_ir_artifact_ref": content_artifact.artifact_ref,
            "content_ir_artifact_sha256": (
                golden_content_digest_override or content_artifact.sha256
            ),
            "content_ir_contract_sha256": content_contract_sha256,
            "gold_plan_artifact_ref": gold_plan_artifact.artifact_ref,
            "gold_plan_artifact_sha256": gold_plan_artifact.sha256,
            "gold_plan_contract_sha256": gold_plan_contract_sha256,
            "reviewer_id": "plan-reviewer",
            "verified_at": "2026-09-10T09:00:00+09:00",
            "decision": "approved",
        }
        if plan_attestation_override is not None:
            field_name, value = plan_attestation_override
            plan_attestation_payload[field_name] = value
        plan_attestation = _artifact(
            f"artifact://golden/plan-verifications/{document_id}.json",
            _json_bytes(plan_attestation_payload),
        )
        plan_attestations[document_id] = plan_attestation
        if document_id in selected_golden:
            golden_bindings.append(
                {
                    "document_id": document_id,
                    "lineage_id": lineage_id,
                    "source_document_sha256": source_sha256,
                    "content_ir_artifact_ref": content_artifact.artifact_ref,
                    "content_ir_artifact_sha256": (
                        golden_content_digest_override or content_artifact.sha256
                    ),
                    "content_ir_contract_sha256": content_contract_sha256,
                    "content_verification_artifact_ref": (content_attestation.artifact_ref),
                    "content_verification_artifact_sha256": content_attestation.sha256,
                    "gold_plan_artifact_ref": gold_plan_artifact.artifact_ref,
                    "gold_plan_artifact_sha256": gold_plan_artifact.sha256,
                    "gold_plan_contract_sha256": gold_plan_contract_sha256,
                    "plan_verification": {
                        "reviewer_id": "plan-reviewer",
                        "verified_at": "2026-09-10T09:00:00+09:00",
                        "verification_artifact_ref": plan_attestation.artifact_ref,
                        "verification_artifact_sha256": plan_attestation.sha256,
                    },
                }
            )
    golden_artifact = _artifact(
        "artifact://golden/model-b-manifest.json",
        _json_bytes(
            {
                "schema_version": "model-b-golden/1.0",
                "example_only": golden_example_only,
                "dataset_manifest_ref": dataset_artifact.artifact_ref,
                "dataset_manifest_sha256": dataset_artifact.sha256,
                "documents": golden_bindings,
            }
        ),
    )

    selected_predictions = tuple(prediction_document_ids or document_ids)
    prediction_bindings = []
    for document_id in selected_predictions:
        prediction_artifact = raw_records[document_id][-1]
        prediction_bindings.append(
            {
                "document_id": document_id,
                "prediction_artifact_ref": prediction_artifact.artifact_ref,
                "prediction_artifact_sha256": (
                    prediction_digest_override or prediction_artifact.sha256
                ),
            }
        )
    prediction_manifest_payload: dict[str, Any] = {
        "schema_version": "model-b-predictions/1.0",
        "model_bundle_id": "candidate-bundle-1",
        "model_bundle_manifest_ref": bundle_artifact.artifact_ref,
        "model_bundle_manifest_sha256": bundle_artifact.sha256,
        "dataset_manifest_ref": dataset_artifact.artifact_ref,
        "dataset_manifest_sha256": dataset_artifact.sha256,
        "golden_manifest_ref": golden_artifact.artifact_ref,
        "golden_manifest_sha256": golden_artifact.sha256,
        "metric_set_id": MODEL_B_PLAN_METRIC_SET_ID,
        "metric_definition_ids": metric_definitions or MODEL_B_METRIC_DEFINITION_IDS,
        "documents": prediction_bindings,
    }
    if include_aggregate_measurements:
        prediction_manifest_payload["measurements"] = {"model_b.schema_valid_rate": 1.0}
    prediction_manifest_artifact = _artifact(
        "artifact://predictions/model-b-manifest.json",
        _json_bytes(prediction_manifest_payload),
    )

    selected_inputs = tuple(input_document_ids or document_ids)
    document_inputs = []
    for document_id in selected_inputs:
        (
            _,
            _,
            _,
            content_artifact,
            content_attestation,
            gold_plan_artifact,
            prediction_artifact,
        ) = raw_records[document_id]
        document_inputs.append(
            VerifiedModelBDocumentArtifacts(
                document_id=document_id,
                content_ir=content_artifact,
                content_verification_attestation=content_attestation,
                gold_plan=gold_plan_artifact,
                gold_plan_verification_attestation=plan_attestations[document_id],
                prediction=prediction_artifact,
            )
        )
    evaluator_artifact = _artifact(
        VERIFIED_MODEL_B_EVALUATOR_ARTIFACT_REF,
        Path(model_b_module.__file__).read_bytes(),
    )
    return VerifiedModelBEvaluationInputs(
        dataset_manifest=dataset_artifact,
        model_bundle_manifest=bundle_artifact,
        model_b_artifact=model_b_artifact,
        golden_manifest=golden_artifact,
        prediction_manifest=prediction_manifest_artifact,
        evaluator_artifact=evaluator_artifact,
        documents=tuple(document_inputs),
    )


def test_recomputes_metrics_from_raw_plans_and_allows_design_changes() -> None:
    report = evaluate_verified_model_b_records(_make_inputs())

    assert report.promotable is False
    assert report.review_identity_assurance == "unsigned_manifest_consistency_only"
    assert report.metric_set_id == MODEL_B_PLAN_METRIC_SET_ID
    assert report.metric_definition_ids == MODEL_B_METRIC_DEFINITION_IDS
    assert report.measurements == {
        "model_b.schema_valid_rate": 1.0,
        "model_b.content_reference_coverage": 1.0,
        "model_b.unauthorized_content_changes": 0,
    }
    assert report.metric_document_counts == {
        metric_id: 1 for metric_id in MODEL_B_METRIC_DEFINITION_IDS
    }
    assert report.documents[0].prediction_schema_valid is True
    assert report.documents[0].content_reference_integrity_exact is True
    assert report.documents[0].prediction_plan_contract_sha256 is not None


def test_report_contract_cannot_claim_promotion() -> None:
    report = evaluate_verified_model_b_records(_make_inputs())
    payload = report.model_dump()
    payload["promotable"] = True

    with pytest.raises(ValidationError, match="promotable"):
        VerifiedModelBMeasurementReport.model_validate(payload)


def test_reordered_content_is_schema_valid_and_covered_but_unauthorized() -> None:
    report = evaluate_verified_model_b_records(_make_inputs(prediction_mode="reordered"))

    assert report.measurements == {
        "model_b.schema_valid_rate": 1.0,
        "model_b.content_reference_coverage": 1.0,
        "model_b.unauthorized_content_changes": 2,
    }


def test_missing_content_ref_fails_exact_coverage_and_counts_one_deletion() -> None:
    report = evaluate_verified_model_b_records(_make_inputs(prediction_mode="missing"))

    assert report.measurements["model_b.schema_valid_rate"] == 1.0
    assert report.measurements["model_b.content_reference_coverage"] == 0.0
    assert report.measurements["model_b.unauthorized_content_changes"] == 1


def test_changed_content_lineage_is_an_unauthorized_change() -> None:
    report = evaluate_verified_model_b_records(_make_inputs(prediction_mode="wrong_lineage"))

    assert report.measurements["model_b.schema_valid_rate"] == 1.0
    assert report.measurements["model_b.content_reference_coverage"] == 0.0
    assert report.measurements["model_b.unauthorized_content_changes"] == 1


@pytest.mark.parametrize(
    "prediction_mode",
    [
        "invalid_json",
        "extra_content_field",
        "duplicate_key",
        "excessive_nesting",
        "excessive_integer",
    ],
)
def test_invalid_prediction_is_measured_without_weakening_trusted_inputs(
    prediction_mode: str,
) -> None:
    report = evaluate_verified_model_b_records(_make_inputs(prediction_mode=prediction_mode))

    assert report.measurements == {
        "model_b.schema_valid_rate": 0.0,
        "model_b.content_reference_coverage": 0.0,
        "model_b.unauthorized_content_changes": 2,
    }
    assert report.documents[0].prediction_plan_contract_sha256 is None


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("golden", "golden manifest document coverage mismatch"),
        ("prediction", "prediction manifest document coverage mismatch"),
        ("input", "evaluation inputs document coverage mismatch"),
    ],
)
def test_every_input_layer_must_exactly_cover_dataset_test_documents(
    selection: str,
    message: str,
) -> None:
    if selection == "golden":
        inputs = _make_inputs(
            ("doc-001", "doc-002"),
            golden_document_ids=("doc-001",),
        )
    elif selection == "prediction":
        inputs = _make_inputs(
            ("doc-001", "doc-002"),
            prediction_document_ids=("doc-001",),
        )
    else:
        inputs = _make_inputs(
            ("doc-001", "doc-002"),
            input_document_ids=("doc-001",),
        )

    with pytest.raises(ValueError, match=message):
        evaluate_verified_model_b_records(inputs)


def test_prediction_digest_mismatch_aborts_the_evaluation() -> None:
    inputs = _make_inputs(prediction_digest_override="e" * 64)

    with pytest.raises(ValueError, match="prediction artifact digest"):
        evaluate_verified_model_b_records(inputs)


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("golden_content_digest_override", "gold ContentIR artifact digest"),
        ("golden_content_contract_override", "gold ContentIR.*contract digest"),
        ("gold_plan_contract_override", "gold plan.*contract digest"),
    ],
)
def test_gold_bytes_and_contract_digests_are_fail_closed(
    argument: str,
    message: str,
) -> None:
    if argument == "golden_content_digest_override":
        inputs = _make_inputs(golden_content_digest_override="d" * 64)
    elif argument == "golden_content_contract_override":
        inputs = _make_inputs(golden_content_contract_override="d" * 64)
    else:
        inputs = _make_inputs(gold_plan_contract_override="d" * 64)

    with pytest.raises(ValueError, match=message):
        evaluate_verified_model_b_records(inputs)


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("lineage_id", "different-lineage"),
        ("content_ir_contract_sha256", "a" * 64),
        ("gold_plan_contract_sha256", "b" * 64),
        ("reviewer_id", "different-reviewer"),
    ],
)
def test_plan_attestation_must_bind_gold_lineage_and_reviewer(
    field_name: str,
    wrong_value: str,
) -> None:
    inputs = _make_inputs(
        plan_attestation_override=(field_name, wrong_value),
    )

    with pytest.raises(ValueError, match="plan attestation.*disagrees"):
        evaluate_verified_model_b_records(inputs)


def test_invalid_plan_attestation_aborts_instead_of_becoming_a_failed_sample() -> None:
    inputs = _make_inputs(plan_attestation_override=("decision", "rejected"))

    with pytest.raises(ValidationError, match="decision"):
        evaluate_verified_model_b_records(inputs)


def test_model_b_artifact_must_match_model_bundle() -> None:
    inputs = _make_inputs()
    different_model = _artifact(
        inputs.model_b_artifact.artifact_ref,
        b"different-model-b-weights",
    )
    invalid = inputs.model_copy(update={"model_b_artifact": different_model})

    with pytest.raises(ValueError, match="model B artifact digest"):
        evaluate_verified_model_b_records(invalid)


def test_placeholder_model_b_is_rejected() -> None:
    inputs = _make_inputs(placeholder_model_b=True)

    with pytest.raises(ValueError, match="Model B component is a placeholder"):
        evaluate_verified_model_b_records(inputs)


def test_evaluator_bytes_must_match_the_running_evaluator() -> None:
    inputs = _make_inputs()
    different_evaluator = _artifact(
        VERIFIED_MODEL_B_EVALUATOR_ARTIFACT_REF,
        b"different-evaluator",
    )
    invalid = inputs.model_copy(update={"evaluator_artifact": different_evaluator})

    with pytest.raises(ValueError, match="running evaluator bytes"):
        evaluate_verified_model_b_records(invalid)


def test_prediction_manifest_cannot_change_metrics_or_supply_aggregates() -> None:
    changed = dict(MODEL_B_METRIC_DEFINITION_IDS)
    changed["model_b.schema_valid_rate"] = "different-definition/v1"
    with pytest.raises(ValidationError, match="canonical Model B definitions"):
        evaluate_verified_model_b_records(_make_inputs(metric_definitions=changed))

    with pytest.raises(ValidationError, match="measurements"):
        evaluate_verified_model_b_records(_make_inputs(include_aggregate_measurements=True))


def test_example_only_golden_manifest_is_rejected() -> None:
    with pytest.raises(ValidationError, match="example-only Model B golden"):
        evaluate_verified_model_b_records(_make_inputs(golden_example_only=True))


def test_artifact_payload_rejects_unbound_raw_prediction_bytes() -> None:
    with pytest.raises(ValidationError, match="artifact payload digest mismatch"):
        ArtifactPayload(
            artifact_ref="artifact://predictions/doc-001.json",
            sha256="0" * 64,
            payload=b"raw prediction",
        )
