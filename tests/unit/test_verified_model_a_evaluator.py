from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scan2hwpx.contracts.models import ContentIR, EvidenceIR, contract_sha256
from scan2hwpx.evaluation import verified as verified_module
from scan2hwpx.evaluation.model_a import (
    MODEL_A_METRIC_DEFINITION_IDS,
    MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID,
)
from scan2hwpx.evaluation.verified import (
    VERIFIED_MODEL_A_EVALUATOR_ARTIFACT_REF,
    ArtifactPayload,
    VerifiedModelADocumentArtifacts,
    VerifiedModelAEvaluationInputs,
    evaluate_verified_model_a_records,
)


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


def _evidence(document_id: str, source_sha256: str) -> tuple[EvidenceIR, bytes]:
    payload = {
        "schema_version": "evidence-ir/1.0",
        "id": f"evidence-{document_id}",
        "source_document_sha256": source_sha256,
        "sources": [
            {
                "id": "page-image",
                "kind": "page_image",
                "artifact_ref": f"artifact://pages/{document_id}-001.png",
                "producer": "fixture",
                "sha256": _sha(f"page:{document_id}"),
            },
            {
                "id": "ocr-source",
                "kind": "ocr_provider",
                "artifact_ref": f"artifact://ocr/{document_id}-001.json",
                "producer": "fixture-ocr",
                "sha256": _sha(f"ocr:{document_id}"),
            },
        ],
        "pages": [
            {
                "id": "page-001",
                "page_no": 1,
                "width": 1000.0,
                "height": 1400.0,
                "rotation": 0,
                "image_source_ref": "page-image",
                "observations": [
                    {
                        "id": "obs-q1",
                        "kind": "text_line",
                        "bbox": {
                            "pixel": [10.0, 10.0, 500.0, 50.0],
                            "normalized": [0.01, 0.01, 0.5, 0.04],
                        },
                        "confidence": 0.99,
                        "source_refs": ["ocr-source"],
                        "ocr_candidates": [
                            {
                                "text": "1. 문제 [2점]",
                                "provider": "fixture-ocr",
                                "confidence": 0.99,
                                "source_ref": "ocr-source",
                                "selected": True,
                            }
                        ],
                    },
                    {
                        "id": "obs-c1",
                        "kind": "text_line",
                        "bbox": {
                            "pixel": [20.0, 60.0, 500.0, 100.0],
                            "normalized": [0.02, 0.04, 0.5, 0.08],
                        },
                        "confidence": 0.98,
                        "source_refs": ["ocr-source"],
                        "ocr_candidates": [
                            {
                                "text": "① 정답",
                                "provider": "fixture-ocr",
                                "confidence": 0.98,
                                "source_ref": "ocr-source",
                                "selected": True,
                            }
                        ],
                    },
                ],
            }
        ],
    }
    evidence = EvidenceIR.model_validate(payload)
    return evidence, evidence.model_dump_json().encode("utf-8")


def _content(
    document_id: str,
    evidence: EvidenceIR,
    *,
    prediction: bool,
    evidence_sha256: str | None = None,
) -> tuple[ContentIR, bytes]:
    payload = {
        "schema_version": "content-ir/1.0",
        "id": ("prediction-" if prediction else "annotation-") + document_id,
        "evidence_ir_id": evidence.id,
        "evidence_ir_sha256": evidence_sha256 or contract_sha256(evidence),
        "revision": 1 if prediction else 2,
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
    content = ContentIR.model_validate(payload)
    return content, content.model_dump_json().encode("utf-8")


def _bundle_payload(
    dataset_ref: str,
    dataset_sha256: str,
    model_a_ref: str,
    model_a_sha256: str,
) -> bytes:
    return _json_bytes(
        {
            "schema_version": "1.0",
            "bundle_id": "candidate-bundle-1",
            "status": "candidate",
            "created_at": "2026-09-09T09:00:00+09:00",
            "model_a": {
                "status": "candidate",
                "model_id": "document-understanding-model-a-v1",
                "revision": "candidate-v1",
                "artifact_ref": model_a_ref,
                "artifact_sha256": model_a_sha256,
            },
            "model_b": {
                "status": "placeholder",
                "model_id": "document-author-model-b-v1",
                "revision": "not-trained",
                "artifact_ref": None,
                "artifact_sha256": None,
            },
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
    dataset_document_ids: Sequence[str] = ("doc-001",),
    *,
    prediction_document_ids: Sequence[str] | None = None,
    input_document_ids: Sequence[str] | None = None,
    annotation_digest_override: str | None = None,
    prediction_digest_override: str | None = None,
    prediction_evidence_digest_override: str | None = None,
    attestation_reviewer_override: str | None = None,
    page_count_override: int | None = None,
    metric_definitions: dict[str, str] | None = None,
    include_aggregate_measurements: bool = False,
) -> VerifiedModelAEvaluationInputs:
    records: dict[str, tuple[ArtifactPayload, ArtifactPayload, ArtifactPayload, ArtifactPayload]] = {}
    dataset_documents: list[dict[str, Any]] = []
    for index, document_id in enumerate(dataset_document_ids, start=1):
        source_sha256 = _sha(f"source:{document_id}")
        evidence, evidence_bytes = _evidence(document_id, source_sha256)
        _, annotation_bytes = _content(document_id, evidence, prediction=False)
        _, prediction_bytes = _content(
            document_id,
            evidence,
            prediction=True,
            evidence_sha256=prediction_evidence_digest_override,
        )
        evidence_artifact = _artifact(
            f"artifact://evidence/{document_id}.json", evidence_bytes
        )
        annotation_artifact = _artifact(
            f"artifact://golden/annotations/{document_id}.json", annotation_bytes
        )
        prediction_artifact = _artifact(
            f"artifact://predictions/{document_id}.json", prediction_bytes
        )
        manifest_annotation_sha256 = (
            annotation_digest_override or annotation_artifact.sha256
        )
        attestation_ref = f"artifact://golden/verifications/{document_id}.json"
        attestation_artifact = _artifact(
            attestation_ref,
            _json_bytes(
                {
                    "schema_version": "1.0",
                    "document_id": document_id,
                    "source_sha256": source_sha256,
                    "annotation_sha256": manifest_annotation_sha256,
                    "reviewer_id": attestation_reviewer_override or "reviewer-001",
                    "verified_at": "2026-09-09T09:00:00+09:00",
                    "decision": "approved",
                    "usage_rights_verified": True,
                }
            ),
        )
        records[document_id] = (
            evidence_artifact,
            annotation_artifact,
            prediction_artifact,
            attestation_artifact,
        )
        dataset_documents.append(
            {
                "id": document_id,
                "lineage_id": f"lineage-{index:03d}",
                "source_sha256": source_sha256,
                "template_family": f"held-out-{index:03d}",
                "split": "test",
                "capture": "real_scan",
                "page_count": page_count_override if page_count_override is not None else 1,
                "annotation_artifact_ref": annotation_artifact.artifact_ref,
                "annotation_sha256": manifest_annotation_sha256,
                "verified": True,
                "usage_rights_verified": True,
                "verification_attestation": {
                    "reviewer_id": "reviewer-001",
                    "verified_at": "2026-09-09T09:00:00+09:00",
                    "verification_artifact_ref": attestation_ref,
                    "verification_artifact_sha256": attestation_artifact.sha256,
                },
            }
        )

    dataset_ref = "artifact://golden/dataset-manifest.json"
    dataset_bytes = _json_bytes(
        {
            "schema_version": "1.0",
            "example_only": False,
            "documents": dataset_documents,
        }
    )
    dataset_artifact = _artifact(dataset_ref, dataset_bytes)
    bundle_ref = "artifact://models/candidate-bundle-1.json"
    model_a_artifact = _artifact(
        "artifact://models/model-a-v1.bin",
        b"model-a-weights-fixture",
    )
    bundle_artifact = _artifact(
        bundle_ref,
        _bundle_payload(
            dataset_ref,
            dataset_artifact.sha256,
            model_a_artifact.artifact_ref,
            model_a_artifact.sha256,
        ),
    )

    selected_predictions = tuple(prediction_document_ids or dataset_document_ids)
    prediction_bindings = []
    for document_id in selected_predictions:
        evidence_artifact, _, prediction_artifact, _ = records[document_id]
        prediction_bindings.append(
            {
                "document_id": document_id,
                "evidence_artifact_ref": evidence_artifact.artifact_ref,
                "evidence_artifact_sha256": evidence_artifact.sha256,
                "prediction_artifact_ref": prediction_artifact.artifact_ref,
                "prediction_artifact_sha256": (
                    prediction_digest_override or prediction_artifact.sha256
                ),
            }
        )
    prediction_manifest_payload: dict[str, Any] = {
        "schema_version": "model-a-predictions/1.0",
        "model_bundle_id": "candidate-bundle-1",
        "model_bundle_manifest_ref": bundle_ref,
        "model_bundle_manifest_sha256": bundle_artifact.sha256,
        "dataset_manifest_ref": dataset_ref,
        "dataset_manifest_sha256": dataset_artifact.sha256,
        "metric_set_id": MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID,
        "metric_definition_ids": metric_definitions or MODEL_A_METRIC_DEFINITION_IDS,
        "documents": prediction_bindings,
    }
    if include_aggregate_measurements:
        prediction_manifest_payload["measurements"] = {
            metric_id: 1.0 for metric_id in MODEL_A_METRIC_DEFINITION_IDS
        }
    prediction_manifest_artifact = _artifact(
        "artifact://predictions/manifest.json",
        _json_bytes(prediction_manifest_payload),
    )

    selected_inputs = tuple(input_document_ids or selected_predictions)
    document_inputs = []
    for document_id in selected_inputs:
        (
            evidence_artifact,
            annotation_artifact,
            prediction_artifact,
            attestation_artifact,
        ) = records[document_id]
        document_inputs.append(
            VerifiedModelADocumentArtifacts(
                document_id=document_id,
                evidence=evidence_artifact,
                annotation=annotation_artifact,
                prediction=prediction_artifact,
                verification_attestation=attestation_artifact,
            )
        )
    evaluator_artifact = _artifact(
        VERIFIED_MODEL_A_EVALUATOR_ARTIFACT_REF,
        Path(verified_module.__file__).read_bytes(),
    )
    return VerifiedModelAEvaluationInputs(
        dataset_manifest=dataset_artifact,
        model_bundle_manifest=bundle_artifact,
        prediction_manifest=prediction_manifest_artifact,
        model_a_artifact=model_a_artifact,
        evaluator_artifact=evaluator_artifact,
        documents=tuple(document_inputs),
    )


def test_verified_evaluator_recomputes_metrics_from_raw_ir_records() -> None:
    inputs = _make_inputs()

    report = evaluate_verified_model_a_records(inputs)

    assert report.evidence_level == "verified_records"
    assert (
        report.review_identity_assurance == "unsigned_manifest_consistency_only"
    )
    assert report.metric_set_id == MODEL_A_TEXT_STRUCTURE_METRIC_SET_ID
    assert report.metric_definition_ids == MODEL_A_METRIC_DEFINITION_IDS
    assert report.measurements == {
        "model_a.overall_cer": 0.0,
        "model_a.critical_token_error_rate": 0.0,
        "model_a.role_macro_f1": 1.0,
        "model_a.reading_order_exact_match": 1.0,
        "model_a.reading_order_kendall_tau": 1.0,
    }
    assert report.metric_document_counts == {
        metric_id: 1 for metric_id in MODEL_A_METRIC_DEFINITION_IDS
    }
    assert report.evaluated_document_ids == ("doc-001",)
    assert report.documents[0].source_document_sha256 == _sha("source:doc-001")
    assert report.documents[0].annotation_artifact_sha256 == (
        inputs.documents[0].annotation.sha256
    )


def test_empty_prediction_records_are_rejected() -> None:
    inputs = _make_inputs()
    payload = json.loads(inputs.prediction_manifest.payload)
    payload["documents"] = []
    empty_manifest = _artifact(
        inputs.prediction_manifest.artifact_ref,
        _json_bytes(payload),
    )
    invalid_inputs = inputs.model_copy(update={"prediction_manifest": empty_manifest})

    with pytest.raises(ValidationError, match="documents"):
        evaluate_verified_model_a_records(invalid_inputs)


def test_prediction_document_coverage_must_match_all_test_documents() -> None:
    inputs = _make_inputs(
        ("doc-001", "doc-002"),
        prediction_document_ids=("doc-001",),
    )

    with pytest.raises(ValueError, match="prediction manifest document coverage mismatch"):
        evaluate_verified_model_a_records(inputs)


def test_evaluation_input_document_ids_must_match_manifest() -> None:
    inputs = _make_inputs(
        ("doc-001", "doc-002"),
        input_document_ids=("doc-001",),
    )

    with pytest.raises(ValueError, match="evaluation inputs document coverage mismatch"):
        evaluate_verified_model_a_records(inputs)


def test_annotation_digest_must_match_golden_manifest() -> None:
    inputs = _make_inputs(annotation_digest_override="f" * 64)

    with pytest.raises(ValueError, match="annotation artifact digest"):
        evaluate_verified_model_a_records(inputs)


def test_prediction_digest_must_match_prediction_manifest() -> None:
    inputs = _make_inputs(prediction_digest_override="e" * 64)

    with pytest.raises(ValueError, match="prediction artifact digest"):
        evaluate_verified_model_a_records(inputs)


def test_artifact_payload_rejects_bytes_that_do_not_match_claimed_digest() -> None:
    with pytest.raises(ValidationError, match="artifact payload digest mismatch"):
        ArtifactPayload(
            artifact_ref="artifact://predictions/doc-001.json",
            sha256="0" * 64,
            payload=b"not-the-claimed-artifact",
        )


def test_content_ir_must_be_bound_to_the_document_evidence() -> None:
    inputs = _make_inputs(prediction_evidence_digest_override="d" * 64)

    with pytest.raises(ValueError, match="ContentIR EvidenceIR digest mismatch"):
        evaluate_verified_model_a_records(inputs)


def test_prediction_manifest_cannot_change_canonical_metric_definitions() -> None:
    changed = dict(MODEL_A_METRIC_DEFINITION_IDS)
    changed["model_a.overall_cer"] = "different-definition/v1"
    inputs = _make_inputs(metric_definitions=changed)

    with pytest.raises(ValidationError, match="canonical Model A definitions"):
        evaluate_verified_model_a_records(inputs)


def test_prediction_manifest_cannot_supply_aggregate_measurements() -> None:
    inputs = _make_inputs(include_aggregate_measurements=True)

    with pytest.raises(ValidationError, match="measurements"):
        evaluate_verified_model_a_records(inputs)


def test_model_artifact_digest_must_match_model_bundle() -> None:
    inputs = _make_inputs()
    different_model = _artifact(
        inputs.model_a_artifact.artifact_ref,
        b"different-model-weights",
    )
    invalid_inputs = inputs.model_copy(update={"model_a_artifact": different_model})

    with pytest.raises(ValueError, match="model A artifact digest"):
        evaluate_verified_model_a_records(invalid_inputs)


def test_evaluator_artifact_must_match_running_evaluator_bytes() -> None:
    inputs = _make_inputs()
    different_evaluator = _artifact(
        VERIFIED_MODEL_A_EVALUATOR_ARTIFACT_REF,
        b"different-evaluator-build",
    )
    invalid_inputs = inputs.model_copy(
        update={"evaluator_artifact": different_evaluator}
    )

    with pytest.raises(ValueError, match="running evaluator bytes"):
        evaluate_verified_model_a_records(invalid_inputs)


def test_logical_artifact_refs_reject_traversal_segments() -> None:
    payload = b"artifact"

    with pytest.raises(ValidationError, match="artifact_ref"):
        ArtifactPayload(
            artifact_ref="artifact://predictions/../secret.json",
            sha256=_sha(payload),
            payload=payload,
        )


def test_human_attestation_must_match_golden_manifest() -> None:
    inputs = _make_inputs(attestation_reviewer_override="different-reviewer")

    with pytest.raises(ValueError, match="attestation.*disagrees with manifest"):
        evaluate_verified_model_a_records(inputs)


def test_evidence_page_count_must_match_golden_manifest() -> None:
    inputs = _make_inputs(page_count_override=2)

    with pytest.raises(ValueError, match="EvidenceIR page count"):
        evaluate_verified_model_a_records(inputs)
