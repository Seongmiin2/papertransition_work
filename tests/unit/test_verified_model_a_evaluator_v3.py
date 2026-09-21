from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import scan2hwpx.evaluation.model_a_v2 as evaluator_v2_module
import scan2hwpx.evaluation.model_a_v3 as evaluator_v3_module
from scan2hwpx.contracts.models import ContentIR, EvidenceIR, contract_sha256
from scan2hwpx.evaluation.model_a_v2 import (
    MODEL_A_FULL_RECORD_METRIC_SET_ID,
    MODEL_A_V2_METRIC_DEFINITION_IDS,
    VerifiedModelAV2DocumentArtifacts,
    VerifiedModelAV2EvaluationInputs,
)
from scan2hwpx.evaluation.model_a_v3 import (
    MODEL_A_V3_METRIC_SET_ID,
    VERIFIED_MODEL_A_V3_EVALUATOR_ARTIFACT_REF,
    ModelADocumentExecutionManifest,
    ModelAHighRiskAnnotation,
    ModelAHighRiskSurfaceLabel,
    ModelAPageExecutionArtifacts,
    ModelAPageExecutionBinding,
    ModelARawResponsePayload,
    ModelARiskSurface,
    ModelARiskSurfaceKind,
    VerifiedModelAV3DocumentArtifacts,
    VerifiedModelAV3EvaluationInputs,
    canonical_model_a_risk_surfaces,
    canonical_model_a_v3_report_bytes,
    evaluate_verified_model_a_v3_records,
)
from scan2hwpx.evaluation.verified import ArtifactPayload
from scan2hwpx.model_a.inference import (
    BuiltModelAInferenceRequest,
    BuiltModelAInferenceResult,
    ModelAModelArtifact,
    ModelAPageImage,
    assemble_model_a_document,
    build_model_a_inference_request,
    validate_model_a_inference_response,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nmodel-a-v3-fixture"
NodeMutator = Callable[[list[dict[str, Any]]], None]
RiskSelector = Callable[[ModelARiskSurface], bool]


def _sha(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _artifact(ref: str, payload: bytes) -> ArtifactPayload:
    return ArtifactPayload(artifact_ref=ref, sha256=_sha(payload), payload=payload)


def _response_artifact(ref: str, payload: bytes) -> ModelARawResponsePayload:
    return ModelARawResponsePayload(
        artifact_ref=ref,
        sha256=_sha(payload),
        payload=payload,
    )


def _evidence() -> EvidenceIR:
    return EvidenceIR.model_validate(
        {
            "id": "evidence-doc-001",
            "source_document_sha256": _sha("source-doc-001"),
            "sources": [
                {
                    "id": "source-document",
                    "kind": "original_document",
                    "artifact_ref": "artifact://sources/doc-001.pdf",
                    "producer": "fixture",
                    "sha256": _sha("source-doc-001"),
                },
                {
                    "id": "page-image",
                    "kind": "page_image",
                    "artifact_ref": "artifact://pages/doc-001.png",
                    "producer": "fixture",
                    "sha256": _sha(PNG_BYTES),
                },
                {
                    "id": "page-image-2",
                    "kind": "page_image",
                    "artifact_ref": "artifact://pages/doc-002.png",
                    "producer": "fixture",
                    "sha256": _sha(PNG_BYTES),
                },
                {
                    "id": "crop-image",
                    "kind": "crop",
                    "artifact_ref": "artifact://crops/doc-001.png",
                    "producer": "fixture",
                    "sha256": _sha("crop-image"),
                },
                {
                    "id": "ocr-source",
                    "kind": "ocr_provider",
                    "artifact_ref": "artifact://ocr/doc-001.json",
                    "producer": "fixture",
                    "sha256": _sha("ocr-source"),
                },
            ],
            "pages": [
                {
                    "id": "page-001",
                    "page_no": 1,
                    "width": 1000.0,
                    "height": 1400.0,
                    "image_source_ref": "page-image",
                    "observations": [
                        _observation("obs-t1", "text_line", "A"),
                        _observation("obs-t2", "text_line", "B"),
                        _observation("obs-table", "table_grid"),
                        _observation("obs-formula", "formula"),
                        _observation(
                            "obs-image",
                            "image",
                            source_refs=["page-image", "crop-image"],
                        ),
                    ],
                },
                {
                    "id": "page-002",
                    "page_no": 2,
                    "width": 1000.0,
                    "height": 1400.0,
                    "image_source_ref": "page-image-2",
                    "observations": [],
                },
            ],
        }
    )


def _observation(
    observation_id: str,
    kind: str,
    text: str | None = None,
    *,
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": observation_id,
        "kind": kind,
        "bbox": {
            "pixel": [1.0, 1.0, 20.0, 20.0],
            "normalized": [0.001, 0.001, 0.02, 0.02],
        },
        "confidence": 0.99,
        "source_refs": source_refs or ["ocr-source"],
    }
    if text is not None:
        value["ocr_candidates"] = [
            {
                "text": text,
                "provider": "fixture",
                "confidence": 0.99,
                "source_ref": "ocr-source",
                "selected": True,
            }
        ]
    return value


def _nodes() -> list[dict[str, Any]]:
    prefix = "evidence-doc-001:content:page-001:"
    return [
        {
            "kind": "text",
            "id": prefix + "t1",
            "evidence_refs": ["obs-t1"],
            "confidence": 0.8,
            "needs_review": False,
            "role": "question",
            "text": "A",
        },
        {
            "kind": "text",
            "id": prefix + "t2",
            "evidence_refs": ["obs-t2"],
            "confidence": 0.9,
            "needs_review": False,
            "role": "choice",
            "text": "B",
        },
        {
            "kind": "table",
            "id": prefix + "table",
            "evidence_refs": ["obs-table"],
            "confidence": 0.85,
            "needs_review": False,
            "role": "table",
            "rows": 1,
            "columns": 1,
            "cells": [
                {
                    "row": 0,
                    "column": 0,
                    "text": "",
                    "evidence_refs": ["obs-table"],
                }
            ],
        },
        {
            "kind": "formula",
            "id": prefix + "formula",
            "evidence_refs": ["obs-formula"],
            "confidence": 0.95,
            "needs_review": True,
            "role": "formula",
            "expression": "x",
            "format": "latex",
        },
        {
            "kind": "image",
            "id": prefix + "image",
            "evidence_refs": ["obs-image"],
            "confidence": 1.0,
            "needs_review": False,
            "role": "image",
            "asset_ref": "page-image",
        },
    ]


def _make_inputs(
    *,
    mutate_gold: NodeMutator | None = None,
    mutate_prediction: NodeMutator | None = None,
    risk_selector: RiskSelector = lambda _surface: False,
    blocked: bool = False,
    reverse_resolved_pages: bool = False,
    prediction_reading_order: tuple[str, ...] | None = None,
) -> VerifiedModelAV3EvaluationInputs:
    document_id = "doc-001"
    evidence = _evidence()
    evidence_artifact = _artifact(
        "artifact://evidence/doc-001.json",
        _json_bytes(evidence.model_dump(mode="json")),
    )
    gold_nodes = copy.deepcopy(_nodes())
    prediction_nodes = copy.deepcopy(_nodes())
    if mutate_gold is not None:
        mutate_gold(gold_nodes)
    if mutate_prediction is not None:
        mutate_prediction(prediction_nodes)
    gold = ContentIR.model_validate(
        {
            "id": "gold-doc-001",
            "evidence_ir_id": evidence.id,
            "evidence_ir_sha256": contract_sha256(evidence),
            "revision": 2,
            "nodes": gold_nodes,
            "reading_order": [node["id"] for node in gold_nodes],
        }
    )
    gold.assert_evidence_integrity(evidence)
    gold_artifact = _artifact(
        "artifact://gold/doc-001.json",
        _json_bytes(gold.model_dump(mode="json")),
    )
    model_artifact = _artifact(
        "artifact://models/model-a.bin",
        b"model-a-v3-fixture",
    )
    model = ModelAModelArtifact(
        model_id="model-a",
        model_revision="revision-1",
        artifact_sha256=model_artifact.sha256,
    )
    page_runs: list[tuple[BuiltModelAInferenceRequest, bytes, BuiltModelAInferenceResult]] = []
    for page in sorted(evidence.pages, key=lambda item: item.page_no):
        request = build_model_a_inference_request(
            evidence,
            page_image=ModelAPageImage(
                page_id=page.id,
                page_no=page.page_no,
                image_source_ref=page.image_source_ref,
                media_type="image/png",
                raw_bytes=PNG_BYTES,
                sha256=_sha(PNG_BYTES),
            ),
            model=model,
        )
        page_nodes = prediction_nodes if page.page_no == 1 else []
        if blocked and page.page_no == 1:
            raw_response = b"{"
        else:
            raw_response = _json_bytes(
                {
                    "id": request.artifact.source.expected_page_analysis_id,
                    "evidence_ir_id": evidence.id,
                    "evidence_ir_sha256": contract_sha256(evidence),
                    "page_id": page.id,
                    "page_no": page.page_no,
                    "nodes": page_nodes,
                    "reading_order": (
                        list(prediction_reading_order)
                        if page.page_no == 1 and prediction_reading_order is not None
                        else [node["id"] for node in page_nodes]
                    ),
                }
            )
        page_runs.append(
            (
                request,
                raw_response,
                validate_model_a_inference_response(request, raw_response),
            )
        )
    assembly = assemble_model_a_document(
        evidence,
        tuple(result for _request, _response, result in page_runs),
    )
    prediction_payload = (
        _json_bytes(assembly.artifact.content_ir.model_dump(mode="json"))
        if assembly.artifact.content_ir is not None
        else b"{}"
    )
    prediction_artifact = _artifact(
        "artifact://predictions/doc-001.json",
        prediction_payload,
    )

    attestation_ref = "artifact://gold/verifications/doc-001.json"
    attestation_artifact = _artifact(
        attestation_ref,
        _json_bytes(
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "source_sha256": evidence.source_document_sha256,
                "annotation_sha256": gold_artifact.sha256,
                "reviewer_id": "reviewer-001",
                "verified_at": "2026-09-10T09:00:00+09:00",
                "decision": "approved",
                "usage_rights_verified": True,
            }
        ),
    )
    dataset_ref = "artifact://gold/dataset.json"
    dataset_artifact = _artifact(
        dataset_ref,
        _json_bytes(
            {
                "schema_version": "1.0",
                "example_only": False,
                "documents": [
                    {
                        "id": document_id,
                        "lineage_id": "lineage-001",
                        "source_sha256": evidence.source_document_sha256,
                        "template_family": "held-out-001",
                        "split": "test",
                        "capture": "real_scan",
                        "page_count": 2,
                        "annotation_artifact_ref": gold_artifact.artifact_ref,
                        "annotation_sha256": gold_artifact.sha256,
                        "verified": True,
                        "usage_rights_verified": True,
                        "verification_attestation": {
                            "reviewer_id": "reviewer-001",
                            "verified_at": "2026-09-10T09:00:00+09:00",
                            "verification_artifact_ref": attestation_ref,
                            "verification_artifact_sha256": attestation_artifact.sha256,
                        },
                    }
                ],
            }
        ),
    )
    bundle_ref = "artifact://models/bundle.json"
    bundle_artifact = _artifact(
        bundle_ref,
        _json_bytes(
            {
                "schema_version": "1.0",
                "bundle_id": "bundle-1",
                "status": "candidate",
                "created_at": "2026-09-10T09:00:00+09:00",
                "model_a": {
                    "status": "candidate",
                    "model_id": model.model_id,
                    "revision": model.model_revision,
                    "artifact_ref": model_artifact.artifact_ref,
                    "artifact_sha256": model_artifact.sha256,
                },
                "model_b": {
                    "status": "placeholder",
                    "model_id": "model-b",
                    "revision": "none",
                    "artifact_ref": None,
                    "artifact_sha256": None,
                },
                "dataset_manifest_ref": dataset_ref,
                "dataset_manifest_sha256": dataset_artifact.sha256,
                "release_gate_set_id": "initial_v1",
                "capability_profile_ref": "artifact://profiles/default.json",
                "capability_profile_sha256": "2" * 64,
                "hancom_knowledge_manifest_ref": "artifact://hancom/manifest.json",
                "hancom_knowledge_manifest_sha256": "3" * 64,
                "hancom_knowledge_corpus_ref": "artifact://hancom/corpus.json",
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
        ),
    )
    prediction_manifest = _artifact(
        "artifact://predictions/manifest-v2.json",
        _json_bytes(
            {
                "schema_version": "model-a-predictions/2.0",
                "model_bundle_id": "bundle-1",
                "model_bundle_manifest_ref": bundle_ref,
                "model_bundle_manifest_sha256": bundle_artifact.sha256,
                "dataset_manifest_ref": dataset_ref,
                "dataset_manifest_sha256": dataset_artifact.sha256,
                "metric_set_id": MODEL_A_FULL_RECORD_METRIC_SET_ID,
                "metric_definition_ids": MODEL_A_V2_METRIC_DEFINITION_IDS,
                "documents": [
                    {
                        "document_id": document_id,
                        "evidence_artifact_ref": evidence_artifact.artifact_ref,
                        "evidence_artifact_sha256": evidence_artifact.sha256,
                        "prediction_artifact_ref": prediction_artifact.artifact_ref,
                        "prediction_artifact_sha256": prediction_artifact.sha256,
                    }
                ],
            }
        ),
    )
    v2_inputs = VerifiedModelAV2EvaluationInputs(
        dataset_manifest=dataset_artifact,
        model_bundle_manifest=bundle_artifact,
        model_a_artifact=model_artifact,
        prediction_manifest=prediction_manifest,
        evaluator_artifact=_artifact(
            "repo://src/scan2hwpx/evaluation/model_a_v2.py",
            Path(evaluator_v2_module.__file__).read_bytes(),
        ),
        documents=(
            VerifiedModelAV2DocumentArtifacts(
                document_id=document_id,
                evidence=evidence_artifact,
                annotation=gold_artifact,
                prediction=prediction_artifact,
                verification_attestation=attestation_artifact,
            ),
        ),
    )

    surfaces = canonical_model_a_risk_surfaces(evidence, gold)
    risk_annotation = ModelAHighRiskAnnotation(
        document_id=document_id,
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        gold_content_ir_id=gold.id,
        gold_content_ir_revision=gold.revision,
        gold_content_ir_sha256=contract_sha256(gold),
        labels=tuple(
            ModelAHighRiskSurfaceLabel(
                surface=surface,
                high_risk=risk_selector(surface),
            )
            for surface in surfaces
        ),
    )
    risk_artifact = _artifact(
        "artifact://risk/doc-001.json",
        _json_bytes(risk_annotation.model_dump(mode="json")),
    )
    resolved_pages: list[ModelAPageExecutionArtifacts] = []
    page_bindings: list[ModelAPageExecutionBinding] = []
    for request, raw_response, result in page_runs:
        page_no = request.artifact.source.target_page_no
        page_id = request.artifact.source.target_page_id
        request_artifact = _artifact(
            f"artifact://executions/doc-001/page-{page_no}/request.json",
            request.raw_request_bytes,
        )
        response_artifact = _response_artifact(
            f"artifact://executions/doc-001/page-{page_no}/response.json",
            raw_response,
        )
        result_artifact = _artifact(
            f"artifact://executions/doc-001/page-{page_no}/result.json",
            result.result_bytes,
        )
        page_bindings.append(
            ModelAPageExecutionBinding(
                page_id=page_id,
                page_no=page_no,
                response_byte_limit=result.artifact.response_byte_limit,
                request_artifact_ref=request_artifact.artifact_ref,
                request_artifact_sha256=request_artifact.sha256,
                raw_response_artifact_ref=response_artifact.artifact_ref,
                raw_response_artifact_sha256=response_artifact.sha256,
                validation_result_artifact_ref=result_artifact.artifact_ref,
                validation_result_artifact_sha256=result_artifact.sha256,
            )
        )
        resolved_pages.append(
            ModelAPageExecutionArtifacts(
                page_id=page_id,
                page_no=page_no,
                request=request_artifact,
                raw_response=response_artifact,
                validation_result=result_artifact,
            )
        )
    assembly_artifact = _artifact(
        "artifact://executions/doc-001/assembly.json",
        assembly.result_bytes,
    )
    execution_manifest = ModelADocumentExecutionManifest(
        document_id=document_id,
        evidence_artifact_ref=evidence_artifact.artifact_ref,
        evidence_artifact_sha256=evidence_artifact.sha256,
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        gold_annotation_artifact_ref=gold_artifact.artifact_ref,
        gold_annotation_artifact_sha256=gold_artifact.sha256,
        prediction_artifact_ref=prediction_artifact.artifact_ref,
        prediction_artifact_sha256=prediction_artifact.sha256,
        high_risk_annotation_artifact_ref=risk_artifact.artifact_ref,
        high_risk_annotation_artifact_sha256=risk_artifact.sha256,
        model=model,
        pages=tuple(page_bindings),
        assembly_artifact_ref=assembly_artifact.artifact_ref,
        assembly_artifact_sha256=assembly_artifact.sha256,
    )
    return VerifiedModelAV3EvaluationInputs(
        v2_inputs=v2_inputs,
        evaluator_artifact=_artifact(
            VERIFIED_MODEL_A_V3_EVALUATOR_ARTIFACT_REF,
            Path(evaluator_v3_module.__file__).read_bytes(),
        ),
        documents=(
            VerifiedModelAV3DocumentArtifacts(
                document_id=document_id,
                execution_manifest=_artifact(
                    "artifact://executions/doc-001/manifest.json",
                    _json_bytes(execution_manifest.model_dump(mode="json")),
                ),
                high_risk_annotation=risk_artifact,
                assembly=assembly_artifact,
                pages=(
                    tuple(reversed(resolved_pages))
                    if reverse_resolved_pages
                    else tuple(resolved_pages)
                ),
            ),
        ),
    )


def _gold_text_mismatch(nodes: list[dict[str, Any]]) -> None:
    nodes[0]["text"] = "gold-only"


def _gold_table_mismatch(nodes: list[dict[str, Any]]) -> None:
    nodes[2]["cells"][0]["text"] = "gold-only"


def _gold_formula_mismatch(nodes: list[dict[str, Any]]) -> None:
    nodes[3]["expression"] = "y"


def _gold_image_mismatch(nodes: list[dict[str, Any]]) -> None:
    nodes[4]["asset_ref"] = "crop-image"


def _gold_evidence_mismatch(nodes: list[dict[str, Any]]) -> None:
    nodes[0]["evidence_refs"] = ["obs-t2"]


def _prediction_role_mismatch(nodes: list[dict[str, Any]]) -> None:
    nodes[0]["role"] = "instruction"


def _prediction_reviewed_role_mismatch(nodes: list[dict[str, Any]]) -> None:
    _prediction_role_mismatch(nodes)
    nodes[0]["needs_review"] = True


def _prediction_missing_and_extra(nodes: list[dict[str, Any]]) -> None:
    nodes[0]["id"] += "-extra"


def _prediction_confidence_boundaries(nodes: list[dict[str, Any]]) -> None:
    for node, confidence in zip(nodes, (0.0, 0.1, 0.8, 0.9, 1.0), strict=True):
        node["confidence"] = confidence
        node["needs_review"] = confidence < 0.75 or node["kind"] == "formula"


def _reverse_physical_node_order(nodes: list[dict[str, Any]]) -> None:
    nodes.reverse()


def _two_cell_table(nodes: list[dict[str, Any]]) -> None:
    table = nodes[2]
    table["columns"] = 2
    table["cells"] = [
        {
            "row": 0,
            "column": 0,
            "text": "",
            "evidence_refs": ["obs-table"],
        },
        {
            "row": 0,
            "column": 1,
            "text": "",
            "evidence_refs": ["obs-table"],
        },
    ]


def _reversed_two_cell_table(nodes: list[dict[str, Any]]) -> None:
    _two_cell_table(nodes)
    nodes[2]["cells"].reverse()


def _replace_risk_labels(
    inputs: VerifiedModelAV3EvaluationInputs,
    transform: Callable[[list[dict[str, Any]]], None],
) -> VerifiedModelAV3EvaluationInputs:
    document = inputs.documents[0]
    risk_payload = json.loads(document.high_risk_annotation.payload)
    transform(risk_payload["labels"])
    risk_artifact = _artifact(
        document.high_risk_annotation.artifact_ref,
        _json_bytes(risk_payload),
    )
    manifest_payload = json.loads(document.execution_manifest.payload)
    manifest_payload["high_risk_annotation_artifact_sha256"] = risk_artifact.sha256
    manifest_artifact = _artifact(
        document.execution_manifest.artifact_ref,
        _json_bytes(manifest_payload),
    )
    replacement = document.model_copy(
        update={
            "execution_manifest": manifest_artifact,
            "high_risk_annotation": risk_artifact,
        }
    )
    return inputs.model_copy(update={"documents": (replacement,)})


def _replace_risk_payload(
    inputs: VerifiedModelAV3EvaluationInputs,
    payload: bytes,
) -> VerifiedModelAV3EvaluationInputs:
    document = inputs.documents[0]
    risk_artifact = _artifact(document.high_risk_annotation.artifact_ref, payload)
    manifest_payload = json.loads(document.execution_manifest.payload)
    manifest_payload["high_risk_annotation_artifact_sha256"] = risk_artifact.sha256
    manifest_artifact = _artifact(
        document.execution_manifest.artifact_ref,
        _json_bytes(manifest_payload),
    )
    replacement = document.model_copy(
        update={
            "execution_manifest": manifest_artifact,
            "high_risk_annotation": risk_artifact,
        }
    )
    return inputs.model_copy(update={"documents": (replacement,)})


def _mutate_execution_manifest(
    inputs: VerifiedModelAV3EvaluationInputs,
    transform: Callable[[dict[str, Any]], None],
) -> VerifiedModelAV3EvaluationInputs:
    document = inputs.documents[0]
    payload = json.loads(document.execution_manifest.payload)
    transform(payload)
    replacement = document.model_copy(
        update={
            "execution_manifest": _artifact(
                document.execution_manifest.artifact_ref,
                _json_bytes(payload),
            )
        }
    )
    return inputs.model_copy(update={"documents": (replacement,)})


def _replace_first_raw_response(
    inputs: VerifiedModelAV3EvaluationInputs,
    payload: bytes,
) -> VerifiedModelAV3EvaluationInputs:
    document = inputs.documents[0]
    first_page = document.pages[0]
    response = _response_artifact(first_page.raw_response.artifact_ref, payload)
    replacement_page = first_page.model_copy(update={"raw_response": response})
    manifest_payload = json.loads(document.execution_manifest.payload)
    manifest_payload["pages"][0]["raw_response_artifact_sha256"] = response.sha256
    manifest_artifact = _artifact(
        document.execution_manifest.artifact_ref,
        _json_bytes(manifest_payload),
    )
    replacement_document = document.model_copy(
        update={
            "execution_manifest": manifest_artifact,
            "pages": (replacement_page, *document.pages[1:]),
        }
    )
    return inputs.model_copy(update={"documents": (replacement_document,)})


def test_v3_replays_execution_and_preserves_v2_metrics() -> None:
    report = evaluate_verified_model_a_v3_records(_make_inputs())

    assert report.metric_set_id == MODEL_A_V3_METRIC_SET_ID
    assert report.promotable is False
    for metric_id in MODEL_A_V2_METRIC_DEFINITION_IDS:
        assert report.measurements[metric_id] == report.base_v2_report.measurements[metric_id]
    assert report.measurements["model_a.high_risk_review_recall"] is None
    assert report.metric_sample_counts["model_a.confidence_ece"] == 5
    assert report.measurements["model_a.confidence_ece"] == pytest.approx(0.1)
    assert report.calibration_bins[8].sample_count == 2
    assert report.calibration_bins[9].sample_count == 3


@pytest.mark.parametrize(
    ("kind", "gold_mutator", "expected_recall"),
    [
        (ModelARiskSurfaceKind.TEXT_VALUE, _gold_text_mismatch, 0.0),
        (ModelARiskSurfaceKind.TABLE_CELL_TEXT, _gold_table_mismatch, 0.0),
        (ModelARiskSurfaceKind.FORMULA_EXPRESSION, _gold_formula_mismatch, 1.0),
        (ModelARiskSurfaceKind.IMAGE_ASSET_REF, _gold_image_mismatch, 0.0),
        (ModelARiskSurfaceKind.NODE_EVIDENCE_REFS, _gold_evidence_mismatch, 0.0),
    ],
)
def test_high_risk_text_table_formula_image_and_evidence_are_raw_recomputed(
    kind: ModelARiskSurfaceKind,
    gold_mutator: NodeMutator,
    expected_recall: float,
) -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(
            mutate_gold=gold_mutator,
            risk_selector=lambda surface: surface.kind == kind,
        )
    )

    document = report.documents[0]
    assert document.high_risk_mismatch_count == 1
    assert report.measurements["model_a.high_risk_review_recall"] == expected_recall
    expected_route = "node_needs_review" if expected_recall == 1.0 else None
    assert document.high_risk_mismatches[0].route == expected_route


def test_actual_validator_block_is_a_route_and_has_no_confidence_samples() -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(
            blocked=True,
            risk_selector=lambda surface: (
                surface.kind == ModelARiskSurfaceKind.PAGE_OUTPUT_VALID
                and surface.page_ids == ("page-001",)
            ),
        )
    )

    assert report.measurements["model_a.high_risk_review_recall"] == 1.0
    assert report.documents[0].high_risk_mismatches[0].route == "validator_block"
    assert report.documents[0].blocked_page_ids == ("page-001",)
    assert report.metric_sample_counts["model_a.confidence_ece"] == 0
    assert report.measurements["model_a.confidence_ece"] is None


def test_exact_node_review_routes_but_global_mandatory_review_does_not() -> None:
    selector = lambda surface: surface.kind == ModelARiskSurfaceKind.NODE_ROLE
    reviewed = evaluate_verified_model_a_v3_records(
        _make_inputs(
            mutate_prediction=_prediction_reviewed_role_mismatch,
            risk_selector=selector,
        )
    )
    unreviewed_inputs = _make_inputs(
        mutate_prediction=_prediction_role_mismatch,
        risk_selector=selector,
    )
    raw_request = json.loads(
        unreviewed_inputs.documents[0].pages[0].request.payload.decode("utf-8")
    )
    assert raw_request["human_review_required"] is True
    unreviewed = evaluate_verified_model_a_v3_records(unreviewed_inputs)

    assert reviewed.measurements["model_a.high_risk_review_recall"] == 1.0
    assert reviewed.documents[0].high_risk_mismatches[0].route == "node_needs_review"
    assert unreviewed.measurements["model_a.high_risk_review_recall"] == 0.0
    assert unreviewed.documents[0].high_risk_mismatches[0].route is None


def test_high_risk_recall_zero_error_denominator_is_null() -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(risk_selector=lambda surface: surface.kind == ModelARiskSurfaceKind.TEXT_VALUE)
    )

    assert report.metric_sample_counts["model_a.high_risk_review_recall"] == 0
    assert report.measurements["model_a.high_risk_review_recall"] is None


def test_unlabeled_mismatch_is_not_inferred_as_high_risk() -> None:
    report = evaluate_verified_model_a_v3_records(_make_inputs(mutate_gold=_gold_text_mismatch))

    assert report.documents[0].high_risk_surface_count == 0
    assert report.documents[0].high_risk_mismatch_count == 0
    assert report.measurements["model_a.high_risk_review_recall"] is None


def test_ten_bin_boundaries_and_ece_are_recomputed() -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(mutate_prediction=_prediction_confidence_boundaries)
    )

    assert tuple(item.index for item in report.calibration_bins) == tuple(range(10))
    assert report.calibration_bins[0].sample_count == 1
    assert report.calibration_bins[1].sample_count == 1
    assert report.calibration_bins[8].sample_count == 1
    assert report.calibration_bins[9].sample_count == 2
    assert report.calibration_bins[9].upper_inclusive is True
    assert report.measurements["model_a.confidence_ece"] == pytest.approx(0.44)


def test_table_cell_storage_order_does_not_change_calibration_correctness() -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(
            mutate_gold=_two_cell_table,
            mutate_prediction=_reversed_two_cell_table,
        )
    )

    assert report.calibration_bins[8].sample_count == 2
    assert report.calibration_bins[8].correct_count == 2
    assert report.measurements["model_a.confidence_ece"] == pytest.approx(0.1)


def test_table_cell_mismatch_evidence_preserves_identity_and_rejects_forgery() -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(
            mutate_gold=_gold_table_mismatch,
            risk_selector=lambda surface: surface.kind == ModelARiskSurfaceKind.TABLE_CELL_TEXT,
        )
    )
    document = report.documents[0]
    mismatch = document.high_risk_mismatches[0]

    assert mismatch.table_cell == (0, 0, 1, 1)
    forged_mismatch = mismatch.model_copy(update={"surface_id": "risk:" + "0" * 64})
    forged_document = document.model_copy(update={"high_risk_mismatches": (forged_mismatch,)})
    forged_report = report.model_copy(update={"documents": (forged_document,)})
    with pytest.raises(ValueError, match="not a valid strict model"):
        canonical_model_a_v3_report_bytes(forged_report)


def test_missing_gold_and_extra_prediction_nodes_are_not_hidden() -> None:
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(
            mutate_prediction=_prediction_missing_and_extra,
            risk_selector=lambda surface: (
                surface.kind
                in {
                    ModelARiskSurfaceKind.NODE_PRESENCE,
                    ModelARiskSurfaceKind.EXTRA_NODES,
                }
            ),
        )
    )

    document = report.documents[0]
    assert document.missing_gold_node_count == 1
    assert document.extra_predicted_node_count == 1
    assert document.high_risk_mismatch_count == 2
    assert document.high_risk_routed_count == 0
    assert report.measurements["model_a.high_risk_review_recall"] == 0.0
    assert report.measurements["model_a.confidence_ece"] == pytest.approx(0.16)


def test_out_of_order_resolved_page_results_are_deterministic() -> None:
    ordered = evaluate_verified_model_a_v3_records(_make_inputs())
    reversed_pages = evaluate_verified_model_a_v3_records(_make_inputs(reverse_resolved_pages=True))

    assert canonical_model_a_v3_report_bytes(ordered) == canonical_model_a_v3_report_bytes(
        reversed_pages
    )


def test_blocked_page_membership_is_precomputed_once_per_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = evaluator_v3_module._surface_route
    routed_blocked_sets: list[object] = []

    def record_blocked_set(*args: Any, **kwargs: Any) -> Any:
        routed_blocked_sets.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluator_v3_module, "_surface_route", record_blocked_set)
    evaluate_verified_model_a_v3_records(
        _make_inputs(blocked=True, risk_selector=lambda _surface: True)
    )

    assert routed_blocked_sets
    assert all(type(value) is frozenset for value in routed_blocked_sets)
    assert len({id(value) for value in routed_blocked_sets}) == 1


def test_reading_order_uses_page_reading_order_not_physical_node_storage() -> None:
    canonical_order = tuple(node["id"] for node in _nodes())
    report = evaluate_verified_model_a_v3_records(
        _make_inputs(
            mutate_prediction=_reverse_physical_node_order,
            prediction_reading_order=canonical_order,
            risk_selector=lambda surface: surface.kind == ModelARiskSurfaceKind.READING_ORDER,
        )
    )

    assert report.metric_sample_counts["model_a.high_risk_review_recall"] == 0
    assert report.measurements["model_a.high_risk_review_recall"] is None
    assert report.base_v2_report.measurements["model_a.reading_order_exact_match"] == 1.0


@pytest.mark.parametrize(
    "transform",
    [
        lambda labels: labels.pop(),
        lambda labels: labels.reverse(),
        lambda labels: labels.append(copy.deepcopy(labels[0])),
    ],
)
def test_risk_surface_coverage_must_be_exact(
    transform: Callable[[list[dict[str, Any]]], None],
) -> None:
    inputs = _replace_risk_labels(_make_inputs(), transform)

    with pytest.raises(ValueError):
        evaluate_verified_model_a_v3_records(inputs)


def test_stale_validation_result_is_a_trusted_artifact_abort() -> None:
    inputs = _replace_first_raw_response(_make_inputs(), b"{")

    with pytest.raises(ValueError, match="validation result cannot be reproduced"):
        evaluate_verified_model_a_v3_records(inputs)


def test_stale_exact_model_identity_is_a_trusted_artifact_abort() -> None:
    inputs = _mutate_execution_manifest(
        _make_inputs(),
        lambda payload: payload["model"].__setitem__("model_revision", "stale"),
    )

    with pytest.raises(ValueError, match="model identity mismatch"):
        evaluate_verified_model_a_v3_records(inputs)


@pytest.mark.parametrize("mode", ["duplicate", "incomplete"])
def test_duplicate_or_incomplete_resolved_execution_is_rejected(mode: str) -> None:
    inputs = _make_inputs()
    document = inputs.documents[0]
    pages = (
        (document.pages[0], document.pages[0], *document.pages[1:]) if mode == "duplicate" else ()
    )
    forged_document = document.model_copy(update={"pages": pages})
    forged = inputs.model_copy(update={"documents": (forged_document,)})

    with pytest.raises(ValueError, match="not a valid strict model"):
        evaluate_verified_model_a_v3_records(forged)


def test_nested_payload_subclass_and_forged_digest_are_rejected() -> None:
    class PayloadSubclass(ArtifactPayload):
        pass

    inputs = _make_inputs()
    document = inputs.documents[0]
    page = document.pages[0]
    subclass_request = PayloadSubclass.model_validate(
        page.request.model_dump(mode="python"),
        strict=True,
    )
    subclass_page = page.model_copy(update={"request": subclass_request})
    subclass_document = document.model_copy(update={"pages": (subclass_page, *document.pages[1:])})
    subclass_inputs = inputs.model_copy(update={"documents": (subclass_document,)})
    with pytest.raises(TypeError, match="exact payload models"):
        evaluate_verified_model_a_v3_records(subclass_inputs)

    forged_request = ArtifactPayload.model_construct(
        artifact_ref=page.request.artifact_ref,
        sha256="0" * 64,
        payload=page.request.payload,
    )
    forged_page = page.model_copy(update={"request": forged_request})
    forged_document = document.model_copy(update={"pages": (forged_page, *document.pages[1:])})
    forged_inputs = inputs.model_copy(update={"documents": (forged_document,)})
    with pytest.raises(ValueError, match="not a valid strict model"):
        evaluate_verified_model_a_v3_records(forged_inputs)


def test_top_level_subclass_and_mutated_report_are_rejected() -> None:
    class InputsSubclass(VerifiedModelAV3EvaluationInputs):
        pass

    inputs = _make_inputs()
    subclass = InputsSubclass.model_validate(inputs.model_dump(mode="python"), strict=True)
    with pytest.raises(TypeError, match="inputs must be"):
        evaluate_verified_model_a_v3_records(subclass)

    report = evaluate_verified_model_a_v3_records(inputs)
    report.measurements["model_a.confidence_ece"] = 0.0
    with pytest.raises(ValueError, match="not a valid strict model"):
        canonical_model_a_v3_report_bytes(report)


def test_in_memory_cap_allows_measured_v4_request_and_blocks_oversize_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert evaluator_v3_module._MEASURED_V4_PAGE_IMAGE_BYTES == 368_458_051
    assert evaluator_v3_module._MEASURED_V4_IMAGE_BASE64_BYTES == 491_277_404
    assert (
        evaluator_v3_module._MEASURED_V4_IMAGE_BASE64_BYTES
        < evaluator_v3_module.MAX_MODEL_A_V3_TOTAL_ARTIFACT_BYTES
        == 1024 * 1024 * 1024
    )

    inputs = _make_inputs()
    monkeypatch.setattr(evaluator_v3_module, "MAX_MODEL_A_V3_TOTAL_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="future streaming evaluator"):
        evaluate_verified_model_a_v3_records(inputs)


def test_total_size_cap_precedes_forged_payload_digest_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_inputs()
    document = inputs.documents[0]
    page = document.pages[0]
    forged_request = ArtifactPayload.model_construct(
        artifact_ref=page.request.artifact_ref,
        sha256="0" * 64,
        payload=page.request.payload,
    )
    forged_page = page.model_copy(update={"request": forged_request})
    forged_document = document.model_copy(update={"pages": (forged_page, *document.pages[1:])})
    forged_inputs = inputs.model_copy(update={"documents": (forged_document,)})
    monkeypatch.setattr(evaluator_v3_module, "MAX_MODEL_A_V3_TOTAL_ARTIFACT_BYTES", 1)

    with pytest.raises(ValueError, match="future streaming evaluator"):
        evaluate_verified_model_a_v3_records(forged_inputs)


@pytest.mark.parametrize(
    ("owner", "attribute", "metric_id"),
    [
        (
            evaluator_v3_module,
            "MODEL_A_V3_METRIC_DEFINITION_IDS",
            "model_a.confidence_ece",
        ),
        (
            evaluator_v3_module,
            "MODEL_A_V3_METRIC_SAMPLE_UNITS",
            "model_a.confidence_ece",
        ),
        (
            evaluator_v2_module,
            "MODEL_A_V2_METRIC_DEFINITION_IDS",
            "model_a.overall_cer",
        ),
        (
            evaluator_v2_module,
            "MODEL_A_V2_METRIC_SAMPLE_UNITS",
            "model_a.overall_cer",
        ),
    ],
)
def test_runtime_metric_table_mutation_aborts_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    owner: Any,
    attribute: str,
    metric_id: str,
) -> None:
    inputs = _make_inputs()
    table = getattr(owner, attribute)
    monkeypatch.setitem(table, metric_id, "tampered-runtime-definition")

    with pytest.raises(ValueError, match="runtime metric tables were mutated"):
        evaluate_verified_model_a_v3_records(inputs)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            (
                b'{"schema_version":"model-a-high-risk-annotations/1.0",'
                b'"schema_version":"model-a-high-risk-annotations/1.0"}'
            ),
            "duplicate object key",
        ),
        (b'{"value":NaN}', "non-finite number"),
        (b'{"value":9223372036854775808}', "signed 64-bit"),
        (b"[" * 65 + b"0" + b"]" * 65, "nesting limit"),
    ],
)
def test_risk_annotation_uses_bounded_unambiguous_strict_json(
    payload: bytes,
    message: str,
) -> None:
    inputs = _replace_risk_payload(_make_inputs(), payload)

    with pytest.raises(ValueError, match=message):
        evaluate_verified_model_a_v3_records(inputs)


@pytest.mark.parametrize(
    ("constant_name", "message"),
    [
        ("MAX_TRUSTED_JSON_BYTES", "maximum byte size"),
        ("MAX_TRUSTED_JSON_NODES", "JSON node limit"),
        ("MAX_MODEL_A_V3_RISK_SURFACES", "count exceeds limit"),
        ("MAX_MODEL_A_COMPARISON_WORK_UNITS", "comparison work exceeds limit"),
    ],
)
def test_bytes_nodes_counts_and_work_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    message: str,
) -> None:
    inputs = _make_inputs()
    monkeypatch.setattr(evaluator_v3_module, constant_name, 1)

    with pytest.raises(ValueError, match=message):
        evaluate_verified_model_a_v3_records(inputs)
