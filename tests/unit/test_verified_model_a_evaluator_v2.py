from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import scan2hwpx.evaluation.model_a_v2 as evaluator_module
from scan2hwpx.contracts.models import ContentIR, EvidenceIR, contract_sha256
from scan2hwpx.evaluation.model_a_v2 import (
    MAX_MODEL_A_PREDICTION_BYTES,
    MODEL_A_FULL_RECORD_METRIC_SET_ID,
    MODEL_A_V2_METRIC_DEFINITION_IDS,
    MODEL_A_V2_METRIC_SAMPLE_UNITS,
    VERIFIED_MODEL_A_V2_EVALUATOR_ARTIFACT_REF,
    ArtifactPayload,
    VerifiedModelAV2DocumentArtifacts,
    VerifiedModelAV2EvaluationInputs,
    evaluate_verified_model_a_v2_records,
)


def _sha(value: str | bytes) -> str:
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


def _evidence_payload(document_id: str, source_sha256: str) -> dict[str, Any]:
    return {
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
                    {
                        "id": "obs-table",
                        "kind": "table_grid",
                        "bbox": {
                            "pixel": [20.0, 120.0, 800.0, 500.0],
                            "normalized": [0.02, 0.09, 0.8, 0.36],
                        },
                        "confidence": 0.97,
                        "source_refs": ["ocr-source"],
                    },
                    {
                        "id": "obs-formula",
                        "kind": "formula",
                        "bbox": {
                            "pixel": [100.0, 520.0, 700.0, 620.0],
                            "normalized": [0.1, 0.37, 0.7, 0.45],
                        },
                        "confidence": 0.96,
                        "source_refs": ["ocr-source"],
                    },
                ],
            }
        ],
    }


def _content_payload(
    document_id: str,
    evidence: EvidenceIR,
    *,
    prediction: bool,
    include_structured: bool,
    include_image: bool = False,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
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
    ]
    if include_structured:
        nodes.extend(
            [
                {
                    "kind": "table",
                    "id": "table1",
                    "evidence_refs": ["obs-table"],
                    "confidence": 0.97,
                    "role": "table",
                    "rows": 2,
                    "columns": 2,
                    "cells": [
                        {
                            "row": row,
                            "column": column,
                            "text": (
                                "3. 표 셀 [1점]"
                                if (row, column) == (0, 0)
                                else f"{row},{column}"
                            ),
                            "evidence_refs": ["obs-table"],
                        }
                        for row in range(2)
                        for column in range(2)
                    ],
                },
                {
                    "kind": "formula",
                    "id": "formula1",
                    "evidence_refs": ["obs-formula"],
                    "confidence": 0.96,
                    "role": "formula",
                    "expression": r"\frac{x + 1}{2}",
                    "format": "latex",
                },
            ]
        )
    if include_image:
        nodes.append(
            {
                "kind": "image",
                "id": "image1",
                "evidence_refs": ["obs-table"],
                "confidence": 0.95,
                "role": "image",
                "asset_ref": "page-image",
            }
        )
    return {
        "schema_version": "content-ir/1.0",
        "id": ("prediction-" if prediction else "annotation-") + document_id,
        "evidence_ir_id": evidence.id,
        "evidence_ir_sha256": contract_sha256(evidence),
        "revision": 1 if prediction else 2,
        "nodes": nodes,
        "reading_order": [node["id"] for node in nodes],
    }


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
    *,
    prediction_mutator: Callable[[dict[str, Any]], None] | None = None,
    include_structured: bool = True,
    include_image: bool = False,
    evidence_source_override: str | None = None,
    annotation_evidence_sha_override: str | None = None,
    attestation_reviewer_override: str | None = None,
    page_count_override: int | None = None,
    prediction_manifest_dataset_sha_override: str | None = None,
) -> VerifiedModelAV2EvaluationInputs:
    document_id = "doc-001"
    source_sha256 = _sha(f"source:{document_id}")
    evidence_payload = _evidence_payload(
        document_id,
        evidence_source_override or source_sha256,
    )
    evidence = EvidenceIR.model_validate(evidence_payload)
    evidence_artifact = _artifact(
        f"artifact://evidence/{document_id}.json",
        evidence.model_dump_json().encode("utf-8"),
    )

    annotation_payload = _content_payload(
        document_id,
        evidence,
        prediction=False,
        include_structured=include_structured,
        include_image=include_image,
    )
    if annotation_evidence_sha_override is not None:
        annotation_payload["evidence_ir_sha256"] = annotation_evidence_sha_override
    annotation = ContentIR.model_validate(annotation_payload)
    annotation_artifact = _artifact(
        f"artifact://golden/annotations/{document_id}.json",
        annotation.model_dump_json().encode("utf-8"),
    )

    prediction_payload = _content_payload(
        document_id,
        evidence,
        prediction=True,
        include_structured=include_structured,
        include_image=include_image,
    )
    if prediction_mutator is not None:
        prediction_mutator(prediction_payload)
    prediction_artifact = _artifact(
        f"artifact://predictions/{document_id}.json",
        _json_bytes(prediction_payload),
    )

    attestation_ref = f"artifact://golden/verifications/{document_id}.json"
    attestation_artifact = _artifact(
        attestation_ref,
        _json_bytes(
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "source_sha256": source_sha256,
                "annotation_sha256": annotation_artifact.sha256,
                "reviewer_id": attestation_reviewer_override or "reviewer-001",
                "verified_at": "2026-09-09T09:00:00+09:00",
                "decision": "approved",
                "usage_rights_verified": True,
            }
        ),
    )

    dataset_ref = "artifact://golden/dataset-manifest.json"
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
                        "source_sha256": source_sha256,
                        "template_family": "held-out-001",
                        "split": "test",
                        "capture": "real_scan",
                        "page_count": page_count_override or 1,
                        "annotation_artifact_ref": annotation_artifact.artifact_ref,
                        "annotation_sha256": annotation_artifact.sha256,
                        "verified": True,
                        "usage_rights_verified": True,
                        "verification_attestation": {
                            "reviewer_id": "reviewer-001",
                            "verified_at": "2026-09-09T09:00:00+09:00",
                            "verification_artifact_ref": attestation_ref,
                            "verification_artifact_sha256": attestation_artifact.sha256,
                        },
                    }
                ],
            }
        ),
    )
    model_a_artifact = _artifact(
        "artifact://models/model-a-v1.bin",
        b"model-a-weights-fixture",
    )
    bundle_ref = "artifact://models/candidate-bundle-1.json"
    bundle_artifact = _artifact(
        bundle_ref,
        _bundle_payload(
            dataset_ref,
            dataset_artifact.sha256,
            model_a_artifact.artifact_ref,
            model_a_artifact.sha256,
        ),
    )
    prediction_manifest_artifact = _artifact(
        "artifact://predictions/manifest-v2.json",
        _json_bytes(
            {
                "schema_version": "model-a-predictions/2.0",
                "model_bundle_id": "candidate-bundle-1",
                "model_bundle_manifest_ref": bundle_ref,
                "model_bundle_manifest_sha256": bundle_artifact.sha256,
                "dataset_manifest_ref": dataset_ref,
                "dataset_manifest_sha256": (
                    prediction_manifest_dataset_sha_override or dataset_artifact.sha256
                ),
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
    evaluator_artifact = _artifact(
        VERIFIED_MODEL_A_V2_EVALUATOR_ARTIFACT_REF,
        Path(evaluator_module.__file__).read_bytes(),
    )
    return VerifiedModelAV2EvaluationInputs(
        dataset_manifest=dataset_artifact,
        model_bundle_manifest=bundle_artifact,
        model_a_artifact=model_a_artifact,
        prediction_manifest=prediction_manifest_artifact,
        evaluator_artifact=evaluator_artifact,
        documents=(
            VerifiedModelAV2DocumentArtifacts(
                document_id=document_id,
                evidence=evidence_artifact,
                annotation=annotation_artifact,
                prediction=prediction_artifact,
                verification_attestation=attestation_artifact,
            ),
        ),
    )


def _replace_prediction_bytes(
    inputs: VerifiedModelAV2EvaluationInputs,
    payload: bytes,
) -> VerifiedModelAV2EvaluationInputs:
    replacement = _artifact(inputs.documents[0].prediction.artifact_ref, payload)
    manifest = json.loads(inputs.prediction_manifest.payload)
    manifest["documents"][0]["prediction_artifact_sha256"] = replacement.sha256
    manifest_artifact = _artifact(
        inputs.prediction_manifest.artifact_ref,
        _json_bytes(manifest),
    )
    document = inputs.documents[0].model_copy(update={"prediction": replacement})
    return inputs.model_copy(
        update={
            "prediction_manifest": manifest_artifact,
            "documents": (document,),
        }
    )


def test_perfect_records_cover_all_eight_metrics_but_cannot_promote() -> None:
    report = evaluate_verified_model_a_v2_records(_make_inputs())

    assert report.promotable is False
    assert report.metric_definition_ids == MODEL_A_V2_METRIC_DEFINITION_IDS
    assert report.metric_sample_units == MODEL_A_V2_METRIC_SAMPLE_UNITS
    assert report.measurements == {
        "model_a.overall_cer": 0.0,
        "model_a.critical_token_error_rate": 0.0,
        "model_a.role_macro_f1": 1.0,
        "model_a.reading_order_exact_match": 1.0,
        "model_a.reading_order_kendall_tau": 1.0,
        "model_a.table_topology_exact_match": 1.0,
        "model_a.formula_detection_recall": 1.0,
        "model_a.normalized_formula_similarity": 1.0,
    }
    assert report.metric_sample_counts["model_a.table_topology_exact_match"] == 1
    assert report.metric_sample_counts["model_a.formula_detection_recall"] == 1
    assert report.unmeasured_metric_ids == ()
    assert report.structural_hard_failure_document_ids == ()
    assert report.unimplemented_required_gate_ids == (
        "model_a.high_risk_review_recall",
        "model_a.confidence_ece",
    )


def test_absent_gold_table_and_formula_samples_are_unmeasured_not_perfect() -> None:
    report = evaluate_verified_model_a_v2_records(
        _make_inputs(include_structured=False)
    )

    for metric_id in (
        "model_a.table_topology_exact_match",
        "model_a.formula_detection_recall",
        "model_a.normalized_formula_similarity",
    ):
        assert report.metric_sample_counts[metric_id] == 0
        assert report.measurements[metric_id] is None
        assert metric_id in report.unmeasured_metric_ids


def test_missing_text_node_is_measured_conservatively() -> None:
    def remove_question(payload: dict[str, Any]) -> None:
        payload["nodes"] = [node for node in payload["nodes"] if node["id"] != "q1"]
        payload["reading_order"].remove("q1")

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=remove_question)
    )
    document = report.documents[0]

    assert document.prediction_record_valid
    assert document.missing_node_count == 1
    assert document.extra_node_count == 0
    assert report.measurements["model_a.overall_cer"] > 0
    assert report.measurements["model_a.role_macro_f1"] < 1
    assert report.measurements["model_a.reading_order_exact_match"] == 0
    assert report.measurements["model_a.reading_order_kendall_tau"] == 0


def test_extra_text_node_is_measured_in_text_role_and_order_metrics() -> None:
    def add_extra(payload: dict[str, Any]) -> None:
        payload["nodes"].append(
            {
                "kind": "text",
                "id": "extra1",
                "evidence_refs": ["obs-c1"],
                "confidence": 0.8,
                "role": "other",
                "text": "99. 잘못 추가됨",
            }
        )
        payload["reading_order"].append("extra1")

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=add_extra)
    )
    document = report.documents[0]

    assert document.extra_node_count == 1
    assert report.measurements["model_a.overall_cer"] > 0
    assert report.measurements["model_a.role_macro_f1"] < 1
    assert report.measurements["model_a.reading_order_exact_match"] == 0
    assert report.measurements["model_a.reading_order_kendall_tau"] == 0


def test_wrong_table_topology_scores_zero_without_aborting() -> None:
    def change_topology(payload: dict[str, Any]) -> None:
        table = next(node for node in payload["nodes"] if node["id"] == "table1")
        table["rows"] = 1
        table["columns"] = 4
        table["cells"] = [
            {
                "row": 0,
                "column": column,
                "text": str(column),
                "evidence_refs": ["obs-table"],
            }
            for column in range(4)
        ]

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=change_topology)
    )

    assert report.documents[0].prediction_record_valid
    assert report.measurements["model_a.table_topology_exact_match"] == 0


def test_wrong_table_cell_text_is_counted_by_cer_and_critical_token_rate() -> None:
    def change_cell_text(payload: dict[str, Any]) -> None:
        table = next(node for node in payload["nodes"] if node["id"] == "table1")
        table["cells"][0]["text"] = "4. 다른 셀 [2점]"

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=change_cell_text)
    )

    assert report.measurements["model_a.table_topology_exact_match"] == 1
    assert report.measurements["model_a.overall_cer"] > 0
    assert report.measurements["model_a.critical_token_error_rate"] > 0


def test_wrong_evidence_grounding_is_a_structural_hard_failure() -> None:
    def change_evidence_refs(payload: dict[str, Any]) -> None:
        replacements = {
            "q1": "obs-c1",
            "c1": "obs-q1",
            "table1": "obs-formula",
            "formula1": "obs-table",
        }
        for node in payload["nodes"]:
            node["evidence_refs"] = [replacements[node["id"]]]
            for cell in node.get("cells", []):
                cell["evidence_refs"] = ["obs-q1"]

    baseline = evaluate_verified_model_a_v2_records(_make_inputs())
    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=change_evidence_refs)
    )

    assert report.measurements == baseline.measurements
    assert report.documents[0].evidence_grounding_mismatch_count == 8
    assert report.documents[0].image_asset_mismatch_count == 0
    assert report.structural_hard_failure_document_ids == ("doc-001",)


def test_wrong_image_asset_is_a_structural_hard_failure() -> None:
    def change_asset(payload: dict[str, Any]) -> None:
        image = next(node for node in payload["nodes"] if node["id"] == "image1")
        image["asset_ref"] = "ocr-source"

    baseline = evaluate_verified_model_a_v2_records(_make_inputs(include_image=True))
    report = evaluate_verified_model_a_v2_records(
        _make_inputs(include_image=True, prediction_mutator=change_asset)
    )

    assert report.measurements == baseline.measurements
    assert report.documents[0].evidence_grounding_mismatch_count == 0
    assert report.documents[0].image_asset_mismatch_count == 1
    assert report.structural_hard_failure_document_ids == ("doc-001",)


def test_missing_table_cell_text_is_aligned_to_empty_prediction() -> None:
    def clear_cell_text(payload: dict[str, Any]) -> None:
        table = next(node for node in payload["nodes"] if node["id"] == "table1")
        table["cells"][0]["text"] = ""

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=clear_cell_text)
    )

    assert report.measurements["model_a.table_topology_exact_match"] == 1
    assert report.measurements["model_a.overall_cer"] > 0
    assert report.measurements["model_a.critical_token_error_rate"] > 0


def test_extra_table_cells_are_aligned_from_empty_reference_text() -> None:
    def add_column(payload: dict[str, Any]) -> None:
        table = next(node for node in payload["nodes"] if node["id"] == "table1")
        table["columns"] = 3
        table["cells"].extend(
            [
                {
                    "row": 0,
                    "column": 2,
                    "text": "5. 추가 셀 [3점]",
                    "evidence_refs": ["obs-table"],
                },
                {
                    "row": 1,
                    "column": 2,
                    "text": "추가",
                    "evidence_refs": ["obs-table"],
                },
            ]
        )

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=add_column)
    )

    assert report.measurements["model_a.table_topology_exact_match"] == 0
    assert report.measurements["model_a.overall_cer"] > 0
    assert report.measurements["model_a.critical_token_error_rate"] > 0


def test_formula_kind_mismatch_is_a_miss_and_not_an_evaluator_error() -> None:
    def replace_formula_with_text(payload: dict[str, Any]) -> None:
        index = next(
            index for index, node in enumerate(payload["nodes"]) if node["id"] == "formula1"
        )
        payload["nodes"][index] = {
            "kind": "text",
            "id": "formula1",
            "evidence_refs": ["obs-formula"],
            "confidence": 0.7,
            "role": "formula",
            "text": "x+1",
        }

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=replace_formula_with_text)
    )
    document = report.documents[0]

    assert document.prediction_record_valid
    assert document.kind_mismatch_count == 1
    assert report.measurements["model_a.formula_detection_recall"] == 0
    assert report.measurements["model_a.normalized_formula_similarity"] == 0
    assert report.measurements["model_a.role_macro_f1"] < 1


def test_missed_formula_and_extra_formula_are_both_visible() -> None:
    def replace_with_extra(payload: dict[str, Any]) -> None:
        payload["nodes"] = [node for node in payload["nodes"] if node["id"] != "formula1"]
        payload["reading_order"].remove("formula1")
        payload["nodes"].append(
            {
                "kind": "formula",
                "id": "formula-extra",
                "evidence_refs": ["obs-formula"],
                "confidence": 0.8,
                "role": "formula",
                "expression": "x+1",
                "format": "latex",
            }
        )
        payload["reading_order"].append("formula-extra")

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=replace_with_extra)
    )
    document = report.documents[0]

    assert document.missing_node_count == 1
    assert document.extra_node_count == 1
    assert document.extra_formula_count == 1
    assert report.measurements["model_a.formula_detection_recall"] == 0
    assert report.measurements["model_a.normalized_formula_similarity"] == 0
    assert report.measurements["model_a.reading_order_exact_match"] == 0


def test_formula_similarity_normalizes_whitespace_but_counts_symbol_changes() -> None:
    def whitespace_only(payload: dict[str, Any]) -> None:
        formula = next(node for node in payload["nodes"] if node["id"] == "formula1")
        formula["expression"] = "  \\frac { x+1 } { 2 }  "

    normalized = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=whitespace_only)
    )
    assert normalized.measurements["model_a.normalized_formula_similarity"] == 1

    def changed_symbol(payload: dict[str, Any]) -> None:
        formula = next(node for node in payload["nodes"] if node["id"] == "formula1")
        formula["expression"] = r"\frac{x+2}{2}"

    changed = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=changed_symbol)
    )
    similarity = changed.measurements["model_a.normalized_formula_similarity"]
    assert similarity is not None
    assert 0 < similarity < 1


@pytest.mark.parametrize(
    ("payload_factory", "failure"),
    [
        (lambda valid: b"{", "invalid_json"),
        (
            lambda valid: b'{"schema_version":"content-ir/1.0",' + valid[1:],
            "invalid_json",
        ),
        (lambda valid: valid.replace(b'"confidence":0.99', b'"confidence":NaN'), "invalid_json"),
        (
            lambda valid: valid.replace(
                b'"revision":1', b'"revision":12345678901234567890'
            ),
            "invalid_json",
        ),
        (
            lambda valid: valid.replace(
                b'"revision":1', b'"revision":9223372036854775808'
            ),
            "invalid_json",
        ),
        (
            lambda valid: valid.replace(b'"confidence":0.99', b'"confidence":1e309'),
            "invalid_json",
        ),
        (lambda valid: b"[" * 100 + b"0" + b"]" * 100, "invalid_json"),
        (lambda valid: b" " * (MAX_MODEL_A_PREDICTION_BYTES + 1), "size_limit"),
        (lambda valid: b"{}", "schema_invalid"),
    ],
    ids=[
        "malformed",
        "duplicate-key",
        "nan",
        "integer-digits",
        "signed64-overflow",
        "float-overflow",
        "deep",
        "huge",
        "schema-invalid",
    ],
)
def test_invalid_predictions_are_measured_not_raised(
    payload_factory: Callable[[bytes], bytes],
    failure: str,
) -> None:
    inputs = _make_inputs()
    invalid_inputs = _replace_prediction_bytes(
        inputs,
        payload_factory(inputs.documents[0].prediction.payload),
    )

    report = evaluate_verified_model_a_v2_records(invalid_inputs)
    document = report.documents[0]

    assert not document.prediction_record_valid
    assert document.prediction_failure == failure
    assert document.missing_node_count == document.gold_node_count
    assert document.predicted_node_count is None
    assert report.measurements["model_a.overall_cer"] == 1
    assert report.measurements["model_a.role_macro_f1"] == 0
    assert report.measurements["model_a.reading_order_exact_match"] == 0
    assert report.measurements["model_a.table_topology_exact_match"] == 0
    assert report.measurements["model_a.formula_detection_recall"] == 0
    assert report.measurements["model_a.normalized_formula_similarity"] == 0


def test_prediction_json_node_limit_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_inputs()
    monkeypatch.setattr(evaluator_module, "MAX_MODEL_A_PREDICTION_JSON_NODES", 1)

    report = evaluate_verified_model_a_v2_records(inputs)

    assert report.documents[0].prediction_failure == "invalid_json"


def test_prediction_content_node_limit_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def add_extra(payload: dict[str, Any]) -> None:
        payload["nodes"].append(
            {
                "kind": "text",
                "id": "extra-limit",
                "evidence_refs": ["obs-c1"],
                "confidence": 0.8,
                "role": "other",
                "text": "extra",
            }
        )
        payload["reading_order"].append("extra-limit")

    inputs = _make_inputs(prediction_mutator=add_extra)
    monkeypatch.setattr(evaluator_module, "MAX_CONTENT_NODES", 4)

    report = evaluate_verified_model_a_v2_records(inputs)

    assert report.documents[0].prediction_failure == "collection_limit"
    assert report.documents[0].missing_node_count == 4


def test_prediction_table_cell_limit_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def add_column(payload: dict[str, Any]) -> None:
        table = next(node for node in payload["nodes"] if node["id"] == "table1")
        table["columns"] = 3
        table["cells"].extend(
            [
                {
                    "row": row,
                    "column": 2,
                    "text": "extra",
                    "evidence_refs": ["obs-table"],
                }
                for row in range(2)
            ]
        )

    inputs = _make_inputs(prediction_mutator=add_column)
    monkeypatch.setattr(evaluator_module, "MAX_TABLE_CELLS_PER_DOCUMENT", 4)

    report = evaluate_verified_model_a_v2_records(inputs)

    assert report.documents[0].prediction_failure == "collection_limit"


def test_prediction_table_grid_limit_is_checked_before_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def enlarge_grid(payload: dict[str, Any]) -> None:
        table = next(node for node in payload["nodes"] if node["id"] == "table1")
        table["rows"] = 3

    inputs = _make_inputs(prediction_mutator=enlarge_grid)
    monkeypatch.setattr(evaluator_module, "MAX_TABLE_GRID_POSITIONS", 4)

    report = evaluate_verified_model_a_v2_records(inputs)

    assert report.documents[0].prediction_failure == "collection_limit"


def test_prediction_comparison_work_limit_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_inputs()
    monkeypatch.setattr(
        evaluator_module,
        "MAX_MODEL_A_COMPARISON_WORK_UNITS",
        1,
        raising=False,
    )

    report = evaluate_verified_model_a_v2_records(inputs)

    assert report.documents[0].prediction_failure == "comparison_limit"
    assert report.measurements["model_a.overall_cer"] == 1
    assert report.measurements["model_a.formula_detection_recall"] == 0


def test_trusted_json_node_limit_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_inputs()
    monkeypatch.setattr(evaluator_module, "MAX_TRUSTED_JSON_NODES", 1)

    with pytest.raises(ValueError, match="maximum JSON node count"):
        evaluate_verified_model_a_v2_records(inputs)


def test_prediction_evidence_lineage_error_is_measured_not_trusted() -> None:
    def break_prediction_lineage(payload: dict[str, Any]) -> None:
        payload["evidence_ir_sha256"] = "f" * 64

    report = evaluate_verified_model_a_v2_records(
        _make_inputs(prediction_mutator=break_prediction_lineage)
    )

    assert report.documents[0].prediction_failure == "evidence_integrity_invalid"
    assert report.documents[0].missing_node_count == report.documents[0].gold_node_count
    assert report.measurements["model_a.overall_cer"] == 1
    assert report.measurements["model_a.table_topology_exact_match"] == 0
    assert report.measurements["model_a.formula_detection_recall"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evidence_source_override": "d" * 64}, "EvidenceIR source digest"),
        ({"annotation_evidence_sha_override": "e" * 64}, "ContentIR EvidenceIR digest"),
        ({"attestation_reviewer_override": "different-reviewer"}, "attestation.*disagrees"),
        ({"page_count_override": 2}, "EvidenceIR page count"),
        (
            {"prediction_manifest_dataset_sha_override": "f" * 64},
            "prediction manifest dataset manifest artifact digest",
        ),
    ],
    ids=["evidence", "golden", "attestation", "dataset", "manifest"],
)
def test_trusted_input_mismatches_abort(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_verified_model_a_v2_records(_make_inputs(**kwargs))


def test_prediction_digest_binding_mismatch_aborts_instead_of_scoring_bytes() -> None:
    inputs = _make_inputs()
    manifest = json.loads(inputs.prediction_manifest.payload)
    manifest["documents"][0]["prediction_artifact_sha256"] = "f" * 64
    changed_manifest = _artifact(
        inputs.prediction_manifest.artifact_ref,
        _json_bytes(manifest),
    )

    with pytest.raises(ValueError, match="prediction artifact digest"):
        evaluate_verified_model_a_v2_records(
            inputs.model_copy(update={"prediction_manifest": changed_manifest})
        )


def test_artifact_digest_is_rechecked_at_the_evaluation_boundary() -> None:
    inputs = _make_inputs()
    bypassed_validation = inputs.documents[0].annotation.model_copy(
        update={"payload": b'{"tampered":true}'}
    )
    changed_document = inputs.documents[0].model_copy(
        update={"annotation": bypassed_validation}
    )

    with pytest.raises(ValueError, match="artifact payload digest mismatch"):
        evaluate_verified_model_a_v2_records(
            inputs.model_copy(update={"documents": (changed_document,)})
        )


def test_trusted_manifest_duplicate_key_aborts() -> None:
    inputs = _make_inputs()
    duplicate = b'{"schema_version":"1.0",' + inputs.dataset_manifest.payload[1:]
    changed_dataset = _artifact(inputs.dataset_manifest.artifact_ref, duplicate)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        evaluate_verified_model_a_v2_records(
            inputs.model_copy(update={"dataset_manifest": changed_dataset})
        )


def test_forged_mutable_artifact_payload_is_rejected_at_public_boundary() -> None:
    inputs = _make_inputs()
    forged_evaluator = inputs.evaluator_artifact.model_copy(
        update={"payload": bytearray(inputs.evaluator_artifact.payload)}
    )
    forged_inputs = inputs.model_copy(update={"evaluator_artifact": forged_evaluator})

    with pytest.raises(ValueError, match="valid strict evaluation inputs"):
        evaluate_verified_model_a_v2_records(forged_inputs)


def test_invalid_prediction_evidence_cannot_claim_success_contributions() -> None:
    inputs = _make_inputs()
    report = evaluate_verified_model_a_v2_records(
        _replace_prediction_bytes(inputs, b"{")
    )
    forged = report.documents[0].model_dump(mode="python")
    forged.update(
        {
            "reading_order_exact": True,
            "reading_order_kendall_tau": 1.0,
        }
    )

    with pytest.raises(ValueError, match="invalid prediction contributions"):
        type(report.documents[0]).model_validate(forged, strict=True)


def test_valid_prediction_diagnostics_must_match_role_alignment_sentinels() -> None:
    report = evaluate_verified_model_a_v2_records(_make_inputs())
    forged = report.documents[0].model_dump(mode="python")
    forged["kind_mismatch_count"] = 1

    with pytest.raises(ValueError, match="role alignment diagnostics"):
        type(report.documents[0]).model_validate(forged, strict=True)


def test_measurement_report_rejects_duplicate_document_sample_inflation() -> None:
    report = evaluate_verified_model_a_v2_records(_make_inputs())
    forged = report.model_dump(mode="python")
    forged["evaluated_document_ids"] = (report.evaluated_document_ids[0],) * 2
    forged["documents"] = (forged["documents"][0],) * 2
    forged["metric_sample_counts"] = {
        metric_id: sample_count * 2
        for metric_id, sample_count in report.metric_sample_counts.items()
    }

    with pytest.raises(ValueError, match="duplicate report document ids"):
        type(report).model_validate(forged, strict=True)


def test_measurement_report_recomputes_structural_hard_failure_ids() -> None:
    report = evaluate_verified_model_a_v2_records(_make_inputs())
    forged = report.model_dump(mode="python")
    forged["structural_hard_failure_document_ids"] = ("doc-001",)

    with pytest.raises(ValueError, match="structural hard-failure ids"):
        type(report).model_validate(forged, strict=True)
