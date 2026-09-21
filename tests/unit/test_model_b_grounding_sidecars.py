from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scan2hwpx.evaluation.model_b_grounding_sidecars as grounding_module
from scan2hwpx.contracts import (
    ContentIR,
    ContentPlanItem,
    HwpDocumentPlan,
    contract_sha256,
)
from scan2hwpx.evaluation.model_b_grounding_sidecars import (
    CHUNK_CANONICALIZATION,
    CHUNK_RECORD_FIELDS,
    ModelBGroundingSidecarBuildError,
    build_model_b_grounding_sidecars,
    canonical_hancom_chunk_record_bytes,
    load_verified_model_b_grounding_sidecar,
    retrieved_chunk_sha256,
)
from scan2hwpx.evaluation.model_b_plan_review import ModelBPlanGroundingEvidence
from scan2hwpx.knowledge.hancom import load_chunks


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contract_binding(path: Path, value: ContentIR | HwpDocumentPlan, root: Path) -> dict[str, str]:
    payload = _write_json(path, value.model_dump(mode="json"))
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(payload),
        "contract_sha256": contract_sha256(value),
    }


def _artifact_binding(path: Path, payload: bytes, root: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(payload),
    }


def _content(index: int) -> ContentIR:
    return ContentIR.model_validate(
        {
            "id": f"content-{index}",
            "evidence_ir_id": f"evidence-{index}",
            "evidence_ir_sha256": "1" * 64,
            "revision": 1,
            "nodes": [
                {
                    "id": f"text-{index}",
                    "kind": "text",
                    "role": "question",
                    "text": f"immutable content {index}",
                    "evidence_refs": [f"obs-{index}"],
                    "confidence": 1.0,
                    "needs_review": True,
                }
            ],
            "reading_order": [f"text-{index}"],
        }
    )


def _plan(
    index: int,
    content: ContentIR,
    *,
    official_ref: str = "official-hwpx-fixture",
) -> HwpDocumentPlan:
    return HwpDocumentPlan(
        id=f"plan-{index}",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        capability_profile_id="exam2hwpx-authoring-v1",
        design_profile_id="fixture-v1",
        official_spec_refs=(official_ref,),
        page_layout={
            "width_mm": 210.0,
            "height_mm": 297.0,
            "margin_top_mm": 15.0,
            "margin_right_mm": 15.0,
            "margin_bottom_mm": 15.0,
            "margin_left_mm": 15.0,
        },
        flow=(
            ContentPlanItem(
                id=f"flow-{index}",
                render_as="paragraph",
                content_ref=f"text-{index}",
            ),
        ),
    )


@dataclass(frozen=True)
class _Fixture:
    candidate_root: Path
    corpus_path: Path
    capability_path: Path
    lineages: tuple[str, ...]


def _write_fixture(
    tmp_path: Path,
    *,
    document_count: int = 1,
    bad_plan_index: int | None = None,
) -> _Fixture:
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
    capability_path = tmp_path / "capability" / "profile.json"
    capability_bytes = _write_json(capability_path, capability_value)
    chunk_value = {
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
    corpus_path = tmp_path / "knowledge" / "chunks.jsonl"
    corpus_path.parent.mkdir(parents=True)
    corpus_bytes = (json.dumps(chunk_value, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    corpus_path.write_bytes(corpus_bytes)

    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    documents: list[dict[str, Any]] = []
    lineages: list[str] = []
    for index in range(document_count):
        lineage = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        lineages.append(lineage)
        document_dir = candidate_root / lineage
        document_dir.mkdir()
        content = _content(index)
        official_ref = "not-retrieved" if index == bad_plan_index else "official-hwpx-fixture"
        plan = _plan(index, content, official_ref=official_ref)
        content_binding = _contract_binding(
            document_dir / "content_ir.candidate.json",
            content,
            candidate_root,
        )
        plan_binding = _contract_binding(
            document_dir / "hwp_document_plan.candidate.json",
            plan,
            candidate_root,
        )
        projection_binding = _artifact_binding(
            document_dir / "projection_source.json",
            b"{}\n",
            candidate_root,
        )
        evidence_binding = {
            **_artifact_binding(
                document_dir / "evidence_ir.json",
                b"{}\n",
                candidate_root,
            ),
            "contract_sha256": "2" * 64,
        }
        report_binding = _artifact_binding(
            document_dir / "report.json",
            b"{}\n",
            candidate_root,
        )
        page_binding = _artifact_binding(
            document_dir / "pages" / "page-0001.png",
            b"fixture-page",
            candidate_root,
        )
        documents.append(
            {
                "document_id": f"candidate-{lineage[:24]}",
                "lineage_id": lineage,
                "split": "train",
                "page_count": 1,
                "status": "needs_human_review",
                "source_artifacts": {
                    "hwp": {"sha256": lineage},
                    "hwpx": {"sha256": hashlib.sha256(f"hwpx-{index}".encode()).hexdigest()},
                    "pdf": {"sha256": hashlib.sha256(f"pdf-{index}".encode()).hexdigest()},
                },
                "artifacts": {
                    "projection_source": projection_binding,
                    "evidence_ir": evidence_binding,
                    "content_ir_candidate": content_binding,
                    "hwp_document_plan_candidate": plan_binding,
                    "report": report_binding,
                    "page_images": [page_binding],
                    "image_crops": [],
                },
                "issue_codes": ["human_review_required"],
            }
        )
    manifest = {
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
            "manifest_sha256": "3" * 64,
            "source_dataset_report_sha256": "4" * 64,
            "source_inventory_sha256": "5" * 64,
            "producer": {"name": "fixture", "version": "1"},
        },
        "producer": {
            "name": "fixture-candidate-builder",
            "version": "1",
            "pymupdf_version": "1",
            "dpi": 300,
            "pdf_text_order": "native",
        },
        "capability_profile": {
            "id": "exam2hwpx-authoring-v1",
            "sha256": _sha256(capability_bytes),
        },
        "knowledge_corpus_sha256": _sha256(corpus_bytes),
        "contract_versions": {
            "projection_source": "hwpx-projection-source/1.0",
            "evidence_ir": "evidence-ir/1.0",
            "content_ir": "content-ir/1.0",
            "hwp_document_plan": "hwp-document-plan/1.0",
        },
        "document_count": document_count,
        "page_count": document_count,
        "splits": {
            "train": {"documents": document_count, "pages": document_count},
            "validation": {"documents": 0, "pages": 0},
            "test": {"documents": 0, "pages": 0},
        },
        "issue_counts": {"human_review_required": document_count},
        "documents": documents,
    }
    _write_json(candidate_root / "manifest.json", manifest)
    return _Fixture(
        candidate_root=candidate_root,
        corpus_path=corpus_path,
        capability_path=capability_path,
        lineages=tuple(lineages),
    )


def test_builds_reproducible_noneligible_sidecars_and_verified_handoff(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "sidecars"
    source_snapshots = {
        path: path.read_bytes()
        for path in (
            fixture.candidate_root / "manifest.json",
            fixture.corpus_path,
            fixture.capability_path,
            fixture.candidate_root / fixture.lineages[0] / "content_ir.candidate.json",
            fixture.candidate_root / fixture.lineages[0] / "hwp_document_plan.candidate.json",
        )
    }

    manifest = build_model_b_grounding_sidecars(
        fixture.candidate_root,
        fixture.corpus_path,
        fixture.capability_path,
        output,
        max_workers=2,
    )

    assert manifest["document_count"] == 1
    assert manifest["training_eligible"] is False
    assert manifest["release_eligible"] is False
    assert manifest["rights_status"] == "unverified"
    assert manifest["retrieval"]["retrieved_chunk_count"] == 1
    assert manifest["retrieval"]["retrieved_chunk_digest"] == {
        "algorithm": "sha256",
        "canonical_record_fields": list(CHUNK_RECORD_FIELDS),
        "canonicalization": CHUNK_CANONICALIZATION,
    }
    assert len(list(output.rglob("*.*"))) == 3

    chunk = load_chunks(fixture.corpus_path)[0]
    expected_canonical = (
        '{"id":"official-hwpx-fixture","section":"manifest header style reference",'
        '"source_id":"hancom-hwpx-format","source_sha256":"'
        + "f"
        * 64
        + '","source_url":"https://tech.hancom.com/hwpxformat/","tags":'
        '["hwpx","package","paragraph","table","style-reference","validation"],'
        '"text":"HWPX package manifest header section paragraph table style validation DVC.",'
        '"title":"HWPX package paragraph table validation"}'
    ).encode()
    assert canonical_hancom_chunk_record_bytes(chunk) == expected_canonical
    assert retrieved_chunk_sha256(chunk) == _sha256(expected_canonical)

    lineage = fixture.lineages[0]
    retrieval_path = output / lineage / "retrieval.json"
    evidence_path = output / lineage / "grounding_evidence.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    evidence = ModelBPlanGroundingEvidence.model_validate_json(
        evidence_path.read_bytes(),
        strict=True,
    )
    assert evidence.retrieval_artifact_sha256 == _sha256(retrieval_path.read_bytes())
    assert evidence.official_spec_refs[0].retrieved_chunk_sha256 == _sha256(expected_canonical)
    assert retrieval["candidate_bundle"]["manifest_sha256"] == _sha256(
        (fixture.candidate_root / "manifest.json").read_bytes()
    )

    verified = load_verified_model_b_grounding_sidecar(
        fixture.candidate_root,
        output,
        fixture.corpus_path,
        fixture.capability_path,
        lineage_id=lineage,
    )
    assert verified.content_ir.id == "content-0"
    assert verified.hwp_document_plan.id == "plan-0"
    assert verified.grounding_evidence == evidence
    assert all(path.read_bytes() == before for path, before in source_snapshots.items())

    second_output = tmp_path / "sidecars-second"
    second_manifest = build_model_b_grounding_sidecars(
        fixture.candidate_root,
        fixture.corpus_path,
        fixture.capability_path,
        second_output,
        max_workers=1,
    )
    assert second_manifest == manifest
    first_files = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_output).as_posix(): path.read_bytes()
        for path in second_output.rglob("*")
        if path.is_file()
    }
    assert second_files == first_files

    with pytest.raises(FileExistsError, match="already exists"):
        build_model_b_grounding_sidecars(
            fixture.candidate_root,
            fixture.corpus_path,
            fixture.capability_path,
            output,
            max_workers=1,
        )


def test_verified_loader_rejects_self_consistent_retrieval_tamper(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "sidecars"
    build_model_b_grounding_sidecars(
        fixture.candidate_root,
        fixture.corpus_path,
        fixture.capability_path,
        output,
        max_workers=1,
    )
    lineage = fixture.lineages[0]
    retrieval_path = output / lineage / "retrieval.json"
    evidence_path = output / lineage / "grounding_evidence.json"
    manifest_path = output / "manifest.json"

    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieval["retrieved_chunks"][0]["record"]["text"] = "forged official text"
    retrieval_bytes = _write_json(retrieval_path, retrieval)
    evidence_value = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_value["retrieval_artifact_sha256"] = _sha256(retrieval_bytes)
    evidence = ModelBPlanGroundingEvidence.model_validate_json(
        _json_bytes(evidence_value),
        strict=True,
    )
    evidence_bytes = _write_json(evidence_path, evidence.model_dump(mode="json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_document = manifest["documents"][0]
    manifest_document["artifacts"]["retrieval"]["sha256"] = _sha256(retrieval_bytes)
    manifest_document["artifacts"]["grounding_evidence"].update(
        {
            "sha256": _sha256(evidence_bytes),
            "contract_sha256": contract_sha256(evidence),
        }
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ModelBGroundingSidecarBuildError,
        match="cannot be reproduced",
    ):
        load_verified_model_b_grounding_sidecar(
            fixture.candidate_root,
            output,
            fixture.corpus_path,
            fixture.capability_path,
            lineage_id=lineage,
        )


def test_rejects_candidate_corpus_binding_mismatch_before_output(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    manifest_path = fixture.candidate_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["knowledge_corpus_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    output = tmp_path / "sidecars"

    with pytest.raises(
        ModelBGroundingSidecarBuildError,
        match="knowledge corpus SHA-256",
    ):
        build_model_b_grounding_sidecars(
            fixture.candidate_root,
            fixture.corpus_path,
            fixture.capability_path,
            output,
            max_workers=1,
        )

    assert not output.exists()


def test_document_failure_isolated_and_no_partial_bundle_is_published(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path, document_count=2, bad_plan_index=1)
    output = tmp_path / "sidecars"

    with pytest.raises(
        ModelBGroundingSidecarBuildError,
        match=fixture.lineages[1],
    ) as caught:
        build_model_b_grounding_sidecars(
            fixture.candidate_root,
            fixture.corpus_path,
            fixture.capability_path,
            output,
            max_workers=2,
        )

    assert caught.value.__cause__ is not None
    assert "official_spec_refs" in str(caught.value.__cause__)
    assert fixture.lineages[0] not in str(caught.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".sidecars.*.partial"))


@pytest.mark.parametrize(
    ("limit_name", "message"),
    [
        ("_MAX_RETRIEVAL_ARTIFACT_BYTES", "retrieval artifact exceeds"),
        ("_MAX_GROUNDING_EVIDENCE_BYTES", "grounding evidence exceeds"),
    ],
)
def test_build_rejects_artifacts_that_its_loader_cannot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "sidecars"
    monkeypatch.setattr(grounding_module, limit_name, 1)

    with pytest.raises(ModelBGroundingSidecarBuildError) as caught:
        build_model_b_grounding_sidecars(
            fixture.candidate_root,
            fixture.corpus_path,
            fixture.capability_path,
            output,
            max_workers=1,
        )

    assert caught.value.__cause__ is not None
    assert message in str(caught.value.__cause__)
    assert not output.exists()
    assert not list(tmp_path.glob(".sidecars.*.partial"))


def test_rejects_content_plan_contract_lineage_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    lineage = fixture.lineages[0]
    plan_path = fixture.candidate_root / lineage / "hwp_document_plan.candidate.json"
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_value["content_ir_sha256"] = "9" * 64
    plan_bytes = _write_json(plan_path, plan_value)
    plan = HwpDocumentPlan.model_validate(plan_value)
    manifest_path = fixture.candidate_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_binding = manifest["documents"][0]["artifacts"]["hwp_document_plan_candidate"]
    plan_binding["sha256"] = _sha256(plan_bytes)
    plan_binding["contract_sha256"] = contract_sha256(plan)
    _write_json(manifest_path, manifest)

    with pytest.raises(ModelBGroundingSidecarBuildError) as caught:
        build_model_b_grounding_sidecars(
            fixture.candidate_root,
            fixture.corpus_path,
            fixture.capability_path,
            tmp_path / "sidecars",
            max_workers=1,
        )

    assert caught.value.__cause__ is not None
    assert "contract lineage mismatch" in str(caught.value.__cause__)


def test_input_change_before_publish_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "sidecars"
    original_worker = grounding_module._build_document_sidecars

    def mutating_worker(*args: Any, **kwargs: Any) -> Any:
        result = original_worker(*args, **kwargs)
        fixture.capability_path.write_bytes(fixture.capability_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        grounding_module,
        "_build_document_sidecars",
        mutating_worker,
    )

    with pytest.raises(RuntimeError, match="capability profile changed"):
        build_model_b_grounding_sidecars(
            fixture.candidate_root,
            fixture.corpus_path,
            fixture.capability_path,
            output,
            max_workers=1,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".sidecars.*.partial"))


def test_descriptor_read_rejects_file_replacement_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    replacement = tmp_path / "replacement.json"
    source.write_bytes(b'{"value":"original"}')
    replacement.write_bytes(b'{"value":"replacement"}')
    original_open = grounding_module.os.open
    swapped = False

    def swapping_open(path: Path, flags: int) -> int:
        nonlocal swapped
        if not swapped and Path(path) == source:
            swapped = True
            source.unlink()
            replacement.replace(source)
        return original_open(path, flags)

    monkeypatch.setattr(grounding_module.os, "open", swapping_open)

    with pytest.raises(ModelBGroundingSidecarBuildError, match="changed while it was read"):
        grounding_module._read_bounded_file(source, 1_024, "fixture input")


def test_directory_publish_never_replaces_existing_empty_destination(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".bundle.partial"
    destination = tmp_path / "bundle"
    staging.mkdir()
    (staging / "manifest.json").write_text("staged", encoding="utf-8")
    destination.mkdir()

    with pytest.raises(FileExistsError, match="already exists|appeared"):
        grounding_module._publish_directory_create_only(staging, destination)

    assert staging.joinpath("manifest.json").read_text(encoding="utf-8") == "staged"
    assert not list(destination.iterdir())
