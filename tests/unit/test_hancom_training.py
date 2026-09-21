from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ai.knowledge.build_model_b_training_data import main as build_model_b_main
from scan2hwpx.contracts import ContentIR, contract_sha256
from scan2hwpx.knowledge.training import build_planner_training_dataset


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _chunk() -> dict[str, Any]:
    return {
        "id": "official-style-1",
        "source_id": "hancom-hwpx-format",
        "title": "HWPX format",
        "section": "styles",
        "text": "header.xml에 정의된 문단 모양은 section.xml에서 참조한다.",
        "tags": ["hwpx", "style"],
        "source_url": "https://tech.hancom.com/hwpxformat/",
        "source_sha256": "a" * 64,
    }


def _example(
    *, verified: bool = True, spec_refs: list[str] | None = None
) -> dict[str, Any]:
    active_spec_refs = spec_refs or ["official-style-1"]
    record: dict[str, Any] = {
        "id": "example-1",
        "verified": verified,
        "content_ir": {
            "schema_version": "content-ir/1.0",
            "id": "content-1",
            "evidence_ir_id": "evidence-1",
            "evidence_ir_sha256": "a" * 64,
            "revision": 1,
            "nodes": [
                {
                    "id": "q1",
                    "kind": "text",
                    "role": "question",
                    "text": "문항",
                    "evidence_refs": ["observation-1"],
                    "confidence": 1.0,
                    "needs_review": False,
                }
            ],
            "reading_order": ["q1"],
        },
        "target_plan": {
            "schema_version": "hwp-document-plan/1.0",
            "id": "plan-1",
            "content_ir_id": "content-1",
            "content_ir_revision": 1,
            "capability_profile_id": "test-profile",
            "design_profile_id": "test-design",
            "official_spec_refs": active_spec_refs,
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
                    "id": "flow-1",
                    "kind": "content",
                    "render_as": "paragraph",
                    "content_ref": "q1",
                }
            ],
        },
        "retrieval_query": "header.xml 문단 모양",
        "retrieval_tags": ["hwpx", "style"],
        "spec_refs": active_spec_refs,
    }
    record["target_plan"]["content_ir_sha256"] = contract_sha256(
        ContentIR.model_validate(record["content_ir"])
    )
    return record


def test_default_rejects_self_asserted_verified_example(tmp_path: Path) -> None:
    corpus = tmp_path / "chunks.jsonl"
    examples = tmp_path / "examples.jsonl"
    capability = tmp_path / "capability.json"
    output = tmp_path / "research.jsonl"
    _write_jsonl(corpus, [_chunk()])
    _write_jsonl(examples, [_example()])
    capability.write_text(
        json.dumps({"profile_id": "test-profile", "target_format": "hwpx"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trusted verifier unavailable"):
        build_planner_training_dataset(examples, corpus, capability, output)

    assert not output.exists()


def test_builds_flagged_research_input_only_with_explicit_opt_in(tmp_path: Path) -> None:
    corpus = tmp_path / "chunks.jsonl"
    examples = tmp_path / "examples.jsonl"
    capability = tmp_path / "capability.json"
    output = tmp_path / "research.jsonl"
    _write_jsonl(corpus, [_chunk()])
    _write_jsonl(examples, [_example()])
    capability.write_text(
        json.dumps({"profile_id": "test-profile", "target_format": "hwpx"}),
        encoding="utf-8",
    )

    report = build_planner_training_dataset(
        examples,
        corpus,
        capability,
        output,
        allow_untrusted_research_examples=True,
    )
    generated = json.loads(output.read_text(encoding="utf-8"))

    assert report["examples"] == 1
    assert report["research_only"] is True
    assert report["training_eligible"] is False
    assert report["human_verified_only"] is False
    assert report["identity_assurance"] == "self_asserted_untrusted"
    assert generated["target"]["spec_refs"] == ["official-style-1"]
    assert generated["provenance"]["retrieved_chunk_ids"] == ["official-style-1"]
    assert generated["provenance"]["research_only"] is True
    assert generated["provenance"]["training_eligible"] is False
    assert generated["provenance"]["human_verified_only"] is False
    assert generated["provenance"]["identity_assurance"] == "self_asserted_untrusted"
    assert "HANCOM_OFFICIAL_REFERENCE_CONTEXT" in generated["input"]["official_context"]


def test_cli_forwards_only_explicit_untrusted_research_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "chunks.jsonl"
    examples = tmp_path / "examples.jsonl"
    capability = tmp_path / "capability.json"
    output = tmp_path / "research.jsonl"
    _write_jsonl(corpus, [_chunk()])
    _write_jsonl(examples, [_example()])
    capability.write_text(json.dumps({"profile_id": "test-profile"}), encoding="utf-8")
    argv = [
        "build_model_b_training_data.py",
        str(examples),
        "--out",
        str(output),
        "--corpus",
        str(corpus),
        "--capability-profile",
        str(capability),
    ]

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="trusted verifier unavailable"):
        build_model_b_main()
    assert not output.exists()

    monkeypatch.setattr(sys, "argv", [*argv, "--allow-untrusted-research-examples"])
    assert build_model_b_main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["research_only"] is True
    assert report["training_eligible"] is False
    assert report["identity_assurance"] == "self_asserted_untrusted"


def test_rejects_unverified_or_ungrounded_training_targets(tmp_path: Path) -> None:
    corpus = tmp_path / "chunks.jsonl"
    capability = tmp_path / "capability.json"
    _write_jsonl(corpus, [_chunk()])
    capability.write_text(json.dumps({"profile_id": "test-profile"}), encoding="utf-8")

    unverified = tmp_path / "unverified.jsonl"
    _write_jsonl(unverified, [_example(verified=False)])
    with pytest.raises(ValueError, match="self-asserted verified=true"):
        build_planner_training_dataset(
            unverified,
            corpus,
            capability,
            tmp_path / "unverified-output.jsonl",
            allow_untrusted_research_examples=True,
        )

    ungrounded = tmp_path / "ungrounded.jsonl"
    _write_jsonl(ungrounded, [_example(spec_refs=["missing-reference"])])
    with pytest.raises(ValueError, match="not present in retrieved context"):
        build_planner_training_dataset(
            ungrounded,
            corpus,
            capability,
            tmp_path / "ungrounded-output.jsonl",
            allow_untrusted_research_examples=True,
        )


def test_rejects_schema_invalid_hwpx_plan(tmp_path: Path) -> None:
    corpus = tmp_path / "chunks.jsonl"
    examples = tmp_path / "invalid-plan.jsonl"
    capability = tmp_path / "capability.json"
    _write_jsonl(corpus, [_chunk()])
    capability.write_text(json.dumps({"profile_id": "test-profile"}), encoding="utf-8")
    example = _example()
    example["target_plan"]["flow"][0]["text"] = "모델이 생성한 본문"
    _write_jsonl(examples, [example])

    with pytest.raises(ValueError, match="invalid HwpDocumentPlan"):
        build_planner_training_dataset(
            examples,
            corpus,
            capability,
            tmp_path / "invalid-plan-output.jsonl",
            allow_untrusted_research_examples=True,
        )


def test_rejects_plan_bound_to_stale_content(tmp_path: Path) -> None:
    corpus = tmp_path / "chunks.jsonl"
    examples = tmp_path / "stale-plan.jsonl"
    capability = tmp_path / "capability.json"
    _write_jsonl(corpus, [_chunk()])
    capability.write_text(
        json.dumps({"profile_id": "test-profile"}),
        encoding="utf-8",
    )
    example = _example()
    example["target_plan"]["content_ir_sha256"] = "f" * 64
    _write_jsonl(examples, [example])

    with pytest.raises(ValueError, match="ContentIR digest mismatch"):
        build_planner_training_dataset(
            examples,
            corpus,
            capability,
            tmp_path / "stale-plan-output.jsonl",
            allow_untrusted_research_examples=True,
        )


def test_research_mode_rejects_content_still_marked_for_review(tmp_path: Path) -> None:
    corpus = tmp_path / "chunks.jsonl"
    examples = tmp_path / "needs-review.jsonl"
    capability = tmp_path / "capability.json"
    _write_jsonl(corpus, [_chunk()])
    capability.write_text(json.dumps({"profile_id": "test-profile"}), encoding="utf-8")
    example = _example()
    example["content_ir"]["nodes"][0]["needs_review"] = True
    _write_jsonl(examples, [example])

    with pytest.raises(ValueError, match="every ContentIR node must have needs_review=false"):
        build_planner_training_dataset(
            examples,
            corpus,
            capability,
            tmp_path / "needs-review-output.jsonl",
            allow_untrusted_research_examples=True,
        )
