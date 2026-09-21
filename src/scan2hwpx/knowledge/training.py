from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scan2hwpx.contracts import ContentIR, HwpDocumentPlan
from scan2hwpx.knowledge.hancom import format_planner_context, load_chunks, retrieve_chunks


def build_planner_training_dataset(
    examples_path: Path,
    corpus_path: Path,
    capability_profile_path: Path,
    output_path: Path,
    *,
    context_limit: int = 8,
    allow_untrusted_research_examples: bool = False,
) -> dict[str, Any]:
    """Build a non-trainable research dataset from self-asserted Model B examples."""
    if allow_untrusted_research_examples is not True:
        raise ValueError(
            "trusted verifier unavailable: self-asserted verified=true examples are rejected; "
            "set allow_untrusted_research_examples=True only for research output"
        )
    if context_limit < 1:
        raise ValueError("context_limit must be positive")
    chunks = load_chunks(corpus_path)
    capability = json.loads(capability_profile_path.read_text(encoding="utf-8"))
    if not isinstance(capability, dict) or not capability.get("profile_id"):
        raise ValueError("capability profile must be a JSON object with profile_id")
    corpus_sha256 = _sha256(corpus_path)
    capability_sha256 = _sha256(capability_profile_path)

    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with examples_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError(f"line {line_number}: empty training example")
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: example must be an object")  # noqa: TRY004
            example_id = _required_string(record.get("id"), "id", line_number)
            if example_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id: {example_id}")
            seen_ids.add(example_id)
            if record.get("verified") is not True:
                raise ValueError(
                    f"line {line_number}: research example must contain the self-asserted "
                    "verified=true marker"
                )
            content_ir_payload = _required_object(
                record.get("content_ir"), "content_ir", line_number
            )
            target_plan_payload = _required_object(
                record.get("target_plan"), "target_plan", line_number
            )
            query = _required_string(
                record.get("retrieval_query"), "retrieval_query", line_number
            )
            tags = _string_list(record.get("retrieval_tags", []), "retrieval_tags", line_number)
            spec_refs = _string_list(record.get("spec_refs"), "spec_refs", line_number)
            if not spec_refs:
                raise ValueError(f"line {line_number}: spec_refs must not be empty")

            content_ir = _validate_content_ir(content_ir_payload, line_number)
            pending_review = sorted(node.id for node in content_ir.nodes if node.needs_review)
            if pending_review:
                raise ValueError(
                    f"line {line_number}: every ContentIR node must have needs_review=false: "
                    + ", ".join(pending_review)
                )
            target_plan = _validate_hwpx_plan(target_plan_payload, line_number)
            try:
                target_plan.assert_content_integrity(content_ir)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number}: invalid ContentIR/Plan refs: {exc}"
                ) from exc
            if target_plan.capability_profile_id != capability["profile_id"]:
                raise ValueError(
                    f"line {line_number}: target plan capability_profile_id does not match "
                    "the selected capability profile"
                )
            if set(target_plan.official_spec_refs) != set(spec_refs):
                raise ValueError(
                    f"line {line_number}: target plan official_spec_refs must match spec_refs"
                )

            selected = retrieve_chunks(
                query,
                limit=context_limit,
                tags=tags,
                chunks=chunks,
            )
            if not selected:
                raise ValueError(f"line {line_number}: official context retrieval returned no chunks")
            selected_ids = {chunk.id for chunk in selected}
            missing_refs = sorted(set(spec_refs) - selected_ids)
            if missing_refs:
                raise ValueError(
                    f"line {line_number}: spec_refs not present in retrieved context: "
                    + ", ".join(missing_refs)
                )
            output.append(
                {
                    "id": example_id,
                    "input": {
                        "content_ir": content_ir.model_dump(mode="json"),
                        "capability_profile": capability,
                        "official_context": format_planner_context(selected),
                    },
                    "target": {
                        "hwpx_plan": target_plan.model_dump(mode="json"),
                        "spec_refs": spec_refs,
                    },
                    "provenance": {
                        "knowledge_corpus_sha256": corpus_sha256,
                        "capability_profile_sha256": capability_sha256,
                        "retrieved_chunk_ids": [chunk.id for chunk in selected],
                        "research_only": True,
                        "training_eligible": False,
                        "human_verified_only": False,
                        "identity_assurance": "self_asserted_untrusted",
                    },
                }
            )

    if not output:
        raise ValueError("training examples file contains no records")
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in output
    ).encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_path, encoded)
    return {
        "schema_version": "1.0",
        "examples": len(output),
        "output": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "knowledge_corpus_sha256": corpus_sha256,
        "capability_profile_sha256": capability_sha256,
        "research_only": True,
        "training_eligible": False,
        "human_verified_only": False,
        "identity_assurance": "self_asserted_untrusted",
    }


def _required_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    return value.strip()


def _required_object(value: Any, field: str, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"line {line_number}: {field} must be a non-empty object")
    return value


def _validate_content_ir(payload: dict[str, Any], line_number: int) -> ContentIR:
    try:
        return ContentIR.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"line {line_number}: invalid ContentIR: {exc}") from exc


def _validate_hwpx_plan(payload: dict[str, Any], line_number: int) -> HwpDocumentPlan:
    try:
        return HwpDocumentPlan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"line {line_number}: invalid HwpDocumentPlan: {exc}") from exc


def _string_list(value: Any, field: str, line_number: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"line {line_number}: {field} must be a list")  # noqa: TRY004
    return [_required_string(item, field, line_number) for item in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(path)
