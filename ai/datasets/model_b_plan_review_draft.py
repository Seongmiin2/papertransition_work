"""Start, verify, view, patch or complete an immutable Model B plan review.

No revision is promoted to Golden, training or release eligibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.model_b_plan_review import (
    apply_persisted_model_b_plan_review_patch,
    complete_persisted_model_b_plan_review_draft,
    load_model_b_plan_review_completion_request,
    load_model_b_plan_review_patch_request,
    reopen_model_b_plan_review_draft,
    reopen_model_b_plan_review_draft_view,
    start_persisted_model_b_plan_review_draft,
)

SUMMARY_FIELDS = (
    "schema_version",
    "document_id",
    "draft_revision",
    "source_candidate_contract_sha256",
    "grounding_evidence_contract_sha256",
    "rights_status",
    "identity_assurance",
    "golden_eligible",
    "training_eligible",
    "release_eligible",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser(
        "start", description="Start an immutable, noneligible Model B plan review revision"
    )
    start.add_argument("content_ir", type=Path)
    start.add_argument("base_plan", type=Path)
    start.add_argument("grounding_evidence", type=Path)
    start.add_argument("document_id")
    start.add_argument("--reviewer-label", required=True)
    start.add_argument("--out", type=Path, required=True)
    for name, description, writes_revision in (
        ("verify", "Reopen and verify a Model B review against immutable sources", False),
        ("view", "Emit a bounded view of a source-verified Model B review revision", False),
        ("patch", "Apply one typed patch as a new immutable Model B review revision", True),
        (
            "complete",
            (
                "Complete one exact immutable Model B review revision without "
                "promoting it to Golden or training eligibility"
            ),
            True,
        ),
    ):
        command = commands.add_parser(name, description=description)
        command.add_argument("draft", type=Path)
        command.add_argument("content_ir", type=Path)
        command.add_argument("base_plan", type=Path)
        command.add_argument("grounding_evidence", type=Path)
        if writes_revision:
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "start":
        draft = start_persisted_model_b_plan_review_draft(
            args.content_ir,
            args.base_plan,
            args.grounding_evidence,
            document_id=args.document_id,
            reviewer_label=args.reviewer_label,
            output_path=args.out,
        )
        _print_summary(draft, output=str(args.out.resolve()))
        return 0
    sources = (args.draft, args.content_ir, args.base_plan, args.grounding_evidence)
    if args.command == "verify":
        _print_summary(reopen_model_b_plan_review_draft(*sources))
    elif args.command == "view":
        view = reopen_model_b_plan_review_draft_view(*sources)
        sys.stdout.buffer.write(
            (
                json.dumps(
                    view.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8")
        )
    elif args.command == "patch":
        request = load_model_b_plan_review_patch_request(sys.stdin.buffer)
        _print_summary(apply_persisted_model_b_plan_review_patch(*sources, request, args.out))
    else:
        completion = load_model_b_plan_review_completion_request(sys.stdin.buffer)
        completed = complete_persisted_model_b_plan_review_draft(*sources, completion, args.out)
        _print_summary(
            completed,
            status=completed.status,
            reviewed_hwp_document_plan_contract_sha256=(
                completed.reviewed_hwp_document_plan_contract_sha256
            ),
        )
    return 0


def _print_summary(draft: Any, **extra: str) -> None:
    summary = {**extra, **{name: getattr(draft, name) for name in SUMMARY_FIELDS}}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
