"""Start, verify, view, patch or complete a candidate review draft.

Every draft stays self-asserted and non-training-eligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.review_draft import (
    CandidateReviewCompletionRequest,
    CandidateReviewPatchRequest,
    apply_candidate_review_patch,
    build_candidate_review_draft_view,
    complete_candidate_review_draft,
    load_candidate_review_draft,
    start_candidate_review_draft,
    verify_candidate_review_draft,
)

SUMMARY_FIELDS = (
    "schema_version",
    "candidate_manifest_sha256",
    "document_id",
    "lineage_id",
    "status",
    "draft_revision",
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
        "start",
        description=(
            "Start a self-asserted, non-training-eligible review draft from a "
            "registered candidate document"
        ),
    )
    start.add_argument("candidate_root", type=Path)
    start.add_argument("document_id")
    start.add_argument("--reviewer-label", required=True)
    start.add_argument("--expected-manifest-sha256", required=True)
    start.add_argument("--expected-lineage-id", required=True)
    start.add_argument("--out", type=Path, required=True)
    for name, description, writes_revision in (
        (
            "verify",
            (
                "Revalidate a self-asserted review draft against its immutable candidate "
                "without granting training eligibility"
            ),
            False,
        ),
        ("view", "Emit a bounded sanitized view of a candidate-verified review draft", False),
        (
            "patch",
            (
                "Apply a strict typed patch from stdin to a verified, non-training-eligible "
                "candidate review draft"
            ),
            True,
        ),
        (
            "complete",
            (
                "Complete one exact revision of a verified, non-training-eligible "
                "candidate review draft"
            ),
            True,
        ),
    ):
        command = commands.add_parser(name, description=description)
        command.add_argument("candidate_root", type=Path)
        command.add_argument("draft", type=Path)
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--expected-lineage-id", required=True)
        if writes_revision:
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshot = {
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "expected_lineage_id": args.expected_lineage_id,
    }

    if args.command == "start":
        draft = start_candidate_review_draft(
            args.candidate_root, args.document_id, args.reviewer_label, args.out, **snapshot
        )
        _print_summary(draft, output=str(args.out.resolve()))
    elif args.command in {"verify", "view"}:
        verified = verify_candidate_review_draft(
            load_candidate_review_draft(args.draft), candidate_root=args.candidate_root
        )
        if (
            verified.candidate_manifest_sha256 != args.expected_manifest_sha256
            or verified.lineage_id != args.expected_lineage_id
        ):
            raise ValueError("review draft does not match the selected candidate snapshot")
        if args.command == "verify":
            _print_summary(verified)
        else:
            view = build_candidate_review_draft_view(verified)
            sys.stdout.buffer.write(
                (
                    json.dumps(
                        view.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
                    )
                    + "\n"
                ).encode("utf-8")
            )
    elif args.command == "patch":
        request = CandidateReviewPatchRequest.model_validate_json(
            sys.stdin.buffer.read(), strict=True
        )
        _print_summary(
            apply_candidate_review_patch(
                args.candidate_root, args.draft, request, args.out, **snapshot
            )
        )
    else:
        completion = CandidateReviewCompletionRequest.model_validate_json(
            sys.stdin.buffer.read(), strict=True
        )
        _print_summary(
            complete_candidate_review_draft(
                args.candidate_root, args.draft, completion, args.out, **snapshot
            )
        )
    return 0


def _print_summary(draft: Any, **extra: str) -> None:
    summary = {**extra, **{name: getattr(draft, name) for name in SUMMARY_FIELDS}}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
