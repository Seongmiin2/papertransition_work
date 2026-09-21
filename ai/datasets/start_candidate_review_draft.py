from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.review_draft import start_candidate_review_draft


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Start a self-asserted, non-training-eligible review draft from a "
            "registered candidate document"
        )
    )
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("document_id")
    parser.add_argument("--reviewer-label", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-lineage-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    draft = start_candidate_review_draft(
        args.candidate_root,
        args.document_id,
        args.reviewer_label,
        args.out,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_lineage_id=args.expected_lineage_id,
    )
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "schema_version": draft.schema_version,
                "candidate_manifest_sha256": draft.candidate_manifest_sha256,
                "document_id": draft.document_id,
                "lineage_id": draft.lineage_id,
                "status": draft.status,
                "draft_revision": draft.draft_revision,
                "rights_status": draft.rights_status,
                "identity_assurance": draft.identity_assurance,
                "golden_eligible": draft.golden_eligible,
                "training_eligible": draft.training_eligible,
                "release_eligible": draft.release_eligible,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
