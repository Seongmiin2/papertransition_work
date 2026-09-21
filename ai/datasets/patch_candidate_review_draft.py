from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.review_draft import (
    CandidateReviewPatchRequest,
    apply_candidate_review_patch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a strict typed patch from stdin to a verified, non-training-eligible "
            "candidate review draft"
        )
    )
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-lineage-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    request = CandidateReviewPatchRequest.model_validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    patched = apply_candidate_review_patch(
        args.candidate_root,
        args.draft,
        request,
        args.out,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_lineage_id=args.expected_lineage_id,
    )
    print(
        json.dumps(
            {
                "schema_version": patched.schema_version,
                "candidate_manifest_sha256": patched.candidate_manifest_sha256,
                "document_id": patched.document_id,
                "lineage_id": patched.lineage_id,
                "status": patched.status,
                "draft_revision": patched.draft_revision,
                "rights_status": patched.rights_status,
                "identity_assurance": patched.identity_assurance,
                "golden_eligible": patched.golden_eligible,
                "training_eligible": patched.training_eligible,
                "release_eligible": patched.release_eligible,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
