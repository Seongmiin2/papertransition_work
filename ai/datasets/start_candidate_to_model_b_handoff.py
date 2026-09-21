from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.candidate_to_model_b_handoff import (
    start_candidate_to_model_b_handoff,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Start an immutable, noneligible Model B review from one completed "
            "candidate review and a verified grounding sidecar"
        )
    )
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("completed_candidate_review", type=Path)
    parser.add_argument("sidecar_root", type=Path)
    parser.add_argument("knowledge_corpus", type=Path)
    parser.add_argument("capability_profile", type=Path)
    parser.add_argument("--reviewer-label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-review-sha256")
    parser.add_argument("--expected-review-revision", type=int)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-lineage-id")
    args = parser.parse_args()

    verified = start_candidate_to_model_b_handoff(
        args.candidate_root,
        args.completed_candidate_review,
        args.sidecar_root,
        args.knowledge_corpus,
        args.capability_profile,
        reviewer_label=args.reviewer_label,
        output_path=args.out,
        expected_candidate_review_artifact_sha256=args.expected_review_sha256,
        expected_candidate_review_revision=args.expected_review_revision,
        expected_candidate_manifest_sha256=args.expected_manifest_sha256,
        expected_lineage_id=args.expected_lineage_id,
    )
    envelope = verified.envelope
    model_b = verified.model_b_plan_review_draft
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "schema_version": envelope.schema_version,
                "artifact_sha256": verified.artifact_sha256,
                "document_id": envelope.document_id,
                "lineage_id": envelope.lineage_id,
                "candidate_review_revision": (
                    envelope.completed_candidate_review.draft_revision
                ),
                "candidate_review_artifact_sha256": (
                    envelope.completed_candidate_review.artifact_sha256
                ),
                "sidecar_manifest_sha256": envelope.grounding_sidecar.manifest_sha256,
                "model_b_draft_revision": model_b.draft_revision,
                "rights_status": envelope.rights_status,
                "identity_assurance": envelope.identity_assurance,
                "golden_eligible": envelope.golden_eligible,
                "training_eligible": envelope.training_eligible,
                "release_eligible": envelope.release_eligible,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
