from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.model_b_plan_review import (
    complete_persisted_model_b_plan_review_draft,
    load_model_b_plan_review_completion_request,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Complete one exact immutable Model B review revision without "
            "promoting it to Golden or training eligibility"
        )
    )
    parser.add_argument("draft", type=Path)
    parser.add_argument("content_ir", type=Path)
    parser.add_argument("base_plan", type=Path)
    parser.add_argument("grounding_evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    request = load_model_b_plan_review_completion_request(sys.stdin.buffer)
    completed = complete_persisted_model_b_plan_review_draft(
        args.draft,
        args.content_ir,
        args.base_plan,
        args.grounding_evidence,
        request,
        args.out,
    )
    print(
        json.dumps(
            {
                "schema_version": completed.schema_version,
                "document_id": completed.document_id,
                "status": completed.status,
                "draft_revision": completed.draft_revision,
                "source_candidate_contract_sha256": (
                    completed.source_candidate_contract_sha256
                ),
                "grounding_evidence_contract_sha256": (
                    completed.grounding_evidence_contract_sha256
                ),
                "reviewed_hwp_document_plan_contract_sha256": (
                    completed.reviewed_hwp_document_plan_contract_sha256
                ),
                "rights_status": completed.rights_status,
                "identity_assurance": completed.identity_assurance,
                "golden_eligible": completed.golden_eligible,
                "training_eligible": completed.training_eligible,
                "release_eligible": completed.release_eligible,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
