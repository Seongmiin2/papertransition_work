from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.model_b_plan_review import reopen_model_b_plan_review_draft


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reopen and verify a Model B review against immutable sources"
    )
    parser.add_argument("draft", type=Path)
    parser.add_argument("content_ir", type=Path)
    parser.add_argument("base_plan", type=Path)
    parser.add_argument("grounding_evidence", type=Path)
    args = parser.parse_args()

    draft = reopen_model_b_plan_review_draft(
        args.draft,
        args.content_ir,
        args.base_plan,
        args.grounding_evidence,
    )
    print(
        json.dumps(
            {
                "schema_version": draft.schema_version,
                "document_id": draft.document_id,
                "draft_revision": draft.draft_revision,
                "source_candidate_contract_sha256": (
                    draft.source_candidate_contract_sha256
                ),
                "grounding_evidence_contract_sha256": (
                    draft.grounding_evidence_contract_sha256
                ),
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
