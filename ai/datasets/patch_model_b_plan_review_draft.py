from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.model_b_plan_review import (
    apply_persisted_model_b_plan_review_patch,
    load_model_b_plan_review_patch_request,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply one typed patch as a new immutable Model B review revision"
    )
    parser.add_argument("draft", type=Path)
    parser.add_argument("content_ir", type=Path)
    parser.add_argument("base_plan", type=Path)
    parser.add_argument("grounding_evidence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    request = load_model_b_plan_review_patch_request(sys.stdin.buffer)
    patched = apply_persisted_model_b_plan_review_patch(
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
                "schema_version": patched.schema_version,
                "document_id": patched.document_id,
                "draft_revision": patched.draft_revision,
                "source_candidate_contract_sha256": (
                    patched.source_candidate_contract_sha256
                ),
                "grounding_evidence_contract_sha256": (
                    patched.grounding_evidence_contract_sha256
                ),
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
