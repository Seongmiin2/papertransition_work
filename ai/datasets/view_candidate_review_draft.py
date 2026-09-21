from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.review_draft import (
    build_candidate_review_draft_view,
    load_candidate_review_draft,
    verify_candidate_review_draft,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit a bounded sanitized view of a candidate-verified review draft"
    )
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-lineage-id", required=True)
    args = parser.parse_args()

    draft = load_candidate_review_draft(args.draft)
    verified = verify_candidate_review_draft(draft, candidate_root=args.candidate_root)
    if (
        verified.candidate_manifest_sha256 != args.expected_manifest_sha256
        or verified.lineage_id != args.expected_lineage_id
    ):
        raise ValueError("review draft does not match the selected candidate snapshot")
    view = build_candidate_review_draft_view(verified)
    sys.stdout.buffer.write(
        (
            json.dumps(
                view.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
