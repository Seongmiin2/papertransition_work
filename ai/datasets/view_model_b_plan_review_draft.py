from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.evaluation.model_b_plan_review import (
    reopen_model_b_plan_review_draft_view,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit a bounded view of a source-verified Model B review revision"
    )
    parser.add_argument("draft", type=Path)
    parser.add_argument("content_ir", type=Path)
    parser.add_argument("base_plan", type=Path)
    parser.add_argument("grounding_evidence", type=Path)
    args = parser.parse_args()

    view = reopen_model_b_plan_review_draft_view(
        args.draft,
        args.content_ir,
        args.base_plan,
        args.grounding_evidence,
    )
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
