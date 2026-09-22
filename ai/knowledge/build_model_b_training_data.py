from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.knowledge.training import build_planner_training_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an untrusted, research-only Model B intermediate dataset with "
            "official Hancom context"
        )
    )
    parser.add_argument("examples", type=Path, help="self-asserted research example JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "output" / "datasets" / "hancom-knowledge" / "processed" / "chunks.jsonl",
    )
    parser.add_argument(
        "--capability-profile",
        type=Path,
        default=PROJECT_ROOT / "ai" / "knowledge" / "hancom" / "capability_profile.json",
    )
    parser.add_argument("--context-limit", type=int, default=8)
    parser.add_argument(
        "--allow-untrusted-research-examples",
        action="store_true",
        help=(
            "allow self-asserted examples only for a research-only, non-trainable output"
        ),
    )
    args = parser.parse_args()

    report = build_planner_training_dataset(
        args.examples,
        args.corpus,
        args.capability_profile,
        args.out,
        context_limit=args.context_limit,
        allow_untrusted_research_examples=args.allow_untrusted_research_examples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
