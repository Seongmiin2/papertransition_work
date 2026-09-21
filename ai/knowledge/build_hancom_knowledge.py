from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scan2hwpx.knowledge.build import build_knowledge_corpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a checksum-verified retrieval corpus from pinned Hancom sources"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "ai" / "knowledge" / "hancom" / "sources.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "hancom-knowledge" / "raw",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "output" / "hancom-knowledge" / "processed",
    )
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=160)
    args = parser.parse_args()

    report = build_knowledge_corpus(
        args.manifest,
        args.raw_dir,
        args.out,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
