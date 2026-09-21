from __future__ import annotations

import argparse
import json
from pathlib import Path

from scan2hwpx.model_a.bootstrap import bootstrap_model_a_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap non-trained Model A candidates for human annotation; "
            "outputs are never golden data"
        )
    )
    parser.add_argument("document_ir", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    args = parser.parse_args()

    destinations = bootstrap_model_a_candidates(
        args.document_ir,
        args.output,
        max_workers=args.max_workers,
    )
    print(
        json.dumps(
            {
                "completed_count": len(destinations),
                "source_document_sha256": [destination.name for destination in destinations],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
