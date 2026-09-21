from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scan2hwpx.evaluation.model_b_grounding_sidecars import (
    build_model_b_grounding_sidecars,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable, unverified Model B retrieval and grounding sidecars "
            "for an HWPX projection candidate bundle"
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--knowledge-corpus", type=Path, required=True)
    parser.add_argument("--capability-profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    args = parser.parse_args()

    manifest = build_model_b_grounding_sidecars(
        args.candidates,
        args.knowledge_corpus,
        args.capability_profile,
        args.out,
        max_workers=args.max_workers,
    )
    manifest_sha256 = hashlib.sha256((args.out / "manifest.json").read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "document_count": manifest["document_count"],
                "retrieved_chunk_count": manifest["retrieval"]["retrieved_chunk_count"],
                "manifest_sha256": manifest_sha256,
                "training_eligible": manifest["training_eligible"],
                "release_eligible": manifest["release_eligible"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
