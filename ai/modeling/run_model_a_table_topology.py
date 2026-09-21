from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from scan2hwpx.contracts import EvidenceIR, EvidencePage, contract_sha256
from scan2hwpx.model_a.inference import ModelAPageImage
from scan2hwpx.model_a.table_topology import (
    TABLE_TOPOLOGY_ARTIFACT_VERSION,
    TABLE_TOPOLOGY_DETECTOR_VERSION,
    TABLE_TOPOLOGY_POLICY_SHA256,
    BuiltModelATableTopology,
    build_model_a_table_topology,
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local deterministic table topology for exact EvidenceIR pages."
    )
    parser.add_argument("--evidence-ir", type=Path, required=True)
    parser.add_argument("--pages-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-id", action="append", dest="page_ids")
    return parser.parse_args(argv)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _media_type(path: Path) -> Literal["image/png", "image/jpeg", "image/webp"]:
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"unsupported page image extension: {path.name}")


def _page_image_path(pages_root: Path, page_id: str) -> Path:
    matches: list[Path] = []
    for suffix in _IMAGE_SUFFIXES:
        candidate = (pages_root / f"{page_id}{suffix}").resolve()
        if candidate.parent != pages_root:
            raise ValueError("page id cannot escape the configured pages root")
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            f"page image must resolve exactly once for {page_id}; found {len(matches)}"
        )
    return matches[0]


def _page_output_key(page_no: int, page_id: str) -> str:
    digest = _sha256(page_id.encode("utf-8"))[:12]
    return f"page-{page_no:04d}-{digest}"


def _selected_pages(
    evidence_ir: EvidenceIR,
    page_ids: list[str] | None,
) -> tuple[EvidencePage, ...]:
    ordered = tuple(sorted(evidence_ir.pages, key=lambda page: (page.page_no, page.id)))
    if page_ids is None:
        return ordered
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("--page-id values must be unique")
    page_by_id = {page.id: page for page in ordered}
    unknown = sorted(set(page_ids) - set(page_by_id))
    if unknown:
        raise ValueError("unknown --page-id values: " + ", ".join(unknown))
    requested = set(page_ids)
    return tuple(page for page in ordered if page.id in requested)


def _eligibility() -> dict[str, object]:
    return {
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
    }


def _page_binding(
    evidence_ir: EvidenceIR,
    page: EvidencePage,
    *,
    page_key: str,
    actual_image_sha256: str | None,
    image_file_name: str | None,
) -> dict[str, object]:
    source_by_id = {source.id: source for source in evidence_ir.sources}
    expected_sha256 = source_by_id[page.image_source_ref].sha256
    if actual_image_sha256 is None:
        status = "unresolved"
    elif actual_image_sha256 == expected_sha256:
        status = "exact_digest_match"
    else:
        status = "digest_mismatch"
    return {
        "schema_version": "model-a-table-topology-page-binding/1.0",
        **_eligibility(),
        "status": status,
        "evidence_ir_id": evidence_ir.id,
        "evidence_ir_contract_sha256": contract_sha256(evidence_ir),
        "source_document_sha256": evidence_ir.source_document_sha256,
        "page_id": page.id,
        "page_no": page.page_no,
        "page_output_key": page_key,
        "page_image_source_ref": page.image_source_ref,
        "expected_page_image_sha256": expected_sha256,
        "actual_page_image_sha256": actual_image_sha256,
        "image_file_name": image_file_name,
    }


def _success_summary(
    page: EvidencePage,
    page_key: str,
    built: BuiltModelATableTopology,
) -> dict[str, object]:
    tables = built.artifact.tables
    return {
        "page_id": page.id,
        "page_no": page.page_no,
        "page_output_key": page_key,
        "outcome": "generated_unverified",
        "table_count": len(tables),
        "atomic_cell_count": sum(table.rows * table.columns for table in tables),
        "output_cell_count": sum(len(table.cells) for table in tables),
        "merged_cell_count": sum(
            cell.row_span > 1 or cell.column_span > 1 for table in tables for cell in table.cells
        ),
        "unique_text_observation_count": sum(
            len(table.ordered_observation_ids) for table in tables
        ),
        "cell_observation_assignment_count": sum(
            len(cell.ordered_observation_ids) for table in tables for cell in table.cells
        ),
        "empty_cell_count": sum(
            not cell.ordered_observation_ids for table in tables for cell in table.cells
        ),
        "topology_sha256": built.artifact_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_path = args.evidence_ir.resolve()
    evidence_raw = evidence_path.read_bytes()
    evidence_ir = EvidenceIR.model_validate_json(evidence_raw, strict=True)
    pages = _selected_pages(evidence_ir, args.page_ids)
    if not pages:
        raise ValueError("at least one page must be selected")
    pages_root = (
        args.pages_root.resolve()
        if args.pages_root is not None
        else (evidence_path.parent / "pages").resolve()
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    page_summaries: list[dict[str, object]] = []
    failed = False
    for page in pages:
        page_key = _page_output_key(page.page_no, page.id)
        page_dir = output_dir / page_key
        page_dir.mkdir()
        actual_image_sha256: str | None = None
        image_file_name: str | None = None
        try:
            image_path = _page_image_path(pages_root, page.id)
            image_file_name = image_path.name
            image_bytes = image_path.read_bytes()
            actual_image_sha256 = _sha256(image_bytes)
            page_image = ModelAPageImage(
                page_id=page.id,
                page_no=page.page_no,
                image_source_ref=page.image_source_ref,
                media_type=_media_type(image_path),
                raw_bytes=image_bytes,
                sha256=actual_image_sha256,
            )
            _write_json(
                page_dir / "page-binding.json",
                _page_binding(
                    evidence_ir,
                    page,
                    page_key=page_key,
                    actual_image_sha256=actual_image_sha256,
                    image_file_name=image_file_name,
                ),
            )
            built = build_model_a_table_topology(evidence_ir, page_image=page_image)
            (page_dir / "topology.json").write_bytes(built.artifact_bytes)
            (page_dir / "topology.sha256").write_bytes(
                f"{built.artifact_sha256}  topology.json\n".encode("ascii")
            )
            page_summaries.append(_success_summary(page, page_key, built))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failed = True
            binding_path = page_dir / "page-binding.json"
            if not binding_path.exists():
                _write_json(
                    binding_path,
                    _page_binding(
                        evidence_ir,
                        page,
                        page_key=page_key,
                        actual_image_sha256=actual_image_sha256,
                        image_file_name=image_file_name,
                    ),
                )
            failure = {
                "schema_version": "model-a-table-topology-failure/1.0",
                **_eligibility(),
                "status": "execution_error",
                "page_id": page.id,
                "page_no": page.page_no,
                "page_output_key": page_key,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            _write_json(page_dir / "failure.json", failure)
            page_summaries.append(
                {
                    "page_id": page.id,
                    "page_no": page.page_no,
                    "page_output_key": page_key,
                    "outcome": "execution_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    summary = {
        "schema_version": "model-a-table-topology-batch/1.0",
        **_eligibility(),
        "execution_policy": "sequential_local_opencv_no_network_no_retry",
        "evidence_ir_id": evidence_ir.id,
        "evidence_ir_raw_sha256": _sha256(evidence_raw),
        "evidence_ir_contract_sha256": contract_sha256(evidence_ir),
        "source_document_sha256": evidence_ir.source_document_sha256,
        "artifact_version": TABLE_TOPOLOGY_ARTIFACT_VERSION,
        "detector_version": TABLE_TOPOLOGY_DETECTOR_VERSION,
        "detector_policy_sha256": TABLE_TOPOLOGY_POLICY_SHA256,
        "page_count": len(pages),
        "generated_unverified_count": sum(
            item["outcome"] == "generated_unverified" for item in page_summaries
        ),
        "execution_error_count": sum(
            item["outcome"] == "execution_error" for item in page_summaries
        ),
        "limitations": [
            "No human-verified table labels are used; every output requires review.",
            "The detector is a deterministic OpenCV baseline, not a learned table model.",
            "Topology artifacts do not mutate EvidenceIR or semantic role artifacts.",
            "No paid API, network provider, retry, or hidden repair is used.",
        ],
        "pages": page_summaries,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
