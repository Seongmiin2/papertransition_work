from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scan2hwpx.contracts.legacy import evidence_from_document
from scan2hwpx.contracts.models import ContentIR, EvidenceIR
from scan2hwpx.ir.models import Document

from .baseline import RuleBasedModelABaseline

_SOURCE_HASH = re.compile(r"^[0-9a-f]{64}$")


def bootstrap_model_a_candidates(
    document_ir_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    max_workers: int,
) -> tuple[Path, ...]:
    """Create isolated human-review candidates with the non-trained baseline.

    This is an annotation bootstrap utility, not a training or golden-data
    generator. A failed document never publishes its staging directory, while
    successful documents from the same batch remain available for review.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    paths = tuple(Path(path) for path in document_ir_paths)
    if not paths:
        raise ValueError("at least one document_ir.json path is required")
    documents = tuple(Document.model_validate_json(path.read_bytes()) for path in paths)
    _validate_source_hashes(documents)

    destination_root = Path(output_dir)
    existing = [
        document.source_hash
        for document in documents
        if (destination_root / document.source_hash).exists()
    ]
    if existing:
        raise FileExistsError("candidate output already exists for: " + ", ".join(existing))
    destination_root.mkdir(parents=True, exist_ok=True)

    results: list[Path | None] = [None] * len(documents)
    failures: list[tuple[int, str, BaseException]] = []
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="model-a-bootstrap",
    ) as executor:
        future_metadata = {
            executor.submit(_bootstrap_document, document, destination_root): (
                index,
                document.source_hash,
            )
            for index, document in enumerate(documents)
        }
        for future in as_completed(future_metadata):
            index, source_hash = future_metadata[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate arbitrary worker failures
                failures.append((index, source_hash, exc))

    if failures:
        failures.sort(key=lambda item: item[0])
        failed_hashes = ", ".join(item[1] for item in failures)
        raise RuntimeError(f"candidate bootstrap failed for source hashes: {failed_hashes}") from (
            failures[0][2]
        )
    return tuple(result for result in results if result is not None)


def _validate_source_hashes(documents: tuple[Document, ...]) -> None:
    invalid = [
        document.source_hash
        for document in documents
        if not _SOURCE_HASH.fullmatch(document.source_hash)
    ]
    if invalid:
        raise ValueError("source_hash must be 64 lowercase hexadecimal characters")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for document in documents:
        if document.source_hash in seen:
            duplicates.add(document.source_hash)
        seen.add(document.source_hash)
    if duplicates:
        raise ValueError("duplicate source_hash values: " + ", ".join(sorted(duplicates)))


def _bootstrap_document(document: Document, destination_root: Path) -> Path:
    evidence = evidence_from_document(document)
    content = RuleBasedModelABaseline().analyze(evidence)
    content.assert_evidence_integrity(evidence)

    evidence_payload = _contract_json(evidence)
    content_payload = _contract_json(content)
    report_payload = _report_json(
        document.source_hash, evidence, content, evidence_payload, content_payload
    )

    destination = destination_root / document.source_hash
    staging = Path(
        tempfile.mkdtemp(
            dir=destination_root,
            prefix=f".{document.source_hash}.",
            suffix=".partial",
        )
    )
    try:
        _atomic_write(staging / "evidence_ir.json", evidence_payload)
        _atomic_write(staging / "content_ir.candidate.json", content_payload)
        _atomic_write(staging / "report.json", report_payload)
        if destination.exists():
            raise FileExistsError(
                f"candidate output already exists for source hash {document.source_hash}"
            )
        staging.replace(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _contract_json(contract: EvidenceIR | ContentIR) -> bytes:
    return (contract.model_dump_json(indent=2) + "\n").encode("utf-8")


def _report_json(
    source_hash: str,
    evidence: EvidenceIR,
    content: ContentIR,
    evidence_payload: bytes,
    content_payload: bytes,
) -> bytes:
    observations = tuple(
        observation for page in evidence.pages for observation in page.observations
    )
    report = {
        "source_document_sha256": source_hash,
        "evidence_ir_sha256": hashlib.sha256(evidence_payload).hexdigest(),
        "content_ir_candidate_sha256": hashlib.sha256(content_payload).hexdigest(),
        "page_count": len(evidence.pages),
        "observation_count": len(observations),
        "candidate_count": sum(len(item.ocr_candidates) for item in observations),
        "content_node_count": len(content.nodes),
        "review_count": sum(node.needs_review for node in content.nodes),
        "human_review_required": True,
        "golden_eligible": False,
    }
    return (json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(path)
