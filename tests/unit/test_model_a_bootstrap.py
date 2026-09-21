from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import scan2hwpx.model_a.bootstrap as bootstrap_module
from scan2hwpx.contracts import ContentIR, EvidenceIR
from scan2hwpx.model_a.bootstrap import bootstrap_model_a_candidates
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider

FIXTURE = Path("tests/fixtures/ocr_page_1.json")


def _write_document(path: Path, source_hash: str, *, private_text: str | None = None) -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    document.id = f"document-{source_hash[:8]}"
    document.source_hash = source_hash
    document.metadata["source"] = rf"C:\private\{source_hash[:8]}\student.pdf"
    if private_text is not None:
        document.pages[0].blocks[0].text = private_text
    path.write_text(document.model_dump_json(), encoding="utf-8")


def test_bootstrap_writes_review_only_candidates_without_report_leakage(
    tmp_path: Path,
) -> None:
    first_hash = "1" * 64
    second_hash = "2" * 64
    private_text = "PRIVATE ORIGINAL STUDENT TEXT"
    first_path = tmp_path / "student-kim-document_ir.json"
    second_path = tmp_path / "student-lee-document_ir.json"
    _write_document(first_path, first_hash, private_text=private_text)
    _write_document(second_path, second_hash)
    output = tmp_path / "candidates"

    destinations = bootstrap_model_a_candidates(
        [first_path, second_path],
        output,
        max_workers=2,
    )

    assert destinations == (output / first_hash, output / second_hash)
    for destination in destinations:
        evidence_path = destination / "evidence_ir.json"
        content_path = destination / "content_ir.candidate.json"
        report_path = destination / "report.json"
        evidence = EvidenceIR.model_validate_json(evidence_path.read_bytes())
        content = ContentIR.model_validate_json(content_path.read_bytes())
        content.assert_evidence_integrity(evidence)
        report = json.loads(report_path.read_text(encoding="utf-8"))

        assert set(report) == {
            "source_document_sha256",
            "evidence_ir_sha256",
            "content_ir_candidate_sha256",
            "page_count",
            "observation_count",
            "candidate_count",
            "content_node_count",
            "review_count",
            "human_review_required",
            "golden_eligible",
        }
        assert report["source_document_sha256"] == destination.name
        assert report["evidence_ir_sha256"] == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        assert report["content_ir_candidate_sha256"] == hashlib.sha256(
            content_path.read_bytes()
        ).hexdigest()
        assert report["human_review_required"] is True
        assert report["golden_eligible"] is False

        serialized_report = report_path.read_text(encoding="utf-8")
        assert private_text not in serialized_report
        assert "student-kim" not in serialized_report
        assert "student-lee" not in serialized_report
        assert "C:\\private" not in serialized_report
    assert not list(output.glob("*.partial"))


def test_bootstrap_rejects_duplicate_source_hash_before_writing(tmp_path: Path) -> None:
    source_hash = "a" * 64
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_document(first_path, source_hash)
    _write_document(second_path, source_hash)
    output = tmp_path / "candidates"

    with pytest.raises(ValueError, match="duplicate source_hash"):
        bootstrap_model_a_candidates(
            [first_path, second_path],
            output,
            max_workers=2,
        )

    assert not output.exists()


def test_failed_document_cleans_staging_without_corrupting_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_hash = "b" * 64
    bad_hash = "c" * 64
    good_path = tmp_path / "good.json"
    bad_path = tmp_path / "bad.json"
    _write_document(good_path, good_hash)
    _write_document(bad_path, bad_hash)
    output = tmp_path / "candidates"
    original_atomic_write = bootstrap_module._atomic_write

    def fail_bad_content(path: Path, payload: bytes) -> None:
        if bad_hash in path.parent.name and path.name == "content_ir.candidate.json":
            raise OSError("simulated isolated write failure")
        original_atomic_write(path, payload)

    monkeypatch.setattr(bootstrap_module, "_atomic_write", fail_bad_content)

    with pytest.raises(RuntimeError, match=bad_hash):
        bootstrap_model_a_candidates(
            [good_path, bad_path],
            output,
            max_workers=2,
        )

    assert sorted(path.name for path in (output / good_hash).iterdir()) == [
        "content_ir.candidate.json",
        "evidence_ir.json",
        "report.json",
    ]
    assert not (output / bad_hash).exists()
    assert not list(output.glob("*.partial"))


def test_bootstrap_passes_explicit_worker_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "document.json"
    _write_document(document_path, "d" * 64)
    observed: dict[str, Any] = {}

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers: int, **kwargs: Any) -> None:
            observed["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(bootstrap_module, "ThreadPoolExecutor", RecordingExecutor)

    bootstrap_model_a_candidates(
        [document_path],
        tmp_path / "candidates",
        max_workers=3,
    )

    assert observed == {"max_workers": 3}


def test_bootstrap_requires_positive_explicit_worker_limit(tmp_path: Path) -> None:
    document_path = tmp_path / "document.json"
    _write_document(document_path, "e" * 64)

    with pytest.raises(ValueError, match="at least 1"):
        bootstrap_model_a_candidates(
            [document_path],
            tmp_path / "candidates",
            max_workers=0,
        )
