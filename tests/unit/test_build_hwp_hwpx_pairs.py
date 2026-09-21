from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from ai.datasets.build_hwp_hwpx_pairs import (
    OPEN_ARGUMENTS,
    PyhwpxComAdapter,
    build_hwp_hwpx_pairs,
)

_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "src/scan2hwpx/hwpx/canonical_template.b64"
)
_VALID_HWPX = base64.b64decode("".join(_TEMPLATE_PATH.read_text(encoding="ascii").split()))



class FakeAdapter:
    producer = "fake-pyhwpx-com"
    version = "1.2.3-test"

    def __init__(self, *, fail: bool = False, mutate_source: bool = False) -> None:
        self.fail = fail
        self.mutate_source = mutate_source
        self.calls: list[tuple[Path, Path, Path]] = []
        self.closed = False

    def convert(self, source_hwp: Path, output_hwpx: Path, output_pdf: Path) -> int:
        self.calls.append((source_hwp, output_hwpx, output_pdf))
        payload = source_hwp.read_bytes()
        output_hwpx.write_bytes(_VALID_HWPX)
        if self.fail:
            raise RuntimeError("injected conversion failure")
        with pymupdf.open() as pdf:  # type: ignore[no-untyped-call]
            pdf.new_page()
            pdf.save(output_pdf)
        if self.mutate_source:
            source_hwp.write_bytes(payload + b"changed")
        return 1

    def close(self) -> None:
        self.closed = True


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_sources(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    refs: dict[str, str] = {}
    root.mkdir()
    for source_ref, payload in files.items():
        path = root.joinpath(*source_ref.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        refs[_sha256_bytes(payload)] = source_ref
    return refs


def _write_report(path: Path, documents: list[object]) -> Path:
    split_counts = {
        split: sum(
            isinstance(document, dict) and document.get("split") == split
            for document in documents
        )
        for split in ("train", "validation", "test")
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "splits": {
                    split: {"documents": count} for split, count in split_counts.items()
                },
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return path


def _document(source_sha256: str, split: str, *, pages: int = 1) -> dict[str, str | int]:
    return {
        "source": "../../stale-and-untrusted/source.hwp",
        "sha256": source_sha256,
        "split": split,
        "pages": pages,
    }


def test_builds_sha_mapped_projection_pairs_and_manifest(tmp_path: Path) -> None:
    payloads = {
        "grade2/zeta.hwp": b"first source",
        "grade1/nested/alpha.hwp": b"second source",
        "grade2/beta.HWP": b"third source",
    }
    source = tmp_path / "source"
    refs_by_sha = _write_sources(source, payloads)
    split_by_sha = {
        _sha256_bytes(b"first source"): "validation",
        _sha256_bytes(b"second source"): "train",
        _sha256_bytes(b"third source"): "test",
    }
    report = _write_report(
        tmp_path / "dataset_report.json",
        [
            _document(source_sha256, split_by_sha[source_sha256])
            for source_sha256 in reversed(list(split_by_sha))
        ],
    )
    adapter = FakeAdapter()
    factory_calls = 0

    def factory() -> FakeAdapter:
        nonlocal factory_calls
        factory_calls += 1
        return adapter

    output = tmp_path / "pairs"
    manifest = build_hwp_hwpx_pairs(
        report,
        source,
        output,
        adapter_factory=factory,
    )

    assert factory_calls == 1
    assert adapter.closed
    assert len(adapter.calls) == 3
    assert [call[0] for call in adapter.calls] == [
        source.joinpath(*refs_by_sha[source_sha256].split("/")).resolve()
        for source_sha256 in sorted(split_by_sha)
    ]
    assert manifest["schema_version"] == "hwp-hwpx-projection-pairs/1.0"
    assert manifest["artifact_role"] == "paired_source_for_projection"
    assert manifest["research_only"] is True
    assert manifest["rights_manifest"] == {"status": "missing"}
    assert manifest["producer"] == {"name": "fake-pyhwpx-com", "version": "1.2.3-test"}
    assert manifest["document_count"] == 3
    assert manifest["splits"] == {"test": 1, "train": 1, "validation": 1}
    assert manifest["source_dataset_report_sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()

    assert [item["source_sha256"] for item in manifest["documents"]] == sorted(split_by_sha)
    for item in manifest["documents"]:
        source_sha256 = item["source_sha256"]
        assert item["source_ref"] == refs_by_sha[source_sha256]
        assert item["split"] == split_by_sha[source_sha256]
        assert item["page_count"] == 1
        for artifact_name in ("hwpx", "pdf"):
            artifact = item[artifact_name]
            artifact_path = output.joinpath(*artifact["path"].split("/"))
            assert artifact_path.is_file()
            assert artifact["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    serialized = json.dumps(manifest).casefold()
    assert all(term not in serialized for term in ("gold", "contentir", "plan"))
    assert not list(tmp_path.glob(".pairs.*.partial"))


def test_pyhwpx_adapter_reopens_hwpx_before_exporting_pdf(tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []

    class FakeHwp:
        @property
        def PageCount(self) -> int:
            events.append(("page_count",))
            return 7

        def open(self, path: str, *, arg: str) -> bool:
            events.append(("open", path, arg))
            return True

        def save_as(self, path: str, *, format: str) -> bool:
            events.append(("save_as", path, format))
            Path(path).write_bytes(format.encode("ascii"))
            return True

        def close(self, *, is_dirty: bool) -> None:
            events.append(("close", is_dirty))

        def quit(self) -> None:
            events.append(("quit",))

    source = tmp_path / "source.hwp"
    source.write_bytes(b"source")
    hwpx = tmp_path / "source.hwpx"
    pdf = tmp_path / "source.pdf"
    adapter = PyhwpxComAdapter(hwp_factory=FakeHwp, producer_version="test-version")

    assert adapter.convert(source.resolve(), hwpx.resolve(), pdf.resolve()) == 7
    adapter.close()

    assert adapter.producer == "pyhwpx"
    assert adapter.version == "test-version"
    assert events == [
        ("open", str(source.resolve()), OPEN_ARGUMENTS),
        ("save_as", str(hwpx.resolve()), "HWPX"),
        ("close", False),
        ("open", str(hwpx.resolve()), OPEN_ARGUMENTS),
        ("page_count",),
        ("save_as", str(pdf.resolve()), "PDF"),
        ("close", False),
        ("quit",),
    ]


@pytest.mark.parametrize(
    "documents",
    [
        [],
        ["not-an-object"],
        [_document("not-a-sha", "train")],
        [_document("a" * 64, "dev")],
        [_document("a" * 64, "train"), _document("a" * 64, "test")],
    ],
)
def test_rejects_malformed_or_duplicate_report_documents(
    tmp_path: Path, documents: list[object]
) -> None:
    source = tmp_path / "source"
    _write_sources(source, {"source.hwp": b"source"})
    report = _write_report(tmp_path / "dataset_report.json", documents)

    with pytest.raises((TypeError, ValueError)):
        build_hwp_hwpx_pairs(
            report,
            source,
            tmp_path / "pairs",
            adapter_factory=lambda: pytest.fail("adapter must not be created"),
        )


def test_requires_exact_one_to_one_source_sha_mapping(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_sources(source, {"expected.hwp": b"expected", "extra.hwp": b"extra"})
    report = _write_report(
        tmp_path / "dataset_report.json",
        [_document(_sha256_bytes(b"expected"), "train")],
    )

    with pytest.raises(ValueError, match=r"missing=0, unexpected=1"):
        build_hwp_hwpx_pairs(
            report,
            source,
            tmp_path / "pairs",
            adapter_factory=lambda: pytest.fail("adapter must not be created"),
        )


def test_rejects_duplicate_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_sources(source, {"first.hwp": b"same", "nested/second.hwp": b"same"})
    report = _write_report(
        tmp_path / "dataset_report.json",
        [_document(_sha256_bytes(b"same"), "train")],
    )

    with pytest.raises(ValueError, match="duplicate HWP bytes"):
        build_hwp_hwpx_pairs(report, source, tmp_path / "pairs", adapter_factory=FakeAdapter)


@pytest.mark.parametrize("failure_mode", ["partial", "mutation", "page_count", "report_pages"])
def test_failure_never_publishes_partial_output(tmp_path: Path, failure_mode: str) -> None:
    source = tmp_path / "source"
    _write_sources(source, {"source.hwp": b"source"})
    report = _write_report(
        tmp_path / "dataset_report.json",
        [_document(_sha256_bytes(b"source"), "train", pages=2 if failure_mode == "report_pages" else 1)],
    )
    adapter = FakeAdapter(
        fail=failure_mode == "partial",
        mutate_source=failure_mode == "mutation",
    )
    if failure_mode == "page_count":
        adapter.convert = lambda source_hwp, output_hwpx, output_pdf: 0  # type: ignore[method-assign]
    output = tmp_path / "pairs"

    with pytest.raises((RuntimeError, ValueError)):
        build_hwp_hwpx_pairs(report, source, output, adapter_factory=lambda: adapter)

    assert adapter.closed
    assert not output.exists()
    assert not list(tmp_path.glob(".pairs.*.partial"))


def test_rejects_existing_or_source_nested_output_before_adapter_creation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_sources(source, {"source.hwp": b"source"})
    report = _write_report(
        tmp_path / "dataset_report.json",
        [_document(_sha256_bytes(b"source"), "train")],
    )
    existing = tmp_path / "existing"
    existing.mkdir()

    for output in (existing, source / "derived"):
        with pytest.raises((FileExistsError, ValueError)):
            build_hwp_hwpx_pairs(
                report,
                source,
                output,
                adapter_factory=lambda: pytest.fail("adapter must not be created"),
            )


def test_rejects_symlink_in_source_tree_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_sources(source, {"source.hwp": b"source"})
    link = source / "linked.hwp"
    try:
        link.symlink_to(source / "source.hwp")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    report = _write_report(
        tmp_path / "dataset_report.json",
        [_document(_sha256_bytes(b"source"), "train")],
    )

    with pytest.raises(ValueError, match="symlinks"):
        build_hwp_hwpx_pairs(report, source, tmp_path / "pairs", adapter_factory=FakeAdapter)
