from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image, ImageDraw

from scan2hwpx.contracts import (
    EvidenceIR,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
)


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "ai/modeling/run_model_a_table_topology.py"
    spec = importlib.util.spec_from_file_location("run_model_a_table_topology", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _png() -> bytes:
    image = Image.new("RGB", (1_000, 1_400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 700, 700), outline="black", width=5)
    draw.line((400, 100, 400, 700), fill="black", width=5)
    draw.line((100, 400, 700, 400), fill="black", width=5)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _write_fixture(
    root: Path,
    *,
    page_ids: tuple[str, ...] = ("page-0001",),
    present_page_ids: tuple[str, ...] = ("page-0001",),
) -> tuple[Path, Path]:
    png = _png()
    image_sha256 = _sha256(png)
    sources = [
        EvidenceSource(
            id="source-document",
            kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
            artifact_ref="artifact://fixture/source",
            producer="fixture",
            sha256="a" * 64,
        )
    ]
    pages = []
    for page_no, page_id in enumerate(page_ids, start=1):
        image_source_ref = f"page-image-{page_no}"
        sources.append(
            EvidenceSource(
                id=image_source_ref,
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref=f"artifact://fixture/page-{page_no}",
                producer="fixture",
                sha256=image_sha256,
            )
        )
        pages.append(
            EvidencePage(
                id=page_id,
                page_no=page_no,
                width=1_000,
                height=1_400,
                image_source_ref=image_source_ref,
            )
        )
    evidence = EvidenceIR(
        id="evidence-fixture",
        source_document_sha256="a" * 64,
        sources=tuple(sources),
        pages=tuple(pages),
    )
    evidence_path = root / "evidence.json"
    evidence_path.write_bytes(
        json.dumps(
            evidence.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    pages_root = root / "pages"
    pages_root.mkdir()
    for page_id in present_page_ids:
        (pages_root / f"{page_id}.png").write_bytes(png)
    return evidence_path, pages_root


def _args(evidence_path: Path, pages_root: Path, output_dir: Path) -> list[str]:
    return [
        "--evidence-ir",
        str(evidence_path),
        "--pages-root",
        str(pages_root),
        "--output-dir",
        str(output_dir),
    ]


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def test_runner_writes_exact_deterministic_artifacts_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    evidence_path, pages_root = _write_fixture(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    assert runner.main(_args(evidence_path, pages_root, first_output)) == 0
    assert runner.main(_args(evidence_path, pages_root, second_output)) == 0

    first_tree = _tree_bytes(first_output)
    assert first_tree == _tree_bytes(second_output)
    summary = json.loads(first_tree["summary.json"])
    assert summary["execution_policy"] == "sequential_local_opencv_no_network_no_retry"
    assert summary["generated_unverified_count"] == 1
    assert summary["execution_error_count"] == 0
    assert summary["research_only"] is True
    assert summary["golden_eligible"] is False
    page_key = summary["pages"][0]["page_output_key"]
    topology = first_tree[f"{page_key}/topology.json"]
    assert topology == json.dumps(
        json.loads(topology),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    topology_sha256 = _sha256(topology)
    assert first_tree[f"{page_key}/topology.sha256"] == (
        f"{topology_sha256}  topology.json\n".encode("ascii")
    )
    binding = json.loads(first_tree[f"{page_key}/page-binding.json"])
    assert binding["status"] == "exact_digest_match"
    assert binding["actual_page_image_sha256"] == binding["expected_page_image_sha256"]

    before = _tree_bytes(first_output)
    with pytest.raises(FileExistsError, match="already exists"):
        runner.main(_args(evidence_path, pages_root, first_output))
    assert _tree_bytes(first_output) == before


def test_runner_preserves_success_when_another_page_fails_and_supports_selector(
    tmp_path: Path,
) -> None:
    evidence_path, pages_root = _write_fixture(
        tmp_path,
        page_ids=("page-0001", "page-0002"),
        present_page_ids=("page-0001",),
    )
    batch_output = tmp_path / "batch"

    assert runner.main(_args(evidence_path, pages_root, batch_output)) == 1

    summary = json.loads((batch_output / "summary.json").read_bytes())
    assert summary["generated_unverified_count"] == 1
    assert summary["execution_error_count"] == 1
    success, failure = summary["pages"]
    assert (batch_output / success["page_output_key"] / "topology.json").is_file()
    failure_dir = batch_output / failure["page_output_key"]
    assert (failure_dir / "failure.json").is_file()
    assert (failure_dir / "page-binding.json").is_file()
    assert not (failure_dir / "topology.json").exists()

    selected_output = tmp_path / "selected"
    selected_args = [
        *_args(evidence_path, pages_root, selected_output),
        "--page-id",
        "page-0001",
    ]
    assert runner.main(selected_args) == 0
    selected_summary = json.loads((selected_output / "summary.json").read_bytes())
    assert selected_summary["page_count"] == 1
    assert selected_summary["pages"][0]["page_id"] == "page-0001"

    unknown_output = tmp_path / "unknown"
    with pytest.raises(ValueError, match="unknown --page-id"):
        runner.main(
            [
                *_args(evidence_path, pages_root, unknown_output),
                "--page-id",
                "missing",
            ]
        )
    assert not unknown_output.exists()


def test_runner_contains_hostile_page_id_inside_hashed_page_directory(tmp_path: Path) -> None:
    evidence_path, pages_root = _write_fixture(
        tmp_path,
        page_ids=("../escape",),
        present_page_ids=(),
    )
    output_dir = tmp_path / "output"

    assert runner.main(_args(evidence_path, pages_root, output_dir)) == 1

    summary = json.loads((output_dir / "summary.json").read_bytes())
    page = summary["pages"][0]
    assert page["outcome"] == "execution_error"
    assert "escape" not in page["page_output_key"]
    assert (output_dir / page["page_output_key"] / "failure.json").is_file()
    assert not (tmp_path / "escape.png").exists()
