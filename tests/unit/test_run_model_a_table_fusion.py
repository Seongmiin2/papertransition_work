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
    BBox,
    ContentIR,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
    contract_sha256,
)
from scan2hwpx.model_a.block_role_vector import (
    build_model_a_block_role_vector_requests,
    compile_model_a_block_role_vector_responses,
)
from scan2hwpx.model_a.inference import ModelAModelArtifact, ModelAPageImage

_WIDTH = 1_000
_HEIGHT = 1_400


def _load_script(name: str, relative_path: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


topology_runner = _load_script(
    "run_model_a_table_topology_for_fusion_test",
    "ai/modeling/run_model_a_table_topology.py",
)
fusion_runner = _load_script(
    "run_model_a_table_fusion_test_target",
    "ai/modeling/run_model_a_table_fusion.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _png(*, table: bool) -> bytes:
    image = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    if table:
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 100, 700, 700), outline="black", width=5)
        draw.line((400, 100, 400, 700), fill="black", width=5)
        draw.line((100, 400, 700, 400), fill="black", width=5)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _observation(
    page_image_source: str,
    observation_id: str,
    text: str,
    bbox: tuple[float, float, float, float],
) -> EvidenceObservation:
    x0, y0, x1, y1 = bbox
    return EvidenceObservation(
        id=observation_id,
        kind=ObservationKind.TEXT_LINE,
        bbox=BBox(
            pixel=bbox,
            normalized=(x0 / _WIDTH, y0 / _HEIGHT, x1 / _WIDTH, y1 / _HEIGHT),
        ),
        confidence=0.9,
        source_refs=(page_image_source, "ocr-source"),
        ocr_candidates=(
            OcrCandidate(
                text=text,
                provider="fixture-ocr",
                confidence=0.9,
                source_ref="ocr-source",
                selected=True,
            ),
        ),
    )


def _fixture(tmp_path: Path) -> tuple[EvidenceIR, Path, Path, dict[str, bytes]]:
    page_images = {"page-0001": _png(table=True), "page-0002": _png(table=False)}
    image_sources = {"page-0001": "page-image-1", "page-0002": "page-image-2"}
    page_one_observations = (
        _observation("page-image-1", "before", "before", (20, 20, 80, 45)),
        _observation("page-image-1", "cell-00", "A", (120, 150, 180, 180)),
        _observation("page-image-1", "cell-01", "B", (420, 150, 480, 180)),
        _observation("page-image-1", "cell-10", "C", (120, 450, 180, 480)),
        _observation("page-image-1", "cell-11", "D", (420, 450, 480, 480)),
        _observation("page-image-1", "after", "after", (100, 800, 180, 830)),
    )
    page_two_observations = (
        _observation("page-image-2", "page-two-text", "second", (20, 20, 100, 50)),
    )
    evidence = EvidenceIR(
        id="evidence-fusion-runner",
        source_document_sha256="1" * 64,
        sources=(
            EvidenceSource(
                id="document",
                kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
                artifact_ref="artifact://fixture/document",
                producer="fixture",
                sha256="1" * 64,
            ),
            EvidenceSource(
                id="page-image-1",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/page-1",
                producer="fixture",
                sha256=_sha256(page_images["page-0001"]),
            ),
            EvidenceSource(
                id="page-image-2",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/page-2",
                producer="fixture",
                sha256=_sha256(page_images["page-0002"]),
            ),
            EvidenceSource(
                id="ocr-source",
                kind=EvidenceSourceKind.OCR_PROVIDER,
                artifact_ref="artifact://fixture/ocr",
                producer="fixture-ocr",
            ),
        ),
        pages=(
            EvidencePage(
                id="page-0001",
                page_no=1,
                width=_WIDTH,
                height=_HEIGHT,
                image_source_ref=image_sources["page-0001"],
                observations=page_one_observations,
            ),
            EvidencePage(
                id="page-0002",
                page_no=2,
                width=_WIDTH,
                height=_HEIGHT,
                image_source_ref=image_sources["page-0002"],
                observations=page_two_observations,
            ),
        ),
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_bytes(_json_bytes(evidence.model_dump(mode="json")))
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    for page_id, raw in page_images.items():
        (pages_root / f"{page_id}.png").write_bytes(raw)
    return evidence, evidence_path, pages_root, page_images


def _page_key(page: EvidencePage) -> str:
    return f"page-{page.page_no:04d}-{_sha256(page.id.encode('utf-8'))[:12]}"


def _write_block_role_batch(
    root: Path,
    evidence: EvidenceIR,
    page_images: dict[str, bytes],
) -> None:
    root.mkdir()
    model = ModelAModelArtifact(
        model_id="fixture-model",
        model_revision="fixture-revision",
        artifact_sha256="9" * 64,
    )
    page_summaries = []
    for page in evidence.pages:
        page_key = _page_key(page)
        page_dir = root / page_key
        page_dir.mkdir()
        raw_image = page_images[page.id]
        page_image = ModelAPageImage(
            page_id=page.id,
            page_no=page.page_no,
            image_source_ref=page.image_source_ref,
            media_type="image/png",
            raw_bytes=raw_image,
            sha256=_sha256(raw_image),
        )
        requests = build_model_a_block_role_vector_requests(
            evidence,
            page_image=page_image,
            model=model,
        )
        responses = tuple(
            _json_bytes({"roles": ["passage"] * len(request.artifact.target_blocks)})
            for request in requests
        )
        result = compile_model_a_block_role_vector_responses(requests, responses)
        assert result.compiled_page_analysis_bytes is not None
        (page_dir / "result.json").write_bytes(result.result_bytes)
        (page_dir / "compiled-page-analysis.json").write_bytes(result.compiled_page_analysis_bytes)
        (page_dir / "page-binding.json").write_bytes(
            _json_bytes(
                {
                    "page_id": page.id,
                    "page_no": page.page_no,
                    "page_output_key": page_key,
                    "page_image_sha256": page_image.sha256,
                }
            )
        )
        segments_dir = page_dir / "segments"
        segments_dir.mkdir()
        for request, response, trace in zip(
            requests,
            responses,
            result.artifact.segments,
            strict=True,
        ):
            segment_dir = segments_dir / f"segment-{request.artifact.segment_index + 1:04d}"
            segment_dir.mkdir()
            (segment_dir / "request.json").write_bytes(request.raw_request_bytes)
            (segment_dir / "raw-assistant-content.bin").write_bytes(response)
            (segment_dir / "trace.json").write_bytes(_json_bytes(trace.model_dump(mode="json")))
            (segment_dir / "execution.json").write_bytes(
                _json_bytes(
                    {
                        "status": "received",
                        "request_sha256": request.raw_request_sha256,
                        "assistant_content_sha256": _sha256(response),
                        "assistant_content_size_bytes": len(response),
                    }
                )
            )
        page_summaries.append(
            {
                "page_id": page.id,
                "page_no": page.page_no,
                "page_output_key": page_key,
                "outcome": "valid_unverified",
                "result_sha256": result.result_sha256,
            }
        )
    (root / "summary.json").write_bytes(
        _json_bytes(
            {
                "schema_version": "model-a-block-role-vector-smoke/2.0",
                "research_only": True,
                "golden_eligible": False,
                "training_eligible": False,
                "release_eligible": False,
                "evidence_ir_contract_sha256": contract_sha256(evidence),
                "page_count": len(evidence.pages),
                "valid_unverified_count": len(evidence.pages),
                "blocked_count": 0,
                "execution_error_count": 0,
                "model_stability": {"status": "stable"},
                "pages": page_summaries,
            }
        )
    )


def _prepare_batches(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    evidence, evidence_path, pages_root, images = _fixture(tmp_path)
    topology_dir = tmp_path / "topology"
    assert (
        topology_runner.main(
            [
                "--evidence-ir",
                str(evidence_path),
                "--pages-root",
                str(pages_root),
                "--output-dir",
                str(topology_dir),
            ]
        )
        == 0
    )
    block_dir = tmp_path / "block-role"
    _write_block_role_batch(block_dir, evidence, images)
    return evidence_path, pages_root, topology_dir, block_dir


def _runner_args(
    evidence_path: Path,
    pages_root: Path,
    topology_dir: Path,
    block_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "--evidence-ir",
        str(evidence_path),
        "--pages-root",
        str(pages_root),
        "--topology-batch-dir",
        str(topology_dir),
        "--block-role-batch-dir",
        str(block_dir),
        "--output-dir",
        str(output_dir),
    ]


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def test_runner_strictly_fuses_and_writes_byte_deterministic_document(
    tmp_path: Path,
) -> None:
    evidence_path, pages_root, topology_dir, block_dir = _prepare_batches(tmp_path)
    first_output = tmp_path / "fusion-first"
    second_output = tmp_path / "fusion-second"

    assert (
        fusion_runner.main(
            _runner_args(
                evidence_path,
                pages_root,
                topology_dir,
                block_dir,
                first_output,
            )
        )
        == 0
    )
    assert (
        fusion_runner.main(
            _runner_args(
                evidence_path,
                pages_root,
                topology_dir,
                block_dir,
                second_output,
            )
        )
        == 0
    )

    first_tree = _tree_bytes(first_output)
    assert first_tree == _tree_bytes(second_output)
    for name in (
        "promotion.json",
        "promoted-evidence-ir.json",
        "assembly.json",
        "content-ir.json",
        "summary.json",
    ):
        digest = _sha256(first_tree[name])
        sha_name = str(Path(name).with_suffix(".sha256"))
        assert first_tree[sha_name] == f"{digest}  {name}\n".encode("ascii")
    summary = json.loads(first_tree["summary.json"])
    assert summary["promotion_outcome"] == "generated_unverified"
    assert summary["assembly_outcome"] == "generated_unverified"
    assert summary["fused_unverified_count"] == 2
    assert summary["blocked_page_count"] == 0
    assert summary["research_only"] is True
    assert summary["release_eligible"] is False

    promoted = EvidenceIR.model_validate_json(first_tree["promoted-evidence-ir.json"], strict=True)
    assert any(
        observation.kind == ObservationKind.TABLE_GRID
        for page in promoted.pages
        for observation in page.observations
    )
    content = ContentIR.model_validate_json(first_tree["content-ir.json"], strict=True)
    assert sum(node.kind == "table" for node in content.nodes) == 1
    content.assert_evidence_integrity(promoted)
    page_one_key = summary["pages"][0]["page_output_key"]
    assert f"{page_one_key}/fusion.json" in first_tree
    assert f"{page_one_key}/fused-page-analysis.json" in first_tree

    before = _tree_bytes(first_output)
    with pytest.raises(FileExistsError, match="already exists"):
        fusion_runner.main(
            _runner_args(
                evidence_path,
                pages_root,
                topology_dir,
                block_dir,
                first_output,
            )
        )
    assert _tree_bytes(first_output) == before


def test_runner_keeps_promotion_and_good_page_when_one_semantic_page_fails(
    tmp_path: Path,
) -> None:
    evidence_path, pages_root, topology_dir, block_dir = _prepare_batches(tmp_path)
    evidence = EvidenceIR.model_validate_json(evidence_path.read_bytes(), strict=True)
    failed_page = evidence.pages[1]
    failed_page_dir = block_dir / _page_key(failed_page)
    (failed_page_dir / "compiled-page-analysis.json").unlink()
    output_dir = tmp_path / "partial-fusion"

    assert (
        fusion_runner.main(
            _runner_args(
                evidence_path,
                pages_root,
                topology_dir,
                block_dir,
                output_dir,
            )
        )
        == 1
    )

    summary = json.loads((output_dir / "summary.json").read_bytes())
    assert summary["validated_topology_count"] == 2
    assert summary["validated_semantic_count"] == 1
    assert summary["promotion_outcome"] == "generated_unverified"
    assert summary["fused_unverified_count"] == 1
    assert summary["blocked_page_count"] == 1
    assert summary["assembly_outcome"] == "blocked"
    assert (output_dir / "promotion.json").is_file()
    assert (output_dir / "promoted-evidence-ir.json").is_file()
    assert not (output_dir / "assembly.json").exists()
    assert not (output_dir / "content-ir.json").exists()
    good_page_dir = output_dir / summary["pages"][0]["page_output_key"]
    bad_page_dir = output_dir / summary["pages"][1]["page_output_key"]
    assert (good_page_dir / "fusion.json").is_file()
    assert (bad_page_dir / "failure.json").is_file()
    failure = json.loads((bad_page_dir / "failure.json").read_bytes())
    assert failure["release_eligible"] is False
    assert any("compiled semantic page analysis" in issue for issue in failure["issues"])


def test_runner_blocks_promotion_when_topology_sha_is_forged(tmp_path: Path) -> None:
    evidence_path, pages_root, topology_dir, block_dir = _prepare_batches(tmp_path)
    evidence = EvidenceIR.model_validate_json(evidence_path.read_bytes(), strict=True)
    forged_page = evidence.pages[0]
    sha_path = topology_dir / _page_key(forged_page) / "topology.sha256"
    sha_path.write_text(f"{'0' * 64}  topology.json\n", encoding="ascii")
    output_dir = tmp_path / "forged-topology"

    assert (
        fusion_runner.main(
            _runner_args(
                evidence_path,
                pages_root,
                topology_dir,
                block_dir,
                output_dir,
            )
        )
        == 1
    )

    summary = json.loads((output_dir / "summary.json").read_bytes())
    assert summary["validated_topology_count"] == 1
    assert summary["validated_semantic_count"] == 2
    assert summary["promotion_outcome"] == "blocked"
    assert summary["fused_unverified_count"] == 0
    assert not (output_dir / "promoted-evidence-ir.json").exists()
    assert (output_dir / "promotion-failure.json").is_file()
