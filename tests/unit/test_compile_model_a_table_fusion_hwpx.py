from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from scan2hwpx.contracts import (
    BBox,
    ContentIR,
    ContentPlanItem,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    HwpDocumentPlan,
    ImageContentNode,
    LayoutIntent,
    ObservationKind,
    OcrCandidate,
    PageBreakPlanItem,
    PageLayoutIntent,
    StyleIntent,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.hwpx import IMAGE_DESIGN_PROFILE_ID, validate_hwpx
from scan2hwpx.model_a.inference import ModelAPageAnalysis
from scan2hwpx.model_a.table_fusion import (
    ModelATableDocumentAssemblyArtifact,
    ModelATableDocumentAssemblyPage,
    ModelATableDocumentAssemblySource,
    ModelATablePageFusionArtifact,
    ModelATablePageFusionSource,
)


def _load_script(name: str, relative_path: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script(
    "compile_model_a_table_fusion_hwpx_test_target",
    "ai/modeling/compile_model_a_table_fusion_hwpx.py",
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


def _png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (20, 10), color)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _write_artifact(path: Path, value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = _json_bytes(value)
    path.write_bytes(payload)
    path.with_suffix(".sha256").write_bytes(
        f"{_sha256(payload)}  {path.name}\n".encode("ascii")
    )
    return payload


def _bbox(x0: float, y0: float, x1: float, y1: float) -> BBox:
    return BBox(
        pixel=(x0, y0, x1, y1),
        normalized=(x0 / 100, y0 / 100, x1 / 100, y1 / 100),
    )


def _text_observation(observation_id: str, text: str, page_source: str) -> EvidenceObservation:
    return EvidenceObservation(
        id=observation_id,
        kind=ObservationKind.TEXT_LINE,
        bbox=_bbox(5, 5, 40, 15),
        confidence=0.9,
        source_refs=(page_source, "ocr-source"),
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


def _fixture(
    tmp_path: Path,
    *,
    image_bbox: tuple[float, float, float, float] = (
        10.0,
        90.0,
        14.733333333333334,
        95.0,
    ),
) -> tuple[Path, Path]:
    image_bytes = _png((30, 80, 120))
    image_file_name = "page-0002-image-0001.png"
    image_asset_ref = "page-0002-image-0001-crop"
    evidence = EvidenceIR(
        id="evidence-promoted",
        source_document_sha256="a" * 64,
        sources=(
            EvidenceSource(
                id="document",
                kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
                artifact_ref="artifact://fixture/document",
                producer="fixture",
                sha256="a" * 64,
            ),
            EvidenceSource(
                id="page-image-1",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/pages/page-0001.png",
                producer="fixture",
                sha256="1" * 64,
            ),
            EvidenceSource(
                id="page-image-2",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/pages/page-0002.png",
                producer="fixture",
                sha256="2" * 64,
            ),
            EvidenceSource(
                id="ocr-source",
                kind=EvidenceSourceKind.OCR_PROVIDER,
                artifact_ref="artifact://fixture/ocr",
                producer="fixture-ocr",
            ),
            EvidenceSource(
                id=image_asset_ref,
                kind=EvidenceSourceKind.CROP,
                artifact_ref=f"artifact://fixture/assets/{image_file_name}",
                producer="fixture-crop",
                sha256=_sha256(image_bytes),
            ),
        ),
        pages=(
            EvidencePage(
                id="page-0001",
                page_no=1,
                width=100,
                height=100,
                image_source_ref="page-image-1",
                observations=(
                    _text_observation("text-observation", "paragraph", "page-image-1"),
                    EvidenceObservation(
                        id="table-grid-observation",
                        kind=ObservationKind.TABLE_GRID,
                        bbox=_bbox(5, 20, 90, 70),
                        confidence=0.9,
                        source_refs=("page-image-1",),
                    ),
                    _text_observation("cell-observation", "cell", "page-image-1"),
                ),
            ),
            EvidencePage(
                id="page-0002",
                page_no=2,
                width=100,
                height=100,
                image_source_ref="page-image-2",
                observations=(
                    EvidenceObservation(
                        id="image-observation",
                        kind=ObservationKind.IMAGE,
                        bbox=_bbox(*image_bbox),
                        confidence=0.9,
                        source_refs=(image_asset_ref,),
                    ),
                ),
            ),
        ),
    )
    evidence_sha256 = contract_sha256(evidence)
    text = TextContentNode(
        id="content-text",
        role=ContentRole.PASSAGE,
        text="paragraph",
        evidence_refs=("text-observation",),
        confidence=0.9,
    )
    table = TableContentNode(
        id="content-table",
        evidence_refs=("table-grid-observation",),
        confidence=0.9,
        rows=1,
        columns=1,
        cells=(
            TableCell(
                row=0,
                column=0,
                text="cell",
                evidence_refs=("cell-observation",),
            ),
        ),
    )
    image = ImageContentNode(
        id="content-image",
        evidence_refs=("image-observation",),
        confidence=0.9,
        asset_ref=image_asset_ref,
    )
    page_analyses = (
        ModelAPageAnalysis(
            id="analysis-page-1",
            evidence_ir_id=evidence.id,
            evidence_ir_sha256=evidence_sha256,
            page_id="page-0001",
            page_no=1,
            nodes=(text, table),
            reading_order=(text.id, table.id),
        ),
        ModelAPageAnalysis(
            id="analysis-page-2",
            evidence_ir_id=evidence.id,
            evidence_ir_sha256=evidence_sha256,
            page_id="page-0002",
            page_no=2,
            nodes=(image,),
            reading_order=(image.id,),
        ),
    )
    content = ContentIR(
        id="fused-content",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=evidence_sha256,
        nodes=(text, table, image),
        reading_order=(text.id, table.id, image.id),
    )
    promotion_sha256 = "b" * 64
    fusion_artifacts = tuple(
        ModelATablePageFusionArtifact(
            source=ModelATablePageFusionSource(
                base_evidence_ir_id=evidence.id,
                base_evidence_ir_sha256="c" * 64,
                promoted_evidence_ir_sha256=evidence_sha256,
                promotion_artifact_sha256=promotion_sha256,
                semantic_page_analysis_sha256=str(page.page_no) * 64,
                topology_artifact_sha256=str(page.page_no + 2) * 64,
                target_page_id=page.page_id,
                target_page_no=page.page_no,
            ),
            semantic_role_sidecars=(),
            shared_text_resolutions=(),
            fused_page_analysis=page,
            fused_page_analysis_sha256=contract_sha256(page),
        )
        for page in page_analyses
    )
    fusion_bytes = tuple(
        _json_bytes(artifact.model_dump(mode="json")) for artifact in fusion_artifacts
    )
    assembly = ModelATableDocumentAssemblyArtifact(
        source=ModelATableDocumentAssemblySource(
            promoted_evidence_ir_id=evidence.id,
            promoted_evidence_ir_sha256=evidence_sha256,
            promotion_artifact_sha256=promotion_sha256,
            source_document_sha256=evidence.source_document_sha256,
            content_ir_id=content.id,
            content_ir_revision=1,
        ),
        pages=tuple(
            ModelATableDocumentAssemblyPage(
                page_id=page.page_id,
                page_no=page.page_no,
                fusion_artifact_sha256=_sha256(raw),
                fused_page_analysis_sha256=contract_sha256(page),
            )
            for page, raw in zip(page_analyses, fusion_bytes, strict=True)
        ),
        content_ir=content,
        content_ir_sha256=contract_sha256(content),
    )

    fusion_root = tmp_path / "fusion"
    fusion_root.mkdir()
    _write_artifact(fusion_root / "assembly.json", assembly)
    _write_artifact(fusion_root / "promoted-evidence-ir.json", evidence)
    _write_artifact(fusion_root / "content-ir.json", content)
    for page, fusion in zip(page_analyses, fusion_artifacts, strict=True):
        page_dir = fusion_root / runner._page_output_key(page.page_no, page.page_id)
        page_dir.mkdir()
        _write_artifact(page_dir / "fusion.json", fusion)
        _write_artifact(page_dir / "fused-page-analysis.json", page)

    candidate_root = tmp_path / "candidate"
    assets_root = candidate_root / "assets"
    assets_root.mkdir(parents=True)
    (assets_root / image_file_name).write_bytes(image_bytes)
    source_style = StyleIntent(
        id="source-style",
        semantic_role="source",
        font_size_pt=9.0,
    )
    candidate_plan = HwpDocumentPlan(
        id="source-candidate-plan",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        capability_profile_id="source-capability",
        design_profile_id="source-projection-candidate-v1",
        official_spec_refs=("hancom:test-one", "hancom:test-two"),
        page_layout=PageLayoutIntent(
            width_mm=210.0,
            height_mm=297.0,
            margin_top_mm=10.0,
            margin_right_mm=10.0,
            margin_bottom_mm=10.0,
            margin_left_mm=10.0,
        ),
        styles=(source_style,),
        flow=(
            ContentPlanItem(
                id="source-text",
                render_as="paragraph",
                content_ref=text.id,
                style_ref=source_style.id,
            ),
            ContentPlanItem(
                id="source-table",
                render_as="table",
                content_ref=table.id,
                layout=LayoutIntent(width_fraction=0.5),
            ),
            PageBreakPlanItem(id="source-break"),
            ContentPlanItem(
                id="source-image",
                render_as="image",
                content_ref=image.id,
            ),
        ),
    )
    (candidate_root / "hwp_document_plan.candidate.json").write_text(
        json.dumps(candidate_plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return fusion_root, candidate_root


def _args(fusion_root: Path, candidate_root: Path, output: Path) -> list[str]:
    return [
        "--fusion-batch-dir",
        str(fusion_root),
        "--candidate-document-dir",
        str(candidate_root),
        "--output-hwpx",
        str(output),
    ]


def _partial_directories(parent: Path) -> list[Path]:
    return [path for path in parent.iterdir() if path.name.endswith(".partial")]


def test_runner_builds_canonical_plan_and_compiles_deterministically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fusion_root, candidate_root = _fixture(tmp_path)
    first = tmp_path / "first.hwpx"
    second = tmp_path / "second.hwpx"

    assert runner.main(_args(fusion_root, candidate_root, first)) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert runner.main(_args(fusion_root, candidate_root, second)) == 0
    second_summary = json.loads(capsys.readouterr().out)

    assert first.read_bytes() == second.read_bytes()
    assert first_summary == second_summary
    assert first_summary["status"] == "compiled_unverified_smoke"
    assert first_summary["release_eligible"] is False
    assert first_summary["content_node_count"] == 3
    assert first_summary["table_node_count"] == 1
    assert first_summary["image_node_count"] == 1
    assert first_summary["page_break_count"] == 1
    assert validate_hwpx(first).valid

    batch = runner._load_fusion_batch(fusion_root.resolve())
    refs = runner._load_official_spec_refs(candidate_root.resolve())
    plan = runner._build_smoke_plan(batch, refs)
    content_items = tuple(item for item in plan.flow if isinstance(item, ContentPlanItem))
    assert tuple(item.content_ref for item in content_items) == batch.content_ir.reading_order
    assert plan.design_profile_id == IMAGE_DESIGN_PROFILE_ID
    assert plan.styles == ()
    assert plan.official_spec_refs == ("hancom:test-one", "hancom:test-two")
    assert plan.page_layout.margin_left_mm == 30.0
    assert sum(isinstance(item, PageBreakPlanItem) for item in plan.flow) == 1
    image_item = next(item for item in content_items if item.render_as == "image")
    assert image_item.layout.width_fraction == pytest.approx(0.14)
    assert all(
        item.layout == LayoutIntent()
        for item in content_items
        if item.render_as != "image"
    )

    before = first.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        runner.main(_args(fusion_root, candidate_root, first))
    assert first.read_bytes() == before
    assert _partial_directories(tmp_path) == []


def test_plan_clamps_a_page_spanning_image_to_one_column_width(tmp_path: Path) -> None:
    fusion_root, candidate_root = _fixture(
        tmp_path,
        image_bbox=(5.0, 10.0, 95.0, 25.0),
    )

    batch = runner._load_fusion_batch(fusion_root.resolve())
    refs = runner._load_official_spec_refs(candidate_root.resolve())
    plan = runner._build_smoke_plan(batch, refs)

    image_item = next(
        item
        for item in plan.flow
        if isinstance(item, ContentPlanItem) and item.render_as == "image"
    )
    assert image_item.layout.width_fraction == 1.0


def test_image_width_requires_one_direct_image_observation_on_the_same_page(
    tmp_path: Path,
) -> None:
    fusion_root, _candidate_root = _fixture(tmp_path)
    batch = runner._load_fusion_batch(fusion_root.resolve())
    image = next(
        node for node in batch.content_ir.nodes if isinstance(node, ImageContentNode)
    )

    with pytest.raises(
        runner.TableFusionHwpxSmokeError,
        match="exactly one IMAGE observation",
    ):
        runner._image_width_fraction(
            batch.promoted_evidence_ir,
            batch.page_analyses[1],
            image.model_copy(
                update={
                    "evidence_refs": ("image-observation", "text-observation"),
                }
            ),
        )

    with pytest.raises(
        runner.TableFusionHwpxSmokeError,
        match="on its analysis page",
    ):
        runner._image_width_fraction(
            batch.promoted_evidence_ir,
            batch.page_analyses[0],
            image,
        )


def test_image_width_rejects_zero_width_bbox(tmp_path: Path) -> None:
    fusion_root, _candidate_root = _fixture(tmp_path)
    batch = runner._load_fusion_batch(fusion_root.resolve())
    image = next(
        node for node in batch.content_ir.nodes if isinstance(node, ImageContentNode)
    )
    image_page = batch.promoted_evidence_ir.pages[1]
    observation = image_page.observations[0]
    zero_width_bbox = BBox.model_construct(
        pixel=(10.0, 90.0, 10.0, 95.0),
        normalized=(0.1, 0.9, 0.1, 0.95),
    )
    invalid_observation = observation.model_copy(update={"bbox": zero_width_bbox})
    invalid_page = image_page.model_copy(
        update={"observations": (invalid_observation,)}
    )
    invalid_evidence = batch.promoted_evidence_ir.model_copy(
        update={"pages": (batch.promoted_evidence_ir.pages[0], invalid_page)}
    )

    with pytest.raises(
        runner.TableFusionHwpxSmokeError,
        match="cannot be projected",
    ):
        runner._image_width_fraction(
            invalid_evidence,
            batch.page_analyses[1],
            image,
        )


def test_runner_rejects_forged_page_sha_without_partial_output(tmp_path: Path) -> None:
    fusion_root, candidate_root = _fixture(tmp_path)
    assembly = ModelATableDocumentAssemblyArtifact.model_validate_json(
        (fusion_root / "assembly.json").read_bytes(),
        strict=True,
    )
    page = assembly.pages[0]
    page_dir = fusion_root / runner._page_output_key(page.page_no, page.page_id)
    (page_dir / "fused-page-analysis.sha256").write_text(
        f"{'0' * 64}  fused-page-analysis.json\n",
        encoding="ascii",
    )
    output = tmp_path / "forged.hwpx"

    with pytest.raises(runner.TableFusionHwpxSmokeError, match="SHA file mismatch"):
        runner.main(_args(fusion_root, candidate_root, output))

    assert not output.exists()
    assert _partial_directories(tmp_path) == []


@pytest.mark.parametrize("tamper", ["mismatch", "extra"])
def test_runner_requires_exact_candidate_png_coverage(
    tmp_path: Path,
    tamper: str,
) -> None:
    fusion_root, candidate_root = _fixture(tmp_path)
    assets_root = candidate_root / "assets"
    if tamper == "mismatch":
        (assets_root / "page-0002-image-0001.png").write_bytes(_png((200, 20, 20)))
    else:
        (assets_root / "extra.png").write_bytes(_png((20, 200, 20)))
    output = tmp_path / f"{tamper}.hwpx"

    with pytest.raises(runner.TableFusionHwpxSmokeError):
        runner.main(_args(fusion_root, candidate_root, output))

    assert not output.exists()
    assert _partial_directories(tmp_path) == []


def test_runner_rejects_non_strict_candidate_plan_json(tmp_path: Path) -> None:
    fusion_root, candidate_root = _fixture(tmp_path)
    (candidate_root / "hwp_document_plan.candidate.json").write_bytes(
        b'{"id":"first","id":"second"}'
    )
    output = tmp_path / "duplicate-key.hwpx"

    with pytest.raises(ValueError, match="duplicate"):
        runner.main(_args(fusion_root, candidate_root, output))

    assert not output.exists()
    assert _partial_directories(tmp_path) == []
