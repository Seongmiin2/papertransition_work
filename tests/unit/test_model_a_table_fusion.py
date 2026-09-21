from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from scan2hwpx.contracts import (
    BBox,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.contracts.models import StrictContractModel
from scan2hwpx.model_a import table_fusion
from scan2hwpx.model_a.inference import ModelAPageAnalysis, ModelAPageImage
from scan2hwpx.model_a.table_fusion import (
    ModelATableDocumentAssemblyArtifact,
    ModelATableEvidencePromotionArtifact,
    ModelATableFusionError,
    ModelATablePageFusionArtifact,
    assemble_model_a_table_fused_document,
    fuse_model_a_table_page_analysis,
    promote_model_a_table_evidence,
)
from scan2hwpx.model_a.table_topology import build_model_a_table_topology

_WIDTH = 1_000
_HEIGHT = 1_400
_DOCUMENT_SHA = "1" * 64


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _png(*, table: bool = True) -> bytes:
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
        source_refs=("page-image", "ocr-source"),
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
    *,
    table: bool = True,
    shared_text: str | None = None,
) -> tuple[EvidenceIR, ModelAPageImage, ModelAPageAnalysis]:
    image_bytes = _png(table=table)
    observations = [
        _observation("outside-before", "before", (20, 20, 90, 45)),
        _observation("outside-after", "after", (100, 800, 180, 830)),
    ]
    if table:
        observations[1:1] = [
            _observation(
                "cell-shared" if shared_text is not None else "cell-00",
                shared_text if shared_text is not None else "A",
                (350, 150, 450, 180) if shared_text is not None else (120, 150, 180, 180),
            ),
            _observation("cell-10", "C", (120, 450, 180, 480)),
            _observation("cell-11", "D", (420, 450, 480, 480)),
        ]
        if shared_text is None:
            observations.insert(2, _observation("cell-01", "B", (420, 150, 480, 180)))
    evidence = EvidenceIR(
        id="evidence-fixture",
        source_document_sha256=_DOCUMENT_SHA,
        sources=(
            EvidenceSource(
                id="document",
                kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
                artifact_ref="artifact://fixture/document",
                producer="fixture",
                sha256=_DOCUMENT_SHA,
            ),
            EvidenceSource(
                id="page-image",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/page-image",
                producer="fixture",
                sha256=_sha256(image_bytes),
            ),
            EvidenceSource(
                id="ocr-source",
                kind=EvidenceSourceKind.OCR_PROVIDER,
                artifact_ref="artifact://fixture/ocr",
                producer="fixture",
            ),
        ),
        pages=(
            EvidencePage(
                id="page-0001",
                page_no=1,
                width=float(_WIDTH),
                height=float(_HEIGHT),
                image_source_ref="page-image",
                observations=tuple(observations),
            ),
        ),
    )
    page_image = ModelAPageImage(
        page_id="page-0001",
        page_no=1,
        image_source_ref="page-image",
        media_type="image/png",
        raw_bytes=image_bytes,
        sha256=_sha256(image_bytes),
    )
    nodes = tuple(
        TextContentNode(
            id=f"{evidence.id}:content:page-0001:{observation.id}",
            role=(
                ContentRole.CHOICE
                if observation.id.startswith("cell")
                else ContentRole.PASSAGE
            ),
            text=observation.ocr_candidates[0].text,
            evidence_refs=(observation.id,),
            confidence=0.0,
            needs_review=True,
        )
        for observation in observations
    )
    analysis = ModelAPageAnalysis(
        id=f"{evidence.id}:analysis:page-0001",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        page_id="page-0001",
        page_no=1,
        nodes=nodes,
        reading_order=tuple(node.id for node in nodes),
    )
    return evidence, page_image, analysis


def _reversed_two_page_no_table_fixture() -> tuple[
    EvidenceIR,
    tuple[ModelAPageImage, ModelAPageImage],
]:
    image_bytes = _png(table=False)
    image_sha = _sha256(image_bytes)
    evidence = EvidenceIR(
        id="evidence-two-page",
        source_document_sha256=_DOCUMENT_SHA,
        sources=(
            EvidenceSource(
                id="document",
                kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
                artifact_ref="artifact://fixture/document",
                producer="fixture",
                sha256=_DOCUMENT_SHA,
            ),
            EvidenceSource(
                id="page-image-0001",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/page-image-0001",
                producer="fixture",
                sha256=image_sha,
            ),
            EvidenceSource(
                id="page-image-0002",
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref="artifact://fixture/page-image-0002",
                producer="fixture",
                sha256=image_sha,
            ),
        ),
        pages=(
            EvidencePage(
                id="page-0002",
                page_no=2,
                width=float(_WIDTH),
                height=float(_HEIGHT),
                image_source_ref="page-image-0002",
            ),
            EvidencePage(
                id="page-0001",
                page_no=1,
                width=float(_WIDTH),
                height=float(_HEIGHT),
                image_source_ref="page-image-0001",
            ),
        ),
    )
    images = tuple(
        ModelAPageImage(
            page_id=page.id,
            page_no=page.page_no,
            image_source_ref=page.image_source_ref,
            media_type="image/png",
            raw_bytes=image_bytes,
            sha256=image_sha,
        )
        for page in evidence.pages
    )
    return evidence, images


def test_promotes_grid_and_fuses_table_without_duplicate_text_nodes() -> None:
    evidence, image, semantic = _fixture()
    topology = build_model_a_table_topology(evidence, page_image=image)

    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))
    fused = fuse_model_a_table_page_analysis(
        promotion,
        semantic_page_analysis=semantic,
    )

    assert promotion.promoted_evidence_ir.id == evidence.id
    assert contract_sha256(promotion.promoted_evidence_ir) != contract_sha256(evidence)
    assert len(promotion.artifact.added_source_ids) == 1
    assert len(promotion.artifact.added_table_grid_observation_ids) == 1
    promoted_page = promotion.promoted_evidence_ir.pages[0]
    grid = next(
        item for item in promoted_page.observations if item.kind == ObservationKind.TABLE_GRID
    )
    assert grid.id == topology.artifact.tables[0].table_grid_observation_id
    assert grid.source_refs[-1] == promotion.artifact.added_source_ids[0]
    assert promotion.promoted_evidence_ir.sources[: len(evidence.sources)] == evidence.sources
    original_ids = {item.id for item in evidence.pages[0].observations}
    assert tuple(
        item for item in promoted_page.observations if item.id in original_ids
    ) == evidence.pages[0].observations

    analysis = fused.artifact.fused_page_analysis
    assert tuple(node.kind for node in analysis.nodes) == ("text", "table", "text")
    table = analysis.nodes[1]
    assert table.kind == "table"
    assert (table.rows, table.columns, len(table.cells)) == (2, 2, 4)
    assert table.evidence_refs == (
        grid.id,
        "cell-00",
        "cell-01",
        "cell-10",
        "cell-11",
    )
    assert tuple(cell.text for cell in table.cells) == ("A", "B", "C", "D")
    assert all(cell.evidence_refs[0] == grid.id for cell in table.cells)
    assert {
        ref
        for node in analysis.nodes
        if node.kind == "text"
        for ref in node.evidence_refs
    } == {"outside-before", "outside-after"}
    assert {role.semantic_role for role in fused.artifact.semantic_role_sidecars[0].roles} == {
        ContentRole.CHOICE
    }
    assert fused.artifact.shared_text_resolutions == ()
    assert fused.artifact_bytes == fuse_model_a_table_page_analysis(
        promotion,
        semantic_page_analysis=semantic,
    ).artifact_bytes
    assembly = assemble_model_a_table_fused_document(
        promotion,
        page_fusions=(fused,),
    )
    assert assembly.artifact.content_ir.nodes == analysis.nodes
    assert assembly.artifact.content_ir.reading_order == analysis.reading_order
    assembly.artifact.content_ir.assert_evidence_integrity(promotion.promoted_evidence_ir)
    assert assembly.artifact_bytes == assemble_model_a_table_fused_document(
        promotion,
        page_fusions=(fused,),
    ).artifact_bytes
    equivalent_promotion = promote_model_a_table_evidence(
        evidence,
        topologies=(topology,),
    )
    rebound = assemble_model_a_table_fused_document(
        equivalent_promotion,
        page_fusions=(fused,),
    )
    assert rebound.artifact.source.promotion_artifact_sha256 == promotion.artifact_sha256


def test_no_table_promotion_is_a_noop_and_rebinds_same_digest() -> None:
    evidence, image, semantic = _fixture(table=False)
    topology = build_model_a_table_topology(evidence, page_image=image)

    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))
    fused = fuse_model_a_table_page_analysis(
        promotion,
        semantic_page_analysis=semantic,
    )

    assert promotion.promoted_evidence_ir == evidence
    assert promotion.artifact.added_source_ids == ()
    assert promotion.artifact.added_table_grid_observation_ids == ()
    assert fused.artifact.fused_page_analysis == semantic


def test_no_table_promotion_preserves_reversed_base_page_order() -> None:
    evidence, images = _reversed_two_page_no_table_fixture()
    topologies = tuple(
        build_model_a_table_topology(evidence, page_image=image) for image in images
    )

    promotion = promote_model_a_table_evidence(evidence, topologies=topologies)

    assert promotion.promoted_evidence_ir == evidence
    assert tuple(page.id for page in promotion.promoted_evidence_ir.pages) == (
        "page-0002",
        "page-0001",
    )
    assert promotion.promoted_evidence_ir.sources == evidence.sources
    assert tuple(page.page_id for page in promotion.artifact.pages) == (
        "page-0001",
        "page-0002",
    )


def test_promotion_hashes_base_evidence_once_per_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, image, _semantic = _fixture(table=False)
    topology = build_model_a_table_topology(evidence, page_image=image)
    original_sha256 = table_fusion.contract_sha256
    base_calls = 0

    def counting_sha256(model: StrictContractModel) -> str:
        nonlocal base_calls
        if model is evidence:
            base_calls += 1
        return original_sha256(model)

    monkeypatch.setattr(table_fusion, "contract_sha256", counting_sha256)
    promote_model_a_table_evidence(evidence, topologies=(topology,))

    # Public construction derives once and Built* replay derives once.
    assert base_calls == 2


def test_shared_horizontal_ocr_records_conservative_token_resolution() -> None:
    evidence, image, semantic = _fixture(shared_text="11. ②")
    topology = build_model_a_table_topology(evidence, page_image=image)
    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))

    fused = fuse_model_a_table_page_analysis(
        promotion,
        semantic_page_analysis=semantic,
    )

    resolution = fused.artifact.shared_text_resolutions[0]
    assert resolution.observation_id == "cell-shared"
    assert tuple(item.text for item in resolution.cells) == ("11.", "②")
    table = next(node for node in fused.artifact.fused_page_analysis.nodes if node.kind == "table")
    assert tuple(cell.text for cell in table.cells[:2]) == ("11.", "②")
    assert table.evidence_refs.count("cell-shared") == 1
    assert sum("cell-shared" in cell.evidence_refs for cell in table.cells) == 2


def test_unresolved_shared_ocr_and_stale_semantic_are_rejected() -> None:
    evidence, image, semantic = _fixture(shared_text="joined")
    topology = build_model_a_table_topology(evidence, page_image=image)
    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))

    with pytest.raises(ModelATableFusionError, match="cannot be split"):
        fuse_model_a_table_page_analysis(
            promotion,
            semantic_page_analysis=semantic,
        )

    stale = semantic.model_copy(update={"evidence_ir_sha256": "0" * 64})
    with pytest.raises(ModelATableFusionError, match="does not bind base"):
        fuse_model_a_table_page_analysis(
            promotion,
            semantic_page_analysis=stale,
        )


def test_requires_complete_page_topology_coverage() -> None:
    evidence, _image, _semantic = _fixture()

    with pytest.raises(ModelATableFusionError, match="cover every"):
        promote_model_a_table_evidence(evidence, topologies=())


def test_document_assembly_requires_complete_page_fusion_coverage() -> None:
    evidence, image, _semantic = _fixture()
    topology = build_model_a_table_topology(evidence, page_image=image)
    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))

    with pytest.raises(ModelATableFusionError, match="cover every"):
        assemble_model_a_table_fused_document(promotion, page_fusions=())


def test_embedded_models_are_strictly_revalidated() -> None:
    evidence, image, semantic = _fixture()
    topology = build_model_a_table_topology(evidence, page_image=image)
    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))
    fused = fuse_model_a_table_page_analysis(
        promotion,
        semantic_page_analysis=semantic,
    )
    assembly = assemble_model_a_table_fused_document(
        promotion,
        page_fusions=(fused,),
    )

    invalid_page = fused.artifact.fused_page_analysis.model_copy(update={"id": ""})
    with pytest.raises(ValidationError, match="valid strict model instance"):
        ModelATablePageFusionArtifact(
            source=fused.artifact.source,
            semantic_role_sidecars=fused.artifact.semantic_role_sidecars,
            shared_text_resolutions=fused.artifact.shared_text_resolutions,
            fused_page_analysis=invalid_page,
            fused_page_analysis_sha256=contract_sha256(invalid_page),
        )

    invalid_content = assembly.artifact.content_ir.model_copy(update={"id": ""})
    with pytest.raises(ValidationError, match="valid strict model instance"):
        ModelATableDocumentAssemblyArtifact(
            source=assembly.artifact.source,
            pages=assembly.artifact.pages,
            content_ir=invalid_content,
            content_ir_sha256=contract_sha256(invalid_content),
        )


def test_page_fusion_enforces_each_aggregate_resource_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, image, semantic = _fixture()
    topology = build_model_a_table_topology(evidence, page_image=image)
    promotion = promote_model_a_table_evidence(evidence, topologies=(topology,))
    artifact = fuse_model_a_table_page_analysis(
        promotion,
        semantic_page_analysis=semantic,
    ).artifact

    limit_names = (
        "_MAX_PAGES",
        "_MAX_TABLES_PER_DOCUMENT",
        "_MAX_TABLE_CELLS_PER_DOCUMENT",
        "_MAX_EVIDENCE_REFS_PER_DOCUMENT",
        "_MAX_TEXT_CHARS_PER_DOCUMENT",
        "_MAX_SIDECAR_ITEMS_PER_DOCUMENT",
    )
    for limit_name in limit_names:
        with monkeypatch.context() as local_patch:
            local_patch.setattr(table_fusion, limit_name, 0)
            with pytest.raises(ValidationError, match="aggregate resource budget"):
                ModelATablePageFusionArtifact(
                    source=artifact.source,
                    semantic_role_sidecars=artifact.semantic_role_sidecars,
                    shared_text_resolutions=artifact.shared_text_resolutions,
                    fused_page_analysis=artifact.fused_page_analysis,
                    fused_page_analysis_sha256=artifact.fused_page_analysis_sha256,
                )


def test_artifact_collections_publish_max_items() -> None:
    promotion = ModelATableEvidencePromotionArtifact.model_json_schema()["properties"]
    fusion = ModelATablePageFusionArtifact.model_json_schema()["properties"]
    assembly = ModelATableDocumentAssemblyArtifact.model_json_schema()["properties"]

    assert promotion["pages"]["maxItems"] == 2_000
    assert promotion["added_source_ids"]["maxItems"] == 2_000
    assert promotion["added_table_grid_observation_ids"]["maxItems"] == 10_000
    assert fusion["semantic_role_sidecars"]["maxItems"] == 1_000
    assert fusion["shared_text_resolutions"]["maxItems"] == 20_000
    assert assembly["pages"]["maxItems"] == 2_000
