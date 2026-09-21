from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, ValidationError, model_validator

from scan2hwpx.contracts import (
    BBox,
    ContentIR,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    FormulaContentNode,
    ObservationKind,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.contracts.models import ContentNode, StrictContractModel

from .inference import ModelAPageAnalysis, validate_model_a_page_analysis
from .table_topology import BuiltModelATableTopology, ModelATableTopologyTable

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedText = Annotated[str, Field(min_length=1, max_length=1_024)]

_MAX_PAGES = 2_000
_MAX_TABLES_PER_PAGE = 1_000
_MAX_TABLES_PER_DOCUMENT = 10_000
_MAX_TABLE_CELLS_PER_DOCUMENT = 1_000_000
_MAX_EVIDENCE_REFS_PER_DOCUMENT = 2_000_000
_MAX_TEXT_CHARS_PER_DOCUMENT = 16 * 1024 * 1024
_MAX_SIDECAR_ITEMS_PER_DOCUMENT = 1_000_000
_MAX_ROLE_BINDINGS_PER_SIDECAR = 20_000
_MAX_SHARED_TEXT_RESOLUTIONS_PER_PAGE = 20_000
_MAX_SHARED_CELL_FRAGMENTS = 10_000

TABLE_EVIDENCE_PROMOTION_VERSION: Literal["model-a-table-evidence-promotion/1.0"] = (
    "model-a-table-evidence-promotion/1.0"
)
TABLE_PAGE_FUSION_VERSION: Literal["model-a-table-page-fusion/1.0"] = (
    "model-a-table-page-fusion/1.0"
)


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


def _strict_page_analysis_sha256(value: ModelAPageAnalysis) -> str:
    if type(value) is not ModelAPageAnalysis:
        raise TypeError("fused_page_analysis must be ModelAPageAnalysis")
    try:
        raw = _canonical_json_bytes(value.model_dump(mode="json"))
        restored = ModelAPageAnalysis.model_validate_json(raw, strict=True)
    except (TypeError, ValueError, ValidationError, OverflowError) as exc:
        raise ValueError("fused page analysis is not a valid strict model instance") from exc
    if restored != value:
        raise ValueError("fused page analysis is not canonical")
    return _sha256(raw)


def _strict_content_ir_sha256(value: ContentIR) -> str:
    if type(value) is not ContentIR:
        raise TypeError("content_ir must be ContentIR")
    try:
        raw = _canonical_json_bytes(value.model_dump(mode="json"))
        restored = ContentIR.model_validate_json(raw, strict=True)
    except (TypeError, ValueError, ValidationError, OverflowError) as exc:
        raise ValueError("ContentIR is not a valid strict model instance") from exc
    if restored != value:
        raise ValueError("ContentIR is not canonical")
    return _sha256(raw)


def _content_resource_usage(
    nodes: tuple[ContentNode, ...],
) -> tuple[int, int, int, int]:
    table_count = 0
    cell_count = 0
    ref_count = 0
    text_chars = 0
    for node in nodes:
        ref_count += len(node.evidence_refs)
        if isinstance(node, TextContentNode):
            text_chars += len(node.text)
        elif isinstance(node, FormulaContentNode):
            text_chars += len(node.expression)
        elif isinstance(node, TableContentNode):
            table_count += 1
            cell_count += len(node.cells)
            ref_count += sum(len(cell.evidence_refs) for cell in node.cells)
            text_chars += sum(len(cell.text) for cell in node.cells)
    return table_count, cell_count, ref_count, text_chars


def _require_aggregate_budget(
    *,
    label: str,
    pages: int,
    tables: int,
    cells: int,
    refs: int,
    text_chars: int,
    sidecar_items: int,
) -> None:
    limits = (
        ("pages", pages, _MAX_PAGES),
        ("tables", tables, _MAX_TABLES_PER_DOCUMENT),
        ("cells", cells, _MAX_TABLE_CELLS_PER_DOCUMENT),
        ("evidence refs", refs, _MAX_EVIDENCE_REFS_PER_DOCUMENT),
        ("text characters", text_chars, _MAX_TEXT_CHARS_PER_DOCUMENT),
        ("sidecar items", sidecar_items, _MAX_SIDECAR_ITEMS_PER_DOCUMENT),
    )
    for resource, value, maximum in limits:
        if value > maximum:
            raise ModelATableFusionError(
                f"{label} {resource} exceed the aggregate resource budget"
            )


class ModelATableFusionError(ValueError):
    """Raised when table evidence or semantic ownership cannot be fused safely."""


class _StrictModel(StrictContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class _NonEligibleBoundary(_StrictModel):
    research_only: Literal[True] = True
    human_review_required: Literal[True] = True
    verification_status: Literal["unverified"] = "unverified"
    rights_status: Literal["unverified"] = "unverified"
    identity_assurance: Literal["self_asserted_untrusted"] = "self_asserted_untrusted"
    golden_eligible: Literal[False] = False
    training_eligible: Literal[False] = False
    release_eligible: Literal[False] = False


class ModelATableEvidencePromotionPage(_StrictModel):
    page_id: BoundedText
    page_no: int = Field(ge=1)
    topology_artifact_sha256: Sha256
    layout_source_id: BoundedText | None
    table_grid_observation_ids: tuple[BoundedText, ...] = Field(
        max_length=_MAX_TABLES_PER_PAGE
    )


class ModelATableEvidencePromotionArtifact(_NonEligibleBoundary):
    schema_version: Literal["model-a-table-evidence-promotion/1.0"] = (
        TABLE_EVIDENCE_PROMOTION_VERSION
    )
    artifact_role: Literal["deterministic_table_grid_evidence_promotion"] = (
        "deterministic_table_grid_evidence_promotion"
    )
    compiler_id: Literal["scan2hwpx.model_a.table_fusion"] = (
        "scan2hwpx.model_a.table_fusion"
    )
    base_evidence_ir_id: BoundedText
    base_evidence_ir_sha256: Sha256
    source_document_sha256: Sha256
    pages: tuple[ModelATableEvidencePromotionPage, ...] = Field(max_length=_MAX_PAGES)
    promoted_evidence_ir_id: BoundedText
    promoted_evidence_ir_sha256: Sha256
    added_source_ids: tuple[BoundedText, ...] = Field(max_length=_MAX_PAGES)
    added_table_grid_observation_ids: tuple[BoundedText, ...] = Field(
        max_length=_MAX_TABLES_PER_DOCUMENT
    )

    @model_validator(mode="after")
    def validate_promotion_summary(self) -> Self:
        order = tuple((page.page_no, page.page_id) for page in self.pages)
        if order != tuple(sorted(order)):
            raise ValueError("promotion pages must be in canonical document order")
        source_ids = tuple(
            page.layout_source_id for page in self.pages if page.layout_source_id is not None
        )
        if source_ids != self.added_source_ids:
            raise ValueError("promotion added source summary does not match pages")
        grid_ids = tuple(
            grid_id for page in self.pages for grid_id in page.table_grid_observation_ids
        )
        if grid_ids != self.added_table_grid_observation_ids:
            raise ValueError("promotion added grid summary does not match pages")
        if len(source_ids) != len(set(source_ids)) or len(grid_ids) != len(set(grid_ids)):
            raise ValueError("promotion source and grid ids must be unique")
        _require_aggregate_budget(
            label="table evidence promotion",
            pages=len(self.pages),
            tables=len(grid_ids),
            cells=0,
            refs=0,
            text_chars=0,
            sidecar_items=0,
        )
        return self


@dataclass(frozen=True, slots=True)
class BuiltModelATableEvidencePromotion:
    artifact: ModelATableEvidencePromotionArtifact
    artifact_bytes: bytes
    artifact_sha256: str
    base_evidence_ir: EvidenceIR
    promoted_evidence_ir: EvidenceIR
    topologies: tuple[BuiltModelATableTopology, ...]

    def __post_init__(self) -> None:
        if type(self.artifact) is not ModelATableEvidencePromotionArtifact:
            raise TypeError("artifact must be ModelATableEvidencePromotionArtifact")
        expected_artifact, expected_evidence, ordered = _derive_table_evidence_promotion(
            self.base_evidence_ir,
            self.topologies,
        )
        if self.topologies != ordered:
            raise ValueError("promotion topologies must be in canonical document order")
        if self.artifact != expected_artifact or self.promoted_evidence_ir != expected_evidence:
            raise ValueError("table evidence promotion cannot be reproduced from inputs")
        expected_bytes = _canonical_json_bytes(expected_artifact.model_dump(mode="json"))
        if self.artifact_bytes != expected_bytes:
            raise ValueError("table evidence promotion bytes are not canonical")
        if _sha256(self.artifact_bytes) != self.artifact_sha256:
            raise ValueError("table evidence promotion digest mismatch")


class ModelATableSemanticRoleBinding(_StrictModel):
    observation_id: BoundedText
    semantic_role: ContentRole


class ModelATableSemanticRoleSidecar(_StrictModel):
    table_grid_observation_id: BoundedText
    roles: tuple[ModelATableSemanticRoleBinding, ...] = Field(
        max_length=_MAX_ROLE_BINDINGS_PER_SIDECAR
    )


class ModelATableSharedCellFragment(_StrictModel):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    text: BoundedText


class ModelATableSharedTextResolution(_StrictModel):
    table_grid_observation_id: BoundedText
    observation_id: BoundedText
    method: Literal["whitespace_tokens_equal_horizontal_cell_count"] = (
        "whitespace_tokens_equal_horizontal_cell_count"
    )
    source_text_sha256: Sha256
    cells: tuple[ModelATableSharedCellFragment, ...] = Field(
        min_length=2,
        max_length=_MAX_SHARED_CELL_FRAGMENTS,
    )


class ModelATablePageFusionSource(_StrictModel):
    base_evidence_ir_id: BoundedText
    base_evidence_ir_sha256: Sha256
    promoted_evidence_ir_sha256: Sha256
    promotion_artifact_sha256: Sha256
    semantic_page_analysis_sha256: Sha256
    topology_artifact_sha256: Sha256
    target_page_id: BoundedText
    target_page_no: int = Field(ge=1)


class ModelATablePageFusionArtifact(_NonEligibleBoundary):
    schema_version: Literal["model-a-table-page-fusion/1.0"] = TABLE_PAGE_FUSION_VERSION
    artifact_role: Literal["deterministic_semantic_table_page_fusion"] = (
        "deterministic_semantic_table_page_fusion"
    )
    source: ModelATablePageFusionSource
    semantic_role_sidecars: tuple[ModelATableSemanticRoleSidecar, ...] = Field(
        max_length=_MAX_TABLES_PER_PAGE
    )
    shared_text_resolutions: tuple[ModelATableSharedTextResolution, ...] = Field(
        max_length=_MAX_SHARED_TEXT_RESOLUTIONS_PER_PAGE
    )
    fused_page_analysis: ModelAPageAnalysis
    fused_page_analysis_sha256: Sha256

    @model_validator(mode="after")
    def validate_fused_digest(self) -> Self:
        fused_sha = _strict_page_analysis_sha256(self.fused_page_analysis)
        if fused_sha != self.fused_page_analysis_sha256:
            raise ValueError("fused page analysis digest mismatch")
        tables, cells, refs, text_chars = _content_resource_usage(
            self.fused_page_analysis.nodes
        )
        sidecar_items = (
            len(self.semantic_role_sidecars)
            + sum(len(sidecar.roles) for sidecar in self.semantic_role_sidecars)
            + len(self.shared_text_resolutions)
            + sum(len(resolution.cells) for resolution in self.shared_text_resolutions)
        )
        sidecar_refs = sum(
            1 + len(sidecar.roles) for sidecar in self.semantic_role_sidecars
        ) + sum(
            2 + len(resolution.cells) for resolution in self.shared_text_resolutions
        )
        sidecar_text = sum(
            len(fragment.text)
            for resolution in self.shared_text_resolutions
            for fragment in resolution.cells
        )
        _require_aggregate_budget(
            label="table page fusion",
            pages=1,
            tables=tables,
            cells=cells,
            refs=refs + sidecar_refs,
            text_chars=text_chars + sidecar_text,
            sidecar_items=sidecar_items,
        )
        return self


@dataclass(frozen=True, slots=True)
class BuiltModelATablePageFusion:
    artifact: ModelATablePageFusionArtifact
    artifact_bytes: bytes
    artifact_sha256: str
    promotion: BuiltModelATableEvidencePromotion
    semantic_page_analysis: ModelAPageAnalysis

    def __post_init__(self) -> None:
        if type(self.artifact) is not ModelATablePageFusionArtifact:
            raise TypeError("artifact must be ModelATablePageFusionArtifact")
        expected, topology = _derive_table_page_fusion(
            self.promotion,
            self.semantic_page_analysis,
        )
        if self.artifact != expected:
            raise ValueError("table page fusion cannot be reproduced from inputs")
        if self.artifact.source.topology_artifact_sha256 != topology.artifact_sha256:
            raise ValueError("fused page topology digest mismatch")
        expected_bytes = _canonical_json_bytes(expected.model_dump(mode="json"))
        if self.artifact_bytes != expected_bytes:
            raise ValueError("table page fusion bytes are not canonical")
        if _sha256(self.artifact_bytes) != self.artifact_sha256:
            raise ValueError("table page fusion artifact digest mismatch")


class ModelATableDocumentAssemblyPage(_StrictModel):
    page_id: BoundedText
    page_no: int = Field(ge=1)
    fusion_artifact_sha256: Sha256
    fused_page_analysis_sha256: Sha256


class ModelATableDocumentAssemblySource(_StrictModel):
    promoted_evidence_ir_id: BoundedText
    promoted_evidence_ir_sha256: Sha256
    promotion_artifact_sha256: Sha256
    source_document_sha256: Sha256
    content_ir_id: BoundedText
    content_ir_revision: Literal[1] = 1


class ModelATableDocumentAssemblyArtifact(_NonEligibleBoundary):
    schema_version: Literal["model-a-table-document-assembly/1.0"] = (
        "model-a-table-document-assembly/1.0"
    )
    artifact_role: Literal["deterministic_table_fused_document_assembly"] = (
        "deterministic_table_fused_document_assembly"
    )
    source: ModelATableDocumentAssemblySource
    pages: tuple[ModelATableDocumentAssemblyPage, ...] = Field(max_length=_MAX_PAGES)
    content_ir: ContentIR
    content_ir_sha256: Sha256

    @model_validator(mode="after")
    def validate_content_digest(self) -> Self:
        order = tuple((page.page_no, page.page_id) for page in self.pages)
        if order != tuple(sorted(order)):
            raise ValueError("assembled pages must be in canonical document order")
        content_sha = _strict_content_ir_sha256(self.content_ir)
        if content_sha != self.content_ir_sha256:
            raise ValueError("assembled ContentIR digest mismatch")
        tables, cells, refs, text_chars = _content_resource_usage(self.content_ir.nodes)
        _require_aggregate_budget(
            label="table document assembly",
            pages=len(self.pages),
            tables=tables,
            cells=cells,
            refs=refs,
            text_chars=text_chars,
            sidecar_items=0,
        )
        return self


@dataclass(frozen=True, slots=True)
class BuiltModelATableDocumentAssembly:
    artifact: ModelATableDocumentAssemblyArtifact
    artifact_bytes: bytes
    artifact_sha256: str
    promotion: BuiltModelATableEvidencePromotion
    page_fusions: tuple[BuiltModelATablePageFusion, ...]

    def __post_init__(self) -> None:
        if type(self.artifact) is not ModelATableDocumentAssemblyArtifact:
            raise TypeError("artifact must be ModelATableDocumentAssemblyArtifact")
        expected, ordered = _derive_table_document_assembly(
            self.promotion,
            self.page_fusions,
        )
        if self.page_fusions != ordered:
            raise ValueError("page fusions must be in canonical document order")
        if self.artifact != expected:
            raise ValueError("table document assembly cannot be reproduced from inputs")
        expected_bytes = _canonical_json_bytes(expected.model_dump(mode="json"))
        if self.artifact_bytes != expected_bytes:
            raise ValueError("table document assembly bytes are not canonical")
        if _sha256(self.artifact_bytes) != self.artifact_sha256:
            raise ValueError("table document assembly artifact digest mismatch")


def promote_model_a_table_evidence(
    evidence_ir: EvidenceIR,
    *,
    topologies: tuple[BuiltModelATableTopology, ...],
) -> BuiltModelATableEvidencePromotion:
    """Add deterministic table_grid observations for one complete document."""

    artifact, promoted, ordered = _derive_table_evidence_promotion(evidence_ir, topologies)
    artifact_bytes = _canonical_json_bytes(artifact.model_dump(mode="json"))
    return BuiltModelATableEvidencePromotion(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        artifact_sha256=_sha256(artifact_bytes),
        base_evidence_ir=evidence_ir,
        promoted_evidence_ir=promoted,
        topologies=ordered,
    )


def fuse_model_a_table_page_analysis(
    promotion: BuiltModelATableEvidencePromotion,
    *,
    semantic_page_analysis: ModelAPageAnalysis,
) -> BuiltModelATablePageFusion:
    """Replace table-owned semantic line runs with grounded editable table nodes."""

    artifact, _topology = _derive_table_page_fusion(promotion, semantic_page_analysis)
    artifact_bytes = _canonical_json_bytes(artifact.model_dump(mode="json"))
    return BuiltModelATablePageFusion(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        artifact_sha256=_sha256(artifact_bytes),
        promotion=promotion,
        semantic_page_analysis=semantic_page_analysis,
    )


def assemble_model_a_table_fused_document(
    promotion: BuiltModelATableEvidencePromotion,
    *,
    page_fusions: tuple[BuiltModelATablePageFusion, ...],
) -> BuiltModelATableDocumentAssembly:
    """Assemble one promoted document from exactly one fused result per page."""

    artifact, ordered = _derive_table_document_assembly(promotion, page_fusions)
    artifact_bytes = _canonical_json_bytes(artifact.model_dump(mode="json"))
    return BuiltModelATableDocumentAssembly(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        artifact_sha256=_sha256(artifact_bytes),
        promotion=promotion,
        page_fusions=ordered,
    )


def _require_topology_aggregate_budget(
    evidence_ir: EvidenceIR,
    topologies: tuple[BuiltModelATableTopology, ...],
) -> None:
    table_count = 0
    cell_count = 0
    ref_count = 0
    text_chars = 0
    for topology in topologies:
        table_count += len(topology.artifact.tables)
        for table in topology.artifact.tables:
            cell_count += len(table.cells)
            ref_count += len(table.ordered_observation_ids)
            ref_count += sum(len(cell.ordered_observation_ids) for cell in table.cells)
            text_chars += sum(len(cell.text) for cell in table.cells)
    _require_aggregate_budget(
        label="table topology inputs",
        pages=len(evidence_ir.pages),
        tables=table_count,
        cells=cell_count,
        refs=ref_count,
        text_chars=text_chars,
        sidecar_items=0,
    )


def _require_append_only_promotion(
    base: EvidenceIR,
    promoted: EvidenceIR,
    *,
    added_sources: tuple[EvidenceSource, ...],
    added_grid_ids: tuple[str, ...],
) -> None:
    if (
        promoted.id != base.id
        or promoted.source_document_sha256 != base.source_document_sha256
        or promoted.sources != (*base.sources, *added_sources)
    ):
        raise ModelATableFusionError("promoted EvidenceIR is not append-only over sources")
    if tuple(page.id for page in promoted.pages) != tuple(page.id for page in base.pages):
        raise ModelATableFusionError("promoted EvidenceIR must preserve base page order")

    observed_added_ids: list[str] = []
    allowed_added_ids = set(added_grid_ids)
    for base_page, promoted_page in zip(base.pages, promoted.pages, strict=True):
        base_metadata = (
            base_page.id,
            base_page.page_no,
            base_page.width,
            base_page.height,
            base_page.rotation,
            base_page.image_source_ref,
        )
        promoted_metadata = (
            promoted_page.id,
            promoted_page.page_no,
            promoted_page.width,
            promoted_page.height,
            promoted_page.rotation,
            promoted_page.image_source_ref,
        )
        if promoted_metadata != base_metadata:
            raise ModelATableFusionError("promotion changed immutable base page metadata")
        base_ids = {observation.id for observation in base_page.observations}
        carried = tuple(
            observation
            for observation in promoted_page.observations
            if observation.id in base_ids
        )
        if carried != base_page.observations:
            raise ModelATableFusionError(
                "promotion changed base observations or their relative order"
            )
        added = tuple(
            observation
            for observation in promoted_page.observations
            if observation.id not in base_ids
        )
        if any(
            observation.kind != ObservationKind.TABLE_GRID
            or observation.id not in allowed_added_ids
            for observation in added
        ):
            raise ModelATableFusionError("promotion added non-table or undeclared evidence")
        observed_added_ids.extend(observation.id for observation in added)
    if len(observed_added_ids) != len(set(observed_added_ids)) or set(
        observed_added_ids
    ) != allowed_added_ids:
        raise ModelATableFusionError("promotion grid summary is not append-only evidence")


def _promotion_binding(
    promotion: BuiltModelATableEvidencePromotion,
) -> tuple[str, ...]:
    artifact = promotion.artifact
    return (
        promotion.artifact_sha256,
        artifact.base_evidence_ir_id,
        artifact.base_evidence_ir_sha256,
        artifact.promoted_evidence_ir_id,
        artifact.promoted_evidence_ir_sha256,
        artifact.source_document_sha256,
    )


def _page_fusion_sidecar_usage(
    page_fusions: tuple[BuiltModelATablePageFusion, ...],
) -> tuple[int, int, int]:
    items = 0
    refs = 0
    text_chars = 0
    for fusion in page_fusions:
        artifact = fusion.artifact
        items += len(artifact.semantic_role_sidecars)
        items += sum(len(sidecar.roles) for sidecar in artifact.semantic_role_sidecars)
        items += len(artifact.shared_text_resolutions)
        items += sum(len(item.cells) for item in artifact.shared_text_resolutions)
        refs += sum(
            1 + len(sidecar.roles) for sidecar in artifact.semantic_role_sidecars
        )
        refs += sum(
            2 + len(item.cells) for item in artifact.shared_text_resolutions
        )
        text_chars += sum(
            len(fragment.text)
            for item in artifact.shared_text_resolutions
            for fragment in item.cells
        )
    return items, refs, text_chars


def _derive_table_evidence_promotion(
    evidence_ir: EvidenceIR,
    topologies: tuple[BuiltModelATableTopology, ...],
) -> tuple[
    ModelATableEvidencePromotionArtifact,
    EvidenceIR,
    tuple[BuiltModelATableTopology, ...],
]:
    if type(evidence_ir) is not EvidenceIR:
        raise TypeError("evidence_ir must be EvidenceIR")
    if not isinstance(topologies, tuple):
        raise TypeError("topologies must be a tuple")
    if any(type(topology) is not BuiltModelATableTopology for topology in topologies):
        raise TypeError("topologies must contain BuiltModelATableTopology values")
    if len(topologies) > _MAX_PAGES:
        raise ModelATableFusionError("topology inputs exceed the page resource budget")
    base_sha = contract_sha256(evidence_ir)
    _require_topology_aggregate_budget(evidence_ir, topologies)
    pages = tuple(sorted(evidence_ir.pages, key=lambda page: (page.page_no, page.id)))
    topology_by_page: dict[str, BuiltModelATableTopology] = {}
    for topology in topologies:
        topology_source = topology.artifact.source
        if (
            topology_source.evidence_ir_id != evidence_ir.id
            or topology_source.evidence_ir_contract_sha256 != base_sha
            or topology_source.source_document_sha256
            != evidence_ir.source_document_sha256
        ):
            raise ModelATableFusionError("topology does not bind the exact base EvidenceIR")
        page_id = topology.artifact.source.target_page_id
        if page_id in topology_by_page:
            raise ModelATableFusionError("duplicate page topology")
        topology_by_page[page_id] = topology
    if set(topology_by_page) != {page.id for page in pages}:
        raise ModelATableFusionError("topologies must cover every EvidenceIR page exactly once")
    ordered = tuple(topology_by_page[page.id] for page in pages)

    source_ids = {source.id for source in evidence_ir.sources}
    observation_ids = {
        observation.id for page in evidence_ir.pages for observation in page.observations
    }
    original_sources = tuple(
        source
        for source in evidence_ir.sources
        if source.kind == EvidenceSourceKind.ORIGINAL_DOCUMENT
        and source.sha256 == evidence_ir.source_document_sha256
    )
    if len(original_sources) != 1:
        raise ModelATableFusionError("base EvidenceIR must bind one exact original document")
    original_source_id = original_sources[0].id
    added_sources: list[EvidenceSource] = []
    promoted_page_by_id: dict[str, EvidencePage] = {}
    page_summaries: list[ModelATableEvidencePromotionPage] = []
    added_grid_ids: list[str] = []

    for page, topology in zip(pages, ordered, strict=True):
        source = topology.artifact.source
        if (
            source.target_page_id != page.id
            or source.target_page_no != page.page_no
            or source.evidence_ir_id != evidence_ir.id
            or source.evidence_ir_contract_sha256 != base_sha
        ):
            raise ModelATableFusionError("topology page lineage does not match EvidenceIR")
        if not topology.artifact.tables:
            promoted_page_by_id[page.id] = page
            page_summaries.append(
                ModelATableEvidencePromotionPage(
                    page_id=page.id,
                    page_no=page.page_no,
                    topology_artifact_sha256=topology.artifact_sha256,
                    layout_source_id=None,
                    table_grid_observation_ids=(),
                )
            )
            continue

        layout_source_id = f"{page.id}:source:table-topology:{topology.artifact_sha256[:16]}"
        if layout_source_id in source_ids:
            raise ModelATableFusionError("table topology layout source id collides")
        source_ids.add(layout_source_id)
        added_sources.append(
            EvidenceSource(
                id=layout_source_id,
                kind=EvidenceSourceKind.LAYOUT_PROVIDER,
                artifact_ref=f"artifact://{topology.artifact_sha256}/model-a-table-topology",
                producer=(
                    f"{topology.artifact.detector.detector_id}@"
                    f"{topology.artifact.detector.detector_version};policy="
                    f"{topology.artifact.detector.detector_policy_sha256}"
                ),
                sha256=topology.artifact_sha256,
            )
        )
        page_observation_by_id = {observation.id: observation for observation in page.observations}
        insertions: dict[str, list[EvidenceObservation]] = defaultdict(list)
        page_grid_ids: list[str] = []
        for table in topology.artifact.tables:
            if not table.ordered_observation_ids:
                raise ModelATableFusionError(
                    "table without text evidence cannot be placed safely in fusion v1"
                )
            for observation_id in table.ordered_observation_ids:
                observation = page_observation_by_id.get(observation_id)
                if observation is None or observation.kind != ObservationKind.TEXT_LINE:
                    raise ModelATableFusionError(
                        "table topology ref is not a target-page text_line"
                    )
            grid_id = table.table_grid_observation_id
            if grid_id in observation_ids:
                raise ModelATableFusionError("table_grid observation id collides")
            observation_ids.add(grid_id)
            page_grid_ids.append(grid_id)
            added_grid_ids.append(grid_id)
            nx0, ny0, nx1, ny1 = table.bbox_normalized
            grid_observation = EvidenceObservation(
                id=grid_id,
                kind=ObservationKind.TABLE_GRID,
                bbox=BBox(
                    pixel=(
                        nx0 * page.width,
                        ny0 * page.height,
                        nx1 * page.width,
                        ny1 * page.height,
                    ),
                    normalized=table.bbox_normalized,
                ),
                confidence=0.0,
                source_refs=(original_source_id, page.image_source_ref, layout_source_id),
            )
            insertions[table.ordered_observation_ids[0]].append(grid_observation)
        new_observations: list[EvidenceObservation] = []
        for observation in page.observations:
            new_observations.extend(insertions.get(observation.id, ()))
            new_observations.append(observation)
        promoted_page_by_id[page.id] = (
            EvidencePage(
                id=page.id,
                page_no=page.page_no,
                width=page.width,
                height=page.height,
                rotation=page.rotation,
                image_source_ref=page.image_source_ref,
                observations=tuple(new_observations),
            )
        )
        page_summaries.append(
            ModelATableEvidencePromotionPage(
                page_id=page.id,
                page_no=page.page_no,
                topology_artifact_sha256=topology.artifact_sha256,
                layout_source_id=layout_source_id,
                table_grid_observation_ids=tuple(page_grid_ids),
            )
        )

    promoted = EvidenceIR(
        id=evidence_ir.id,
        source_document_sha256=evidence_ir.source_document_sha256,
        sources=(*evidence_ir.sources, *added_sources),
        pages=tuple(promoted_page_by_id[page.id] for page in evidence_ir.pages),
    )
    added_sources_tuple = tuple(added_sources)
    added_grid_ids_tuple = tuple(added_grid_ids)
    _require_append_only_promotion(
        evidence_ir,
        promoted,
        added_sources=added_sources_tuple,
        added_grid_ids=added_grid_ids_tuple,
    )
    promoted_sha = contract_sha256(promoted)
    artifact = ModelATableEvidencePromotionArtifact(
        base_evidence_ir_id=evidence_ir.id,
        base_evidence_ir_sha256=base_sha,
        source_document_sha256=evidence_ir.source_document_sha256,
        pages=tuple(page_summaries),
        promoted_evidence_ir_id=promoted.id,
        promoted_evidence_ir_sha256=promoted_sha,
        added_source_ids=tuple(source.id for source in added_sources_tuple),
        added_table_grid_observation_ids=added_grid_ids_tuple,
    )
    return artifact, promoted, ordered


def _derive_table_page_fusion(
    promotion: BuiltModelATableEvidencePromotion,
    semantic_page_analysis: ModelAPageAnalysis,
) -> tuple[ModelATablePageFusionArtifact, BuiltModelATableTopology]:
    if type(promotion) is not BuiltModelATableEvidencePromotion:
        raise TypeError("promotion must be BuiltModelATableEvidencePromotion")
    if type(semantic_page_analysis) is not ModelAPageAnalysis:
        raise TypeError("semantic_page_analysis must be ModelAPageAnalysis")
    base = promotion.base_evidence_ir
    promoted = promotion.promoted_evidence_ir
    base_sha = promotion.artifact.base_evidence_ir_sha256
    promoted_sha = promotion.artifact.promoted_evidence_ir_sha256
    if (
        promotion.artifact.base_evidence_ir_id != base.id
        or promotion.artifact.promoted_evidence_ir_id != promoted.id
        or promotion.artifact.source_document_sha256
        != base.source_document_sha256
        or promoted.source_document_sha256 != base.source_document_sha256
    ):
        raise ModelATableFusionError("promotion EvidenceIR identity binding is inconsistent")
    if (
        semantic_page_analysis.evidence_ir_id != base.id
        or semantic_page_analysis.evidence_ir_sha256 != base_sha
    ):
        raise ModelATableFusionError("semantic page analysis does not bind base EvidenceIR")
    base_pages = {page.id: page for page in base.pages}
    page = base_pages.get(semantic_page_analysis.page_id)
    if page is None or page.page_no != semantic_page_analysis.page_no:
        raise ModelATableFusionError("semantic page analysis target does not exist")
    semantic_issues = validate_model_a_page_analysis(
        base,
        semantic_page_analysis,
        target_page_id=page.id,
    )
    if semantic_issues:
        raise ModelATableFusionError("semantic page analysis is invalid for base EvidenceIR")
    topology = next(
        (
            item
            for item in promotion.topologies
            if item.artifact.source.target_page_id == page.id
        ),
        None,
    )
    if topology is None:
        raise ModelATableFusionError("promotion has no topology for semantic page")

    observation_by_id = {observation.id: observation for observation in page.observations}
    node_by_id = {node.id: node for node in semantic_page_analysis.nodes}
    owners: dict[str, list[ContentNode]] = defaultdict(list)
    for node in semantic_page_analysis.nodes:
        for observation_id in node.evidence_refs:
            owners[observation_id].append(node)
    order_index = {
        node_id: index for index, node_id in enumerate(semantic_page_analysis.reading_order)
    }
    replacement_by_node_id: dict[str, TableContentNode] = {}
    removed_node_ids: set[str] = set()
    semantic_sidecars: list[ModelATableSemanticRoleSidecar] = []
    shared_resolutions: list[ModelATableSharedTextResolution] = []
    prefix = f"{base.id}:content:{page.id}:"

    for table in topology.artifact.tables:
        table_owners: list[TextContentNode] = []
        role_bindings: list[ModelATableSemanticRoleBinding] = []
        for observation_id in table.ordered_observation_ids:
            matches = owners.get(observation_id, ())
            if len(matches) != 1 or not isinstance(matches[0], TextContentNode):
                raise ModelATableFusionError(
                    "table evidence must have exactly one semantic text owner"
                )
            owner = matches[0]
            if owner.evidence_refs != (observation_id,):
                raise ModelATableFusionError("table semantic owner must be a single-ref line")
            table_owners.append(owner)
            role_bindings.append(
                ModelATableSemanticRoleBinding(
                    observation_id=observation_id,
                    semantic_role=owner.role,
                )
            )
        positions = sorted(order_index[owner.id] for owner in table_owners)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ModelATableFusionError("table semantic owners are not one contiguous run")
        if any(owner.id in removed_node_ids for owner in table_owners):
            raise ModelATableFusionError("table semantic owner belongs to multiple tables")
        table_node = TableContentNode(
            id=prefix + table.table_grid_observation_id,
            evidence_refs=(
                table.table_grid_observation_id,
                *table.ordered_observation_ids,
            ),
            confidence=0.0,
            needs_review=True,
            rows=table.rows,
            columns=table.columns,
            cells=tuple(
                TableCell(
                    row=cell.row,
                    column=cell.column,
                    row_span=cell.row_span,
                    column_span=cell.column_span,
                    text=cell.text,
                    evidence_refs=(
                        table.table_grid_observation_id,
                        *cell.ordered_observation_ids,
                    ),
                )
                for cell in table.cells
            ),
        )
        first_owner = min(table_owners, key=lambda owner: order_index[owner.id])
        replacement_by_node_id[first_owner.id] = table_node
        removed_node_ids.update(owner.id for owner in table_owners)
        semantic_sidecars.append(
            ModelATableSemanticRoleSidecar(
                table_grid_observation_id=table.table_grid_observation_id,
                roles=tuple(role_bindings),
            )
        )
        shared_resolutions.extend(_shared_text_resolutions(table, observation_by_id))

    fused_nodes: list[ContentNode] = []
    fused_order: list[str] = []
    for node_id in semantic_page_analysis.reading_order:
        replacement = replacement_by_node_id.get(node_id)
        if replacement is not None:
            fused_nodes.append(replacement)
            fused_order.append(replacement.id)
        if node_id in removed_node_ids:
            continue
        node = node_by_id[node_id]
        fused_nodes.append(node)
        fused_order.append(node.id)
    fused = ModelAPageAnalysis(
        id=semantic_page_analysis.id,
        evidence_ir_id=promoted.id,
        evidence_ir_sha256=promoted_sha,
        page_id=semantic_page_analysis.page_id,
        page_no=semantic_page_analysis.page_no,
        nodes=tuple(fused_nodes),
        reading_order=tuple(fused_order),
    )
    fused_issues = validate_model_a_page_analysis(promoted, fused, target_page_id=page.id)
    if fused_issues:
        raise ModelATableFusionError("fused page analysis is invalid for promoted EvidenceIR")
    artifact = ModelATablePageFusionArtifact(
        source=ModelATablePageFusionSource(
            base_evidence_ir_id=base.id,
            base_evidence_ir_sha256=base_sha,
            promoted_evidence_ir_sha256=promoted_sha,
            promotion_artifact_sha256=promotion.artifact_sha256,
            semantic_page_analysis_sha256=contract_sha256(semantic_page_analysis),
            topology_artifact_sha256=topology.artifact_sha256,
            target_page_id=page.id,
            target_page_no=page.page_no,
        ),
        semantic_role_sidecars=tuple(semantic_sidecars),
        shared_text_resolutions=tuple(shared_resolutions),
        fused_page_analysis=fused,
        fused_page_analysis_sha256=contract_sha256(fused),
    )
    return artifact, topology


def _derive_table_document_assembly(
    promotion: BuiltModelATableEvidencePromotion,
    page_fusions: tuple[BuiltModelATablePageFusion, ...],
) -> tuple[
    ModelATableDocumentAssemblyArtifact,
    tuple[BuiltModelATablePageFusion, ...],
]:
    if type(promotion) is not BuiltModelATableEvidencePromotion:
        raise TypeError("promotion must be BuiltModelATableEvidencePromotion")
    if not isinstance(page_fusions, tuple):
        raise TypeError("page_fusions must be a tuple")
    if any(type(item) is not BuiltModelATablePageFusion for item in page_fusions):
        raise TypeError("page_fusions must contain BuiltModelATablePageFusion values")
    if len(page_fusions) > _MAX_PAGES:
        raise ModelATableFusionError("page fusions exceed the page resource budget")
    pages = tuple(
        sorted(promotion.promoted_evidence_ir.pages, key=lambda page: (page.page_no, page.id))
    )
    fusion_by_page: dict[str, BuiltModelATablePageFusion] = {}
    expected_promotion_binding = _promotion_binding(promotion)
    for fusion in page_fusions:
        source = fusion.artifact.source
        if (
            _promotion_binding(fusion.promotion) != expected_promotion_binding
            or source.base_evidence_ir_id
            != promotion.artifact.base_evidence_ir_id
            or source.base_evidence_ir_sha256
            != promotion.artifact.base_evidence_ir_sha256
            or source.promoted_evidence_ir_sha256
            != promotion.artifact.promoted_evidence_ir_sha256
            or source.promotion_artifact_sha256 != promotion.artifact_sha256
        ):
            raise ModelATableFusionError("page fusion does not bind the exact promotion")
        page_id = fusion.artifact.source.target_page_id
        if page_id in fusion_by_page:
            raise ModelATableFusionError("duplicate fused page")
        fusion_by_page[page_id] = fusion
    if set(fusion_by_page) != {page.id for page in pages}:
        raise ModelATableFusionError("page fusions must cover every promoted page exactly once")
    ordered = tuple(fusion_by_page[page.id] for page in pages)
    analyses = tuple(item.artifact.fused_page_analysis for item in ordered)
    node_ids = tuple(node.id for analysis in analyses for node in analysis.nodes)
    if len(node_ids) != len(set(node_ids)):
        raise ModelATableFusionError("fused pages contain duplicate content node ids")
    nodes = tuple(node for analysis in analyses for node in analysis.nodes)
    reading_order = tuple(
        node_id for analysis in analyses for node_id in analysis.reading_order
    )
    if not nodes:
        raise ModelATableFusionError("fused document cannot be empty")
    tables, cells, refs, text_chars = _content_resource_usage(nodes)
    sidecar_items, sidecar_refs, sidecar_text = _page_fusion_sidecar_usage(ordered)
    _require_aggregate_budget(
        label="table document assembly inputs",
        pages=len(pages),
        tables=tables,
        cells=cells,
        refs=refs + sidecar_refs,
        text_chars=text_chars + sidecar_text,
        sidecar_items=sidecar_items,
    )
    evidence = promotion.promoted_evidence_ir
    promoted_sha = promotion.artifact.promoted_evidence_ir_sha256
    content = ContentIR(
        id=f"{evidence.id}:content",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=promoted_sha,
        revision=1,
        nodes=nodes,
        reading_order=reading_order,
    )
    try:
        content.assert_evidence_integrity(evidence)
    except ValueError as exc:
        raise ModelATableFusionError(
            "assembled ContentIR fails promoted evidence integrity"
        ) from exc
    content_sha = contract_sha256(content)
    artifact = ModelATableDocumentAssemblyArtifact(
        source=ModelATableDocumentAssemblySource(
            promoted_evidence_ir_id=evidence.id,
            promoted_evidence_ir_sha256=promoted_sha,
            promotion_artifact_sha256=promotion.artifact_sha256,
            source_document_sha256=evidence.source_document_sha256,
            content_ir_id=content.id,
        ),
        pages=tuple(
            ModelATableDocumentAssemblyPage(
                page_id=page.id,
                page_no=page.page_no,
                fusion_artifact_sha256=fusion.artifact_sha256,
                fused_page_analysis_sha256=fusion.artifact.fused_page_analysis_sha256,
            )
            for page, fusion in zip(pages, ordered, strict=True)
        ),
        content_ir=content,
        content_ir_sha256=content_sha,
    )
    return artifact, ordered


def _shared_text_resolutions(
    table: ModelATableTopologyTable,
    observation_by_id: dict[str, EvidenceObservation],
) -> tuple[ModelATableSharedTextResolution, ...]:
    counts = Counter(
        observation_id
        for cell in table.cells
        for observation_id in cell.ordered_observation_ids
    )
    resolutions: list[ModelATableSharedTextResolution] = []
    for observation_id in table.ordered_observation_ids:
        if counts[observation_id] < 2:
            continue
        cells = tuple(
            cell for cell in table.cells if observation_id in cell.ordered_observation_ids
        )
        if len({cell.row for cell in cells}) != 1:
            raise ModelATableFusionError("shared OCR evidence is not horizontal")
        cells = tuple(sorted(cells, key=lambda cell: cell.column))
        text = _preferred_ocr_text(observation_by_id[observation_id])
        tokens = text.split()
        if len(tokens) != len(cells):
            raise ModelATableFusionError("shared OCR cell text cannot be split conservatively")
        fragments: list[ModelATableSharedCellFragment] = []
        for cell, token in zip(cells, tokens, strict=True):
            if token not in cell.text.splitlines():
                raise ModelATableFusionError("shared OCR token does not match topology cell text")
            fragments.append(
                ModelATableSharedCellFragment(
                    row=cell.row,
                    column=cell.column,
                    text=token,
                )
            )
        resolutions.append(
            ModelATableSharedTextResolution(
                table_grid_observation_id=table.table_grid_observation_id,
                observation_id=observation_id,
                source_text_sha256=_sha256(text.encode("utf-8")),
                cells=tuple(fragments),
            )
        )
    return tuple(resolutions)


def _preferred_ocr_text(observation: EvidenceObservation) -> str:
    selected = next(
        (candidate for candidate in observation.ocr_candidates if candidate.selected),
        None,
    )
    if selected is not None:
        return selected.text
    return max(observation.ocr_candidates, key=lambda candidate: candidate.confidence).text
