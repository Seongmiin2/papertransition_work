from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scan2hwpx.ir.models import BBox as LegacyBBox
from scan2hwpx.ir.models import (
    Block,
    Document,
    LayoutRegion,
)

from .models import (
    BBox,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ObservationKind,
    OcrCandidate,
)


def evidence_from_document(
    document: Document,
    page_image_refs: Mapping[int, str] | None = None,
    *,
    page_image_sha256: Mapping[int, str] | None = None,
) -> EvidenceIR:
    """Map legacy blocks without copying filesystem paths into lineage refs.

    A legacy block exposes only one payload ref even when it contains candidates
    from multiple OCR providers. The candidates therefore share one lineage
    source; their provider names are retained on the candidates instead of
    inventing provider-specific artifacts. Legacy layout regions and formula
    detections become separate observations backed by logical provider refs.

    EvidenceObservation v1 has no typed fields for formula expressions, region
    labels, or table cell spans. Those values are therefore not disguised as
    OCR text or otherwise invented here.
    """
    evidence_id = f"{document.id}:evidence"
    document_source_id = f"{evidence_id}:source:document"
    sources = [
        EvidenceSource(
            id=document_source_id,
            kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
            artifact_ref=f"artifact://{document.source_hash}/source",
            producer="scan2hwpx.ir.models.Document",
            sha256=document.source_hash,
        )
    ]
    pages: list[EvidencePage] = []

    for page in document.pages:
        page_id = f"{evidence_id}:page:{page.page_no}"
        page_source_id = f"{page_id}:source:image"
        page_artifact_ref = (
            page_image_refs[page.page_no]
            if page_image_refs is not None and page.page_no in page_image_refs
            else _page_artifact_ref(document.source_hash, page.page_no, page_image_sha256)
        )
        sources.append(
            EvidenceSource(
                id=page_source_id,
                kind=EvidenceSourceKind.PAGE_IMAGE,
                artifact_ref=page_artifact_ref,
                producer="scan2hwpx legacy page raster",
                sha256=(
                    page_image_sha256.get(page.page_no)
                    if page_image_sha256 is not None
                    else None
                ),
            )
        )

        observations: list[EvidenceObservation] = []
        for block_index, block in enumerate(page.blocks):
            observation_id = f"{page_id}:block:{block_index}:{block.id}"
            payload_source_id = f"{observation_id}:source:legacy-ocr-payload"
            sources.append(
                EvidenceSource(
                    id=payload_source_id,
                    kind=EvidenceSourceKind.OCR_PROVIDER,
                    artifact_ref=(
                        f"artifact://{document.source_hash}/legacy-ocr/"
                        f"pages/{page.page_no}/blocks/{block_index}"
                    ),
                    producer=(
                        "scan2hwpx legacy shared OCR payload; "
                        "per-candidate provider artifacts unavailable"
                    ),
                )
            )
            candidates: list[OcrCandidate] = []
            for candidate in _ocr_candidates(block):
                candidates.append(
                    OcrCandidate(
                        text=candidate["text"],
                        provider=candidate["engine"],
                        confidence=candidate["confidence"],
                        quality=candidate["quality"],
                        source_ref=payload_source_id,
                        selected=candidate["selected"],
                    )
                )
            observations.append(
                EvidenceObservation(
                    id=observation_id,
                    kind=ObservationKind.TEXT_LINE,
                    bbox=_evidence_bbox(block.bbox),
                    confidence=block.confidence,
                    source_refs=(document_source_id, page_source_id, payload_source_id),
                    ocr_candidates=tuple(candidates),
                )
            )

        for region_index, region in enumerate(page.regions):
            observation_id = f"{page_id}:region:{region_index}:{region.id}"
            provider_source_id = f"{observation_id}:source:legacy-layout-payload"
            sources.append(
                EvidenceSource(
                    id=provider_source_id,
                    kind=EvidenceSourceKind.LAYOUT_PROVIDER,
                    artifact_ref=(
                        f"artifact://{document.source_hash}/legacy-layout/"
                        f"pages/{page.page_no}/regions/{region_index}"
                    ),
                    producer="scan2hwpx legacy layout region",
                )
            )
            observations.append(
                EvidenceObservation(
                    id=observation_id,
                    kind=_region_observation_kind(region),
                    bbox=_evidence_bbox(region.bbox),
                    confidence=region.confidence,
                    source_refs=(document_source_id, page_source_id, provider_source_id),
                )
            )

        for formula_index, formula in enumerate(page.formulas):
            observation_id = f"{page_id}:formula:{formula_index}:{formula.id}"
            provider_source_id = f"{observation_id}:source:legacy-formula-payload"
            sources.append(
                EvidenceSource(
                    id=provider_source_id,
                    kind=EvidenceSourceKind.FORMULA_PROVIDER,
                    artifact_ref=(
                        f"artifact://{document.source_hash}/legacy-formula/"
                        f"pages/{page.page_no}/formulas/{formula_index}"
                    ),
                    producer=(
                        formula.source_provider.strip()
                        or "scan2hwpx legacy formula provider unavailable"
                    ),
                )
            )
            observations.append(
                EvidenceObservation(
                    id=observation_id,
                    kind=ObservationKind.FORMULA,
                    bbox=_evidence_bbox(formula.bbox),
                    confidence=formula.confidence,
                    source_refs=(document_source_id, page_source_id, provider_source_id),
                )
            )
        pages.append(
            EvidencePage(
                id=page_id,
                page_no=page.page_no,
                width=page.width,
                height=page.height,
                rotation=page.rotation,
                image_source_ref=page_source_id,
                observations=tuple(observations),
            )
        )

    return EvidenceIR(
        id=evidence_id,
        source_document_sha256=document.source_hash,
        sources=tuple(sources),
        pages=tuple(pages),
    )


def _page_artifact_ref(
    document_sha256: str,
    page_no: int,
    page_image_sha256: Mapping[int, str] | None,
) -> str:
    if page_image_sha256 is not None and page_no in page_image_sha256:
        return f"artifact://{page_image_sha256[page_no]}/page-image"
    return f"artifact://{document_sha256}/pages/{page_no}/image"


def _region_observation_kind(region: LayoutRegion) -> ObservationKind:
    raw_labels = {region.label.strip().lower(), region.source_label.strip().lower()}
    classified_labels = raw_labels & {"table", "image"}
    if classified_labels == {"table"}:
        return ObservationKind.TABLE_GRID
    if classified_labels == {"image"}:
        return ObservationKind.IMAGE
    return ObservationKind.REGION


def _evidence_bbox(bbox: LegacyBBox) -> BBox:
    return BBox(pixel=bbox.pixel, normalized=bbox.normalized)


def _ocr_candidates(block: Block) -> list[dict[str, Any]]:
    raw = block.style.get("ocr_candidates")
    if raw is None or raw == []:
        return [_block_candidate(block)]
    if not isinstance(raw, list):
        raise TypeError("block.style['ocr_candidates'] must be a list")

    candidates: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("each OCR candidate must be an object")
        confidence = item.get("confidence")
        quality = item.get("quality")
        selected = item.get("selected", False)
        if not isinstance(selected, bool):
            raise TypeError("OCR candidate selected must be a boolean")
        candidates.append(
            {
                "engine": str(item.get("engine") or block.source_provider),
                "text": str(item.get("text") or ""),
                "confidence": float(block.confidence if confidence is None else confidence),
                "quality": None if quality is None else float(quality),
                "selected": selected,
            }
        )
    if not any(candidate["selected"] for candidate in candidates):
        for candidate in candidates:
            if candidate["text"] == block.text:
                candidate["selected"] = True
                break
        else:
            candidates.append(_block_candidate(block))
    return candidates


def _block_candidate(block: Block) -> dict[str, Any]:
    return {
        "engine": block.source_provider,
        "text": block.text,
        "confidence": block.confidence,
        "quality": None,
        "selected": True,
    }
