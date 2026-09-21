from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scan2hwpx.contracts import EvidenceSourceKind, ObservationKind
from scan2hwpx.contracts.legacy import evidence_from_document
from scan2hwpx.ir.models import BBox as LegacyBBox
from scan2hwpx.ir.models import (
    Formula,
    LayoutRegion,
    RegionPlacement,
    TableCellSpan,
)
from scan2hwpx.ocr.providers.fixture import FixtureOcrProvider

FIXTURE = Path("tests/fixtures/ocr_page_1.json")


def _bbox(x0: float, y0: float, x1: float, y1: float) -> LegacyBBox:
    return LegacyBBox(
        pixel=(x0, y0, x1, y1),
        normalized=(x0 / 100, y0 / 100, x1 / 100, y1 / 100),
    )


def _region(
    identifier: str,
    bbox: LegacyBBox,
    label: str,
    source_label: str,
    placement: RegionPlacement | None,
) -> LayoutRegion:
    return LayoutRegion(
        id=identifier,
        label=label,
        bbox=bbox,
        confidence=0.8,
        source="C:/private/layout-seed.json",
        source_label=source_label,
        confirmed=placement is not None,
        placement=placement,
        placement_confidence=0.8 if placement is not None else None,
        placement_source="rule_based_v0" if placement is not None else None,
    )


def test_regions_and_formulas_become_distinct_grounded_observations() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    page = document.pages[0]
    table_region = _region(
        "table",
        _bbox(10, 10, 40, 30),
        "table",
        "provider-table",
        RegionPlacement.IMAGE,
    )
    table_region.cell_spans = [
        TableCellSpan(row=0, column=0, col_span=2, confidence=0.75)
    ]
    page.regions = [
        table_region,
        _region(
            "image",
            _bbox(45, 10, 75, 35),
            "provider-image",
            "image",
            RegionPlacement.TABLE,
        ),
        _region(
            "placement-is-not-evidence",
            _bbox(10, 40, 75, 60),
            "unpreserved-region-label",
            "provider-other",
            RegionPlacement.TABLE,
        ),
        _region(
            "conflicting-labels",
            _bbox(10, 82, 75, 92),
            "table",
            "image",
            RegionPlacement.TABLE,
        ),
    ]
    formula = Formula(
        id="formula-1",
        bbox=_bbox(20, 65, 60, 80),
        source_image="C:/private/formula-crop.png",
        expression="FORMULA_EXPRESSION_NOT_IN_V1_CONTRACT",
        confidence=0.73,
        source_provider="formula-provider-v1",
    )
    page.formulas = [formula]

    evidence = evidence_from_document(document)
    observations = evidence.pages[0].observations
    text_observations = [
        item for item in observations if item.kind == ObservationKind.TEXT_LINE
    ]
    non_text_observations = [
        item for item in observations if item.kind != ObservationKind.TEXT_LINE
    ]

    assert len(text_observations) == len(page.blocks)
    assert len(observations) == len(page.blocks) + len(page.regions) + len(page.formulas)
    assert len({item.id for item in text_observations}) == len(page.blocks)
    assert [item.kind for item in non_text_observations] == [
        ObservationKind.TABLE_GRID,
        ObservationKind.IMAGE,
        ObservationKind.REGION,
        ObservationKind.REGION,
        ObservationKind.FORMULA,
    ]
    assert [item.bbox.pixel for item in non_text_observations] == [
        region.bbox.pixel for region in page.regions
    ] + [formula.bbox.pixel]
    assert [item.confidence for item in non_text_observations] == [
        region.confidence for region in page.regions
    ] + [formula.confidence]
    assert all(not item.ocr_candidates for item in non_text_observations)
    assert len({item.id for item in observations}) == len(observations)

    source_by_id = {source.id: source for source in evidence.sources}
    for index, observation in enumerate(non_text_observations):
        resolved_sources = [source_by_id[source_ref] for source_ref in observation.source_refs]
        provider_sources = [
            source
            for source in resolved_sources
            if source.kind
            in {EvidenceSourceKind.LAYOUT_PROVIDER, EvidenceSourceKind.FORMULA_PROVIDER}
        ]
        assert len(provider_sources) == 1
        expected_kind = (
            EvidenceSourceKind.LAYOUT_PROVIDER
            if index < len(page.regions)
            else EvidenceSourceKind.FORMULA_PROVIDER
        )
        assert provider_sources[0].kind == expected_kind
        assert provider_sources[0].artifact_ref.startswith(f"artifact://{document.source_hash}/")

    encoded = evidence.model_dump_json()
    assert "C:/private/layout-seed.json" not in encoded
    assert formula.source_image not in encoded
    assert formula.expression not in encoded
    assert "unpreserved-region-label" not in encoded
    assert "cell_spans" not in encoded


def test_non_text_evidence_rejects_degenerate_legacy_bbox() -> None:
    document = FixtureOcrProvider().convert(FIXTURE)
    document.pages[0].regions = [
        _region("degenerate", _bbox(20, 10, 20, 30), "table", "table", RegionPlacement.IMAGE)
    ]

    with pytest.raises(ValidationError, match="pixel bbox must be non-negative and ordered"):
        evidence_from_document(document)
