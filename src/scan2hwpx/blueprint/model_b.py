from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scan2hwpx.ir.models import BBox, Document, LayoutRegion, RegionPlacement

PLACEMENT_SOURCE = "rule_based_v0"
MIN_CONFIDENCE = 0.45
CV_TABLE_MIN_CONFIDENCE = 0.7


def assign_region_placements(document: Document, layout_seed_path: Path) -> None:
    """Populate each page's `regions` from a layout_seed.json file.

    Rule-based stand-in for a future trained "Model B": reproduces exactly
    the per-annotation bucketing that hwpx/semantic.py's _image_regions /
    _table_regions / _passage_regions used to do independently, so callers
    that switch to reading `Page.regions` see the same placements. Leaves
    `document.pages` untouched (silently returns) if the seed's page count
    does not match, so a caller can always fall back to today's direct
    layout_seed.json parsing.
    """
    payload = json.loads(layout_seed_path.read_text(encoding="utf-8"))
    seed_pages = payload.get("pages", [])
    if len(seed_pages) != len(document.pages):
        return
    for ir_page, seed_page in zip(document.pages, seed_pages, strict=True):
        ir_page.regions = _regions_for_page(seed_page)


def _regions_for_page(seed_page: dict[str, Any]) -> list[LayoutRegion]:
    width = float(seed_page["width"])
    height = float(seed_page["height"])
    regions: list[LayoutRegion] = []
    for index, annotation in enumerate(seed_page.get("annotations", [])):
        placement = _placement_for_annotation(annotation)
        if placement is None:
            continue
        confidence = float(annotation.get("confidence", 0))
        x0, y0, x1, y1 = (float(value) for value in annotation["bbox"])
        regions.append(
            LayoutRegion(
                id=f"{seed_page.get('page_no', 0)}-{index}",
                label=str(annotation.get("label", "")),
                bbox=BBox(
                    pixel=(x0, y0, x1, y1),
                    normalized=(
                        _clamp01(x0 / width),
                        _clamp01(y0 / height),
                        _clamp01(x1 / width),
                        _clamp01(y1 / height),
                    ),
                ),
                confidence=confidence,
                source=str(annotation.get("source", "")),
                source_label=str(annotation.get("label", "")),
                confirmed=True,
                placement=placement,
                placement_confidence=confidence,
                placement_source=PLACEMENT_SOURCE,
            )
        )
    return regions


def _placement_for_annotation(annotation: dict[str, Any]) -> RegionPlacement | None:
    label = annotation.get("label")
    confidence = float(annotation.get("confidence", 0))
    if confidence < MIN_CONFIDENCE:
        return None
    if label == "image":
        return RegionPlacement.IMAGE
    if label == "table":
        if annotation.get("source") == "opencv_ruled_region" and confidence < CV_TABLE_MIN_CONFIDENCE:
            # Same fallback _passage_regions() already applied: an unconfirmed
            # CV-detected grid is never rendered as a table, only ever
            # considered as a passage-box candidate.
            return RegionPlacement.PASSAGE_BOX
        return RegionPlacement.TABLE
    if label == "passage_box":
        return RegionPlacement.PASSAGE_BOX
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
