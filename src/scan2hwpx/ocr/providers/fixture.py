from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scan2hwpx.ir.models import (
    AnnotationState,
    BBox,
    Block,
    BlockKind,
    Column,
    Document,
    Page,
    PageQuality,
    QAReport,
)


class FixtureOcrProvider:
    name = "fixture"

    def convert(self, path: Path) -> Document:
        raw_bytes = path.read_bytes()
        payload: dict[str, Any] = json.loads(raw_bytes)
        pages: list[Page] = []
        for page_data in payload["pages"]:
            width, height = page_data["width"], page_data["height"]
            blocks: list[Block] = []
            for index, item in enumerate(page_data["blocks"]):
                x0, y0, x1, y1 = item["bbox"]
                bbox = BBox(
                    pixel=(x0, y0, x1, y1),
                    normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
                )
                blocks.append(
                    Block(
                        id=item["id"],
                        kind=BlockKind(item["kind"]),
                        bbox=bbox,
                        reading_order=index,
                        text=item["text"],
                        confidence=item["confidence"],
                        source_provider=self.name,
                        source_payload_ref=f"{path.name}#/pages/{page_data['page_no']}/blocks/{index}",
                        annotation_state=AnnotationState(item.get("annotation_state", "printed")),
                    )
                )
            full = BBox(pixel=(0, 0, width, height), normalized=(0, 0, 1, 1))
            pages.append(
                Page(
                    page_no=page_data["page_no"],
                    width=width,
                    height=height,
                    rotation=page_data.get("rotation", 0),
                    columns=[Column(index=0, bbox=full)],
                    blocks=blocks,
                    quality=PageQuality(score=1.0),
                )
            )
        low = [block.id for page in pages for block in page.blocks if block.confidence < 0.8]
        return Document(
            id=payload["document_id"],
            source_hash=hashlib.sha256(raw_bytes).hexdigest(),
            page_size=(pages[0].width, pages[0].height),
            metadata={"fixture": path.name},
            pages=pages,
            qa=QAReport(low_confidence_blocks=low),
        )
