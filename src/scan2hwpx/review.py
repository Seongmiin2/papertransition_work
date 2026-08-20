from __future__ import annotations

import html
import json
from pathlib import Path
from typing import cast

from scan2hwpx.ir.models import Document


def write_review(document: Document, output_dir: Path) -> tuple[Path, Path]:
    blocks = [block for page in document.pages for block in page.blocks]
    low = [block for block in blocks if block.confidence < 0.75]
    page_for = {block.id: page.page_no for page in document.pages for block in page.blocks}
    items = [
        {
            "page": page_for[block.id],
            "block_id": block.id,
            "text": block.text,
            "confidence": block.confidence,
            "bbox": list(block.bbox.pixel),
            "crop_image": f"review_crops/{block.id}.png",
            "reason": "low_ocr_confidence",
        }
        for block in low
    ]
    payload = {
        "summary": {
            "total_pages": len(document.pages),
            "total_blocks": len(blocks),
            "review_required_blocks": len(low),
            "average_confidence": sum(block.confidence for block in blocks) / max(1, len(blocks)),
        },
        "items": items,
    }
    json_path = output_dir / "review.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{item['page']}</td><td>{html.escape(str(item['block_id']))}</td>"
        f"<td>{cast(float, item['confidence']):.3f}</td><td>{html.escape(str(item['text']))}</td></tr>"
        for item in items
    )
    page_cards = "".join(
        f"<section><h2>{page.page_no}페이지</h2>"
        f"<img src='debug/original/page-{page.page_no}.png'><img src='debug/preprocessed/page-{page.page_no}.png'>"
        f"<img src='debug/ocr_overlay/page-{page.page_no}.png'></section>"
        for page in document.pages
    )
    html_path = output_dir / "review.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Scan2HWPX 검토</title>"
        "<style>body{font-family:Malgun Gothic;margin:24px}img{width:30%;margin:1%;vertical-align:top}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px}</style>"
        f"<h1>Scan2HWPX 검토</h1><pre>{html.escape(json.dumps(payload['summary'], ensure_ascii=False, indent=2))}</pre>"
        f"{page_cards}<h2>저신뢰 블록</h2><table><tr><th>페이지</th><th>ID</th><th>신뢰도</th><th>텍스트</th></tr>{rows}</table>",
        encoding="utf-8",
    )
    return json_path, html_path
