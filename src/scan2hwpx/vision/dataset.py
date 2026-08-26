from __future__ import annotations

import json
from pathlib import Path

from scan2hwpx.ir.models import Document


def write_formula_manifest(document: Document, output_path: Path) -> Path:
    """Write JSONL suitable for review now and supervised fine-tuning later."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for page in document.pages:
        for formula in page.formulas:
            image = Path(formula.source_image)
            try:
                image_value = image.resolve().relative_to(output_path.parent.resolve()).as_posix()
            except ValueError:
                image_value = str(image.resolve())
            lines.append(
                json.dumps(
                    {
                        "document_id": document.id,
                        "page": page.page_no,
                        "formula_id": formula.id,
                        "image": image_value,
                        "bbox": list(formula.bbox.pixel),
                        "prediction": formula.expression,
                        "target": None,
                        "format": formula.format.value,
                        "confidence": formula.confidence,
                        "status": formula.status.value,
                    },
                    ensure_ascii=False,
                )
            )
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path
