from __future__ import annotations

import json
from pathlib import Path

from scan2hwpx.reference.verified import import_verified_ocr


def test_import_verified_ocr_copies_crop_and_writes_one_split(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fake-png")
    manifest = tmp_path / "verified.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "status": "verified",
                "target": "컴퓨터용 사인펜",
                "image": str(crop),
                "document_id": "pdf-1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = import_verified_ocr(manifest, tmp_path / "dataset")

    assert report["verified_samples"] == 1
    assert report["failures"] == []
    assert sum(report["splits"].values()) == 1
    labels = "".join(
        (tmp_path / "dataset" / f"{split}.txt").read_text(encoding="utf-8")
        for split in ("train", "validation", "test")
    )
    assert "컴퓨터용 사인펜" in labels
