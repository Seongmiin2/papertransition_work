from __future__ import annotations

import hashlib
import io
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

_SPACE = re.compile(r"[ \t\u00a0]+")
_WORD = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_DEFAULT_FONTS = (
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/batang.ttc"),
    Path("C:/Windows/Fonts/gulim.ttc"),
)


@dataclass(frozen=True)
class SyntheticLine:
    document_id: str
    text: str
    index: int


def build_ocr_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    variants: int = 1,
    limit: int | None = 5000,
    fonts: list[Path] | None = None,
    seed: int = 20260823,
    base_model_dir: Path | None = None,
) -> dict[str, Any]:
    if variants < 1:
        raise ValueError("variants must be at least 1")
    available_fonts = _resolve_fonts(fonts)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    all_lines = load_reference_lines(manifest_path)
    word_counts: Counter[str] = Counter()
    for line in all_lines:
        word_counts.update(_WORD.findall(line.text))
    lines = list(all_lines)
    rng = random.Random(seed)
    rng.shuffle(lines)
    if limit is not None:
        lines = lines[:limit]

    split_rows: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    split_documents: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    dataset_characters: set[str] = set()
    for line in lines:
        split = split_for_document(line.document_id)
        split_documents[split].add(line.document_id)
        dataset_characters.update(line.text)
        for variant in range(variants):
            token = hashlib.sha256(
                f"{seed}:{line.document_id}:{line.index}:{variant}:{line.text}".encode()
            ).hexdigest()[:20]
            relative_path = Path("images") / f"{token}.png"
            line_seed = int(token[:16], 16)
            image = render_text_line(line.text, available_fonts, variant, line_seed)
            image.save(output_dir / relative_path, format="PNG", optimize=True)
            split_rows[split].append(f"{relative_path.as_posix()}\t{line.text}")

    for split, rows in split_rows.items():
        (output_dir / f"{split}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
    lexicon = {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path.resolve()),
        "words": [
            {"text": word, "count": count}
            for word, count in word_counts.most_common(50_000)
        ],
    }
    (output_dir / "lexicon.json").write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dictionary_report = write_training_dictionary(
        dataset_characters,
        output_dir / "korean_exam_dict.txt",
        base_model_dir=base_model_dir,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path.resolve()),
        "seed": seed,
        "fonts": [str(path) for path in available_fonts],
        "unique_text_lines": len(lines),
        "variants": variants,
        "images": sum(len(rows) for rows in split_rows.values()),
        "splits": {
            split: {"images": len(split_rows[split]), "documents": len(split_documents[split])}
            for split in split_rows
        },
        "dictionary": dictionary_report,
        "important_limit": (
            "Synthetic images are domain adaptation data, not a measured replacement for real "
            "PDF crops with human-verified labels."
        ),
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def write_training_dictionary(
    dataset_characters: set[str],
    output_path: Path,
    *,
    base_model_dir: Path | None = None,
) -> dict[str, Any]:
    model_dir = base_model_dir or (
        Path.home() / ".paddlex" / "official_models" / "korean_PP-OCRv5_mobile_rec"
    )
    inference_config = model_dir / "inference.yml"
    base_characters: list[str] = []
    if inference_config.is_file():
        payload = yaml.safe_load(inference_config.read_text(encoding="utf-8"))
        postprocess = payload.get("PostProcess", {}) if isinstance(payload, dict) else {}
        raw_characters = postprocess.get("character_dict", [])
        if isinstance(raw_characters, list):
            base_characters = [str(character) for character in raw_characters]
    usable = {character for character in dataset_characters if character not in {"", "\n", "\r", "\t", " "}}
    existing = set(base_characters)
    missing = sorted(usable - existing)
    combined = list(dict.fromkeys([*base_characters, *missing]))
    output_path.write_text("\n".join(combined) + ("\n" if combined else ""), encoding="utf-8")
    return {
        "path": output_path.name,
        "base_model_dictionary_found": bool(base_characters),
        "base_characters": len(base_characters),
        "dataset_characters": len(usable),
        "appended_characters": len(missing),
        "total_characters": len(combined),
        "appended": "".join(missing),
    }


def load_reference_lines(manifest_path: Path) -> list[SyntheticLine]:
    root = manifest_path.resolve().parent
    seen_text: set[str] = set()
    result: list[SyntheticLine] = []
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    for row in rows:
        if not row.strip():
            continue
        item = json.loads(row)
        if item.get("duplicate_of"):
            continue
        document_id = str(item["sha256"])
        document_path = root / str(item["document_json"])
        document = json.loads(document_path.read_text(encoding="utf-8"))
        line_index = 0
        for paragraph in document.get("paragraphs", []):
            text = paragraph.get("text", "") if isinstance(paragraph, dict) else ""
            for candidate in split_training_text(str(text)):
                if candidate in seen_text:
                    continue
                seen_text.add(candidate)
                result.append(
                    SyntheticLine(document_id=document_id, text=candidate, index=line_index)
                )
                line_index += 1
    return result


def split_training_text(text: str, *, maximum: int = 55) -> list[str]:
    result: list[str] = []
    for source_line in text.replace("\r", "\n").split("\n"):
        remaining = _SPACE.sub(" ", source_line).strip()
        while remaining:
            if len(remaining) <= maximum:
                candidate, remaining = remaining, ""
            else:
                boundary = remaining.rfind(" ", 8, maximum + 1)
                if boundary < 8:
                    boundary = maximum
                candidate, remaining = remaining[:boundary], remaining[boundary:].lstrip()
            candidate = candidate.strip()
            if len(candidate) >= 2 and any(character.isalnum() for character in candidate):
                result.append(candidate)
    return result


def split_for_document(document_id: str) -> str:
    bucket = int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def render_text_line(
    text: str,
    fonts: list[Path],
    variant: int,
    seed: int,
) -> Image.Image:
    rng = random.Random(seed)
    font_path = fonts[rng.randrange(len(fonts))]
    font_size = rng.randint(28, 46)
    font = ImageFont.truetype(str(font_path), font_size)
    probe = Image.new("L", (8, 8), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font, stroke_width=0)
    width = max(8, int(bbox[2] - bbox[0]))
    height = max(8, int(bbox[3] - bbox[1]))
    horizontal = rng.randint(10, 24)
    vertical = rng.randint(6, 13)
    image = Image.new("L", (width + horizontal * 2, height + vertical * 2), rng.randint(247, 255))
    draw = ImageDraw.Draw(image)
    ink = rng.randint(0, 35)
    draw.text((horizontal - bbox[0], vertical - bbox[1]), text, fill=ink, font=font)
    if variant > 0:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.15))
        image = image.rotate(rng.uniform(-1.1, 1.1), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        if rng.random() < 0.7:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.65)))
        if rng.random() < 0.5:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=rng.randint(72, 94))
            buffer.seek(0)
            image = Image.open(buffer).convert("L")
    return image


def _resolve_fonts(fonts: list[Path] | None) -> list[Path]:
    candidates = fonts if fonts else list(_DEFAULT_FONTS)
    available = [path.resolve() for path in candidates if path.is_file()]
    if not available:
        raise FileNotFoundError(
            "No Korean-capable font was found. Pass one or more --font paths to build-ocr-dataset."
        )
    return available
