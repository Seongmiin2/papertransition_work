from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import pymupdf
from lxml import etree  # type: ignore[import-untyped]
from PIL import Image

from scan2hwpx.reference.dataset import normalize_exam_name
from scan2hwpx.reference.hwp5 import read_hwp_reference

_OFFICE_EXTENSIONS = {".hwp", ".hwpx", ".docx"}
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_SUPPORTED_EXTENSIONS = {".pdf", *_OFFICE_EXTENSIONS, *_IMAGE_EXTENSIONS}
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_ANSWER_KEY = re.compile(r"(?:답지|정답|해설)", re.IGNORECASE)


def prepare_training_corpus(
    inputs: list[Path],
    output_dir: Path,
    *,
    pair_config: Path | None = None,
    project_root: Path | None = None,
    copy_files: bool = True,
) -> dict[str, Any]:
    """Build an immutable, leakage-safe local corpus from exam source files.

    Filename matches are retained as candidates only. A pair is promoted to gold solely through
    an explicit pair declaration and a measurable content check.
    """
    root = (project_root or Path.cwd()).resolve()
    destination = output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = _collect_files(inputs)
    configured_pairs = _load_pair_config(pair_config, root)
    forced_roles = _configured_roles(configured_pairs)
    configured_splits = {
        Path(str(pair["source_pdf"])).resolve(): str(pair.get("split", "test"))
        for pair in configured_pairs
    }

    items: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for source in files:
        try:
            digest = _sha256(source)
            role = forced_roles.get(source.resolve(), classify_role(source))
            if role == "source_pdf" and any(
                part.casefold() == "test_data" for part in source.parts
            ):
                role = "regression_source_pdf"
            split = configured_splits.get(source.resolve())
            if split is None and role == "regression_source_pdf":
                split = "test"
            if split is None:
                split = split_for_document(digest, eligible=role == "source_pdf")
            relative_copy = Path("raw") / role / f"{digest[:12]}-{_safe_name(source.name)}"
            canonical = destination / relative_copy
            duplicate_of = seen_hashes.get(digest)
            if copy_files and duplicate_of is None:
                canonical.parent.mkdir(parents=True, exist_ok=True)
                if not canonical.exists() or _sha256(canonical) != digest:
                    shutil.copy2(source, canonical)
            if duplicate_of is None:
                seen_hashes[digest] = relative_copy.as_posix()
            metadata = inspect_training_file(source)
            items.append(
                {
                    "id": digest[:16],
                    "sha256": digest,
                    "source_path": str(source.resolve()),
                    "canonical_path": relative_copy.as_posix() if duplicate_of is None else None,
                    "name": source.name,
                    "extension": source.suffix.lower(),
                    "bytes": source.stat().st_size,
                    "role": role,
                    "split": split,
                    "duplicate_of": duplicate_of,
                    "metadata": metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001 - corpus audit must continue
            failures.append(
                {"path": str(source.resolve()), "error": f"{type(exc).__name__}: {exc}"}
            )

    declared_pairs: list[dict[str, Any]] = []
    for pair in configured_pairs:
        try:
            declared_pairs.append(evaluate_declared_pair(pair))
        except Exception as exc:  # noqa: BLE001 - isolate one invalid declared pair
            pair_id = str(pair.get("id", "unknown"))
            error = f"{type(exc).__name__}: {exc}"
            declared_pairs.append(
                {
                    "id": pair_id,
                    "kind": "declared_scan_transcript_editable",
                    "source_pdf": str(pair.get("source_pdf", "")),
                    "transcript_pdf": str(pair.get("transcript_pdf", "")),
                    "editable_target": str(pair.get("editable_target", "")),
                    "split": str(pair.get("split", "test")),
                    "status": "invalid",
                    "use_for": list(pair.get("use_for", [])),
                    "geometry_target": bool(pair.get("geometry_target", False)),
                    "checks": {
                        "visual_relationship_declared": bool(
                            pair.get("visually_verified", False)
                        ),
                        "transcript_editable_content_verified": False,
                        "held_out_from_training": str(pair.get("split", "test")) == "test",
                    },
                    "error": error,
                    "notes": list(pair.get("notes", [])),
                }
            )
            failures.append({"path": f"configured_pair:{pair_id}", "error": error})
    candidate_pairs = _find_filename_pairs(items)
    _write_jsonl(destination / "manifest.jsonl", items)
    _write_jsonl(destination / "pairs.jsonl", [*declared_pairs, *candidate_pairs])
    split_manifest = _write_split_manifests(destination, items)
    role_counts = Counter(str(item["role"]) for item in items)
    split_counts = Counter(str(item["split"]) for item in items if item["split"])
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "source_roots": [str(path.resolve()) for path in inputs],
        "output_directory": str(destination),
        "summary": {
            "files": len(items),
            "unique_files": sum(item["duplicate_of"] is None for item in items),
            "duplicates": sum(item["duplicate_of"] is not None for item in items),
            "roles": dict(sorted(role_counts.items())),
            "source_splits": dict(sorted(split_counts.items())),
            "declared_pairs": len(declared_pairs),
            "gold_pairs": sum(pair["status"] == "gold" for pair in declared_pairs),
            "filename_pair_candidates": len(candidate_pairs),
            "rejected_filename_pairs": sum(
                pair["status"] == "rejected" for pair in candidate_pairs
            ),
            "failures": len(failures),
        },
        "guardrails": {
            "originals_modified": False,
            "split_unit": "document_sha256",
            "duplicates_share_one_canonical_copy": True,
            "filename_match_is_ground_truth": False,
            "gold_test_is_excluded_from_training": True,
            "auto_ocr_labels_are_gold": False,
        },
        "split_manifests": split_manifest,
        "declared_pairs": declared_pairs,
        "candidate_pairs": candidate_pairs,
        "failures": failures,
    }
    (destination / "corpus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "README.md").write_text(_readme(report), encoding="utf-8")
    return report


def classify_role(path: Path) -> str:
    suffix = path.suffix.lower()
    if _ANSWER_KEY.search(path.stem):
        return "answer_key"
    if suffix == ".pdf":
        return "source_pdf"
    if suffix in _OFFICE_EXTENSIONS:
        return "editable_reference"
    if suffix in _IMAGE_EXTENSIONS:
        return "image_reference"
    return "unsupported"


def split_for_document(digest: str, *, eligible: bool = True) -> str | None:
    if not eligible:
        return None
    bucket = int(hashlib.sha256(f"exam2hwpx-corpus-v2:{digest}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def inspect_training_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _inspect_pdf(path)
    if suffix == ".hwp":
        document = read_hwp_reference(path)
        return {
            "kind": "hwp5",
            "characters": document.character_count,
            "paragraphs": len(document.paragraphs),
            "sections": document.section_count,
            "tables": document.table_count,
            "pictures": document.picture_count,
            "equations": document.equation_count,
        }
    if suffix == ".docx":
        return _inspect_docx(path)
    if suffix == ".hwpx":
        return _inspect_hwpx(path)
    if suffix in _IMAGE_EXTENSIONS:
        with Image.open(path) as image:
            return {
                "kind": "image",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
    return {"kind": "unsupported"}


def evaluate_declared_pair(pair: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(pair["source_pdf"]))
    transcript = Path(str(pair["transcript_pdf"]))
    target = Path(str(pair["editable_target"]))
    source_info = _inspect_pdf(source)
    transcript_info = _inspect_pdf(transcript)
    transcript_text = _normalize_text(_pdf_text(transcript))
    target_text = _normalize_text(_editable_text(target))
    similarity = SequenceMatcher(None, transcript_text, target_text, autojunk=False).ratio()
    declared_verified = bool(pair.get("visually_verified", False))
    content_verified = similarity >= 0.995 and len(transcript_text) >= 500
    status = "gold" if declared_verified and content_verified else "review_required"
    return {
        "id": str(pair["id"]),
        "kind": "declared_scan_transcript_editable",
        "source_pdf": str(source.resolve()),
        "transcript_pdf": str(transcript.resolve()),
        "editable_target": str(target.resolve()),
        "split": str(pair.get("split", "test")),
        "status": status,
        "use_for": list(pair.get("use_for", [])),
        "geometry_target": bool(pair.get("geometry_target", False)),
        "source_pages": source_info["pages"],
        "transcript_pages": transcript_info["pages"],
        "editable_pages_observed": pair.get("editable_pages_observed"),
        "transcript_characters": len(transcript_text),
        "editable_characters": len(target_text),
        "content_similarity": round(similarity, 6),
        "checks": {
            "visual_relationship_declared": declared_verified,
            "transcript_editable_content_verified": content_verified,
            "held_out_from_training": str(pair.get("split", "test")) == "test",
        },
        "notes": list(pair.get("notes", [])),
    }


def _collect_files(inputs: list[Path]) -> list[Path]:
    result: dict[Path, None] = {}
    for value in inputs:
        path = value.resolve()
        candidates = [path] if path.is_file() else path.rglob("*") if path.is_dir() else []
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_EXTENSIONS:
                result[candidate.resolve()] = None
    return sorted(result, key=lambda item: str(item).casefold())


def _load_pair_config(path: Path | None, root: Path) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    result: list[dict[str, Any]] = []
    for raw in raw_pairs:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        for key in ("source_pdf", "transcript_pdf", "editable_target"):
            resolved = (root / str(item[key])).resolve()
            item[key] = str(resolved)
        result.append(item)
    return result


def _configured_roles(pairs: list[dict[str, Any]]) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for pair in pairs:
        result[Path(str(pair["source_pdf"])).resolve()] = "gold_source_pdf"
        result[Path(str(pair["transcript_pdf"])).resolve()] = "gold_transcript_pdf"
        result[Path(str(pair["editable_target"])).resolve()] = "gold_editable_target"
    return result


def _find_filename_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pdfs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    targets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["role"] == "source_pdf":
            pdfs[normalize_exam_name(Path(str(item["name"])).stem)].append(item)
        elif item["role"] == "editable_reference":
            targets[normalize_exam_name(Path(str(item["name"])).stem)].append(item)
    result: list[dict[str, Any]] = []
    for key in sorted(pdfs.keys() & targets.keys()):
        for pdf in pdfs[key]:
            for target in targets[key]:
                assessment = _assess_filename_pair(pdf, target)
                result.append(
                    {
                        "id": f"candidate-{pdf['id']}-{target['id']}",
                        "kind": "filename_match_only",
                        "source_pdf": pdf["source_path"],
                        "editable_target": target["source_path"],
                        **assessment,
                    }
                )
    return result


def _assess_filename_pair(pdf: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    pages = max(1, int(pdf["metadata"].get("pages", 1)))
    characters = int(target["metadata"].get("characters", 0))
    pictures = int(target["metadata"].get("pictures", target["metadata"].get("images", 0)))
    sparse = characters < pages * 180 and pictures < pages
    return {
        "status": "rejected" if sparse else "review_required",
        "reason": (
            "editable target is too sparse for the source page count; likely an answer template"
            if sparse
            else "filename equality is insufficient; render and content verification are required"
        ),
        "source_pages": pages,
        "target_characters": characters,
        "target_pictures": pictures,
        "eligible_for_supervised_training": False,
    }


def _inspect_pdf(path: Path) -> dict[str, Any]:
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        return {
            "kind": "pdf",
            "pages": document.page_count,
            "rotations": sorted({int(page.rotation) for page in document}),
            "native_text_characters": sum(len(page.get_text()) for page in document),
            "embedded_images": sum(len(page.get_images(full=True)) for page in document),
        }


def _inspect_docx(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as package:
            root = etree.fromstring(package.read("word/document.xml"))
            texts = root.findall(f".//{{{_WORD_NS}}}t")
            media = [
                name
                for name in package.namelist()
                if name.startswith("word/media/") and not name.endswith("/")
            ]
            return {
                "kind": "docx",
                "characters": sum(len(node.text or "") for node in texts),
                "paragraphs": len(root.findall(f".//{{{_WORD_NS}}}p")),
                "tables": len(root.findall(f".//{{{_WORD_NS}}}tbl")),
                "pictures": len(root.findall(f".//{{{_WORD_NS}}}drawing")),
                "images": len(media),
                "bold_runs": len(root.findall(f".//{{{_WORD_NS}}}b")),
                "alignment_nodes": len(root.findall(f".//{{{_WORD_NS}}}jc")),
            }
    except (BadZipFile, KeyError, etree.XMLSyntaxError) as exc:
        raise ValueError(f"invalid DOCX package: {exc}") from exc


def _inspect_hwpx(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as package:
            names = package.namelist()
            section_names = [
                name for name in names if re.search(r"Contents/section\d+\.xml$", name)
            ]
            text = ""
            tables = pictures = 0
            for name in section_names:
                root = etree.fromstring(package.read(name))
                text += "".join(root.itertext())
                tables += len(root.xpath("//*[local-name()='tbl']"))
                pictures += len(root.xpath("//*[local-name()='pic']"))
            return {
                "kind": "hwpx",
                "characters": len(text),
                "sections": len(section_names),
                "tables": tables,
                "pictures": pictures,
                "images": sum(
                    name.startswith("BinData/") and not name.endswith("/") for name in names
                ),
            }
    except (BadZipFile, etree.XMLSyntaxError) as exc:
        raise ValueError(f"invalid HWPX package: {exc}") from exc


def _pdf_text(path: Path) -> str:
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        return "".join(page.get_text() for page in document)


def _editable_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        with ZipFile(path) as package:
            root = etree.fromstring(package.read("word/document.xml"))
            return "".join(node.text or "" for node in root.findall(f".//{{{_WORD_NS}}}t"))
    if path.suffix.lower() == ".hwp":
        return read_hwp_reference(path).text
    if path.suffix.lower() == ".hwpx":
        with ZipFile(path) as package:
            result: list[str] = []
            for name in package.namelist():
                if re.search(r"Contents/section\d+\.xml$", name):
                    result.extend(etree.fromstring(package.read(name)).itertext())
            return "".join(result)
    raise ValueError(f"unsupported editable target: {path}")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_split_manifests(output_dir: Path, items: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = output_dir / "splits" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [item for item in items if item["split"] == split]
        _write_jsonl(path, rows)
        result[split] = path.relative_to(output_dir).as_posix()
    return result


def _readme(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return f"""# Training corpus v2

이 디렉터리는 원본을 수정하지 않고 만든 로컬 학습 코퍼스입니다.

- 전체 파일: {summary["files"]}
- gold 쌍: {summary["gold_pairs"]}
- 파일명 일치 후보: {summary["filename_pair_candidates"]}
- 오답 위험으로 자동 제외: {summary["rejected_filename_pairs"]}

`raw/`는 역할별 불변 복사본, `manifest.jsonl`은 파일 단위 메타데이터,
`pairs.jsonl`은 쌍 검증 결과입니다. `splits/test.jsonl`의 gold 문서는 학습에 사용하지 않습니다.
자동 OCR 결과와 레이아웃 의사 라벨은 gold 정답으로 승격하지 않습니다.
"""
