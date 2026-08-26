from __future__ import annotations

import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scan2hwpx.pipeline import inspect_pdf
from scan2hwpx.reference.hwp5 import read_hwp_reference


def audit_reference_directory(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    document_dir = output_dir / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    hwp_files = [path for path in files if path.suffix.lower() == ".hwp"]
    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]

    hwp_items: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in hwp_files:
        try:
            document = read_hwp_reference(path)
            by_hash[document.sha256].append(path.name)
            key = document.sha256[:16]
            json_path = document_dir / f"{key}.json"
            text_path = document_dir / f"{key}.txt"
            if not json_path.exists():
                json_path.write_text(
                    json.dumps(document.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                text_path.write_text(document.text, encoding="utf-8")
            item = document.to_dict(include_paragraphs=False)
            item.update(
                {
                    "name": path.name,
                    "normalized_name": normalize_exam_name(path.stem),
                    "duplicate_of": None,
                    "document_json": _relative(json_path, output_dir),
                    "document_text": _relative(text_path, output_dir),
                }
            )
            hwp_items.append(item)
        except Exception as exc:  # noqa: BLE001 - audit must continue past one bad source file
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    first_by_hash: dict[str, str] = {}
    for item in hwp_items:
        sha256 = str(item["sha256"])
        if sha256 in first_by_hash:
            item["duplicate_of"] = first_by_hash[sha256]
        else:
            first_by_hash[sha256] = str(item["name"])

    pdf_items: list[dict[str, Any]] = []
    for path in pdf_files:
        try:
            info = inspect_pdf(path)
            pdf_items.append(
                {
                    **info,
                    "name": path.name,
                    "sha256": _sha256(path),
                    "normalized_name": normalize_exam_name(path.stem),
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit must continue past one bad source file
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    hwp_names: dict[str, list[str]] = defaultdict(list)
    for item in hwp_items:
        hwp_names[str(item["normalized_name"])].append(str(item["name"]))
    pairs = [
        {
            "pdf": str(pdf["name"]),
            "hwp": candidate,
            "match": "exact_normalized_filename",
        }
        for pdf in pdf_items
        for candidate in hwp_names.get(str(pdf["normalized_name"]), [])
    ]
    duplicate_groups = [names for names in by_hash.values() if len(names) > 1]
    unique_items = [item for item in hwp_items if item["duplicate_of"] is None]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "input_directory": str(source_dir),
        "summary": {
            "hwp_files": len(hwp_files),
            "hwp_parsed": len(hwp_items),
            "unique_hwp_documents": len(unique_items),
            "pdf_files": len(pdf_files),
            "exact_pdf_hwp_pairs": len(pairs),
            "duplicate_groups": len(duplicate_groups),
            "parse_failures": len(failures),
            "total_reference_characters": sum(int(item["character_count"]) for item in unique_items),
            "total_reference_paragraphs": sum(int(item["paragraph_count"]) for item in unique_items),
            "tables": sum(int(item["table_count"]) for item in unique_items),
            "pictures": sum(int(item["picture_count"]) for item in unique_items),
            "equations": sum(int(item["equation_count"]) for item in unique_items),
        },
        "training_readiness": {
            "paired_supervision_ready": bool(pairs),
            "reference_corpus_ready": bool(unique_items),
            "note": (
                "Exact-name PDF/HWP pairs may be used for alignment after content verification."
                if pairs
                else "No exact PDF/HWP pair was found. HWP files are reference-corpus Y examples, "
                "not direct labels for the current PDF."
            ),
        },
        "pairs": pairs,
        "duplicate_groups": duplicate_groups,
        "hwp_documents": hwp_items,
        "pdf_documents": pdf_items,
        "failures": failures,
    }
    (output_dir / "reference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_manifest(hwp_items, output_dir / "reference_manifest.jsonl")
    (output_dir / "reference_report.html").write_text(
        _render_html(report), encoding="utf-8"
    )
    return report


def normalize_exam_name(stem: str) -> str:
    value = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    value = re.sub(r"^\s*\([^)]*\)\s*", "", value)
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).casefold()


def _write_manifest(items: list[dict[str, Any]], path: Path) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['name']))}</td>"
        f"<td>{item['character_count']:,}</td>"
        f"<td>{item['section_count']}</td>"
        f"<td>{item['table_count']}</td>"
        f"<td>{item['picture_count']}</td>"
        f"<td>{item['equation_count']}</td>"
        f"<td>{html.escape(str(item['duplicate_of'] or ''))}</td>"
        "</tr>"
        for item in report["hwp_documents"]
    )
    note = html.escape(str(report["training_readiness"]["note"]))
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><title>HWP 참조 데이터 점검</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.card{{background:#f4f7fb;border-radius:12px;padding:16px}}.n{{font-size:28px;font-weight:700}}
.warning{{margin:20px 0;padding:16px;border-left:5px solid #e99b22;background:#fff6e7}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dde3ec;text-align:left}}
th{{position:sticky;top:0;background:white}}</style></head><body>
<h1>HWP 참조 데이터 점검</h1><div class=\"cards\">
<div class=\"card\"><div class=\"n\">{summary['hwp_parsed']}</div>파싱 HWP</div>
<div class=\"card\"><div class=\"n\">{summary['unique_hwp_documents']}</div>고유 문서</div>
<div class=\"card\"><div class=\"n\">{summary['total_reference_characters']:,}</div>참조 글자</div>
<div class=\"card\"><div class=\"n\">{summary['exact_pdf_hwp_pairs']}</div>정확한 X-Y 후보</div>
</div><div class=\"warning\">{note}</div>
<h2>문서별 추출 결과</h2><table><thead><tr><th>파일</th><th>글자</th><th>구역</th><th>표</th><th>그림</th><th>수식</th><th>중복 원본</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
