from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from scan2hwpx.ir.models import Document


def write_review(document: Document, output_dir: Path) -> tuple[Path, Path]:
    blocks = [block for page in document.pages for block in page.blocks]
    review_ids = set(document.qa.low_confidence_blocks)
    review_blocks = [block for block in blocks if block.id in review_ids]
    page_for = {block.id: page.page_no for page in document.pages for block in page.blocks}
    items: list[dict[str, Any]] = []
    for block in review_blocks:
        reasons = list(block.style.get("review_reasons", []))
        if block.confidence < 0.75 and "low_ocr_confidence" not in reasons:
            reasons.append("low_ocr_confidence")
        relative_crop = Path("review_crops") / f"{block.id}.png"
        items.append(
            {
                "document_id": document.id,
                "source_hash": document.source_hash,
                "page": page_for[block.id],
                "block_id": block.id,
                "prediction": block.text,
                "target": block.text,
                "confidence": block.confidence,
                "bbox": list(block.bbox.pixel),
                "crop_image": relative_crop.as_posix(),
                "image_path": str((output_dir / relative_crop).resolve()),
                "reasons": reasons,
                "ocr_candidates": block.style.get("ocr_candidates", []),
                "verified": False,
            }
        )
    payload = {
        "summary": {
            "total_pages": len(document.pages),
            "total_blocks": len(blocks),
            "review_required_blocks": len(review_blocks),
            "average_confidence": sum(block.confidence for block in blocks) / max(1, len(blocks)),
        },
        "items": items,
    }
    json_path = output_dir / "review.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = output_dir / "review.html"
    html_path.write_text(_render_review_html(payload), encoding="utf-8")
    return json_path, html_path


def _render_review_html(payload: dict[str, Any]) -> str:
    seed = json.dumps(payload["items"], ensure_ascii=False).replace("</", "<\\/")
    summary = html.escape(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'nonce-scan2hwpx-review'; object-src 'none'">
<title>Scan2HWPX OCR 검수</title><style>
body{{font-family:'Malgun Gothic',system-ui,sans-serif;margin:0;background:#f4f6fa;color:#172033}}
header{{position:sticky;top:0;z-index:2;background:#172033;color:white;padding:16px 24px;display:flex;gap:16px;align-items:center}}
header h1{{font-size:20px;margin:0}}button{{cursor:pointer;border:0;border-radius:8px;padding:8px 12px}}
#download{{background:#35c48d;font-weight:700}}main{{max-width:1120px;margin:24px auto;padding:0 16px}}
.card{{background:white;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 2px 10px #17203312}}
.meta{{display:flex;gap:14px;color:#586174;font-size:13px;margin-bottom:10px}}.crop{{max-width:100%;max-height:160px;border:1px solid #ccd3df}}
.candidate{{display:block;width:100%;text-align:left;background:#edf2f8;margin:6px 0}}.candidate:hover{{background:#dce9f7}}
input[type=text]{{width:calc(100% - 20px);padding:10px;font:16px 'Malgun Gothic';border:2px solid #aab5c5;border-radius:8px}}
.verify{{display:inline-flex;gap:7px;align-items:center;margin-top:10px;font-weight:700}}.reasons{{color:#b05c12}}
pre{{white-space:pre-wrap}}small{{color:#657086}}</style></head><body>
<header><h1>OCR 검수·학습 라벨</h1><span id="counter"></span><button id="download">검증 JSONL 다운로드</button></header>
<main><details><summary>변환 요약</summary><pre>{summary}</pre></details><div id="items"></div></main>
<script nonce="scan2hwpx-review">const items={seed};
const root=document.getElementById('items'), counter=document.getElementById('counter');
const escapeText=v=>String(v??'');
function updateCounter(){{const done=items.filter(x=>x.verified).length;counter.textContent=`확정 ${{done}} / ${{items.length}}`;}}
items.forEach((item,index)=>{{
  const card=document.createElement('section');card.className='card';
  const meta=document.createElement('div');meta.className='meta';
  meta.textContent=`${{item.page}}페이지 · ${{item.block_id}} · 신뢰도 ${{Number(item.confidence).toFixed(3)}}`;
  const reasons=document.createElement('div');reasons.className='reasons';reasons.textContent=(item.reasons||[]).join(', ');
  const image=document.createElement('img');image.className='crop';image.src=item.crop_image;image.alt=item.block_id;
  const candidates=document.createElement('div');
  (item.ocr_candidates||[]).forEach(candidate=>{{const button=document.createElement('button');button.className='candidate';
    button.textContent=`${{candidate.engine}} (${{Number(candidate.confidence).toFixed(3)}}): ${{escapeText(candidate.text)}}`;
    button.onclick=()=>{{input.value=candidate.text;item.target=candidate.text;}};candidates.appendChild(button);}});
  const input=document.createElement('input');input.type='text';input.value=item.target;input.oninput=()=>item.target=input.value;
  const label=document.createElement('label');label.className='verify';const check=document.createElement('input');check.type='checkbox';
  check.onchange=()=>{{item.verified=check.checked;updateCounter();}};label.append(check,document.createTextNode('이 텍스트를 사람 정답으로 확정'));
  card.append(meta,reasons,image,candidates,input,label);root.appendChild(card);
}});updateCounter();
document.getElementById('download').onclick=()=>{{const verified=items.filter(x=>x.verified&&x.target.trim()).map(x=>({{
  schema_version:'1.0',document_id:x.document_id,source_hash:x.source_hash,page:x.page,block_id:x.block_id,
  image:x.image_path,bbox:x.bbox,prediction:x.prediction,target:x.target.trim(),status:'verified',candidates:x.ocr_candidates
}}));if(!verified.length){{alert('먼저 한 줄 이상을 정답으로 확정하세요.');return;}}
const body=verified.map(x=>JSON.stringify(x)).join('\\n')+'\\n';const blob=new Blob([body],{{type:'application/x-ndjson;charset=utf-8'}});
const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='verified_ocr_training.jsonl';link.click();
setTimeout(()=>URL.revokeObjectURL(link.href),1000);}};</script></body></html>"""
