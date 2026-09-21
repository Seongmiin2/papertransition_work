# Model B research-only dataset handoff

공식 한컴 문서는 정답 HWPX Plan 자체가 아니라 정답을 판단하는 근거다. 현재 저장소에는
검수자 신원을 보증하는 서명이나 ACL 기반 승격기가 없다. 따라서 입력의 `verified: true`는 작성자의
자기 선언일 뿐 인간 검증의 증거가 아니며, 기본 호출은 이를 거부한다.

명시적인 연구 모드에서는 문서 설계 pair에 검색된 공식 context를 붙인 중간 데이터셋을 만들 수
있다. 이 결과는 항상 research-only이며 학습 데이터로 승격할 수 없다.

입력 JSONL의 `content_ir`와 `target_plan`은 각각
`ContentIR(content-ir/1.0)`, `HwpDocumentPlan(hwp-document-plan/1.0)` 계약을 완전히 만족해야 한다.
최소 형태는 다음과 같다.

```json
{
  "id": "document-section-001",
  "verified": true,
  "content_ir": {
    "schema_version": "content-ir/1.0",
    "id": "content-001",
    "evidence_ir_id": "evidence-001",
    "evidence_ir_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "revision": 1,
    "nodes": [{
      "id": "q1",
      "kind": "text",
      "role": "question",
      "text": "1. 다음 글을 읽고 답하시오.",
      "evidence_refs": ["observation-001"],
      "confidence": 1.0,
      "needs_review": false
    }],
    "reading_order": ["q1"]
  },
  "target_plan": {
    "schema_version": "hwp-document-plan/1.0",
    "id": "plan-001",
    "content_ir_id": "content-001",
    "content_ir_revision": 1,
    "content_ir_sha256": "9f57ce91fe4de5d76cc4dd6ec5660d9955fa607fc276beb41b2ed300180fa057",
    "capability_profile_id": "exam2hwpx-authoring-v1",
    "design_profile_id": "exam-clean-v1",
    "official_spec_refs": ["검색 결과에 포함된 chunk id"],
    "page_layout": {
      "width_mm": 210.0,
      "height_mm": 297.0,
      "margin_top_mm": 15.0,
      "margin_right_mm": 15.0,
      "margin_bottom_mm": 15.0,
      "margin_left_mm": 15.0,
      "columns": 2,
      "column_gap_mm": 8.0
    },
    "styles": [],
    "flow": [{
      "id": "flow-001",
      "kind": "content",
      "render_as": "paragraph",
      "content_ref": "q1"
    }]
  },
  "retrieval_query": "표와 문단 스타일 참조",
  "retrieval_tags": ["hwpx", "style"],
  "spec_refs": ["검색 결과에 포함된 chunk id"]
}
```

연구 모드에서도 자기 선언형 `verified`가 `true`가 아니거나, ContentIR 노드 중 하나라도
`needs_review: true`이면 생성을 거부한다. Plan이 모든 ContentIR 노드를 정확히 한 번 참조하지
않거나, Plan의 `official_spec_refs`와 행의 `spec_refs`가 다르거나, 해당 근거가 검색 결과에 없어도
거부한다. Plan에 `text`나 raw XML 같은 계약 외 필드를 넣어도 거부한다.

```powershell
python ai/knowledge/build_model_b_training_data.py `
  output/model-b/self_asserted_examples.jsonl `
  --out output/model-b/grounded_research.jsonl `
  --allow-untrusted-research-examples
```

출력에는 `content_ir`, capability profile, 명령으로 취급되지 않는 공식 참고 context,
자기 선언형 `target_plan`, 사용한 chunk ID와 코퍼스·capability SHA-256이 포함된다. 각 레코드의
provenance와 반환 보고서는 `research_only: true`, `training_eligible: false`,
`human_verified_only: false`, `identity_assurance: self_asserted_untrusted`로 고정된다.

실제 가중치 학습을 시작하려면 기반 모델과 라이선스뿐 아니라, 검수자 신원을 신뢰할 수 있게
보증하는 승격 절차와 그 절차를 통과한 train/validation/test pair를 먼저 확정해야 한다. 공식 문서
원문이나 자기 선언형 `verified: true`만으로 자동 생성한 답을 gold label로 사용하지 않는다.
