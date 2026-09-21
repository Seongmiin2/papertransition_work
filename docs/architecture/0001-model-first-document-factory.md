# ADR 0001: Model-first 한글 문서 재제작 공장

- **Status:** Accepted / Binding
- **Date:** 2026-09-09
- **Scope:** 모델, 문서 파이프라인, 품질 판정, 웹 배포, 운영·검수 UX

## 결정

이 제품은 OCR 결과를 HWPX로 단순 변환하는 도구가 아니라, 원본 증거를 보존하면서 내용을 이해하고 새 편집형 한글 문서로 재제작하는 공장으로 설계한다.

모든 구현 우선순위는 다음 순서를 따른다.

> **모델 신뢰성 → 제품 완성도 → 배포 안정성 → 서비스 UX**

뒤 단계의 완성도로 앞 단계의 결함을 가릴 수 없다. 모델 품질이 출고 기준을 충족하기 전에는 공개 서비스의 확장, 결제, 화려한 UI를 우선하지 않는다. 여기서 "확실한 모델"과 "완벽한 제품"은 주관적 선언이 아니라 이 문서의 release gate를 재현 가능하게 통과한 상태를 뜻한다.

## 구속력 있는 파이프라인

```text
원본 PDF/이미지
  → EvidenceIR
  → Model A
  → ContentIR
  → Model B
  → HwpDocumentPlan
  → deterministic HWPX compiler
  → validators
  → auto-pass | review | block
```

각 산출물은 버전이 지정된 엄격한 스키마로 저장한다. 입력 hash, 모델·프롬프트·지식 corpus·컴파일러 버전, 각 판단의 confidence와 provenance를 함께 기록해 동일 실행을 추적할 수 있어야 한다.

### EvidenceIR

EvidenceIR은 원본에서 관찰한 사실의 불변 기록이다. 페이지 이미지와 기하 정보, OCR 후보, bbox, confidence, 원본 crop, 엔진 응답과 출처를 담는다. 후속 모델이 EvidenceIR을 덮어쓰거나 출처를 잃게 해서는 안 된다.

### Model A: 문서 분석 모델

Model A는 EvidenceIR을 읽어 ContentIR을 만든다.

책임:

- 텍스트와 수식, 표, 그림 등 콘텐츠 식별
- 제목·지문·문항·선택지·표 셀 등 의미 역할 분류
- 읽기 순서, 그룹, 문항 관계와 페이지 간 연결 추론
- 모든 콘텐츠를 원본 evidence에 연결하고 판단별 confidence 기록
- 불확실하거나 지원하지 않는 항목을 검수 이슈로 명시

금지:

- HWPX/XML, 스타일 ID, package relationship 생성
- 원문을 보기 좋게 고쳐 쓰거나 근거 없는 내용을 추가·삭제
- 낮은 confidence나 모델 간 불일치를 숨김
- 페이지 경계를 넘는 관계를 문서 전체 조립 전에 확정

### ContentIR

ContentIR은 재제작할 문서 내용의 단일 기준이다. 의미 노드, 관계, 읽기 순서, 표 구조, 수식·자산 참조와 provenance를 포함한다. 사람의 수정은 원본 증거를 보존한 새 revision으로 기록한다. Model B와 컴파일러는 본문을 복사해 새 문자열로 만들지 않고 안정적인 `content_ref`로 이 내용을 참조한다.

### Model B: 한글 문서 설계 모델

Model B는 ContentIR, HWPX capability profile, 검색된 공식 한컴 근거를 입력으로 받아 HwpDocumentPlan을 만든다.

책임:

- 페이지, 단, 문단, 표, 그림, 수식과 나눔의 문서 구조 설계
- 의미 역할을 편집 가능한 한글 개체와 스타일 의도로 매핑
- 모든 본문·자산을 유효한 `content_ref`로 연결
- 사용한 공식 근거를 `spec_refs`로 기록
- 지원하지 않거나 모호한 설계를 명시적으로 review/block 처리

금지:

- 본문 생성, 요약, 교정, 번역 또는 ContentIR의 의미 변경
- raw XML, 로컬 경로, 외부 URL, 매크로, 스크립트, 자동화 명령 출력
- XML namespace, 내부 ID, 개수, media 이름, relationship, ZIP 순서 결정
- 존재하지 않는 `content_ref`나 공식 근거를 인용

### HwpDocumentPlan과 결정론적 컴파일러

HwpDocumentPlan은 고수준의 문서 설계도이며 실행 코드가 아니다. XML namespace, 문자·문단 속성 ID, 개체 ID, resource와 relationship, manifest, media 이름, item count, ZIP entry 순서는 결정론적 컴파일러만 소유한다.

같은 승인된 ContentIR, HwpDocumentPlan, 컴파일러 버전과 이미지 자산 번들 digest는
byte-level 차이가 불가피한 명시적 필드를 제외하고 같은 구조의 HWPX를 만들어야 한다.
이미지 번들은 EvidenceIR의 page/crop source SHA와 ContentIR 계보에 결속하며 raw 경로를
계약으로 사용하지 않는다. 모델이 raw HWPX/XML을 직접 생성하는 경로는 허용하지 않는다.

## 병렬 처리와 문서 barrier

- 여러 문서는 서로 격리된 job으로 동시에 처리한다. 한 문서의 실패가 같은 batch의 다른 문서를 중단시키지 않는다.
- 한 문서 안에서는 페이지 전처리, OCR, 시각 evidence 추출과 페이지 단위 Model A 추론을 자원 한도 내에서 병렬 처리한다.
- 모든 페이지의 EvidenceIR과 페이지 분석이 저장되기 전에는 **document barrier**를 통과할 수 없다.
- barrier에서 페이지 순서, 연속 지문·문항, 반복 머리말·꼬리말, 페이지 간 표·그림 관계를 문서 단위로 조립해 ContentIR revision을 확정한다.
- Model B, 컴파일, 최종 검증은 해당 ContentIR revision을 입력으로 수행한다. 수정 시 영향받은 단계부터 재실행하며 이미 검증된 OCR 전체를 불필요하게 반복하지 않는다.
- 각 단계는 idempotent하고 checkpoint에서 재개 가능해야 하며, queue retry가 중복 산출물을 출고해서는 안 된다.

## Validator와 출고 판정

validator는 하나의 평균 confidence가 아니라 독립된 근거를 모아 versioned QualityReport를 만든다.

필수 검증:

1. EvidenceIR, ContentIR, HwpDocumentPlan 스키마와 참조 무결성
2. ContentIR 대비 누락·중복·무단 생성 콘텐츠 0건
3. 읽기 순서, 문항 관계, 표 topology, 수식·자산 연결
4. 원본 대비 페이지 수, 영역별 배치와 렌더 시각 차이
5. 텍스트·표·그림·수식의 실제 편집 가능성
6. HWPX package·XML·manifest·relationship 적합성
7. 지원 대상 한글에서 열기 → 저장 → 재열기 round-trip

판정은 다음 셋뿐이다.

- `auto-pass`: 모든 hard gate를 통과하고 승인된 모델 bundle의 자동 출고 기준을 만족
- `review`: hard failure는 없지만 불확실한 텍스트·구조·레이아웃 또는 표본 검수가 필요
- `block`: 내용 누락·추가, 깨진 참조, 지원 불가 개체, 페이지 불일치, package 오류 또는 round-trip 실패

출고 가능한 artifact는 `auto-pass` 또는 사람의 검수를 거친 `approved` revision뿐이다. draft와 차단 문서는 최종 결과로 표시하지 않는다.

## 웹 배포 구조

```text
Browser
  → API/BFF + authentication/authorization
  → object storage (원본·중간 산출물·버전 artifact)
  → PostgreSQL (batch/job/revision/issue/audit metadata)
  → durable scheduler/queues
       ├─ GPU page-analysis workers
       ├─ Model B planning workers
       ├─ CPU deterministic compiler/validator workers
       └─ isolated Windows Hancom round-trip workers
  → structured job events → SSE/WebSocket → Browser
```

API 서버 프로세스가 모델 추론이나 로컬 파일 경로를 직접 소유하지 않는다. 업로드와 다운로드는 tenant가 제한된 object key와 만료되는 서명 URL을 사용한다. 원본과 중간 산출물에는 암호화, 보관 기간, 즉시 삭제, 감사 로그와 tenant 격리를 적용한다.

GPU·CPU·Windows 검증 lane은 독립적으로 확장하고 각 queue에 concurrency, timeout, retry, dead-letter 정책을 둔다. UI 진행률은 로그 문자열을 추측하지 않고 `{job_id, attempt, stage, page, total, state, timestamp}` 구조화 이벤트를 사용한다.

## 제품 정보 구조와 검수 워크벤치

고객 제품의 핵심 정보 구조는 `Workspace → Batch → Document → Revision → Issue/QualityReport → Artifact`이다. 내부 모델 운영 기능은 고객 UI와 분리한다.

최소 고객 화면:

1. 새 batch: 재개 가능한 다중 업로드, 파일별 사전 검사, 중복 hash, 예상 페이지·시간
2. batch 상세: 문서별 단계와 품질 상태, 필터, 실패 격리, 부분 재시도, 승인 결과 일괄 다운로드
3. 검수 queue: 위험도·영향도순 이슈, 담당자와 승인 상태
4. 문서 검수: 원본과 생성 미리보기 비교, 수정 이력, 재검증, artifact 버전
5. 결과: 품질 근거와 모델 bundle을 포함한 승인 보고서, HWPX 다운로드

검수 화면은 다음 세 영역을 기본으로 한다.

- 좌측: batch·문서·페이지 탐색과 페이지별 이슈 수
- 중앙: 원본/생성본 동기 비교, bbox·읽기 순서·영역 overlay
- 우측: OCR 후보, 텍스트, 의미 역할, 순서, 표 셀·병합, 수식, 배치 속성, confidence와 provenance

수정은 autosave되는 revision과 audit event로 남기고 undo가 가능해야 한다. 항목·페이지·문서 승인, 키보드 중심 검수, 안전한 항목의 일괄 승인을 제공하되 hard block의 일괄 무시는 금지한다. 승인된 교정은 학습 후보가 될 수 있지만 자동으로 학습 정답에 편입하지 않는다.

내부 ML 콘솔은 데이터셋·라벨 검수, 실험 비교, 실패 slice, champion/challenger, 모델 registry와 승격 승인에 집중한다. 고객에게 모델명, DPI, XML 설정 같은 내부 구현 선택을 노출하지 않고 처음에는 하나의 검증된 제품 약속만 제공한다.

## 단계별 release gate

상위 gate를 통과하기 전에는 다음 단계로 진행하지 않는다.

### G0 — 계약과 평가 데이터

- EvidenceIR, ContentIR, HwpDocumentPlan, QualityReport 스키마 고정
- 실제 대상 문서 유형을 포함한 사람 검증 golden corpus 구축
- 문서 단위 train/validation/test 격리와 데이터 provenance 검증
- 실패 유형과 hard/review 판정 기준 확정

### G1 — 모델 신뢰성

- Model A의 OCR/CER, block F1, 역할, 읽기 순서, 문항 관계, 표·수식 지표를 문서 유형별 held-out set에서 측정
- Model B의 schema-valid rate 100%, content-reference coverage 100%, unauthorized content change 0건
- confidence calibration과 review router가 위험 오류를 놓치지 않는지 측정
- champion 대비 핵심 slice 회귀가 없고 실험·가중치·데이터 checksum 재현 가능

### G2 — 제품 완성도

- held-out 문서 전체가 결정론적 compile과 HWPX validation 통과
- 페이지 수와 콘텐츠 보존 hard gate 100% 통과
- 지원 대상 한글 round-trip과 대표 편집 작업 통과
- 원본/결과 시각 diff, 편집 가능 개체 비율, 페이지당 사람 수정 시간을 기록하고 승인 기준 충족
- 수정→부분 재실행→재검증→버전 출고와 audit trail E2E 통과

### G3 — 배포 안정성

- 격리된 worker image와 model bundle checksum 검증·rollback
- durable queue, checkpoint, idempotency, timeout, retry, dead-letter 검증
- 다중 문서 부하·장시간 실행에서 처리량, 자원 한도, crash-free job과 데이터 무손실 기준 충족
- tenant 격리, 접근 제어, 암호화, 보관·삭제, backup/restore 검증

### G4 — 제한 서비스

- mandatory-review 폐쇄 beta로만 운영
- 실제 correction time, review 발생률, 실패 slice와 사용자 이탈 원인을 수집
- 승인 없는 artifact 출고 0건과 운영 대응·감사 가능성 확인

### G5 — 공개 서비스와 자동 출고

- 목표 문서 유형별 auto-pass 정확도와 위험 오류 recall이 사전 합의된 기준 충족
- canary, 관측성, 비용·용량 계획과 SLO 검증
- 공개 범위 밖 문서는 자동 감지해 review/block하고 제품 약속을 확장하지 않음

## 공식 한컴 문서의 사용 원칙

공식 한컴/HWPX/OWPML 문서는 다음 용도로만 사용한다.

- Model B의 검색 증강(RAG) 근거와 `spec_refs`
- capability profile과 컴파일러 구현 규칙
- validator, conformance fixture와 평가 기준

공식 문서 자체는 supervised training target이 아니다. 실제 Model B 학습 target은 사람이 검증한 `(ContentIR + capability profile + 검색된 공식 근거) → HwpDocumentPlan + spec_refs` 쌍이어야 한다. 문서 문장을 모델에 넣었다는 사실만으로 학습·정확도 향상·규격 준수를 주장하지 않는다.

## 현재 구현에 대한 판정

현재 `src/scan2hwpx/blueprint/model_a.py`와 `model_b.py`는 각각 `rule_based_v0`인
자리표시자다. 전자는 텍스트 정규식 기반 역할 분류, 후자는 layout seed의 규칙 기반 영역
배치이며, 이 ADR에서 정의한 학습된 Model A와 문서 설계 Model B가 아니다. 현재 학습·배포된
모델은 OCR recognizer와 품질 보조 page-anomaly 모델뿐이다.

Model A 추론은 특정 provider SDK와 분리된 정확히 1페이지 단위 canonical multimodal
request/response 경계로 구현돼 있다. request는 실제 페이지 이미지 bytes/base64/SHA-256과
source, 전체 EvidenceIR digest, 목표 page와 모델 ID/revision/artifact digest를 고정한다.
response는 bounded strict JSON으로 파싱한 뒤 evidence ownership, node kind, table topology,
image asset, 저신뢰·근거 이탈 검수 routing을 자동수정 없이 검사한다. 페이지 결과는 입력
순서와 무관하게 page number 순으로 조립하는 결정론적 barrier를 거쳐 ContentIR revision 1이
된다. ContentIR 1.0에는 typed cross-page relation이 없으므로 현재 barrier는 관계를 추측하지
않고 page node와 reading order를 결합만 한다. provider-neutral 실행기는 모든 페이지 이미지와
canonical request를 첫 호출 전에 검증하고 page number 순으로 페이지당 정확히 한 번만 호출하며,
retry·repair 없이 원시 응답과 digest를 barrier 결과에 보존한다. 실제 provider adapter와 승인된
가중치는 없고 모든 산출물은 research-only/noneligible이다.

Model A v2 원시 평가기는 digest로 결속된 EvidenceIR, 사람 gold ContentIR, prediction에서
release JSON의 8개 Model A metric을 직접 재계산하며 표 셀 텍스트도 CER·critical token에
포함한다. evidence refs와 image asset 불일치는 점수 밖 structural hard failure로 남긴다.
v3는 canonical 페이지 요청, 원시 응답, validator 결과와 document barrier 조립을 재생하고,
외부에서 공급된 exact 위험 표면 라벨을 사용해 high-risk review recall과 10-bin confidence
ECE를 추가한다. `model-first-v2` 정책은 기존 15개 지표 gate에 이 두 기준을 더한 총
17개이며, 각각 `>= 0.99`, `<= 0.05`를 요구한다. 위험
라벨과 검수자 신원은 아직 인증되지 않았고 typed cross-page 관계 및 release adapter가
남아 있으므로 report는 항상 nonpromotable이다.

`ModelBPlanReviewDraft`는 ContentIR과 `content_ref`를 고정한 채 page layout,
style, flow/break, `render_as`, `style_ref`, grounded official spec refs만
교정하는 typed patch 경계다. `ModelBPlanGroundingEvidence`는 지식 corpus,
retrieval artifact, capability profile과 각 공식문서 원본·검색 chunk의 SHA-256을
고정한다. 공식 corpus sidecar 생성, 완료된 Candidate review의 reviewed ContentIR/Plan을
소비하는 handoff, create-only Plan revision 저장과 4-way CAS 완료 전이까지 구현됐다.
완료 상태도 인증 reviewer·권리 attestation·Golden 승격과 연결되지 않으므로 모든 revision은
명시적으로 noneligible/unverified다. seed-only 검수 초안은 호환용 1.2로 유지하지만 제출할
수 없고, 유효한 Model B 결과 Plan 자체를 base로 시작한 1.3 초안만 출고 컴파일 입력이 된다.
1.3은 request/result/raw response/생성 Plan SHA, handoff SHA와 model ID/revision/artifact
SHA를 patch와 완료 revision 전체에서 보존한다.

Candidate review는 직접 IMAGE observation으로 검증된 PAGE_IMAGE/CROP source에 그림을
재결속하거나 그림 노드를 명시적으로 제외하는 typed patch를 제공한다. 완료본 materializer는
원본 candidate와 EvidenceIR을 다시 읽어 SHA와 grounding을 확인하고 compiler 전용 immutable
PNG bundle을 만든다. 완료된 Candidate review·verified handoff·완료된 Model B Plan review가
동일한 Model B inference result와 모두 일치할 때만 HWPX를 만든다. staged digest를 검증한
뒤 Windows `rename` 또는 Linux `renameat2(RENAME_NOREPLACE)` 한 번으로 create-only
발행하며, commit 뒤 실패나 rollback 경로를 두지 않는다. 외부 시각 비교·DVC·한컴
round-trip 전에는 10개 항목 QualityReport의 최종 판정을 항상 REVIEW로 유지한다.

Model B 추론은 특정 provider SDK와 분리된 canonical JSON request/response 경계로 구현돼
있다. request는 verified handoff, reviewed ContentIR, design seed Plan, capability 원본,
retrieved 공식 chunk, 모델 ID/revision/artifact digest를 함께 고정한다. response는 strict
schema뿐 아니라 ordered `(content_ref, render_as)` projection, ContentIR lineage,
capability operation과 공식 근거 부분집합을 검사한다. raw XML·URL·로컬 경로·자동화
payload는 자동수정하지 않고 구조화된 block으로 반환한다. 현재는 실제 provider adapter와
승인된 가중치가 없으며 valid response도 research-only/noneligible이다. provider-neutral
실행기는 handoff·capability·공식 chunk·response limit과 canonical request를 호출 전에
검증하고 provider를 정확히 한 번만 호출하며, retry·repair 없이 원시 응답과 결과 digest를
보존한다. blocked·변조·다른 handoff의 결과는 검수 시작과 제출 모두에서 거부하며, valid
result의 Plan을 seed 값으로 자동수정하지 않고 새 사람 검수 base Plan으로 그대로 사용한다.

OCR/Model A 학습 entrypoint는 별도의 `ocr-training-rights/1.0` manifest와 실제 evidence
artifact를 필수로 받고, 인증된 외부 caller가 권리 확인자·독립 파생 확인자·신뢰하는
dataset report digest를 주입해야 한다. 원본/이미지 byte inventory,
train/validation/test template-family 격리, dataset report, 라벨·문자 사전 digest가
모두 일치하기 전에 pretrained 다운로드·GPU 초기화·output 생성을 시작하지 않는다.
이는 source→crop을 로컬에서 재현한 암호학적 증명이 아니라 인증된 외부 report
attestation과 artifact 무결성 경계다. 저장소 example manifest나 standalone CLI는 학습을
승인할 수 없다.

결정론적 HWPX compiler v4는 text paragraph와 page break, font family/size, bold/italic,
paragraph alignment와 line spacing의 styled text에 더해 빈 셀·병합 셀·특수 공백/나눔을
보존하는 편집 가능한 표와 명시적 폭의 inline PNG 그림을 지원한다. 이미지 자산 경계는
EvidenceIR/ContentIR 계보, page/crop source SHA, 정적 8-bit RGB/RGBA PNG chunk allowlist와
bounded IDAT decode, 16M 단일·64M bundle declared/unique·64M 문서 occurrence pixel 한도를
검증한다. compiler는 동일 payload digest를 한 번만 BinData에 넣고 flow occurrence는 각각
그림 control로 보존하며, 고정 template digest, 결정적 control ID,
OPF manifest·binary bytes·picture anchor와 최종 candidate digest를 확인한 뒤에만 원자적으로
결과를 교체한다. 폭 미지정 그림, 수식, 명시적 column break 또는 비지원 layout/design
intent는 compile 전에 실패한다.

이 출력도 아직 지원 대상 한글의 open-save-reopen round-trip으로 검증되지 않았다. 또한
v4 후보의 이미지 91건은 모두 사람 검수 대상이고 41문서 모두 권리가 미확인이다. 따라서
현재 후보 전체를 완성 문서로 출고할 수 없으며, 이미지 의미·배치 승인, 수식 지원과 Golden
기반 end-to-end 검증이 필요하다.

실제 후보 대표 1건 검수 smoke는 직접 IMAGE observation이 없는 page fallback 그림을
명시적으로 제외한 뒤 불변 revision 완료와 재열기까지 통과했다. 이 자동 해소는 임시
UI/계약 smoke일 뿐 사람 승인이나 출고 승격이 아니며, 후보 원본과 rights/training/release
eligibility는 바뀌지 않았다.

따라서 기존 데스크톱 앱과 변환기는 데이터·컴파일·검증 자산을 재사용할 수 있는
prototype으로 취급한다. 실제 provider가 연결된 Model A/B, typed cross-page relation을
확정하는 resolver, 외부 validator가 채우는 quality decision, 서버용 영속 검수 workflow가
구현되고 G0~G2를 통과하기 전에는 완성된 한글 문서 재제작 제품으로 표시하거나 공개
웹서비스를 우선 확장하지 않는다. 2026-09-10 감사에서 확인된 학습 가능 Model B target은
0건이며, OCR도
권리 검증과 template-family 격리 실스캔 held-out을 충족하는 데이터가 0건이므로 이 조건이
해소되기 전에는 새 가중치 학습을 시작하지 않는다.

## 결과

이 결정은 모델의 창의성을 HWPX 내부 구현에서 분리하고, 원문 보존과 출고 책임을 결정론적 코드와 검증 gate에 둔다. 추가 비용은 중간 IR, 검수 revision, 별도 worker lane과 품질 데이터 운영이지만, 대량 병렬 처리에서도 오류를 격리하고 동일한 근거로 재현·수정·승인할 수 있다.

이 원칙을 변경하려면 후속 ADR에서 변경 이유, 대체 안전장치, 회귀 평가 결과와 migration 계획을 제시해야 한다.
