# 폴더 구조

저장소는 제품 코드와 AI 수명주기를 분리한다. AI 자산은 `Dataset → Modeling → Results → Production` 순서로 이동하며, 큰 생성물과 민감할 수 있는 원본은 Git에 올리지 않는다.

```text
papertransition_work/
├─ apps/                         Electron 데스크톱 앱
├─ src/scan2hwpx/                Python 제품 파이프라인
│  ├─ contracts/                 EvidenceIR·ContentIR·HwpDocumentPlan 계약
│  ├─ evaluation/                release gate·검수·grounding·handoff·검수 완료본 안전 컴파일 경계
│  ├─ knowledge/                 공식 한컴 자료 RAG/eval 지식 경계
│  ├─ hwpx/                      SHA 결속 PNG assets·결정론적 styled-text/table/image HWPX compiler·검증기
│  ├─ model_a/                   provider-neutral 페이지 실행·추론 검증·document barrier·규칙 기반 대역
│  ├─ model_b/                   provider-neutral 설계 단일 실행·요청·응답 검증 경계
│  └─ blueprint/                 구조 분류·배치 결정 규칙 기반 대역 (Model A/B 자리)
├─ tests/                        단위·통합 테스트
├─ ai/
│  ├─ datasets/
│  │  ├─ source/                 로컬 원본 HWP/PDF/DOCX (Git 제외)
│  │  ├─ configs/                데이터 품질 기준과 정답 쌍 설정
│  │  └─ build_hwp_line_dataset.py
│  ├─ modeling/
│  │  ├─ configs/                PaddleOCR 학습 설정
│  │  ├─ experiments/            실험별 데이터·파라미터·지표·판정
│  │  ├─ pretrained/             다운로드한 기반 가중치 (Git 제외)
│  │  └─ train_korean_ocr.py
│  ├─ results/
│  │  ├─ performance_log.csv     모든 실험의 누적 지표
│  │  ├─ record_metrics.py       벤치마크 JSON을 CSV로 적재
│  │  └─ history/                날짜별 연구·학습 보고서
│  └─ production/
│     ├─ deployed/               앱이 실제 로드하는 승인 모델
│     └─ candidates/             평가했지만 배포하지 않은 후보
├─ docs/                         제품 개발 문서
├─ output/                       실행·학습·벤치마크 생성물 (Git 제외, 아래 구조)
├─ samples/                      재현용 입력 위치 안내
├─ templates/                    HWPX 템플릿
└─ scripts/                      앱 실행·실제 후보 검수 smoke 보조 스크립트
```

데이터 목록은 [`ai/datasets/INVENTORY.md`](ai/datasets/INVENTORY.md), 결과값은
[`ai/results/README.md`](ai/results/README.md)에 정리한다.

## `output/` 구조

생성물은 종류별 하위 폴더에 둔다. 각 산출물 폴더의 이름은 만든 날짜와 버전을 담은 원래 이름을 그대로 쓴다.

```text
output/
├─ datasets/                     학습·평가용으로 만든 데이터 (OCR line crop, HWP↔HWPX 쌍, 한컴 지식 corpus 등)
├─ training/                     학습 실행 결과 (checkpoint, inference export, 학습 로그)
├─ review/                       사람 검수용 후보 bundle과 Model B grounding sidecar
├─ experiments/
│  ├─ model-a/<계열>/            Model A 연구 실험 (block-role, role-vector, table-topology, table-fusion, table-detector, local-smoke)
│  └─ research/                  초기 참조 자료 조사
├─ benchmarks/                   변환기 성능 실측과 샘플 변환 실행
├─ showcase/                     시연용 변환 결과
└─ vendor/                       외부 소스 (PaddleOCR 학습 코드)
```

2026-09-22에 평면 구조에서 옮겼다. 폴더 이름은 그대로이고 앞에 분류만 붙었으므로, 이전
기록(`ai/results/history/`, `docs/DEVELOPMENT_REPORT.md`, 산출물 JSON 안의 절대 경로)에 남은
`output/<이름>`은 아래 규칙으로 찾는다. 산출물 JSON은 해시로 결속된 기록이라 경로를 고쳐 쓰지 않았다.

| 이전 `output/<이름>` | 현재 위치 |
|---|---|
| `hwp-ocr-training`, `hwp-ocr-training-v2`, `hwp-hwpx-pairs-*`, `model-a-line-risk-dataset-*`, `model-a-annotation-candidates`, `hancom-knowledge`, `ocr-training-rights-worklist-*.json` | `output/datasets/<이름>` |
| `trained-*` | `output/training/<이름>` |
| `hwpx-projection-candidates-*`, `model-b-grounding-sidecars-*` | `output/review/<이름>` |
| `model-a-<계열>-*` (예: `model-a-table-fusion-v13-full8-20260917`) | `output/experiments/model-a/<계열>/<이름>` |
| `research` | `output/experiments/research` |
| `performance-current-*`, `model-inference`, `model-inference-v2`, `layout-fix-*`, `layout-flow-*` | `output/benchmarks/<이름>` |
| `showcase-*` | `output/showcase/<이름>` |

`ai/knowledge/hancom/sources.json`의 `raw_directory`도 해시 결속 때문에 이전 경로로 남아 있다. 실제 기본 경로는
`output/datasets/hancom-knowledge/raw`다.

## 관리 원칙

1. 원본 데이터는 `ai/datasets/source/`에 두고 Git에는 올리지 않는다.
2. 데이터셋 생성 결과와 checkpoint는 `output/`의 해당 분류 폴더에 두고, 재현 정보만 `ai/modeling/experiments/`에 기록한다.
3. 비교 가능한 지표는 `ai/results/performance_log.csv`에 누적한다.
4. held-out test와 제품 smoke test를 통과한 모델만 `ai/production/deployed/`로 옮긴다.
5. 탈락한 실험도 삭제하지 않고 메타데이터와 후보 artifact를 남겨 같은 시도를 반복하지 않게 한다.
