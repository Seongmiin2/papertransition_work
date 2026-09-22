# 데이터 목록

2026-09-22 기준으로 로컬에 있는 원본, 가공 데이터셋, 검수 후보, 모델을 정리한다. 원본과 `output/`은
Git에 올라가지 않으므로 이 문서가 무엇이 어디에 있는지 보는 기준표다. 생성 명령은 [`README.md`](README.md),
결과 수치는 [`../results/README.md`](../results/README.md)를 본다.

## 한눈에 보기

| 구분 | 위치 | 규모 | 사용 상태 |
|---|---|---|---|
| 원본 시험지 HWP | `ai/datasets/source/고1-2`, `고2` | 41문서 (고1 21, 고2 20) | 학습 권리 미확인 |
| 참조 자료 | `ai/datasets/source/references` | 2파일 | 참조용 |
| 실제 스캔 샘플 | `samples/` | 1문서, 8쪽 (B4 스캔) | 변환 회귀 확인용 |
| OCR 학습 데이터 | `output/datasets/hwp-ocr-training-v2` | 41문서, 36,622줄 | 연구용, 새 학습 차단 |
| 검수 후보 | `output/review/hwpx-projection-candidates-20260910-v4` | 41문서, 485쪽 | 전부 사람 검수 필요 |
| 배포 모델 | `ai/production/deployed` | OCR 인식기 1, 페이지 이상 탐지 1 | 데스크톱 앱이 사용 |

학습 권리와 사람 검수가 모두 확인된 학습용 데이터는 **0건**이다
([ADR 0002](../../docs/architecture/0002-base-model-bakeoff.md)). 권리 확인 작업 목록은
`output/datasets/ocr-training-rights-worklist-20260910.json`(41문서, `training_authorized: false`)이다.

## 1. 원본 (`ai/datasets/source/`, Git 제외)

폴더 이름은 데이터셋 보고서와 정답 쌍 설정(`configs/training_pairs.json`)이 원본 경로와 해시로
대조하므로 바꾸지 않는다.

### `고1-2/` — 고1 1학기 중간고사 HWP 21개

| 교과서 | 학교 |
|---|---|
| 천재(김) | 고운고, 국제고, 대일외고, 수지고, 진선여고, 한솔고, 현암고, 휘문고 |
| 창비 | 다정고, 대진고, 보평고, 새롬고, 중앙고 |
| 지학사 | 늘푸른고, 돌마고, 이매고 |
| 창비(최) | 야탑고, 홍천고 |
| 천재(김) 수학 | 낙생고, 불곡고 |
| 해냄 | 판교고 |

### `고2/` — 고2 1학기 중간고사 HWP 20개

| 교과서 | 학교 |
|---|---|
| 비상(강) | 두루고, 수지고, 아름고, 양지고, 캠퍼스고, 풍덕고, 홍천고 |
| 미래엔 | 다정고, 대성고, 은광여고, 청담고 |
| 창비(최) | 야탑고, 이매고, 태원고 |
| 지학사 | 국제고, 종촌고 |
| 천재(정) | 단대부고, 서원고 |
| 창비 | 새롬고 |
| 천재(전) | 늘푸른고 |

### `references/` — 2개

| 파일 | 용도 |
|---|---|
| 홍천중학교 2학년 1학기 2차 지필평가 국어 (텍스트 PDF) | 텍스트 레이어가 있는 PDF 변환 확인 |
| 같은 시험의 한글 원본 (DOCX) | 편집 구조 참조 |

### `samples/` — 1개

| 파일 | 용도 |
|---|---|
| (천재박) 홍천중3 1학기 기말고사 | 실제 스캔 8쪽, B4. 변환 품질 회귀 확인의 기준 문서 |

## 2. 가공 데이터셋 (`output/datasets/`)

| 이름 | 내용 | 규모 | 분할 (문서 / 줄) | 비고 |
|---|---|---|---|---|
| `hwp-ocr-training` | HWP 원문을 렌더링한 OCR line crop, 1차 | 20문서, 19,507줄 | train 16 / 15,648 · val 2 / 1,757 · test 2 / 2,102 | 1차 fine-tune에 사용 |
| `hwp-ocr-training-v2` | 같은 방식, 고1·고2 전체 | 41문서, 36,622줄 | train 33 / 30,309 · val 4 / 3,392 · test 4 / 2,921 | 배포 OCR(v2) 학습에 사용 |
| `model-a-line-risk-dataset-20260909` | v2 crop에 배포 OCR 예측을 붙인 오류 위험 데이터 | 31,103줄 (분할 간 중복 5,519줄 제거) | train 26,418 · val 2,584 · test 2,101 | 전체 CER 0.70%, 완전일치 90.8% |
| `hwp-hwpx-pairs-20260909` | HWP를 한글로 HWPX·PDF로 변환한 쌍 | 41쌍 | train 33 · val 4 · test 4 | 연구용, 권리 manifest 없음 |
| `hancom-knowledge` | 공식 한컴/HWPX 문서 검색용 corpus | 10개 출처, 2,554 chunk | — | Model B 근거 검색 전용, 학습 정답 아님 |
| `model-a-annotation-candidates` | 스캔 샘플의 Model A 주석 후보 | 1문서 | — | 사람 검수 전 |
| `ocr-training-rights-worklist-20260910.json` | 원본 41문서의 권리 확인 작업 목록 | 41문서 | — | 학습 승인 없음 |

## 3. 검수 후보 (`output/review/`)

| 이름 | 내용 | 규모 | 상태 |
|---|---|---|---|
| `hwpx-projection-candidates-20260910-v4` | HWP 원문에서 투영한 EvidenceIR·ContentIR·HwpDocumentPlan 후보 | 41문서, 485쪽, 20,383노드, 그림 91개 | 모든 노드 `needs_review`, 권리 manifest 없음, Golden·출고 대상 아님 |
| `model-b-grounding-sidecars-20260910-v4` | 후보별 공식 한컴 근거 검색 결과 | 41문서, 문서당 근거 5 chunk | 검수자 신원 미확인 |

v4 후보에서 41문서 모두에 공통으로 걸린 검수 사유는 머리말·꼬리말 배치, 그림 짝 미확인, 인라인
스타일 단순화, PDF 정렬로 추정한 쪽 나눔 등이다. 수식 관련 사유는 1문서, 단 구성 변경은 22문서다.

## 4. 모델 (`ai/production/`)

| 위치 | 모델 | 상태 | 핵심 수치 |
|---|---|---|---|
| `deployed/korean-exam-ppocrv5` | PP-OCRv5 mobile 한국어 인식기, 시험지 fine-tune v2 | 배포 | test 완전일치 95.57% (test 4문서 중 3문서가 과거 학습 문서와 겹쳐 제품 정확도로 쓰지 않음) |
| `deployed/page-anomaly-v1` | 32×32 썸네일 선형 오토인코더 (rank 32) | 배포, 검수 라우팅 전용 | 이상 표시 train 0/173 · val 2/39 · test 20/54 |
| `candidates/korean-exam-ppocrv5-v3-candidate` | 같은 인식기, lr 1e-5 재학습 | 배포 보류 | test 완전일치 95.54%로 v2보다 낮음 |

Model A와 Model B는 아직 학습된 모델이 없다. 코드의 `model_a`, `model_b`는 실행 경계와 규칙 기반 대역이다.
