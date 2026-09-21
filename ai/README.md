# AI workspace

이 폴더는 데이터 준비부터 실제 앱 배포까지의 AI 수명주기를 한곳에서 추적한다.

> 아래 OCR 이력은 현재 제품의 인식기 자산에 대한 기록이다. 새 문서 재제작 제품의 Model A/B 계약과 승격은 [구속력 있는 ADR](../docs/architecture/0001-model-first-document-factory.md) 및 [새 평가 신뢰 경계](evaluation/README.md)가 우선하며, 집계 점수만으로는 어떤 Model A/B도 승격하지 않는다.

```text
datasets ──> modeling ──> results ──> production/deployed
                 │             │
                 └─────────────┴──> production/candidates
```

## 단계별 책임

| 단계 | 책임 | Git에 남기는 것 | Git에서 제외하는 것 |
|---|---|---|---|
| Dataset | 원본 수집, 분할, 라벨·품질 기준 | 생성 코드, 설정, split seed, 문서·행 수 | 원본 문서, crop, 렌더 PDF |
| Modeling | 학습 설정과 실행 재현 | 학습 코드, 설정, 실험 JSON, checksum | pretrained, checkpoint, optimizer state |
| Results | 모델 간 비교와 의사결정 | 누적 CSV, 성장표, 보고서 | 대용량 디버그 이미지 |
| Production | 앱에서 실제 추론 | 승인 inference artifact, 모델 카드 | 중간 checkpoint |

## 모델 승격 기준

후보는 다음 조건을 모두 만족할 때만 `production/deployed/`로 승격한다.

1. 모든 조상 학습 데이터와 겹치지 않는 문서·template-family 고정 test split에서 비교한다.
2. 핵심 정확도와 정규화 편집 유사도가 모두 기존 배포 모델 이상이다.
3. 가중치 checksum과 학습 설정이 실험 JSON에 기록되어 있다.
4. 데스크톱 앱에서 대표 PDF 변환 smoke test를 통과한다.
5. 배포 경로 변경 후 Python 테스트, TypeScript 테스트와 빌드가 통과한다.
6. 데이터 권리가 검증되고 실제 스캔 최소 50문서·500쪽 평가를 통과한다.

현재 OCR v2/v3의 공통 test는 조상 학습 데이터와 겹치므로 새 1번 기준을
만족하지 않는다. 앱은 호환성을 위해 v2를 계속 읽지만 확정 모델로 간주하지 않는다.
자세한 비교는 [`results/README.md`](results/README.md)에 있다.

2026-09-10 감사 기준으로 로컬 HWP 41건에는 문서별 학습 권리 attestation이 없고,
template-family가 격리된 실제 스캔 held-out은 0문서·0쪽이다. Model B의 검수 전 후보
41건 역시 모두 `training_eligible=false`라서 학습 가능한 target은 0건이다.
권리 증명, 독립 split, 사람 승인 target이 만들어지기 전에는 새 가중치를 실행하거나
기존 수치를 release 성능으로 재사용하지 않는다.

같은 41건에는 공식 한컴 corpus를 digest로 고정한 Model B grounding sidecar가 준비되어
있고, 완료된 Candidate review에서 reviewed ContentIR/Plan만 받는 handoff와 불변 Plan 검수
완료 전이도 구현되어 있다. 이들은 모델 입력·검수 계보를 안전하게 만드는 기반이지 정답
데이터가 아니다. 공식 문서 10개·2,554 chunks는 RAG/compiler/eval 근거로만 사용하며
supervised target으로 취급하지 않는다.

## 새 실험을 남기는 순서

1. `datasets/`에서 데이터 버전과 split을 고정한다.
2. `modeling/`에서 출력 경로를 `output/<run-id>`로 지정해 학습한다.
3. `modeling/experiments/<run-id>.json`에 입력, 파라미터, 지표, checksum을 기록한다.
4. `results/performance_log.csv`에 비교 지표를 추가한다.
5. 승격 또는 보류 사유를 실험 JSON과 결과 문서에 함께 기록한다.
