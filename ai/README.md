# AI workspace

이 폴더는 데이터 준비부터 실제 앱 배포까지의 AI 수명주기를 한곳에서 추적한다.

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

1. 배포 모델과 같은 고정 test split에서 비교한다.
2. 핵심 정확도와 정규화 편집 유사도가 모두 기존 배포 모델 이상이다.
3. 가중치 checksum과 학습 설정이 실험 JSON에 기록되어 있다.
4. 데스크톱 앱에서 대표 PDF 변환 smoke test를 통과한다.
5. 배포 경로 변경 후 Python 테스트, TypeScript 테스트와 빌드가 통과한다.

현재 OCR v3는 1·3번은 만족했지만 test 지표가 v2보다 낮아 승격하지 않았다. 자세한 비교는 [`results/README.md`](results/README.md)에 있다.

## 새 실험을 남기는 순서

1. `datasets/`에서 데이터 버전과 split을 고정한다.
2. `modeling/`에서 출력 경로를 `output/<run-id>`로 지정해 학습한다.
3. `modeling/experiments/<run-id>.json`에 입력, 파라미터, 지표, checksum을 기록한다.
4. `results/performance_log.csv`에 비교 지표를 추가한다.
5. 승격 또는 보류 사유를 실험 JSON과 결과 문서에 함께 기록한다.
