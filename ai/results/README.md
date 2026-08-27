# Results

모델 성능과 배포 판단을 시간 순서로 누적한다. 원시 수치는 `performance_log.csv`, 상세 재현 정보는 `../modeling/experiments/`, 과거 설명 보고서는 `history/`에 둔다.

## OCR 성장 기록

| 날짜 | 모델 | 평가 데이터 | 완전일치 | 편집 유사도 | 상태 |
|---|---|---|---:|---:|---|
| 2026-08-26 | 공식 pretrained | validation 2문서 | 84.09% | 97.26% | 기준선 |
| 2026-08-26 | 1차 fine-tune | test 2문서 | 96.44% | 98.87% | 실험 완료 |
| 2026-08-26 | v2 continued fine-tune | test 4문서, 2,921행 | 95.5701% | 98.1530% | 배포 |
| 2026-08-27 | v3 lr=1e-5 | 같은 test 4문서, 2,921행 | 95.5357% | 98.1052% | 배포 보류 |

1차와 v2는 데이터 분할이 달라 수치를 직접적인 하락으로 해석하지 않는다. v2와 v3는 같은 test split이므로 직접 비교할 수 있다. v3는 완전일치가 0.0343%p 낮아 2,921행 중 정확히 맞힌 행이 한 줄 적고, 편집 유사도도 0.0478%p 낮아 v2를 유지했다.

## 기록 방법

벤치마크 JSON을 기존 CSV 형식으로 추가할 때:

```powershell
python ai/results/record_metrics.py --report output/<run>/ocr_benchmark.json --component ocr --model-config-id <model-id> --notes "<decision context>"
```

같은 날짜·component·model·metric·split 조합은 중복 추가하지 않는다. 모델을 승격하거나 보류할 때는 수치뿐 아니라 판단 이유도 실험 JSON에 남긴다.
