# Model A line-risk head

이 head는 OCR 문자열을 수정하거나 `ContentIR`을 생성하지 않는다. 배포 OCR의 줄 단위 출력과
confidence를 받아 해당 줄이 원문과 다를 위험을 점수화하는 Model A 보조 모델이다. 전체 Model A,
OCR recognizer 또는 Model B로 표시하면 안 된다.

## 새 학습 실행 절차

먼저 배포 OCR로 원본 line crop을 추론하고 split 간 동일 이미지 및 동일
`(NFC(observed_text), NFC(target_text))` 쌍을 모두 제거한다.

```powershell
python ai/datasets/build_model_a_line_risk_dataset.py `
  --dataset output/hwp-ocr-training-v2 `
  --rights-manifest path/to/verified-ocr-training-rights.json `
  --model ai/production/deployed/korean-exam-ppocrv5 `
  --out output/model-a-line-risk-dataset-<run-id> `
  --device gpu:0 `
  --batch-size 32
```

배포 OCR이 학습한 source `train` split은 stacking 누출을 일으키므로 risk head 학습에 사용할 수
없다. OCR이 보지 않은 source `validation`을 risk-train, source `test`를 risk-validation으로만
사용한다. 이 실행에는 독립 risk-test가 없다.

```powershell
python ai/modeling/train_model_a_line_risk.py `
  --dataset output/model-a-line-risk-dataset-<run-id> `
  --source-dataset output/hwp-ocr-training-v2 `
  --character-dict output/hwp-ocr-training-v2/korean_exam_dict.txt `
  --rights-manifest path/to/verified-ocr-training-rights.json `
  --out output/trained-model-a-line-risk-<run-id> `
  --train-split validation `
  --validation-split test `
  --upstream-training-split train `
  --device gpu:0 `
  --epochs 40 `
  --batch-size 256 `
  --learning-rate 0.001 `
  --target-recall 0.98 `
  --seed 20260909
```

두 단계 모두 같은 실제 rights manifest가 필수다. 파생 데이터셋은
`model-a-line-risk-dataset/1.1` 보고서에 manifest, 원본 `dataset_report.json`, 원본 라벨과
이미지 digest를 결속한다. 트레이너는 보고서의 자기주장을 그대로 신뢰하지 않고
`--source-dataset`의 원본 inventory, template-family split, 등록된 라벨·문자 사전과 이미지
digest를 다시 계산한다. 이 검증이 끝나기 전에는 predictor, Paddle 모델, 임시 출력 디렉터리를
만들지 않는다.

## 2026-09-09 판정

아래 결과와 experiment JSON은 당시 실행을 보존한 역사적 기록이다. 권리 검증 결속이 없는 기존
`1.0` 파생 데이터 보고서는 현재 트레이너가 의도적으로 거부하므로, 검증된 권리 manifest로
`1.1` 데이터셋을 새로 만들기 전에는 재실행할 수 없다.

- dataset SHA-256: `eb185eb9679061976c7fb0ec7c747e2690f50f8e324a1745f669c93a4281b832`
- risk-train: 2,584줄, 오류 292줄
- risk-validation: 2,101줄, 오류 193줄
- best epoch: 39/40
- checkpoint SHA-256: `5211f8e5ecd26a4352926a96721e96d7d0cad316ba7c839037dd1e3c4a7ba7db`
- learned AP / confidence-only AP: `0.37748 / 0.25997`
- learned ROC-AUC / confidence-only ROC-AUC: `0.78862 / 0.78825`
- 오류 recall 약 98.45%에서 learned precision / confidence-only precision:
  `9.55% / 13.48%`

순위 전체의 AP는 개선됐지만 서비스가 요구하는 고-recall 구간의 precision과 F1이 baseline보다
낮으므로 배포를 거부했다. 판정과 artifact hash는
`ai/modeling/experiments/model_a_line_risk_20260909_rejected.json`에 고정한다.

다음 비교 전에 source-document lineage와 upstream OCR 학습 ancestry를 데이터 행에 결속하고,
학습 사용권이 확인된 독립 문서 split을 추가해야 한다. 그 전에는 이 checkpoint를 연구 재현 외에
사용하지 않는다.
