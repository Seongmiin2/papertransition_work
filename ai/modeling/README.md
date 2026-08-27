# Modeling

학습 코드, 학습 설정, 실험별 재현 메타데이터를 관리한다. 대용량 pretrained와 checkpoint는 Git에서 제외하고 `pretrained/`와 `output/`에 둔다.

## OCR 학습

```powershell
python ai/modeling/train_korean_ocr.py `
  --paddleocr-repo output/vendor/PaddleOCR `
  --dataset output/hwp-ocr-training-v2 `
  --output output/trained-korean-ocr-<run-id> `
  --epochs 6 --batch-size 32 --learning-rate 0.00002 --warmup-epochs 1
```

기반 가중치가 없으면 `--download-pretrained`를 추가한다. 기본 위치는 `ai/modeling/pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams`다.

## 실험 JSON 필수 항목

- 고유 `model_id`와 상태
- 기반 모델과 외부 코드 revision
- dataset builder, split seed, split별 문서·행 수
- epoch, batch size, learning rate, warmup
- validation/test 지표
- artifact 상대경로와 SHA-256
- 배포 또는 보류 결정과 이유

현재 재현 가능한 실험은 `experiments/`에 있고, 비교 결과는 [`../results/README.md`](../results/README.md)에 정리한다.
