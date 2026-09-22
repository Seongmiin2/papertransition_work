# Modeling

학습 코드, 학습 설정, 실험별 재현 메타데이터를 관리한다. 대용량 pretrained와 checkpoint는 Git에서 제외하고 `pretrained/`와 `output/`에 둔다.

Model A/B 기반 모델의 provisional 후보, 8GB 제약과 승격 gate는 [ADR 0002](../../docs/architecture/0002-base-model-bakeoff.md)에 정의한다.

Model A의 provider-neutral 경계는 `src/scan2hwpx/model_a/inference.py`에 있다. 실제 이미지
bytes·SHA-256·source와 해당 페이지 EvidenceIR, 정확한 model ID/revision/artifact digest를
한 페이지 요청에 고정하고 strict `ModelAPageAnalysis` 응답만 검증한다. 페이지별 결과는
입력 순서와 무관하게 page number 순으로 모아 revision 1 ContentIR을 만드는 document
barrier를 통과한다. 현재 ContentIR 1.0에는 페이지 간 관계 필드가 없으므로 barrier는 관계를
추측하지 않고 순서대로 결합만 한다.

실험용 block-role v2 경계와 로컬 Ollama adapter는 각각
`src/scan2hwpx/model_a/block_role_vector.py`와 `src/scan2hwpx/model_a/ollama.py`에 있다.
실행 CLI는 `ai/modeling/run_model_a_block_role_vector.py`, 미검증 proxy 비교 CLI는
`ai/modeling/compare_model_a_block_role_proxy.py`다. 현재 8쪽 표본에서는 `num_ctx=8192`가
18,241-token 1쪽 요청을 수용하지 못했고 `num_ctx=32768`에서 8/8쪽이 완료되었다. 따라서
32K는 시험한 설정 중 관측된 최소 성공값일 뿐 일반적인 최소 보장이 아니다. 사용한
`qwen3.5:4b-q4_K_M`은 로컬 Ollama base artifact이며 프로젝트가 학습하거나 승격한 Model A
checkpoint가 아니다. 사람 Gold가 없는 실행·proxy 지표·조립 산출물은 모두
research-only/noneligible이고 release 또는 정확도 증거로 사용할 수 없다. 고정 실행 기록은
[`experiments/model_a_block_role_v2_full8_20260917.json`](experiments/model_a_block_role_v2_full8_20260917.json)에 있다.

Model B의 provider-neutral 요청·응답 경계는
`src/scan2hwpx/model_b/inference.py`에 있다. verified handoff와 공식 grounding을 canonical
prompt로 묶고 HwpDocumentPlan 응답을 자동수정 없이 검증하지만, 아직 provider adapter,
로컬 base weight, fine-tuned checkpoint는 없다. 이 경계를 통과한 단일 응답도 사람 Golden
또는 release 증거가 아니며 모든 eligibility가 `false`다.

## OCR 학습

```powershell
python ai/modeling/train_korean_ocr.py `
  --paddleocr-repo output/vendor/PaddleOCR `
  --dataset output/datasets/hwp-ocr-training-v2 `
  --rights-manifest path/to/verified-ocr-training-rights.json `
  --rights-evidence-root path/to/verified-evidence-root `
  --output output/training/trained-korean-ocr-<run-id> `
  --epochs 6 --batch-size 32 --learning-rate 0.00002 --warmup-epochs 1
```

위 명령은 인자 형식을 보여 주는 예시이며 standalone CLI만으로는 학습을 승인할 수 없다.
실제 실행기는 인증 시스템이 만든 `AuthenticatedAttestorContext`를 Python entrypoint에
주입해야 한다. 그 context는 권리 확인자 집합, 별도의 데이터 파생 확인자, 신뢰하는
`dataset_report.json` SHA-256을 고정한다. 인증 context가 없거나 evidence artifact byte,
source/image inventory, split-family, label/dictionary digest가 하나라도 다르면 pretrained
다운로드·GPU 초기화·출력 생성 전에 실패한다. 저장소 example manifest는 스키마 문서일
뿐 학습을 승인하지 않는다.

기반 가중치가 없으면 `--download-pretrained`를 추가한다. 기본 위치는 `ai/modeling/pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams`다.
다운로드 시작 전 `.NAME.download-incomplete.json` transaction marker를 create-only로 만들며,
최종 artifact digest와 파일 identity가 확인된 경우에만 제거한다. marker가 남으면 다음 실행은
해당 pretrained를 격리 상태로 보고 거부한다. 운영자는 destination과 남은 partial을 먼저
격리해 신뢰된 원본 digest와 대조하거나 삭제한 뒤 marker를 수동 제거해야 한다.

## Model A line-risk 보조 head

`train_model_a_line_risk.py`도 `--rights-manifest`, `--rights-evidence-root`,
`--source-dataset`을 필수로 받고 동일한 인증 context 주입을 요구한다. 파생
데이터 보고서의 권리 해시만 신뢰하지 않고 원본 OCR 데이터의 문서 inventory,
template-family split, 라벨·문자 사전과 이미지 digest를 다시 검증한 뒤에만 임시 출력과
가중치를 만든다. 기존 `model-a-line-risk-dataset/1.0`은 이 결속이 없으므로 새 학습 입력으로
사용할 수 없다. 명령과 역사적 실험 판정은 [`MODEL_A_LINE_RISK.md`](MODEL_A_LINE_RISK.md)에
있다.

## 실험 JSON 필수 항목

- 고유 `model_id`와 상태
- 기반 모델과 외부 코드 revision
- dataset builder, split seed, split별 문서·행 수
- epoch, batch size, learning rate, warmup
- validation/test 지표
- artifact 상대경로와 SHA-256
- 배포 또는 보류 결정과 이유

현재 재현 가능한 실험은 `experiments/`에 있고, 비교 결과는 [`../results/README.md`](../results/README.md)에 정리한다.
