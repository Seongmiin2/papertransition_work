# ADR 0002: Model A/B 기반 모델 provisional bake-off

- **Status:** Provisional / Bake-off required
- **Date:** 2026-09-10
- **Scope:** Model A/B 기반 모델, 로컬 8GB 실행 조건, 학습 차단 조건, 승격 평가
- **Supersedes:** 없음. [ADR 0001](0001-model-first-document-factory.md)의 파이프라인과 release gate를 구체화한다.

## 결정

아직 어떤 모델도 제품용으로 확정 채택하지 않는다. 권리와 사람 검수가 완료된 고정 gold set에서 같은 입력, 스키마, 디코딩 제약과 측정 조건으로 비교하기 위한 provisional 우선순위만 정한다.

- **Model A primary:** [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- **Model A fallback:** [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- **EvidenceIR front-end fallback:** [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6). Model A의 의미 분석을 대체하지 않는다.
- **Model B primary:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Model B fallback:** [Kanana-1.5-2.1b-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)

primary는 아래 hard gate를 모두 통과하고 fallback 대비 명확한 품질 또는 운영 이점이 확인될 때만 champion이 된다. 어느 후보도 gate를 통과하지 못하면 확정 모델은 없는 상태를 유지한다. 제작사가 공개한 benchmark는 후보를 줄이는 근거일 뿐 이 제품의 한국어 문서 성능 보장이 아니다.

## 후보 근거와 8GB 판정

`공식 사실`과 `프로젝트의 보수적 판단`을 분리한다. 파일 크기는 모델 저장소의 BF16 weight를 가리키며 실제 VRAM에는 KV cache, activation, image tensor와 runtime overhead가 추가된다.

| 역할 | 후보 | 공식 사실 | 프로젝트의 보수적 판단 |
|---|---|---|---|
| Model A primary | Qwen3.5-2B | native multimodal 2B, 262K context이며 [공식 Qwen 소개](https://qwen.ai/blog?email_hash=23463b99b62a72f26ed677cc556c44e8&id=qwen3.5)에 Korean 지원이 명시되어 있다. [공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-2B)의 동일 표에서 MMLongBench-Doc 45.4, OmniDocBench 1.5 79.8을 보고한다. BF16 weight는 [약 4.55GB](https://huggingface.co/Qwen/Qwen3.5-2B/tree/main)다. | 8GB에서는 4-bit, batch 1, 페이지 단위, 제한된 image/token budget의 추론만 기본 경로로 본다. BF16은 운영 여유가 부족하다. QLoRA는 짧은 입력과 vision encoder freeze 조건의 smoke test만 가능성이 있으며 사전 보장은 없다. 신형 runtime 의존성은 [공식 ms-swift recipe](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)대로 고정하고 native Windows가 아니라 WSL2/Linux에서 검증한다. |
| Model A fallback | Qwen3-VL-2B-Instruct | [공식 저장소](https://github.com/QwenLM/Qwen3-VL)는 OCR/KIE, text와 layout position, document parsing을 명시한다. BF16 weight는 [약 4.26GB](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main)다. | Qwen3.5 runtime이 불안정하거나 gold set에서 열세이면 사용한다. 4-bit 추론은 8GB 후보지만 QLoRA는 batch 1과 강한 token/image 제한이 필요한 조건부 실험이다. [공식 4B LoRA 예](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md)는 2×21GiB를 사용하므로 일반 LoRA 설정을 8GB에 적용하지 않는다. |
| EvidenceIR front-end fallback | PaddleOCR-VL-1.6 | [공식 모델 카드](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)는 약 0.9B, Korean을 포함한 109개 언어, 표·수식·차트 parsing과 JSON/Markdown 출력을 명시한다. BF16 weight는 [약 1.92GB](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/tree/main)다. [공식 pipeline 문서](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)는 layout detector와 VLM을 함께 사용하는 full pipeline을 요구한다. | 8GB 추론 여유가 가장 크다. 다만 작은 language core의 page parser이므로 ContentIR 의미 관계를 만드는 Model A로 단독 승격하지 않는다. 공식 [SFT 안내](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/docs/paddleocr_vl_sft_zh.md)의 기준은 A800-80G full SFT이므로 로컬 학습 가능성을 주장하지 않는다. |
| Model B primary | Qwen3-4B-Instruct-2507 | text-only non-thinking 4B, 262K context이며 [공식 모델 카드](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)는 instruction following, multilingual, tool use 개선과 IFEval 83.4, BFCL-v3 61.9를 보고한다. BF16 weight는 [약 8.06GB](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/main)다. | BF16은 8GB 실사용 대상이 아니다. 4-bit 추론만 bake-off한다. [공식 ms-swift 예](https://github.com/modelscope/ms-swift)의 4B LoRA 메모리는 약 13GB이므로 일반 LoRA는 제외한다. 4-bit QLoRA도 짧은 plan 입력, batch 1의 smoke test로만 취급한다. non-thinking 출력은 HwpDocumentPlan JSON 계약을 단순하게 만드는 이점이 있다. |
| Model B fallback | Kanana-1.5-2.1b-instruct-2505 | [공식 모델 카드](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)는 2.1B, Korean/English, 32K context와 IFEval 68.61, FunctionChatBench 53.70, KoMT-Bench 6.54를 보고한다. BF16 weight는 [약 4.63GB](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505/tree/main)다. | 4-bit 추론은 primary보다 여유가 있고, 짧은 입력의 QLoRA도 후보 중 상대적으로 현실적이다. 그래도 8GB 적합성은 측정 전 추정이며 full fine-tuning은 제외한다. formal Korean과 계획 정확도가 Qwen보다 좋거나 비용·지연 목표를 Qwen이 못 맞출 때 fallback으로 승격한다. |

## 라이선스와 배포 경계

이 절은 기술적 후보 선별이며 법률 자문이 아니다. 배포 전에는 사용 형태와 모델의 정확한 revision을 기준으로 별도 검토한다.

- 선택한 Qwen 후보는 각각 [Qwen3.5-2B Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE), [Qwen3-VL-2B-Instruct Apache-2.0](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/LICENSE), [Qwen3-4B-Instruct-2507 Apache-2.0](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE)이다.
- PaddleOCR-VL-1.6도 [Apache-2.0](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/LICENSE)이다.
- fallback으로 지정한 **정확한** Kanana-1.5-2.1b-instruct-2505 revision은 [Apache-2.0](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505/blob/main/LICENSE)이다. 이름이 비슷한 최신 Kanana checkpoint로 자동 교체하지 않는다.
- [Kanana-1.5-v-3b-instruct의 custom license](https://huggingface.co/kakaocorp/kanana-1.5-v-3b-instruct/blob/main/LICENSE)는 표시·명명 의무와 일부 제3자 API, cloud, SI/on-prem, embedded 제공에 관한 별도 상업 라이선스 조건이 있어 기본 후보에서 제외한다.
- [EXAONE 4.0 공식 라이선스](https://github.com/LG-AI-EXAONE/EXAONE-4.0/blob/main/LICENSE)는 NC 조건을 명시하므로 향후 상업 서버의 기본 후보로 사용하지 않는다.

Apache-2.0도 notice, license 사본, 수정 표시 등 의무가 사라지는 것은 아니다. 모델 registry에는 model ID, exact revision, license digest와 검토 결론을 함께 고정한다.

## 구조화 JSON 계약

모델이 JSON 예시를 잘 생성하는 것과 HwpDocumentPlan 계약 준수는 다르다. 모든 Model A/B 실행에 다음을 적용한다.

1. EvidenceIR, ContentIR, HwpDocumentPlan의 versioned JSON Schema를 사용하고 `additionalProperties: false`를 기본으로 한다.
2. 지원 runtime에서는 [vLLM structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)와 같은 JSON Schema constrained decoding을 사용한다.
3. 출력은 별도 schema validator와 reference validator를 모두 통과해야 한다. 존재하지 않는 `evidence_id`, `content_ref`, `spec_ref`는 즉시 block한다.
4. 파서가 원문이나 참조를 조용히 보정하지 않는다. 실패 시 제한된 재시도 후 `review` 또는 `block`으로 종료한다.
5. raw response, schema version, decoding 설정, model revision, adapter digest와 validator 결과를 보존한다.

constrained decoding은 문법 유효성을 높이지만 의미적 정확성이나 근거 충실성을 보장하지 않는다. 그 부분은 gold-set gate로 판정한다.

## 데이터와 학습 차단 상태

2026-09-10 현재:

- 사람 검수와 학습 권리가 모두 입증된 Model A/B supervised target은 **0건**이다.
- 승격 가능한 새 Model A/B weight 또는 adapter는 **0개**다.
- 후보 41문서, 485페이지는 provenance, 권리와 사람 검수가 끝나기 전에는 training, preference optimization 또는 자동 파생 label 생성에 사용하지 않는다.
- 공식 한컴/HWPX/OWPML 문서는 **RAG corpus, `spec_refs`, capability profile, compiler/validator 근거**다. 문서 문장 자체를 supervised label 또는 target completion으로 사용하지 않는다.

따라서 현재 `training_status`는 `blocked_no_eligible_targets`다. 다음 조건을 모두 만족하기 전에는 base weight 다운로드를 수반하는 학습 job이나 output directory를 만들지 않는다.

1. source별 license/consent, 허용 목적, SHA-256과 reviewer attestation이 고정되어 있다.
2. 사람이 검증한 input/target pair가 있고 target provenance가 원본과 연결되어 있다.
3. 문서와 template family 단위 train/dev/test 격리 검사가 통과한다.
4. immutable gold set과 metric implementation이 먼저 versioned되어 있다.

권리가 승인된 뒤에도 485페이지 규모로 pretraining 또는 full fine-tuning을 정당화하지 않는다. 먼저 zero/few-shot과 RAG baseline을 평가하고 반복되는 오류가 adapter로 해결 가능하다는 증거가 있을 때만 QLoRA SFT를 시도한다. QLoRA는 4-bit frozen base에 adapter만 학습하고 NF4, double quantization, gradient checkpointing을 사용하는 방향이며, 근거는 [Hugging Face PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)와 [bitsandbytes guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes)에 고정한다.

## Gold set과 승격 gate

아래 수치는 후보 성능에 대한 예측이 아니라 이 프로젝트의 provisional 승격 기준이다.

### Gold set 고정 조건

- 평가 권리가 명시된 **실제 스캔** 중 `max(50문서, eligible 문서의 20%)`와 `max(500페이지, eligible 페이지의 20%)`를 모두 만족하는 크기를 사용한다. 이 수량을 확보하지 못하면 `insufficient_evidence`이며 승격하지 않는다.
- native PDF, scan, 표·수식 중심, 그림·도형 혼합의 네 주요 slice에 각각 최소 3문서를 포함한다.
- 문서와 template family를 단위로 격리하며 page-level random split을 금지한다.
- 모든 target은 두 명의 검수와 불일치 adjudication을 거친다. 원본 hash, annotation revision과 reviewer를 고정한다.
- 각 모델 후보와 adapter는 이 gold set을 학습, prompt 선택 또는 threshold tuning에 사용하지 않는다. threshold는 별도 dev set에서 정한다.
- 표 topology, 수식/자산 연결처럼 gold denominator가 30건 미만인 metric은 `insufficient_evidence`로 판정하고 해당 기능의 자동 출고 승격을 보류한다.

### Model A hard gate

기계 판정의 권위 있는 수치와 metric definition은
[`model_release_gates.json`](../../ai/evaluation/model_release_gates.json)이다. 아래 표의 추가
진단 항목은 그 기준을 완화할 수 없으며, 수치를 바꾸려면 gate set을 새 버전으로 올려야 한다.

| Metric | 승격 기준 |
|---|---:|
| ContentIR JSON Schema valid rate | 100% |
| `evidence_id` reference validity와 content evidence coverage | 100% |
| 근거 없는 content node 또는 의미를 바꾸는 누락·중복 | 0건 |
| semantic block detection micro-F1 | ≥ 0.97 |
| semantic role classification macro-F1 | ≥ 0.97 |
| reading-order document exact-match | ≥ 0.99 |
| normalized Kendall agreement mean | ≥ 0.995 |
| table topology exact-match | ≥ 0.95 |
| formula detection recall | ≥ 0.98 |
| normalized formula similarity | ≥ 0.97 |
| Korean text preservation corpus CER | ≤ 0.5% |
| critical-token edit error rate | ≤ 0.1% |
| native PDF/scan slice CER | 별도 기록하며 전체 CER hard gate를 대체하지 않음 |
| high-risk error의 `review`/`block` recall | ≥ 0.99 |
| confidence expected calibration error | ≤ 0.05 |

모든 전체 지표뿐 아니라 네 주요 slice 각각에서 schema/reference/unsupported-content hard gate를 만족해야 한다. aggregate 평균으로 위험 slice 실패를 숨길 수 없다.

### Model B hard gate

| Metric | 승격 기준 |
|---|---:|
| HwpDocumentPlan JSON Schema valid rate | 100% |
| 유효한 `content_ref` coverage | 100% |
| 잘못된 참조, 본문 literal 재생성 또는 ContentIR 의미 변경 | 0건 |
| 필요한 공식 근거에 대한 `spec_ref` precision | 100% |
| gold가 요구하는 `spec_ref` recall | ≥ 0.95 |
| 지원 불가 capability의 `review`/`block` recall | 100% |
| 필수 문서 요소와 page/section 구조 coverage | 100% |
| deterministic compiler와 HWPX validator 성공률 | 100% |
| 지원 대상 한글 열기→저장→재열기 round-trip | 100% |
| 사람 검수에서 수정 없음 또는 minor-only 판정 | ≥ 0.95 |
| 내용 손실, 잘못된 편집 개체, 근거 없는 설계의 critical defect | 0건 |

공식 한컴 문서를 retrieval한 baseline을 먼저 평가한다. fine-tuning 후보는 같은 retrieval snapshot, capability profile과 decoding constraints로 base 대비 모든 hard gate를 유지하고 사람 수정 시간을 10% 이상 줄일 때만 승격한다.

### 로컬 실행 gate

- RTX 5060 Ti 8GB에서 4-bit, batch 1, 고정 image DPI·pixel budget·max sequence length로 gold set 전체를 OOM 없이 완료한다.
- 측정 peak VRAM은 7.5GiB 이하여야 한다. runtime, quantization method, driver/CUDA, peak VRAM, p50/p95 latency를 결과 JSON에 기록한다.
- 8GB에서 full fine-tuning과 일반 LoRA는 승인 대상이 아니다. QLoRA도 첫 20-step smoke test와 한 epoch dry run이 OOM 없이 끝나고 gradient/validation이 유한값일 때만 제한 실험을 계속한다.

## 실행 순서

1. 권리 ledger, schema, gold annotation과 metric implementation을 고정한다.
2. training 없이 Model A primary/fallback과 PaddleOCR-VL front-end 조합을 동일 gold set에서 비교한다.
3. Model B는 동일한 공식 한컴 RAG snapshot으로 primary/fallback zero/few-shot baseline을 비교한다.
4. 모든 결과를 model revision, quantization, prompt, schema, retrieval corpus digest와 함께 저장한다.
5. 권리와 label gate가 해소되고 baseline 오류가 확인된 뒤에만 Model A는 vision encoder를 먼저 freeze한 QLoRA, Model B는 schema/tool/grounding 행동만 학습하는 QLoRA를 시도한다. DPO/GRPO는 독립 preference data가 충분하기 전에는 수행하지 않는다.
6. untouched gold set과 out-of-domain, missing/contradictory evidence set에서 회귀를 검사한다. hard gate 하나라도 실패하면 승격하지 않는다.

## 서버 재검증

로컬 8GB 결과를 향후 서버 용량이나 성능으로 외삽하지 않는다. 배포 대상 Linux GPU가 정해지면 고정 container, exact model/adapter revision, 동일 gold set과 retrieval snapshot으로 다시 측정한다.

- 모든 정확도 hard gate를 다시 통과하고 로컬 기준 대비 어떤 지표도 0.5 percentage point 넘게 하락하지 않아야 한다.
- 목표 concurrency에서 OOM, worker crash와 schema/reference bypass가 0건이어야 한다.
- batch size별 peak VRAM, p50/p95 latency, throughput, queue wait와 비용을 측정한다.
- 제품 SLO와 최대 context/image budget이 승인되기 전에는 server champion으로 승격하지 않는다.
- quantization 또는 runtime 변경은 새 model bundle로 취급해 전체 gate를 다시 실행한다.

## 결과

이 결정은 지금 다운로드하거나 학습할 모델을 확정하는 문서가 아니라, 가장 작은 현실적 후보와 탈락 조건을 고정하는 bake-off 계약이다. 현재 상태는 **Qwen3.5-2B와 Qwen3-4B-Instruct-2507이 provisional primary이지만, eligible target 0건과 미구축 gold set 때문에 학습과 최종 채택이 차단된 상태**다.
