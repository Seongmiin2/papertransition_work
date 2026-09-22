# Dataset

학습 원본, 데이터 품질 규칙, 데이터셋 생성 코드를 관리한다. 원본 문서는 저작권과 개인정보 가능성 때문에 `source/`에 로컬로만 보관하고 Git에는 올리지 않는다.

## 로컬 원본 위치

```text
source/
├─ 고1-2/       고등학교 1학년 계열 HWP
├─ 고2/         고등학교 2학년 HWP
└─ references/  OCR 정답·편집 구조 참조 파일
```

파일명과 실제 문서 내용에 학생 개인정보가 없는지 확인하고, 학습에 사용할 권리가 확인된 자료만 넣는다.

## OCR line dataset 생성

기본 입력은 `source/고2/`, 기본 출력은 `output/hwp-ocr-training/`이다.

```powershell
python ai/datasets/build_hwp_line_dataset.py --input ai/datasets/source/고1-2 --input ai/datasets/source/고2 --output output/hwp-ocr-training-v2
```

생성 결과에는 train/validation/test 라벨, 이미지 crop, 문자 사전, `dataset_report.json`이 포함된다.
split은 `hwp-ocr-line-split/v2` namespace, seed, 원본 SHA-256을 해시한 80/10/10
배정이므로 입력 목록에 새 문서를 추가해도 기존 문서의 split은 바뀌지 않는다. 작은 corpus에서
validation/test가 비더라도 기존 문서를 강제로 옮겨 채우지 않는다.

기존 배정을 그대로 고정하거나 여러 문서를 같은 template family로 격리하려면 새 출력 디렉터리와
`--split-manifest`를 사용한다. 이전 `dataset_report.json`도 그대로 입력할 수 있다.

```powershell
python ai/datasets/build_hwp_line_dataset.py `
  --input ai/datasets/source/고1-2 `
  --input ai/datasets/source/고2 `
  --output output/hwp-ocr-training-v3 `
  --split-manifest ai/datasets/configs/ocr_split_map.json
```

For a training-authorized build, pass a separately verified rights manifest as
`--split-manifest`. Its exact SHA-256 is frozen into `dataset_report.json`; the trainer then
requires that same manifest through `--rights-manifest`. The file
`configs/ocr_training_rights.example.json` has `example_only=true` and placeholder evidence,
so it documents the schema but can never authorize training.

현재 pair manifest에 들어 있는 원본별 확인 항목은 다음 명령으로 작업목록을 만든다.

```powershell
python ai/datasets/prepare_ocr_training_rights_worklist.py `
  --pair-manifest output/hwp-hwpx-pairs-20260909/manifest.json `
  --source-root ai/datasets/source `
  --out output/ocr-training-rights-worklist-20260910.json
```

생성기는 각 상대 `source_ref`의 실제 SHA-256을 확인하며, unsafe 경로·symlink·누락·변조와
기존 출력 덮어쓰기를 거부한다. 결과의 `ocr-training-rights-worklist/1.0` 스키마와
`training_authorized=false`는 사람의 확인이 필요한 체크리스트일 뿐이다. 기존 candidate split은
정보로만 보존되고, `template_family`, 최종 split, 권리 근거·증빙, 확인자 항목은 모두 `null`로
남는다. 이 파일 자체를 `--rights-manifest`에 넘길 수 없으며, 담당자가 증빙을 확인한 뒤 별도의
`ocr-training-rights/1.0` manifest를 작성해야 학습 권리 검증을 통과한다.

전용 split map은 다음처럼 source SHA별 family와 선택적인 고정 split을 기록한다. 같은 family의
다른 항목에 고정 split이 있으면 split을 생략한 항목도 그 배정을 상속하고, 없으면 family 이름을
안정 해시한다.

```json
{
  "schema_version": "hwp-ocr-split-map/1.0",
  "documents": [
    {
      "sha256": "<64자리 source SHA-256>",
      "template_family": "publisher-template-v1",
      "split": "train"
    },
    {
      "sha256": "<다른 64자리 source SHA-256>",
      "template_family": "publisher-template-v1"
    }
  ]
}
```

family가 manifest에 명시되지 않은 문서는 원본 lineage만 격리한다. 파일명으로 family를 추측하지
않으며, 이 경우 cross-document template-family holdout을 보장하지 않는다. 과거
`dataset_report.json`을 고정 입력하는 것은 재현용일 뿐, 이미 발생한 upstream 학습/test
ancestry 누수를 소급해서 정리하지 않는다.

## 재현 규칙

- 문서 단위로 train/validation/test를 분리해 같은 시험지의 행이 여러 split에 섞이지 않게 한다.
- 실험 JSON에 split seed, 문서 수, 행 수, builder 경로를 기록한다.
- test split은 모델 선택에 사용하지 않고 최종 비교에만 사용한다.
- OCR이 만든 silver label과 원본 HWP/PDF에서 얻은 정답 label을 구분한다.
- 누락된 gold 원본은 임의로 대체하지 않고 설정에 미해결 상태로 남긴다.

## Model A line-risk 파생 데이터

OCR line-risk 데이터 생성에도 원본 OCR 데이터셋을 만든 실제 `--rights-manifest`가 필수다.
검증이 끝나기 전에는 OCR predictor나 출력 디렉터리를 만들지 않는다.

```powershell
python ai/datasets/build_model_a_line_risk_dataset.py `
  --dataset output/hwp-ocr-training-v2 `
  --rights-manifest path/to/verified-ocr-training-rights.json `
  --model ai/production/deployed/korean-exam-ppocrv5 `
  --out output/model-a-line-risk-dataset-<run-id> `
  --device gpu:0
```

출력 `report.json`의 `model-a-line-risk-dataset/1.1` 계약은 rights manifest SHA-256,
원본 `dataset_report.json` SHA-256, 문서 수, split별 원본 라벨·이미지 digest를 함께 고정한다.
저장소의 example manifest와 기존 `1.0` 파생 보고서는 새 가중치 학습을 승인하지 않는다.

## Model A annotation 후보 생성

기존 OCR `document_ir.json`을 EvidenceIR과 사람 검수용 ContentIR 후보로 병렬 변환한다.

```powershell
python ai/datasets/bootstrap_model_a_candidates.py `
  output/run-a/document_ir.json output/run-b/document_ir.json `
  --output output/model-a-annotation-candidates `
  --max-workers 4
```

각 원본 SHA-256별로 격리된 디렉터리가 생성되며 실패한 문서는 다른 성공 문서를 지우지 않는다. 산출물은 항상 `human_review_required=true`, `golden_eligible=false`다.

## HWPX/PDF 재제작 후보 생성

같은 HWP에서 변환한 HWPX와 PDF 페어를 이용해 `projection_source.json`, PDF 전용
`EvidenceIR`, 사람 검수용 `ContentIR` 및 `HwpDocumentPlan` 후보를 병렬 생성한다.

```powershell
python ai/datasets/build_hwpx_projection_candidates.py `
  --pairs output/hwp-hwpx-pairs-20260909 `
  --out output/hwpx-projection-candidates-20260910-v4 `
  --max-workers 4 `
  --dpi 300 `
  --capability-profile ai/knowledge/hancom/capability_profile.json `
  --knowledge-corpus output/hancom-knowledge/processed/chunks.jsonl
```

HWPX는 목표 문서의 구조·스타일을 검수하기 위한 target-side sidecar에만 사용한다.
`EvidenceIR`에는 PDF와 PDF에서 렌더링한 page/crop만 들어가며 HWPX 텍스트·locator·asset을
섞지 않는다. 다단 문서 정렬은 PDF native content order를 사용한다. 현재 Plan v1이 직접
표현하지 못하는 빈 간격 문단, 머리말·꼬리말 배치, 쪽 번호, 단 전환, 인라인 스타일과 도형
배치는 issue로 남는다.

번들은 항상 `research_only=true`, `human_review_required=true`,
`training_eligible=false`, `release_eligible=false`이며 권리 manifest가 없는 상태에서는
검수 완료 후에도 학습 데이터로 승격할 수 없다.

v4부터 불확실한 이미지를 PDF image 순번으로 확정하지 않는다. 표시 종횡비가 맞는 후보가
정확히 하나일 때만 해당 crop을 검수 후보로 연결하고, 나머지는 전체 page evidence로
되돌린다. 어느 경우든 `image_pairing_unverified` issue와 `needs_review=true`가 유지된다.
Plan의 `official_spec_refs`는 저장된 Hancom 공식 corpus를 strict schema로 읽은 뒤 HWPX
package, 문단·표·스타일, DVC validation 질의의 관련성 검색 결과만 사용한다.

## Model B 공식 근거 sidecar

후보 bundle을 바꾸지 않고 문서별 검색 결과와 `ModelBPlanGroundingEvidence`를 별도
create-only bundle로 만든다.

```powershell
python ai/datasets/build_model_b_grounding_sidecars.py `
  --candidates output/hwpx-projection-candidates-20260910-v4 `
  --knowledge-corpus output/hancom-knowledge/processed/chunks.jsonl `
  --capability-profile ai/knowledge/hancom/capability_profile.json `
  --out output/model-b-grounding-sidecars-20260910-v4 `
  --max-workers 4
```

loader는 candidate manifest, lineage, corpus, capability profile, retrieval과 공식 source/chunk
digest를 모두 다시 결합해 확인한다. 2026-09-10 v4 실제 산출물은 41문서 모두 검증됐고
5개 공식 chunk를 사용한다. 이 근거는 RAG·compiler·평가 참고자료이며 supervised label이나
권리 증명이 아니다.

## 후보 검수 초안

`CandidateReviewDraft`는 수정한 ContentIR/Plan을 후보 manifest와 원본
EvidenceIR/ContentIR/Plan의 byte SHA 및 contract SHA에 다시 묶는다.

```python
from scan2hwpx.evaluation.review_draft import (
    load_candidate_review_draft,
    verify_candidate_review_draft,
)

draft = load_candidate_review_draft("review-draft.json")
verify_candidate_review_draft(
    draft,
    candidate_root="output/hwpx-projection-candidates-20260910-v4",
)
```

이 검증은 구조와 계보의 일관성만 증명한다. 초안은
`identity_assurance=self_asserted_untrusted`, `rights_status=unverified`,
`golden_eligible=false`, `training_eligible=false`, `release_eligible=false`로 고정된다.
Golden Dataset에 넣으려면 별도의 인증된 사람 검수 증명과 독립적인 학습 권리 증명을
최종 artifact SHA에 결합해야 한다. 자동 후보나 검수 초안을 그대로 정답으로 승격하면
안 된다.

## Candidate review: immutable CLI workflow

Create revision 1 outside the candidate bundle:

```powershell
python ai/datasets/candidate_review_draft.py start `
  output/hwpx-projection-candidates-20260910-v4 candidate-id `
  --reviewer-label reviewer-local `
  --expected-manifest-sha256 $manifestSha256 `
  --expected-lineage-id $lineageId `
  --out output/reviews/candidate-id.r1.json
```

Apply a typed patch from stdin to a new path; the prior revision is never overwritten:

```powershell
'{"expected_draft_revision":1,"operations":[{"op":"set_needs_review","node_id":"text-1","needs_review":false}]}' | `
  python ai/datasets/candidate_review_draft.py patch `
  output/hwpx-projection-candidates-20260910-v4 output/reviews/candidate-id.r1.json `
  --expected-manifest-sha256 $manifestSha256 `
  --expected-lineage-id $lineageId `
  --out output/reviews/candidate-id.r2.json
```

Complete an exact ready revision into another immutable file. Completion changes only
`status`, `draft_revision`, and `updated_at`; it fails while any registered issue is pending
or any ContentIR node still has `needs_review=true`.

```powershell
'{"schema_version":"candidate-review-completion/1.0","expected_draft_revision":2}' | `
  python ai/datasets/candidate_review_draft.py complete `
  output/hwpx-projection-candidates-20260910-v4 output/reviews/candidate-id.r2.json `
  --expected-manifest-sha256 $manifestSha256 `
  --expected-lineage-id $lineageId `
  --out output/reviews/candidate-id.r3.complete.json
```

Reopen through the sanitized Electron view, or run a metadata-only verification:

```powershell
python ai/datasets/candidate_review_draft.py view `
  output/hwpx-projection-candidates-20260910-v4 output/reviews/candidate-id.r3.complete.json `
  --expected-manifest-sha256 $manifestSha256 --expected-lineage-id $lineageId
python ai/datasets/candidate_review_draft.py verify `
  output/hwpx-projection-candidates-20260910-v4 output/reviews/candidate-id.r3.complete.json `
  --expected-manifest-sha256 $manifestSha256 --expected-lineage-id $lineageId
```

These commands never modify the candidate bundle and never grant Golden, training, or
release eligibility. Every patch or completion output must use a new path outside the
candidate root.

## 완료된 Candidate review → Model B

Model B는 base 후보가 아니라 완료된 사람 검수 revision의 reviewed ContentIR/Plan만 받는다.
다음 명령은 candidate bundle, 완료 review artifact, grounding sidecar와 공식 corpus/capability를
다시 검증한 뒤 create-only handoff envelope를 만든다.

```powershell
python ai/datasets/start_candidate_to_model_b_handoff.py `
  output/hwpx-projection-candidates-20260910-v4 `
  output/reviews/$documentId.r3.complete.json `
  output/model-b-grounding-sidecars-20260910-v4 `
  output/hancom-knowledge/processed/chunks.jsonl `
  ai/knowledge/hancom/capability_profile.json `
  --reviewer-label reviewer-local `
  --expected-review-sha256 $reviewSha256 `
  --expected-review-revision 3 `
  --expected-manifest-sha256 $manifestSha256 `
  --expected-lineage-id $lineageId `
  --out output/model-b-handoffs/$documentId.json
```

handoff는 완료 review artifact SHA/revision, candidate manifest/lineage, sidecar manifest,
corpus/capability/retrieval/grounding digest를 고정한다. 그 안의 Model B review revision도
`self_asserted_untrusted`, `unverified`, Golden/학습/배포 부적격으로 유지된다.

Model B Plan 검수의 persisted API/CLI는 ContentIR, base Plan, grounding evidence 세 artifact를
매 revision 다시 결합한다. patch는 page layout/style/flow/spec ref의 typed operation만 받고,
완료는 exact revision·source·grounding·reviewed Plan digest의 4-way CAS가 맞을 때만 새 파일을
만든다. 완료된 revision은 patch하거나 다시 완료할 수 없다.
