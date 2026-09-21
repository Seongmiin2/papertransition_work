# 모델 평가와 승격 정책

이 디렉터리는 모델 점수를 기록하는 곳이 아니라, 어떤 모델 번들이 어떤 데이터와 평가기를 거쳐 출고 가능한지 증명하는 신뢰 경계다.

현재 고정된 순서는 다음과 같다.

```text
Golden Dataset manifest
  + Model bundle manifest
  + versioned release gates
  + measurement evidence
  -> release audit
```

`model_release_gates.json`은 초기 품질 하한을 고정한다.
`model_release_gates.model-first-v2.json`은 기존 15개 지표 gate를 그대로 보존하고
`model_a.high_risk_review_recall >= 0.99`와 `model_a.confidence_ece <= 0.05`를
추가한 총 17개 지표 gate의 별도 정책이다. `model_bundle.example.json`은 스키마 예제일
뿐이며, 현재 Model A와 Model B를 모두 `placeholder`로 표시한다. 실제 학습·검증 artifact가 없는
모델을 배포 모델로 선언하면 안 된다.

## 현재 안전 상태

현재 `MeasurementReport`가 허용하는 증거 수준은 `aggregate_only`뿐이다. 집계 숫자는 다음 계보를 기록하며, manifest·gate·test coverage의 상호 일치 여부를 대조한다.

- 모델 번들 및 manifest SHA-256
- Golden Dataset manifest SHA-256과 전체 test 문서 ID
- gate set 및 metric definition ID
- 평가기 ID, 버전, artifact ref와 SHA-256
- 지표별 sample count

하지만 집계 숫자만으로는 원시 예측을 다시 계산할 수 없으므로, 모든 조건을 만족해도 `release.evidence.aggregate_only`가 실패 gate에 추가되고 `promotable`은 항상 `false`다. 이것은 미구현 상태를 숨기지 않기 위한 의도된 동작이다.

## 원시 Model A 재평가 v2

`src/scan2hwpx/evaluation/model_a_v2.py`는 test 문서별 EvidenceIR과 gold/prediction ContentIR의 실제 bytes에서 다음 8개 지표를 직접 재계산한다. 외부에서 만든 집계 점수는 입력으로 받지 않는다. 공개 진입점과 구성·결과 타입은 `scan2hwpx.evaluation`의 `evaluate_verified_model_a_v2_records` 및 `VerifiedModelAV2*` API다.

- 전체 CER
- 문항 번호·배점·선택지 등 핵심 토큰 오류율
- 역할 macro F1
- 읽기 순서 exact match
- 정규화 Kendall tau
- 표 topology exact match
- 수식 detection recall
- 정규화 수식 similarity

CER와 핵심 토큰 오류율은 일반 텍스트뿐 아니라 표 셀 텍스트도 포함한다. 일반 텍스트는 node ID로, 표 셀은 `(table node ID, row, column, row span, column span)`으로 정렬한다. 누락·추가·wrong-kind 노드는 역할 sentinel과 읽기 순서 실패에 반영한다. Golden에 표나 수식이 없으면 해당 표본 수는 `0`, 측정값은 `null`이며 만점으로 간주하지 않는다. 잘못된 prediction JSON·schema·EvidenceIR 계보와 계산량 한계 초과는 자동 보정하지 않고 보수적 실패로 측정한다.

기존 8개 지표가 직접 표현하지 않는 구조 오류도 숨기지 않는다. 문서별 `evidence_grounding_mismatch_count`는 node evidence refs와 표 셀 evidence refs를 동일 node ID·셀 topology key로 exact 비교하고, `image_asset_mismatch_count`는 동일 image node ID의 asset ref를 비교한다. 둘 중 하나라도 0보다 크면 문서 ID가 canonical 문서 순서대로 `structural_hard_failure_document_ids`에 기록된다. 따라서 8개 숫자가 모두 좋아도 이 목록이 비어 있지 않으면 완전 통과가 아니다.

아직 구현되지 않은 필수 Model A gate도 `unimplemented_required_gate_ids`에 명시한다.

- `model_a.high_risk_review_recall`
- `model_a.confidence_ece`

평가기는 source SHA, 실제 페이지 수, IR 계보, annotation·prediction·Model A bundle·실행 evaluator bytes와 canonical metric definition을 모두 대조한다. `src/scan2hwpx/evaluation/artifacts.py`는 Golden annotation과 검수 증명의 실제 bytes·SHA-256 및 manifest 일관성을 검사한다. 다만 검수 attestation은 전자서명된 신원 증명이 아니며, 검수자 신원이나 사용 권리를 인증하지 않는다. Report는 이 한계를 `review_identity_assurance=unsigned_manifest_consistency_only`로 명시한다.

v2 report는 `promotable=false`로 고정되고 현재 `release.py`가 승격 입력으로 수용하지
않는다. v2에서 미구현으로 표시한 두 지표는 아래 v3 평가기가 계산하지만, 인증된 검수 신원과
사람 Golden에 기반한 Model B·HWPX end-to-end gate 및 실제 승격 경계 연결은 여전히 필요하다.

## 원시 Model A 실행 재평가 v3

`src/scan2hwpx/evaluation/model_a_v3.py`는 v2의 8개 값을 그대로 보존한 채, 문서별로
digest 결속된 canonical 페이지 요청, 원시 모델 응답, validator 결과와 document barrier
조립 결과를 다시 실행한다. 외부에서 공급된 모든 gold 위험 표면 라벨의 exact coverage를
요구하며 위험 여부를 평가기가 추정하거나 보정하지 않는다.

공개 진입점과 구성·결과 타입은 `scan2hwpx.evaluation`의
`evaluate_verified_model_a_v3_records`, `canonical_model_a_risk_surfaces`,
`canonical_model_a_v3_report_bytes` 및 관련 `ModelA*`·`VerifiedModelAV3*` API다.

- high-risk review recall: 실제 validator block 또는 해당 node의 `needs_review`만
  검토 경로로 인정한다.
- confidence ECE: 유효한 예측 최상위 node를 10개 동일 폭 bin으로 나누고 confidence와
  gold semantic exact correctness의 차이를 계산한다. 차단된 페이지에는 가짜 confidence를
  만들지 않는다.

요청·응답·결과·조립 및 모델 ID/revision/artifact digest가 하나라도 재현되지 않으면 trusted
평가 자체를 중단한다. 현재 구현은 최대 1 GiB의 in-memory 평가 경계이며 더 큰 corpus는
streaming 평가기가 필요하다. 위험 라벨은 아직 인증되지 않았고 검수 신원은 unsigned이며,
typed cross-page 관계와 release adapter도 남아 있으므로 v3 report 역시 항상
`promotable=false`다. 새 `model-first-v2` 정책은 이 두 지표의 목표를 고정할 뿐 이러한
blocker를 해제하지 않는다.

## 원시 Model B 재평가

`src/scan2hwpx/evaluation/model_b.py`는 별도의 Golden/Prediction manifest와 실제
ContentIR·gold/prediction Plan byte를 다시 검증해 다음 세 지표를 계산한다.

- schema-valid document rate
- 정확한 `content_ref` 무결성 document rate
- 순서가 있는 `(content_ref, render_as)` projection 및 ContentIR lineage의 무단 변경 수

trusted manifest·Golden·검수 attestation 불일치는 평가 자체를 중단하고, 잘못된 모델
prediction은 schema/coverage 실패와 전체 gold projection 누락으로 계수한다. 이 report도
`promotable=false`로 고정되어 있으며 아직 `release.py` 승격 입력이 아니다. 실제 사람
Golden Plan과 서명된 검수 신원이 생기기 전에는 임계값을 통과했다고 주장할 수 없다.

## 실행

```powershell
python ai/evaluation/check_model_release.py `
  output/evaluation/measurements.json `
  output/evaluation/golden-manifest.json `
  output/evaluation/model-bundle.json `
  --output output/evaluation/release-audit.json
```

종료 코드 `0`만 승격 가능을 뜻한다. 현재 구현된 `aggregate_only` 경로는 정상적으로 검증돼도 종료 코드 `1`을 반환한다.
