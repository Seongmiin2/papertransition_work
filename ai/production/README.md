# Production models

`deployed/`는 데스크톱 앱이 자동으로 읽는 승인 모델, `candidates/`는 평가를 마쳤지만 배포하지 않은 모델이다.

## 현재 배포

| 구성요소 | 모델 | 용도 |
|---|---|---|
| OCR recognizer | `deployed/korean-exam-ppocrv5` | 한국어 시험지 행 인식 |
| Page anomaly | `deployed/page-anomaly-v1` | 품질 라우팅 보조; 문서 내용을 생성하지 않음 |

앱 모델 경로는 `apps/desktop/src/main/main.ts`에서 이 디렉터리를 기준으로 설정한다. 승인 모델을 교체할 때 폴더 이름을 임의로 바꾸지 말고, 후보 평가와 앱 smoke test 후 artifact를 이 경로에 반영한다.

## 후보

`candidates/korean-exam-ppocrv5-v3-candidate`는 2026-08-27 학습을 정상 완료했지만 같은 test split에서 v2보다 두 핵심 지표가 모두 낮아 배포하지 않았다. 후보를 보존하는 이유는 결과 재현과 중복 실험 방지다.

## 배포 체크

1. 실험 JSON의 checksum과 실제 inference weights가 일치한다.
2. `inference.yml`, `inference.json`, `inference.pdiparams`가 함께 존재한다.
3. held-out test 결과와 기존 배포 모델 비교가 `../results/`에 기록돼 있다.
4. 대표 PDF 변환과 HWPX 열기 검증을 통과한다.
5. 전체 관련 테스트와 데스크톱 빌드가 통과한다.
