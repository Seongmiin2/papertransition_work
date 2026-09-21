# Production models

`deployed/`는 데스크톱 앱이 현재 자동으로 읽는 runtime 모델,
`candidates/`는 평가 중이거나 교체하지 않은 모델이다. runtime 경로에 있다는 사실만으로
현재 Model A/B release gate를 통과했다는 뜻은 아니다.

## 현재 배포

| 구성요소 | 모델 | 용도 |
|---|---|---|
| OCR recognizer | `deployed/korean-exam-ppocrv5` | 한국어 시험지 행 인식 |
| Page anomaly | `deployed/page-anomaly-v1` | 품질 라우팅 보조; 문서 내용을 생성하지 않음 |

앱 모델 경로는 `apps/desktop/src/main/main.ts`에서 이 디렉터리를 기준으로 설정한다. 승인 모델을 교체할 때 폴더 이름을 임의로 바꾸지 말고, 후보 평가와 앱 smoke test 후 artifact를 이 경로에 반영한다.

현재 OCR v2의 95.5701% 보고값은 v1 train 문서가 v2 test에 다시 포함된
계보 누출이 있어 독립 holdout 근거로 사용할 수 없다. 따라서 이 모델은 앱 호환성
유지를 위한 legacy prototype이며, 누출 없는 문서·template-family 분할과 실제 스캔
평가를 통과하기 전에는 새 웹서비스의 확정 모델로 승격할 수 없다.

## 후보

`candidates/korean-exam-ppocrv5-v3-candidate`는 2026-08-27 학습을 정상 완료했지만 같은 test split에서 v2보다 두 핵심 지표가 모두 낮아 배포하지 않았다. 후보를 보존하는 이유는 결과 재현과 중복 실험 방지다.

## 배포 체크

1. 실험 JSON의 checksum과 실제 inference weights가 일치한다.
2. `inference.yml`, `inference.json`, `inference.pdiparams`가 함께 존재한다.
3. 모든 조상 학습 데이터와 겹치지 않는 문서·template-family held-out 결과가
   `../results/`에 기록돼 있다.
4. 대표 PDF 변환과 HWPX 열기 검증을 통과한다.
5. 전체 관련 테스트와 데스크톱 빌드가 통과한다.
6. 데이터 사용 권리가 검증되고 실제 스캔 최소 50문서·500쪽 release gate를 통과한다.
