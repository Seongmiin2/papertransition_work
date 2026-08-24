# PDF→HWPX 학습·검증 보고서 (2026-08-25)

## 결론

새 데이터는 모두 원본을 수정하지 않고 문서 단위로 분류·복제했다. 실제 정답으로 확인되지 않은 같은 이름의 PDF/HWP를 억지로 짝지어 학습하지 않았으며, 검증에서 성능이 나빠진 미세조정 OCR 가중치도 제품에 연결하지 않았다.

현재 운영 조합은 다음과 같다.

- OCR: 공식 `korean_PP-OCRv5_mobile_rec` + `PP-OCRv5_mobile_det`
- 레이아웃: `PP-DocLayout-S` + OpenCV 표 격자 검출
- 실행 모드: `fast` (이 gold 자료에서 가장 정확하고 빠름)
- QA: 학습한 PCA 선형 페이지 오토인코더로 분포 이탈 페이지 경고
- 출력: 원본 페이지 수를 고정하고, 고신뢰 OCR 본문을 원래 좌표의 HWPX 글상자로 배치하며, 표와 이미지는 별도 개체로 복원

## 데이터 분류

`output/training-corpus-v2`에 95개 파일을 조사한 manifest와 복사본을 만들었다.

- 고유 파일: 91개
- 완전 중복: 4개
- 스캔 원본 PDF: 20개
- 편집 참고 문서: 57개
- 답안/정답 자료: 14개
- gold 원본 PDF: 1개
- gold 텍스트 PDF: 1개
- gold 편집 DOCX: 1개
- 기존 회귀 PDF: 1개
- split: train 12문서, validation 3문서, test 7문서

`P고2`의 같은 이름 PDF/HWP 20쌍은 HWP가 대체로 2페이지짜리 답안 템플릿이고 PDF는 최대 12페이지 시험지였다. 따라서 20쌍 전부 supervised X→Y 학습에서 제외했다. 잘못된 정답을 학습시키면 레이아웃 축소와 내용 누락을 모델이 정답으로 배우기 때문이다.

홍천중2 자료는 다음 관계를 확인했다.

- 스캔 원본 `홍천중2.pdf`: 7페이지
- native-text transcript PDF: 12페이지
- DOCX: 한글 자동화 렌더 기준 11페이지
- transcript PDF와 DOCX 정규화 텍스트: 각 10,219자, 유사도 0.999902

따라서 transcript/DOCX는 내용과 구조의 gold이지만, 원본 7페이지 좌표를 그대로 가르치는 geometry gold는 아니다. 이 세 문서는 전부 held-out test로 고정했다.

## OCR 적응학습

기존 합성 데이터와 새 스캔 자료에서 고신뢰 teacher crop을 합쳤다.

- 기존: train 5,186 / validation 674 / test 140
- 새 silver: train 2,143 / validation 543
- 합계: train 7,329 / validation 1,217 / test 140
- 공식 11,945자 dictionary 호환 label: train 6,325 / validation 1,103 / test 96
- gold·회귀·source-test 문서는 silver 학습에서 제외
- silver 허용 기준: OCR confidence 0.96 이상, 언어 품질 0.88 이상

PaddleOCR 공식 저장소 commit `2661c7c0ef5c613e8f93c6e93b2e052399f0f854`를 고정하고 PP-OCRv5 mobile recognition을 3 epoch 미세조정했다. 그러나 고정 gold에서 기존 공식 모델보다 나빠져 자동 승격 규칙이 후보 가중치를 거부했다.

| 모델 | Gold CER | 문자 정확도 | 시간 | 판정 |
|---|---:|---:|---:|---|
| 공식 baseline, fast | 9.0936% | 90.9064% | 13.741초(이상탐지 포함 재측정) | 유지 |
| 새 미세조정 candidate | 12.3203% | 87.6797% | 13.854초 | 거부 |

후보 CER은 baseline보다 35.48% 악화됐다. 후보 파일은 재현과 분석을 위해 보관하지만 앱에는 연결하지 않는다.

## 속도 모드 비교

같은 gold에서 세 모드를 비교했다.

| 모드 | CER | 시간 |
|---|---:|---:|
| fast | 9.0936% | 15.775초 |
| balanced | 9.2696% | 15.508초 |
| accurate | 9.8660% | 21.557초 |

이 자료에서는 `fast`가 정확도와 처리량의 최적점이므로 CLI, batch, worker, desktop 기본값을 `fast`로 변경했다.

## 이상탐지 모델

train/validation PDF 페이지만 사용해 32×32 grayscale 페이지를 입력으로 하는 rank-32 PCA 선형 오토인코더를 학습했다. test는 임계값 학습에 사용하지 않았다.

- train: 161페이지, flag 0
- validation: 31페이지, flag 1
- held-out test: 69페이지, flag 25
- threshold: 0.0007239776896

test flag는 손상 확정이 아니라 학습 분포 이탈 경고다. 모델은 OCR 문자를 고치거나 생성하지 않고 QA routing에만 사용한다. 홍천중2 gold에서는 7쪽 중 1·6쪽을 검토 대상으로 표시했고 CER은 변하지 않았다.

## HWPX 렌더 검증

새 좌표편집 렌더러는 고신뢰 OCR 줄을 원본 영상에서 지운 뒤 같은 좌표에 실제 `hp:rect/hp:drawText` 글상자로 넣는다. 저신뢰 줄과 사진 내부 글자는 원본 영상으로 남기며, 검출된 표는 셀 단위 편집 표로 만들고 사진은 별도 이미지 개체로 둔다.

### 홍천중2 gold

- 원본/출력: 7페이지/7페이지
- 편집 글상자: 541개
- 글상자 문자: 12,521자
- 편집 표: 1개
- 분리 이미지: 1개
- 한글 직접 재열기: 성공, 7페이지
- 한글 text export: 14,766자
- HWPX validator: 통과
- 기존 외형우선 SSIM 0.5554, 좌표편집형 SSIM 0.5414

### 기존 홍천중3 회귀자료

- 원본/출력: 8페이지/8페이지
- 편집 글상자: 697개
- 글상자 문자: 19,972자
- 편집 표: 2개
- 분리 이미지: 1개
- 한글 직접 재열기: 성공, 8페이지
- 한글 text export: 23,075자
- HWPX validator: 통과

## 주요 산출물

- corpus: `output/training-corpus-v2/corpus_report.json`
- anomaly weight: `output/trained-models/page-anomaly-v1/page_anomaly_linear_autoencoder.npz`
- OCR adaptation report: `output/ocr-training-v2/adaptation_report.json`
- candidate training: `output/trained-korean-ocr-v2/training_run.json`
- promotion decision: `output/benchmarks/ocr-v2-promotion-decision.json`
- gold OCR benchmark: `output/benchmarks/ocr-baseline-fast-anomaly-hongcheon2/ocr_benchmark.json`
- 7-page coordinate-editable result: `output/hongcheon2-gold-demo/홍천중2_좌표편집형_v3.hwpx`
- 8-page regression result: `output/e2e-semantic-fast-anomaly/result-coordinate.hwpx`

## 남은 한계

현재 결과는 손상 없이 페이지 수를 지키고 본문 대부분을 좌표 글상자로 편집 가능하게 만든 단계다. 그러나 다음 항목은 아직 정답 수준이 아니다.

- 표 안의 제목 크기와 굵기 일부가 원본보다 작다.
- OCR CER 9.09%이므로 약 10,000자당 900자 수준의 문자 편집 차이가 남는다.
- 부분 굵게, 자간, 행간, 문단 정렬을 정확히 학습할 좌표·span style gold가 아직 없다.
- PP-DocLayout pseudo label은 사람 검수 전에는 재학습 정답으로 쓰지 않는다.

다음 성능 상승에 가장 가치 있는 데이터는 PDF 한 페이지와 동일한 페이지의 검수된 HWPX, 그리고 글자/표/그림 객체 좌표가 함께 있는 geometry gold다. 이 자료가 30~50페이지 확보되면 OCR이 아니라 레이아웃·스타일 모델을 정식 fine-tune하고, 문서 단위 held-out 평가로 승격 여부를 결정한다.
