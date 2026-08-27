# 한국어 시험지 OCR 모델 연구·학습 기록 (2026-08-26)

## 결론

현재 배포 모델은 **한국어 PP-OCRv5 mobile recognition 미세조정 모델**로 정한다.
한국어가 공식 지원되지 않는 PP-OCRv6 recognition은 후보에서 제외한다.
PaddleOCR-VL 계열은 속도 목표를 책임지는 기본 인식기가 아니라, 난문자와 구조 라벨을 만드는
교사 모델 후보로 둔다.

이 결정은 다음 프로젝트 목표를 기준으로 한다.

- 입력: 스캔 또는 이미지형 한국어 시험지 PDF
- 중간 출력: 좌표, 읽기 순서, 글자열, 신뢰도를 가진 구조화 문서
- 최종 출력: 원본 이미지를 덧씌우지 않은 편집 가능한 HWPX
- 장기 목표: 비슷한 8쪽 시험지 최대 10개를 60초 안에 처리
- 핵심 불변 조건: 문서 단위 데이터 분리, 원문에 없는 글자 생성 금지, 좌우 2단 읽기 순서 유지

## 모델을 이렇게 나눈다

한 모델에 OCR·레이아웃·HWPX 서식을 모두 억지로 맡기지 않는다.

1. **검출**: PP-OCRv5 mobile detector가 글자 행 좌표를 찾는다.
2. **인식**: 이 저장소에서 학습한 한국어 PP-OCRv5 mobile recognizer가 행 이미지를 글자로 바꾼다.
3. **구조화**: 기존 시험지 파서가 문항, 보기, 선택지, 1단/2단 읽기 순서를 정한다.
4. **서식화**: editable renderer가 좌표와 구조를 HWPX 텍스트 개체로 만든다.
5. **선택적 교사 모델**: PaddleOCR-VL은 Hanja, 특수 기호, 복잡한 박스처럼 작은 인식기가
   확신하지 못하는 표본의 라벨 보조에만 연구한다.

현재 실제로 학습을 끝낸 범위는 2번 인식 모델이다. 레이아웃과 수식 모델의 별도 학습 성능을
이번 결과에 섞어 말하지 않는다.

## 데이터셋

ai/datasets/build_hwp_line_dataset.py가 한컴으로 HWP를 PDF 렌더링한 뒤, PDF 내부의 실제 글자 좌표로
행 이미지를 잘라 정답 글자열과 묶는다. OCR 예측값을 정답으로 재사용하지 않는다.

| 분할 | 문서 | 행 |
|---|---:|---:|
| train | 33 | 30,309 |
| validation | 4 | 3,392 |
| test | 4 | 2,921 |
| 합계 | 41 | 36,622 |

분할은 seed 20260826으로 문서 단위에서 고정했다. 같은 시험지의 행이 train과 test에 섞이지 않는다.
공식 한국어 모델과 같은 11,945자 사전을 사용해 사전 크기 변경으로 pretrained head가 깨지는 일을
피했다.

현재 제외된 데이터도 명확하다.

- 글자 없는 기호 전용 행: 2,692
- 공식 사전에 없는 글자를 포함한 행: 6,721
- 주요 미지원 영역: 일부 한자, 괄호형·원문자형 특수 기호, 일부 문장부호와 PUA 글자

따라서 현재 수치는 지원 사전 안의 한국어 시험지 행 성능이다. 다음 데이터 라운드는 실제 스캔
왜곡·복사 노이즈와 미지원 문자를 별도 hard set으로 수집해야 한다.

## 실제 학습

- 기반 모델: korean_PP-OCRv5_mobile_rec
- PaddleOCR revision: 2661c7c0ef5c613e8f93c6e93b2e052399f0f854
- GPU: NVIDIA GeForce RTX 5060 Ti 8GB
- 1차 학습: 기존 20문서, 12 epoch, learning rate 1e-4
- 2차 이어 학습: 전체 41문서, 6 epoch, learning rate 2e-5
- batch size: 32
- warmup: 1 epoch
- 입력 multi-scale: 320x32, 480x48, 640x48
- 최대 GPU reserved memory: 약 5.66GB

실행 명령:

    python ai\datasets\build_hwp_line_dataset.py --input "ai\datasets\source\고1-2" --input "ai\datasets\source\고2" --output output\hwp-ocr-training-v2

    python ai\modeling\train_korean_ocr.py --paddleocr-repo output\vendor\PaddleOCR --dataset output\hwp-ocr-training-v2 --output output\trained-korean-ocr-hwp-v2-lr2e5 --pretrained output\trained-korean-ocr-hwp-lr1e4\best_accuracy.pdparams --epochs 6 --batch-size 32 --workers 2 --learning-rate 0.00002 --warmup-epochs 1 --eval-every 250

## 결과

| 모델 | 데이터 | 행 완전일치 | 정규화 편집 유사도 | 처리량 |
|---|---|---:|---:|---:|
| 공식 pretrained | validation 2문서 | 84.09% | 97.26% | 151.6행/초 |
| 1차 미세조정 best | validation 2문서 | 95.89% | 98.54% | 157.4행/초 |
| 1차 미세조정 best | test 2문서 | 96.44% | 98.87% | 154.5행/초 |
| 2차 이어 학습 best | validation 4문서 | 94.43% | 98.17% | 154.1행/초 |
| 2차 이어 학습 best | test 4문서 | 95.57% | 98.15% | 155.0행/초 |

1차와 2차는 데이터 분할 자체가 달라 수치를 직접적인 성능 증감으로 비교하지 않는다. 2차 test는
모델 선택에 쓰지 않고 best checkpoint를 정한 뒤 한 번만 실행했다.

최종 추론 모델:

- 앱 경로: ai/production/deployed/korean-exam-ppocrv5
- 학습 checkpoint: output/trained-korean-ocr-hwp-v2-lr2e5/best_accuracy
- 추론 가중치 SHA-256:
  BDCC257F9B9A1D4CAF1F1AC6791E5B3AB981C990BF61177344FD2ED6E4FE716D

## 대표 실제 PDF 추론

samples/(천재박)홍천중3_1학기 기말고사(1).pdf 한 건을 미세조정 모델, GPU, 120 DPI,
editable renderer로 끝까지 처리했다.

- 8쪽
- 실행 시간: 20.3초
- 평균 OCR 신뢰도: 0.9467
- 문항: 14
- 편집 가능 텍스트 상자: 424
- 편집 가능 문자: 20,945
- 결과: output/model-inference-v2/홍천중3-trained-v2-editable.hwpx

실행 명령:

    python -m scan2hwpx convert "samples\(천재박)홍천중3_1학기 기말고사(1).pdf" --output "output\model-inference-v2\홍천중3-trained-v2-editable.hwpx" --mode fast --dpi 120 --device gpu:0 --renderer editable --recognition-model-dir "ai\production\deployed\korean-exam-ppocrv5"

## 10개/1분 목표의 현재 상태

아직 달성하지 못했다. 대표 8쪽 문서 한 건이 20.3초이므로 같은 크기 10건을 현재 단일 앱 워커로
처리하면 약 203초다. 두 워커로 완전히 병렬화된다는 낙관적 산술값도 약 102초다. 이는 실제 10건
실측값이 아니라 이번 한 건을 이용한 추정이다.

인식기는 이미 154.5행/초이므로 다음 성능 작업은 모델을 더 작게 만드는 것보다 다음 순서가 맞다.

1. PDF 페이지 렌더링과 detector 입력을 여러 문서에서 batch 처리한다.
2. 한 GPU에 OCR 엔진을 중복 적재하는 2-process 방식과 단일 process page batch 방식을 비교한다.
3. editable renderer의 개체 생성 시간을 OCR과 분리 계측한다.
4. 정확도가 유지되는 최소 DPI를 120과 100 두 조건에서 실제 문서 한 건으로 정한다.

## 다음 학습 라운드

1. 실제 스캔 PDF와 대응 원본 HWP를 같은 문서 ID로 모은다.
2. 현재 모델이 틀리거나 신뢰도 0.90 미만인 행만 hard set으로 축적한다.
3. 미지원 한자·특수 기호는 기존 사전을 무작정 교체하지 말고 별도 head 확장 실험으로 비교한다.
4. PaddleOCR-VL을 교사로 사용할 때도 사람이 확정한 정답과 분리해 provenance를 남긴다.
5. model registry에는 데이터 hash, split seed, 코드 revision, 파라미터, metric, 가중치 hash를 함께 남긴다.

## 공식 근거

- PP-OCRv5 다국어/한국어 모델:
  https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
- PP-OCRv6 지원 언어와 모델 설명:
  https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md
- PaddleOCR-VL 공식 저장소와 사용 문서:
  https://github.com/PaddlePaddle/PaddleOCR
  https://github.com/paddlepaddle/paddleocr/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md
