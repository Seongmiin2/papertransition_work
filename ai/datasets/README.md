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

## 재현 규칙

- 문서 단위로 train/validation/test를 분리해 같은 시험지의 행이 여러 split에 섞이지 않게 한다.
- 실험 JSON에 split seed, 문서 수, 행 수, builder 경로를 기록한다.
- test split은 모델 선택에 사용하지 않고 최종 비교에만 사용한다.
- OCR이 만든 silver label과 원본 HWP/PDF에서 얻은 정답 label을 구분한다.
- 누락된 gold 원본은 임의로 대체하지 않고 설정에 미해결 상태로 남긴다.
