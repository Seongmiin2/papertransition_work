# Exam2HWPX

스캔된 한국 중·고등학교 시험지 PDF를 로컬 OCR로 분석해 편집 가능한 HWPX로 변환하는 Windows 데스크톱 프로그램입니다. OpenAI API 키나 웹 서버 없이 이 컴퓨터 안에서 처리합니다.

현재는 개발 버전입니다. 데스크톱 앱의 `semantic` 출력은 원본 페이지를 검증 배경으로 보존하면서 검출된 표와 그림을 별도 한글 개체로 겹쳐 만듭니다. 기준 PDF에서 8페이지·그림 9개·표 2개·손상 없음까지 한글 2020으로 확인했습니다. 두 표의 외곽 행·열은 각각 2×4, 4×2이며 한 곳의 가로 병합 셀, 셀 텍스트, 가운데/왼쪽 정렬과 일반/굵게 스타일을 생성합니다. 표 밖 본문 스타일은 아직 사람 검수가 필요합니다. 자세한 현황은 [DEVELOPMENT_REPORT.md](DEVELOPMENT_REPORT.md)를 참고하세요.

## 지원 환경

- Windows 11 64-bit
- Python 3.11
- Node.js 20 이상과 npm
- 한컴오피스 한글 2024 권장
- 한국어 Windows OCR 언어팩

## 새 PC 최초 설치

PowerShell에서 다음 명령을 실행합니다.

```powershell
git clone https://github.com/Seongmiin2/papertransition_work.git
cd papertransition_work
git switch docs/project-development-report

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,cpu]"

npm install
npm run build
```

PaddleOCR 모델은 첫 변환 때 사용자 폴더에 자동 다운로드됩니다. 첫 실행에는 네트워크가 필요할 수 있으며 이후 OCR은 로컬에서 수행됩니다.

NVIDIA GPU가 있는 PC는 CPU 패키지 대신 GPU 패키지를 설치해야 합니다. 이 PC의 RTX 2060과 CUDA 13.1 드라이버에서는 다음 조합을 검증했습니다.

```powershell
python -m pip uninstall -y paddlepaddle
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.device.get_device())"
```

마지막 명령은 `True gpu:0`을 출력해야 합니다. CUDA 버전이 다른 PC는 PaddlePaddle 공식 Windows 설치표에 맞는 `cu118`, `cu126` 또는 `cu129` 저장소를 선택합니다.

## 데스크톱 앱 실행

가상환경을 활성화한 PowerShell에서 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\Activate.ps1
npm start
```

또는 최초 설치 후 다음 스크립트를 한 번 실행해 바탕화면 바로가기를 만듭니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
```

생성된 `Exam2HWPX.lnk`는 PowerShell 창 없이 Electron 앱을 실행합니다.

## 명령행 변환

```powershell
.\.venv\Scripts\Activate.ps1
python -m scan2hwpx convert "C:\문서\시험지.pdf" --output "C:\문서\result.hwpx" --mode balanced --dpi 240
python -m scan2hwpx validate "C:\문서\result.hwpx"
```

렌더러는 다음처럼 구분합니다.

- `fidelity`(기본): 페이지 수와 원본 외형을 우선하며 페이지마다 원본 그림 1개를 넣습니다.
- `semantic`: `fidelity` 배경 위에 검출된 표 셀과 그림 영역을 별도 편집 개체로 추가합니다.
- `editable`: 원본 배경 없이 원본 페이지별 고정 좌표에 텍스트·표·그림을 배치합니다. 데스크톱 앱의 기본값입니다.
- `portable`: OCR 텍스트는 편집할 수 있지만 현재 표·그림·페이지 배치가 원본과 다를 수 있습니다.
- `hancom`: 한글 자동화 기반 편집형 실험 모드이며 보안 모듈 상태에 따라 멈출 수 있습니다.

결과 폴더에는 HWPX 외에 `document_ir.json`, `review.json`, `review.html`, `run.log`, 원본·전처리·OCR overlay 이미지가 생성됩니다.

OCR 모드는 다음과 같습니다.

- `fast`: PaddleOCR만 사용합니다. 가장 빠른 대량 작업용입니다.
- `balanced`: 불확실한 줄만 Windows 한국어 OCR로 다시 읽고 두 후보를 검수 보고서에 남깁니다.
- `accurate`: 페이지 전체를 두 엔진으로 읽습니다. GPU 또는 최종 품질 확인용입니다.

한 폴더의 PDF를 연속 변환할 때는 아래 명령을 사용합니다. 한 번에 최대 10개를 처리하며, GPU에서는 OCR worker 두 개를 사용합니다. 실패 파일 때문에 전체 작업이 멈추지 않으며, 같은 설정으로 재실행하면 완료 파일은 건너뜁니다.

```powershell
python -m scan2hwpx batch-convert "C:\시험지" --out "C:\변환결과" --mode fast --dpi 120
```

배치 변환은 처리량을 위해 HWPX 생성에 필요한 원본 페이지와 `document_ir.json`만 저장하고, 단일 파일 변환에서 제공하는 전처리·OCR overlay·검수 HTML은 생략합니다.

성능 목표 프로필은 `fast`·120 DPI·`fidelity`입니다. 2026-08-26 RTX 5060 Ti 8GB에서 8페이지 시험지 10개(총 80페이지)를 Python 프로세스 및 OCR 모델 초기화 포함 52.254초에 변환했습니다. 파일당 페이지 수나 GPU가 달라지면 같은 시간을 보장하지 않습니다.

### HWP 참조 데이터와 OCR 파인튜닝 준비

HWP 정답 예시와 PDF 입력을 점검하고, HWP 본문으로 한국어 시험지 인식 학습 crop을 만듭니다.

```powershell
python -m scan2hwpx reference-audit test_data --out output\reference-dataset
python -m scan2hwpx build-ocr-dataset output\reference-dataset\reference_manifest.jsonl `
  --out output\ocr-training --limit 3000 --variants 2

# 표·그림·지문 상자 레이아웃 검수용 시드 라벨 생성
python -m scan2hwpx build-layout-dataset "C:\문서\시험지.pdf" `
  --out output\layout-training\시험지 --device gpu:0
```

생성물은 문서 단위 `train.txt`, `validation.txt`, `test.txt`, 6,000개 이미지, HWP 코퍼스 사전과 `korean_exam_dict.txt`입니다. 현재 데이터에서는 PDF와 같은 시험의 HWP가 없어 HWP를 직접 정답쌍으로 간주하지 않습니다. 합성 데이터는 도메인 적응용이며, 실제 PDF crop을 사람이 교정한 평가셋이 생기기 전에는 정확도 향상을 주장할 수 없습니다.

변환 결과의 `review.html`에서는 빨간 검수 crop과 두 OCR 후보를 보고 정답을 수정·확정할 수 있습니다. 내려받은 파일을 다음처럼 가져오면 실제 PDF 기반 학습 label이 됩니다. 서로 다른 PDF가 3개 미만이면 신뢰할 수 있는 train/validation/test 평가가 불가능하다고 보고서에 표시합니다.

```powershell
python -m scan2hwpx import-ocr-labels "$HOME\Downloads\verified_ocr_training.jsonl" `
  --out output\verified-ocr-training
```

NVIDIA GPU가 준비되면 공식 PaddleOCR 저장소를 별도로 clone한 뒤 다음 스크립트로 PP-OCRv5 한국어 mobile recognition 모델을 파인튜닝합니다.

```powershell
python scripts\train_korean_ocr.py --paddleocr-repo C:\src\PaddleOCR --download-pretrained
```

### 수식 모델 비교와 검수

수식 추출을 켜면 crop과 검수용 manifest가 함께 생성됩니다.

```powershell
python -m scan2hwpx convert "C:\문서\시험지.pdf" --formulas --output "output\result.hwpx"
python -m scan2hwpx formula-benchmark "output\formula_training.jsonl" --models PP-FormulaNet_plus-S PP-FormulaNet_plus-M --out "output\formula-benchmark"
```

`output\formula-benchmark\formula_benchmark.html`을 열면 모델별 LaTeX를 비교하고 정답을 확정해 `verified_formula_training.jsonl`로 저장할 수 있습니다. 모델 실행에 필요한 선택 의존성은 `python -m pip install -e ".[vision]"`으로 설치합니다.

## 검사

```powershell
npm run build
npm test
pytest -q
ruff check .
mypy src
```

현재 로컬 기준 통과 수는 TypeScript 6개, Python 42개입니다. OCR 모델이나 환경이 달라질 수 있으므로 새 PC에서도 위 검사를 다시 실행하세요.

## 개인정보와 저장소 제외 항목

원본 PDF, 변환 결과, OCR IR, 검토 crop, 로그, 모델 weight와 API key는 Git에 올리지 않습니다. `.env` 대신 `.env.example`만 포함합니다. 클라우드 OCR은 구현 기본값에서 비활성화돼 있습니다.

## 알려진 한계

- 기본 `fidelity` HWPX package 생성과 검증에는 한글 설치가 필요하지 않습니다.
- 앱 installer와 PyInstaller worker bundle은 아직 없습니다.
- RTX 2060 GPU 기준 홍천중 8페이지 `fidelity` 전체 변환은 31.631초, `semantic` 전체 변환은 40.623초, HWPX 패키징만은 3.459초로 실측했습니다.
- 비정상 한자·라틴 문자열과 특수기호 오류가 일부 남을 수 있습니다.
- 표와 그림은 `fidelity` 출력에서 시각적으로 보존되지만, 표 셀·본문·그림 영역을 각각 편집 가능한 개체로 복원하는 작업은 진행 중입니다.
- 저장소의 시험지 fixture는 합성 JSON이며 실제 시험지 PDF는 저작권·개인정보 때문에 포함하지 않습니다.
