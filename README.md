# Exam2HWPX

스캔된 한국 중·고등학교 시험지 PDF를 로컬 OCR로 분석해 편집 가능한 HWPX로 변환하는 Windows 데스크톱 프로그램입니다. OpenAI API 키나 웹 서버 없이 이 컴퓨터 안에서 처리합니다.

현재는 개발 버전입니다. 문항 1~25 순서, HWPX package validation과 한글 2024 재개방은 기준 8페이지 PDF에서 확인했지만 표·그림·수식과 일부 OCR 오탈자는 아직 완전하지 않습니다. 자세한 현황은 [DEVELOPMENT_REPORT.md](DEVELOPMENT_REPORT.md)를 참고하세요.

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
python -m pip install -e ".[dev]"

npm install
npm run build
```

PaddleOCR 모델은 첫 변환 때 사용자 폴더에 자동 다운로드됩니다. 첫 실행에는 네트워크가 필요할 수 있으며 이후 OCR은 로컬에서 수행됩니다.

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
python -m scan2hwpx convert "C:\문서\시험지.pdf" --output "C:\문서\result.hwpx" --dpi 300
python -m scan2hwpx validate "C:\문서\result.hwpx"
```

결과 폴더에는 HWPX 외에 `document_ir.json`, `review.json`, `review.html`, `run.log`, 원본·전처리·OCR overlay 이미지가 생성됩니다.

## 검사

```powershell
npm run build
npm test
pytest -q
ruff check .
mypy src
```

현재 로컬 기준 통과 수는 TypeScript 2개, Python 17개입니다. OCR 모델이나 환경이 달라질 수 있으므로 새 PC에서도 위 검사를 다시 실행하세요.

## 개인정보와 저장소 제외 항목

원본 PDF, 변환 결과, OCR IR, 검토 crop, 로그, 모델 weight와 API key는 Git에 올리지 않습니다. `.env` 대신 `.env.example`만 포함합니다. 클라우드 OCR은 구현 기본값에서 비활성화돼 있습니다.

## 알려진 한계

- HWPX를 최종 검증하려면 한글 설치가 필요합니다.
- 앱 installer와 PyInstaller worker bundle은 아직 없습니다.
- Paddle와 Windows OCR을 함께 실행해 정확도를 높이는 대신 처리 시간이 깁니다.
- 비정상 한자·라틴 문자열과 특수기호 오류가 일부 남을 수 있습니다.
- 표, 그림, 수식은 원본과 같은 편집 객체로 완벽히 복원되지 않습니다.
- 저장소의 시험지 fixture는 합성 JSON이며 실제 시험지 PDF는 저작권·개인정보 때문에 포함하지 않습니다.
