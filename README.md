# Exam2HWPX

한국어 시험지 PDF를 로컬 OCR로 분석해 편집 가능한 HWPX로 변환하는 Windows 데스크톱 프로젝트입니다. Electron UI와 Python 변환 파이프라인으로 구성되며, OpenAI API나 외부 추론 서버 없이 실행됩니다.

## 현재 상태

- 배포 OCR: `korean-exam-ppocrv5-20260826-v2`
- 동일 held-out test 정확도: 95.5701%, 정규화 편집 유사도: 98.1530%
- 2026-08-27 v3 실험: 테스트 지표가 v2보다 낮아 후보로 보존하고 배포하지 않음
- 데스크톱 기본 출력: 120 DPI, fast OCR, editable HWPX
- AI 데이터·학습·성과·배포 이력: [`ai/`](ai/README.md)

## 저장소 구조

```text
apps/           Electron 데스크톱 앱
src/            PDF→OCR→HWPX Python 제품 코드
tests/          단위·통합 테스트
ai/datasets/    원본 데이터 규칙, 설정, 데이터셋 생성
ai/modeling/    학습 설정·코드·실험 메타데이터
ai/results/     누적 성과와 과거 실험 보고서
ai/production/  앱이 실제 사용하는 모델과 평가 후보
docs/           프로젝트 개발 문서
output/         로컬 실행·학습 생성물(Git 제외)
```

자세한 파일별 설명은 [`STRUCTURE.md`](STRUCTURE.md)를 참고합니다.

## 설치

요구 환경: Windows 11, Python 3.11, Node.js 20 이상, 한컴오피스 한글 2024 권장.

```powershell
git clone https://github.com/Seongmiin2/papertransition_work.git
cd papertransition_work
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,cpu]"
npm install
```

NVIDIA GPU를 사용할 때는 PC의 CUDA 드라이버에 맞는 PaddlePaddle GPU 패키지를 별도로 설치합니다.

## 실행

```powershell
.\.venv\Scripts\Activate.ps1
npm start
```

CLI 변환:

```powershell
python -m scan2hwpx convert "C:/문서/시험지.pdf" --output "output/result.hwpx" --mode fast --dpi 120 --renderer editable --recognition-model-dir "ai/production/deployed/korean-exam-ppocrv5"
```

## 검증

```powershell
pytest -q
ruff check .
mypy src
npm test
npm run build
```

## 주요 문서

- [AI 실험과 제품 승격 흐름](ai/README.md)
- [모델 성과 추이](ai/results/README.md)
- [프로덕션 모델 정책](ai/production/README.md)
- [개발 현황 보고서](docs/DEVELOPMENT_REPORT.md)
- [폴더 구조](STRUCTURE.md)
