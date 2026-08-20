# Exam2HWPX 개발 현황 및 기술 보고서

> 작성일: 2026-08-20
> 목표: 이미지형 한국 중·고등학교 시험지 PDF를 편집 가능한 HWPX로 변환하는 Windows 로컬 데스크톱 프로그램

## 1. 개발 목표

이 프로젝트의 목표는 스캔 PDF를 단순히 이미지로 붙여 넣는 것이 아니라, 지문·문항·선택지를 사용자가 한글에서 직접 고칠 수 있는 HWPX로 변환하는 것이다. 학교명, 시험명, 학생 정보란과 반복 머리말·꼬리말은 제거하고 문제 콘텐츠는 읽기 순서와 2단 구조를 최대한 유지한다.

초기 지원 범위는 Windows 11, 이미지형 A4 PDF, 한국어·영어 혼합 문서, 1단·2단 시험지, 객관식 문항, 빨간 채점 표시다. 웹 서비스, 계정, 결제, 완벽한 수식·손글씨 복원과 픽셀 단위 원본 복제는 현재 범위에서 제외한다.

## 2. 현재 기술 스택

### 데스크톱

- Electron, React, TypeScript, Vite
- Zustand 도입 기반 마련
- SQLite(`better-sqlite3`) 작업·이벤트 영구 저장
- Electron Main / context-isolated Preload / Renderer 분리
- `contextIsolation=true`, `nodeIntegration=false`, `sandbox=true`

### 문서 AI와 변환 worker

- Python 3.11
- PaddleOCR PP-OCRv5 한국어 mobile detection/recognition
- Windows 11 내장 한국어 OCR(WinRT) 교차 인식
- PyMuPDF 300 DPI PDF 렌더링
- OpenCV·NumPy 전처리 및 채점색 마스크
- Pydantic Document IR
- pywin32·pyhwpx 기반 한글 자동화
- ZIP/XML HWPX package validator

### 품질과 테스트

- pytest, Ruff, mypy strict
- Vitest, TypeScript project references
- 실제 한글 2024 COM 재개방 검사
- OCR 원본, 전처리 이미지, block overlay, 검토 crop, JSON/HTML 검토 보고서 저장

## 3. 구현된 기능

### PDF 및 OCR 파이프라인

1. PDF를 300 DPI로 렌더링한다.
2. 원본은 보존하고 OCR용 이미지에만 빨간 채점색 마스크를 적용한다.
3. 페이지와 block의 bbox, confidence, reading order를 Document IR에 저장한다.
4. 2단 문서를 왼쪽 위→아래, 오른쪽 위→아래 순서로 정렬한다.
5. 문항 번호, 배점, ①~⑤ 선택지를 구조화한다.
6. 학교명·시험명·학생 정보 및 반복 header/footer를 정리한다.
7. 한글 자동화로 2단 편집형 HWPX를 생성한다.
8. HWPX ZIP/XML과 필수 참조를 검사하고 한글에서 다시 연다.

### OCR 융합

Paddle 단독 결과에서 한국어 오탈자가 많이 발생해 Windows 내장 한국어 OCR을 추가했다. Windows OCR은 본문 문장에 강하지만 시험지 글꼴의 숫자 `1`을 `7`로 읽는 사례가 있었다. 따라서 한 엔진으로 전면 교체하지 않고 다음처럼 융합한다.

- Paddle: line detection, bbox, 문항·선택지 접두사와 구조
- Windows OCR: 동일 좌표의 한국어 본문 후보
- 좌표의 수직·수평 중첩을 계산해 대응 line 선택
- `1.`, `①` 같은 구조 접두사는 Paddle 값을 보존
- `㉠~㉤` 같은 본문 기호는 Windows 후보를 보존
- 문항 시작은 다음 예상 번호만 허용해 해설 속 숫자 오탐 제거

### 데스크톱 앱

- PDF 다중 선택과 드래그앤드롭
- 20건 이상을 받을 수 있는 bounded queue 설계
- 파일명, 상태, 진행률, 실패 원인 표시
- 작업 취소와 결과 HWPX 열기
- 입력 SHA-256 기록 및 작업용 사본 생성
- SQLite 작업 상태와 이벤트 저장
- 비정상 종료 작업을 `RETRYING` 상태로 복구
- Python worker를 NDJSON 프로토콜로 실행
- PowerShell 창 없이 Electron 또는 `pythonw.exe` 바로가기 실행

## 4. 실제 기준 문서 검증

기준 입력은 8페이지 이미지형 중학교 시험지 PDF다. 원본에는 회전된 페이지, 2단 편집, 한국어·영어 혼합, 원문 기호와 채점 표시가 포함돼 있다.

현재 확인된 결과:

- 8페이지 전체 OCR 및 HWPX 생성
- 문항 번호 1~25 연속성 확인
- HWPX package validation 통과
- 한글 2024 재개방 성공
- 재개방 후 본문 약 21,977자 확인
- Python 자동 테스트 17개 통과
- TypeScript 상태기계 테스트 2개 통과
- Ruff 및 mypy strict 통과
- npm audit 취약점 0건

이 수치는 현재 한 기준 문서의 구조 검증 결과다. 별도 정답 데이터셋을 이용한 CER, block F1, 사람 수정 시간은 아직 측정하지 않았으므로 목표를 달성했다고 주장하지 않는다.

## 5. 개발 과정에서 겪은 어려움

### OCR confidence가 실제 품질을 반영하지 않음

`㉠`을 `@`, `㉡`을 `⑥`, 한글을 영문·한자로 잘못 읽은 block이 confidence 0.9 이상인 사례가 있었다. 단순히 0.75 미만 block만 검토하면 심각한 오류가 누락된다. confidence 외에 비정상 문자, 사전 불일치, 엔진 간 불일치, 문항 번호 연속성과 선택지 개수를 함께 평가해야 한다.

### OCR 엔진별 장단점이 다름

Paddle 한국어 mobile 모델은 문항 번호와 선택지 외곽 구조를 비교적 잘 유지하지만 본문 오탈자가 많다. PP-OCRv5 server recognition 모델은 해당 문서에서 중국어 문자로 오인해 채택하지 않았다. Windows 한국어 OCR은 문장은 개선됐지만 숫자와 일부 원문 기호가 흔들렸다. 이 때문에 좌표 기반 융합과 규칙 기반 품질 라우팅이 필요했다.

### 읽기 순서와 문항 파싱

정규식만 사용하면 해설 속 `2.` 같은 텍스트를 새 문항으로 오인한다. 페이지 위치, 단, bbox, 주변 block, 예상 문항 번호를 함께 사용해야 한다. 페이지를 넘겨 이어지는 지문과 표·그림 연결은 추가 개발이 필요하다.

### 한글 자동화와 HWPX 호환성

첫 저장 HWPX가 package validator에는 통과하지만 한글 GUI에서 불안정하게 열리는 사례가 있었다. 임시 HWPX 저장→한글 재개방→최종 재저장 방식으로 정규화했다. 또한 너무 긴 본문 전체를 exact-match 검증 문자열로 사용하면 재개방에 성공해도 실패로 판정돼, 짧은 실제 출력 구절로 검증하도록 수정했다.

### 처리 성능

Paddle와 Windows OCR을 모든 페이지에서 함께 실행하면 정확도는 좋아지지만 처리 시간이 크게 증가한다. 현재는 안정성을 위해 OCR과 한글 COM 작업을 직렬화했다. 향후에는 Paddle 결과를 먼저 평가하고 위험 block만 Windows OCR로 확대 재인식하며 페이지 cache를 사용해야 한다.

## 6. 현재 한계와 실패 사례

- 본문에 드물게 한자·라틴 잡음 문자열이 남는다.
- `㉠~㉤`, 특수 부호, 작은 각주와 저해상도 영어의 완전 복원은 보장하지 못한다.
- 표, 그림, 도형, 수식은 원본과 동일한 편집 객체로 복원되지 않는다.
- passage가 지나치게 긴 한 문단으로 합쳐지는 경우가 있다.
- 실제 검토 화면과 block별 사용자 수정 저장 기능은 아직 완성되지 않았다.
- 페이지별 정확한 진행률, timeout, checkpoint 단위 재개와 재시도 UI가 부족하다.
- Electron worker의 PyInstaller 패키징과 Windows installer가 아직 없다.
- 20회 연속 실제 변환 crash test와 디스크 부족·긴 한글 경로 E2E가 남아 있다.
- PyMuPDF를 상용 배포할 경우 라이선스 검토가 필요하며 PDFium adapter를 고려해야 한다.
- 자체 데이터셋이 없어 CER 1%, question/choice order 99.5%, block F1 0.95 목표는 미측정 상태다.

## 7. 다음 구현 목표

### P0 안정화

1. OCR 엔진 불일치와 비정상 문자를 점수화하는 Quality Router
2. 위험 line 확대·다중 전처리 재인식
3. 문항별 선택지 ①~⑤ 완전성 검사
4. 검토 화면에서 원본 bbox와 텍스트를 나란히 수정
5. 수정 내용을 로컬 SQLite에 저장
6. `result.partial.hwpx` 검증 후 atomic replace
7. worker heartbeat, timeout, checkpoint, cancel/retry 완성
8. 20회 연속 기준 PDF 변환과 메모리 측정

### 배포 가능한 MVP

1. Python worker PyInstaller one-dir 패키징
2. electron-builder Windows installer
3. 별도 앱 아이콘과 코드 서명 준비
4. 모델 manifest, SHA-256 검증, rollback
5. 로그·개인정보 삭제 UI
6. 긴 경로, 공백, 괄호, 한글 파일명 E2E

### 정확도 개선

1. 권리 정보가 확보된 실제 시험지 golden dataset 구축
2. 한국어·영어·숫자·원문 기호별 CER 분리 측정
3. detection 500페이지, recognition line crop 5,000장 이상 목표
4. 합성 시험지와 채점 표시 augmentation
5. layout block F1 및 실제 사람 수정 시간 측정

## 8. 실행과 검사 명령

```powershell
# Electron 빌드와 테스트
npm install
npm run build
npm test

# Python 검사
pytest -q
ruff check .
mypy src

# 실제 PDF 변환
python -m scan2hwpx convert "input.pdf" --output "output/result.hwpx" --dpi 300

# HWPX package 검사
python -m scan2hwpx validate "output/result.hwpx"
```

## 9. 완료의 정의

코드가 실행되거나 HWPX가 한 번 열리는 것만으로 완료라고 판단하지 않는다. 기준 PDF의 문항·선택지 순서, HWPX open rate, OCR 정확도, crash-free job 비율과 페이지당 사람 수정 시간을 회귀 데이터셋에서 측정한 뒤 품질 게이트를 통과해야 한다.
