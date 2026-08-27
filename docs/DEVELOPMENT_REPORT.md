# Exam2HWPX 개발 현황 및 기술 보고서

> 최초 작성일: 2026-08-20 / 최신화: 2026-08-24
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

## 10. 2026-08-23 실제 데이터 기반 최신화

### HWP 참조 코퍼스

`test_data`의 HWP 37개를 한컴 COM이나 AGPL 라이브러리 없이 HWP 5 OLE/record 형식으로 직접 읽는 파서를 구현했다. 원본 파일은 수정하지 않으며 `test_data/`를 Git 제외 항목에 추가했다.

- 파싱 성공 37/37, 실패 0
- SHA-256 기준 고유 문서 33개, 완전 중복 4그룹
- 고유 문서 본문 795,434자, 21,675문단
- 표 172개, 그림 82개, 수식 record 6개
- 현재 PDF와 파일명 기준 정확한 HWP 짝 0개

따라서 HWP들은 출력 형태·문항 구조·한국어 시험 문체를 학습시키는 참조 코퍼스이지만, 현재 홍천중 PDF의 직접적인 Y label은 아니다. 서로 다른 시험 내용을 억지로 X-Y로 묶는 학습은 금지했다.

### 학습 데이터와 모델 경로

HWP 문장을 네 가지 한글 글꼴, 회전·흐림·대비·JPEG 열화로 렌더링해 6,000개 recognition crop을 만들었다. 동일 문서가 여러 split에 섞이지 않도록 SHA-256 문서 단위로 분할했다.

- train 5,186장 / 28문서
- validation 674장 / 4문서
- test 140장 / 1문서
- 공식 `korean_PP-OCRv5_mobile_rec` dictionary 11,945자를 보존하고 시험 코퍼스에만 있는 한자·옛한글·기호 286자를 뒤에 추가
- 공식 PP-OCRv5 mobile 구조 기반 fine-tune 설정과 학습 실행 스크립트 추가

이 데이터만으로 모델을 배포하지 않는다. 합성 test는 합성 성능만 나타내므로, 실제 PDF line crop과 사람 정답을 추가한 뒤 기존 모델 대비 CER이 좋아질 때만 승격한다.

### Quality Router와 대량 변환

Windows OCR 결과로 Paddle 결과를 무조건 덮어쓰던 방식을 제거했다. 좌표의 가로·세로 겹침, confidence, 비정상 문자, 사전 적합도, 후보 길이와 edit similarity로 후보를 선택하며 엔진 결과가 다르면 confidence가 높아도 검수 대상으로 남긴다.

- `fast`: Paddle만 사용
- `balanced`: 위험 line만 Windows OCR 재인식
- `accurate`: 전체 페이지 이중 인식
- `batch-convert`: 모델 1회 로드, 파일별 결과 격리, 실패 후 계속, 설정·입력 hash 기반 resume, atomic report
- 한컴 보안 모듈 정지 때문에 기본 renderer를 `portable`로 변경하고 `--renderer hancom`만 명시적 선택으로 유지

### 홍천중 8페이지 실측

300 DPI `balanced` CPU 변환에서 25문항, 756 OCR block, 평균 confidence 0.9381, 검수 56 block을 얻었고 최종 HWPX package validation을 통과했다. 67개 line에서 Windows 후보를 얻었고 Quality Router는 그중 6개를 선택했으며, 48개 엔진 불일치를 검수 대상으로 남겼다.

총 시간은 한컴 자동화 정지 대기 약 2분을 포함해 632초였다. OCR 구간만 약 500초로 페이지당 약 1분이었다. 150/200/240 DPI 첫 페이지도 이 Windows CPU 빌드에서는 약 60초로 큰 차이가 없었고 oneDNN 가속은 런타임에서 실패해 자동 비활성화됐다. 결론적으로 CPU 경로는 기능 검증·저속 배치용이고, 돈을 받는 처리량 목표에는 NVIDIA GPU, 고성능 inference backend 또는 별도 GPU worker가 필요하다.

2026-08-24에 이 PC의 RTX 2060이 있음에도 CPU 전용 `paddlepaddle`이 설치되어 있음을 확인했다. 같은 버전의 CUDA 12.9용 `paddlepaddle-gpu 3.3.1`로 교체한 뒤 `cuda=True`, `gpu:0`과 실제 GPU 행렬 연산을 확인했다. 동일한 240 DPI 첫 페이지 `fast` OCR은 모델 첫 로딩을 포함해 6.951초, 102줄로 측정되어 CPU 약 60초 대비 약 8.6배 빨라졌다. 데스크톱 앱은 `device=auto`로 GPU를 자동 선택하며, 페이지 번호를 반영해 진행률을 표시하도록 수정했다.

같은 날 GPU 전체 변환은 8페이지를 29.736초에 완료했지만 기존 `portable` 생성기가 ZIP/XML well-formed 여부만 검사해 한글 2020에서 손상 파일로 거부되는 결함을 발견했다. 문제 파일은 헤더가 123바이트뿐이고 section namespace, OPF namespace, container media type, version/settings namespace가 한글 규격과 달랐다. 한글 2020이 직접 저장한 패키지를 개인정보 제거 후 6.7KB 템플릿으로 내장하고, 본문만 규격 문단으로 교체하도록 생성기를 바꿨다. 복구본은 실제 한글 창에서 열리는 것을 확인했고, validator는 이전 손상 파일에서 14개 구조 오류를 검출하도록 강화했다.

현재 자동 검사는 Python 38개, TypeScript 6개와 Ruff, mypy strict를 통과한다.

## 11. 2026-08-24 페이지·표·그림 보존 기준 재정의

### 실패 기준과 실측

복구된 편집형 HWPX를 한글에서 다시 측정한 결과 원본은 8페이지인데 변환본은 16페이지였고, 한글 표 개체 0개·그림 개체 0개였다. 파일이 열리고 OCR 문자가 들어 있다는 사실만으로는 변환 성공이 아니다. 이 결과는 품질 기준에서 폐기한다.

새 품질 게이트는 다음과 같다.

1. 원본과 변환본의 페이지 수가 같아야 한다.
2. 표·그림·지문 상자·머리말·꼬리말이 같은 페이지와 좌표에 있어야 한다.
3. 굵기, 정렬, 문단 간격, 셀 안쪽 여백을 별도 스타일 속성으로 보존해야 한다.
4. 파일 손상 검사와 한글 실제 재열기 검사를 모두 통과해야 한다.
5. 시각 보존과 편집 가능성을 별도 지표로 보고해야 한다.

### 외형 보존 렌더러

한글 2020이 직접 만든 정상 HWPX의 페이지 그림 구조를 정제해 템플릿으로 내장했다. `fidelity` 렌더러는 PDF 한 페이지를 한글 그림 한 개와 명시적 페이지 나눔 하나로 기록한다. 한글 자동화를 사용하지 않고 ZIP/XML을 직접 생성하므로 숨은 대화상자와 보안 모듈 정지를 피한다.

- 기준 PDF: 8페이지
- 출력 HWPX: 8페이지
- 한글 그림 개체: 8개
- 페이지 나눔: 7개
- 한글 표 개체: 0개
- HWPX validator: 오류 0, 경고 0
- 한글 2020 실제 재열기: 성공, 손상 경고 없음
- 한글 COM 재측정: 8페이지·그림 8개
- HWPX 패키징: 3.459초
- GPU OCR 포함 전체 변환: 31.631초

이 모드는 표·그림·굵기·정렬을 픽셀 수준으로 보존하지만, 페이지 전체가 하나의 그림이므로 표와 본문을 개별 편집할 수 없다. 따라서 상용 결과의 안전한 시각 기준본이며 편집형 복원의 완료본은 아니다. 앱과 CLI의 기본 렌더러를 이 모드로 변경했고, 기존 16페이지 결과는 `result.editable-16page.hwpx`로 보관했다.

### 시험지 전용 레이아웃 학습 데이터

PaddleOCR 3.7의 `PP-DocLayout-S`와 OpenCV 선 구조 검출을 결합한 `build-layout-dataset` 명령을 추가했다. generic 모델이 시험지의 큰 테두리와 지문 상자를 표로 오인하는 문제를 줄이기 위해 페이지 면적 25% 이상인 표 예측은 `page_frame` 검수 후보로 분리하고, 실제 수평·수직 선 개수로 `table`과 `passage_box`를 구분한다.

홍천중 8페이지에서 생성된 시드 라벨은 181개다.

- text 106
- passage_box 23
- page_frame 11
- table 3
- image 1
- footer 22
- page_number 15

COCO 형식 `annotations/instances_seed.json`, 상세 JSON, 페이지별 overlay를 생성한다. 모든 라벨 상태는 `pseudo_label_needs_human_review`로 기록한다. 이 예측을 그대로 다시 학습하면 모델의 오답을 복제하므로, 표·지문 상자·문항·선택지를 사람이 교정하기 전에는 fine-tune을 실행하거나 정확도 향상을 주장하지 않는다.

현재 `test_data` HWP 37개는 표 172개와 그림 82개를 포함하지만 홍천중 PDF의 직접 Y가 아니다. 다음 단계는 HWP를 한글로 렌더링해 페이지 이미지와 실제 개체 좌표를 함께 추출하고, 문항·선택지·지문 상자 라벨을 교정한 문서 단위 train/validation/test로 PP-DocLayout을 fine-tune하는 것이다. 최종 렌더러는 원본 페이지 그림을 검증 배경으로 유지하면서 검증된 표는 한글 표, 그림은 잘라낸 그림 개체, 텍스트는 좌표 고정 문단/글상자로 교체해야 한다.

현재 자동 검사는 Python 42개, TypeScript 6개와 Ruff, mypy strict를 통과한다.

### 편집 개체 오버레이 시제품

한글 표 3×2를 실제 한글 2020으로 생성해 정상 XML 구조와 테두리 스타일을 템플릿화했다. `semantic` 렌더러는 외형 보존 페이지를 배경으로 유지하면서 선 교차점으로 검출한 셀을 한글 표 개체로 만들고, 셀 좌표 안의 OCR block을 셀 텍스트로 넣는다. 레이아웃 모델의 `image` 영역은 원본에서 잘라 별도 그림 개체로 같은 좌표에 배치한다.

홍천중 전체 파이프라인 실측 결과:

- 총 시간: 40.623초
- 페이지: 8
- OCR block: 740
- 문항: 25
- 평균 OCR confidence: 0.93277
- 검수 block: 51
- 한글 그림 개체: 9개(페이지 배경 8 + 분리 그림 1)
- 한글 표 개체: 2개
- 첫 표: 2행×4열로 복원
- 7페이지 표: 부분 선 조각을 합쳐 4행×2열 외곽 구조로 복원
- 1페이지 둘째 행: 끊긴 세로선을 판정해 앞 2칸을 실제 `colSpan=2` 병합 셀로 복원
- 셀 문단: 좁은 제목 칸 가운데 정렬, 본문 칸 왼쪽 정렬, 9pt 일반/굵게 문자 스타일 분리
- HWPX validator: 오류 0
- 한글 2020 실제 재열기와 COM 개체 재측정: 8페이지·그림 9·표 2

페이지 5의 그림 패널이 선 구조만 보면 표처럼 보이는 오인을 막기 위해, Paddle `image` 예측과 45% 이상 겹치는 OpenCV 표 후보는 표 생성에서 제외했다. 데스크톱 앱은 현재 `semantic` 렌더러를 사용한다. 외형 기준본과 이전 실패본은 각각 `result.fidelity-8page.hwpx`, `result.editable-16page.hwpx`, `result.corrupt.hwpx`로 보관했다.

이 단계도 최종 완성은 아니다. 부분 선 조각을 합쳐 누락 행은 보정했지만 병합 셀 토폴로지는 아직 복원하지 않고, 셀 내부 굵기·정렬·문단 간격도 원본에서 직접 추정하지 않는다. 페이지 배경 때문에 외형은 유지되지만 표 밖의 본문은 아직 독립 편집 개체가 아니다.

## 12. 2026-08-27 프로젝트 로드맵과 8개 요청 현황

이 시점 이후 이 프로젝트는 한 세션이 아니라 여러 세션(도구)이 같은 git 저장소 위에서 순차적으로 작업하고 있다. `editable` 렌더러(페이지 배경 없이 좌표 고정 문단·표·그림만으로 구성, 실제 문서로 712 text run·19,268자 검증 완료)가 데스크톱 앱 기본값이 되었고, 2단 시험지 읽기 순서 수정과 일부 죽은 코드 정리가 끝났다. `MODEL_RESEARCH_2026-08-26.md`에 OCR 모델 선정 연구(PP-OCRv5 mobile rec 채택, v6/VL 기각 사유, 파인튜닝 2라운드 84.09%→95.89%/94.43%)가 이미 기록되어 있다 — **모델 선정 근거는 이 문서를 참고하며, 별도 재조사는 하지 않는다.**

아래는 "범용성·동시처리·배치 UI·죽은 코드 정리·폴더 구조·환경·모델링 연구·성과 추적"이라는 8개 요청을 계획(→환경→구조→모델링→실험→성과추적) 순서에 맞게 정리한 현황이다.

| # | 요청 | 상태 | 비고 |
|---|---|---|---|
| 6 | 환경 세팅 | 대부분 완료, 자동화만 추가 | README 설치 절차·RTX 2060 GPU 조합 검증 완료. CI 자동화(`.github/workflows/ci.yml`)만 신규 추가 |
| 5 | 폴더 구조 정리 | 문서화 진행 | `data/README.md` 손상 복구, `STRUCTURE.md` 신규(목표 구조 명시, `output/`·`models/`·`고1-2/`·`고2/`는 이동하지 않고 문서화만) |
| 7 | 모델링 연구·적용 계획 | 완료 | `MODEL_RESEARCH_2026-08-26.md` 참고, 재조사 금지 |
| 8 | 성과 추적 파일(엑셀) | 신규 착수 | `reports/performance_log.csv` + `scripts/log_benchmark_metrics.py`, 기존 벤치마크 JSON을 append 방식으로 통합 |
| 3 | 배치 UI(여러 파일 동시 입력/출력) | UI는 이미 완료, 동시 처리만 남음 | Electron 앱이 이미 다중 파일 드래그/선택(`MAX_BATCH_FILES=10`)과 다중 결과 표시를 지원. 실제로 빠진 건 동시 처리(2번과 동일 원인) |
| 2 | 동시다발 처리 | 미착수, 설계만 확정 | 데스크톱 앱은 작업을 한 번에 1개씩만 처리(`main.ts`의 단일 `activeJobId`). GPU VRAM(RTX 2060 6GB) 한도 실측 후 동시 처리 수 확정 예정 |
| 1 | 범용성(텍스트 기반 PDF도 원본 양식대로) | 미착수 | 현재는 텍스트가 이미 있는 PDF도 무조건 래스터라이즈 후 OCR. `pipeline.py`의 `inspect_pdf()`가 페이지별 텍스트 문자 수를 이미 계산하지만 분기에 쓰이지 않음 |
| 4 | 불필요한 코드/파일 정리 | 1차 완료, 재확인 필요 항목 있음 | Hancom 미사용 함수·Tkinter 프로토타입 삭제 완료. `vision/formulas.py`의 미사용 pluggable 추상화, `ir/models.py`의 `Asset`/`ReprocessRecord`는 재확인 완료, 삭제 대기 |

이번 로드맵에서 `output/`, `models/`, `고1-2/`, `고2/`, `data/`(내용)는 다른 세션이 실시간으로 쓰고 있는 학습 산출물/원본 데이터이므로 물리적으로 옮기거나 이름을 바꾸지 않는다. 1·2·4번(코드 변경)은 `src/scan2hwpx/hwpx/semantic.py` 등 최근 활발히 수정되는 파일과 인접해 있어, 해당 파일들의 수정이 잠잠해진 뒤 순서대로 진행한다.

## 13. 2026-08-27 AI 수명주기 구조 개편

학습 종료 후 위 12절의 동시 작업 제약이 해소되어, AI 자산을 실제로 다음 네 단계로 재구성했다.

- `ai/datasets/`: 로컬 원본, 데이터 품질 설정, 데이터셋 생성기
- `ai/modeling/`: 학습 설정·코드, 실험별 재현 메타데이터
- `ai/results/`: 누적 성과 CSV, 성장표, 과거 보고서
- `ai/production/`: 앱 배포 모델과 평가 후보

`output/`의 약 10GB 생성물은 과거 실험의 절대경로와 재현성을 보존하기 위해 이동하지 않았다. 원본 HWP/PDF/DOCX와 pretrained 가중치는 새 단계 아래로 옮겼지만 계속 Git에서 제외한다.

오늘 완료한 OCR v3 실험은 6 epoch 학습과 inference export까지 성공했으나, 동일한 2,921행 test에서 v2보다 완전일치가 한 줄 적고 편집 유사도도 낮아 배포하지 않았다. v2는 `ai/production/deployed/`에 유지하고 v3는 `ai/production/candidates/`에 보존했다. 비교 지표와 판정은 `ai/results/performance_log.csv` 및 `ai/modeling/experiments/`에서 추적한다.
데스크톱 앱의 모델 로딩 경로도 새 production 위치로 바꾸고, 작고 필수적인 page-anomaly 가중치를 저장소에 포함해 새 clone에서 동일한 품질 라우팅 구성이 재현되도록 했다.
