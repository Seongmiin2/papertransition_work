# 폴더 구조

저장소는 제품 코드와 AI 수명주기를 분리한다. AI 자산은 `Dataset → Modeling → Results → Production` 순서로 이동하며, 큰 생성물과 민감할 수 있는 원본은 Git에 올리지 않는다.

```text
papertransition_work/
├─ apps/                         Electron 데스크톱 앱
├─ src/scan2hwpx/                Python 제품 파이프라인
├─ tests/                        단위·통합 테스트
├─ ai/
│  ├─ datasets/
│  │  ├─ source/                 로컬 원본 HWP/PDF/DOCX (Git 제외)
│  │  ├─ configs/                데이터 품질 기준과 정답 쌍 설정
│  │  └─ build_hwp_line_dataset.py
│  ├─ modeling/
│  │  ├─ configs/                PaddleOCR 학습 설정
│  │  ├─ experiments/            실험별 데이터·파라미터·지표·판정
│  │  ├─ pretrained/             다운로드한 기반 가중치 (Git 제외)
│  │  └─ train_korean_ocr.py
│  ├─ results/
│  │  ├─ performance_log.csv     모든 실험의 누적 지표
│  │  ├─ record_metrics.py       벤치마크 JSON을 CSV로 적재
│  │  └─ history/                날짜별 연구·학습 보고서
│  └─ production/
│     ├─ deployed/               앱이 실제 로드하는 승인 모델
│     └─ candidates/             평가했지만 배포하지 않은 후보
├─ docs/                         제품 개발 문서
├─ output/                       실행·학습·벤치마크 생성물 (Git 제외)
├─ samples/                      재현용 입력 위치 안내
├─ templates/                    HWPX 템플릿
└─ scripts/                      앱 실행·검증 보조 스크립트
```

## 관리 원칙

1. 원본 데이터는 `ai/datasets/source/`에 두고 Git에는 올리지 않는다.
2. 데이터셋 생성 결과와 checkpoint는 `output/`에 두고, 재현 정보만 `ai/modeling/experiments/`에 기록한다.
3. 비교 가능한 지표는 `ai/results/performance_log.csv`에 누적한다.
4. held-out test와 제품 smoke test를 통과한 모델만 `ai/production/deployed/`로 옮긴다.
5. 탈락한 실험도 삭제하지 않고 메타데이터와 후보 artifact를 남겨 같은 시도를 반복하지 않게 한다.
