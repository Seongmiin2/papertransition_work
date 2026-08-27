# 폴더 구조

이 문서는 지금 저장소에 실제로 있는 폴더가 각각 무엇을 위한 것인지, 그리고 목표 상태와 다른 점이 무엇인지 정리한다. `output/`·`models/`·`고1-2/`·`고2/`는 다른 세션이 실시간으로 학습·변환 작업에 쓰고 있어 이 문서는 **설명만 하고 옮기지 않는다.** 실제 이동은 그 작업이 끝난 뒤 별도로 진행한다.

## 소스 코드

| 경로 | 용도 |
|---|---|
| `src/scan2hwpx/` | 변환 파이프라인 Python 패키지(OCR, HWPX 렌더러, 학습 도구 포함) |
| `apps/desktop/` | Electron + React + TypeScript 데스크톱 앱 |
| `scripts/` | 일회성/보조 자동화 스크립트(모델 학습 실행, 바로가기 생성 등) |
| `tests/` | pytest 단위·통합 테스트, `tests/fixtures/`가 유일한 실제 테스트 데이터 폴더 |

## 데이터·모델(참고용, 대부분 Git 제외)

| 경로 | 용도 | 상태 |
|---|---|---|
| `samples/` | 테스트에서 쓰는 기준(golden) PDF 1건 + README | Git 추적(README만) |
| `data/` | `data/raw_pdfs/`에 모을 학습용 원본 PDF 수집 규칙 문서 | Git 추적(README만), 실제 PDF는 제외 |
| `models/` | 배포된 파인튜닝 모델. `models/korean-exam-ppocrv5/`의 3개 파일만 예외적으로 Git 추적, 나머지(사전학습 가중치, 실험용 모델)는 제외 | 혼합 |
| `output/` | 변환 실행 결과, 학습 실행 산출물, 벤치마크 리포트가 모이는 곳. 전체 Git 제외 | 다른 세션이 계속 씀 |
| `고1-2/`, `고2/` | 학년별 실제 시험지 원본(.hwp). 개인정보·저작권 때문에 전체 Git 제외 | 다른 세션이 계속 씀 |
| `templates/` | HWPX 패키지 템플릿 자산 | Git 추적 |
| `configs/` | 벤치마크용 정답 쌍 설정(JSON) | Git 추적 |
| `training/` | PaddleOCR 파인튜닝 설정 | Git 추적 |

## 문서·보고서

| 경로 | 용도 |
|---|---|
| `README.md` | 사용자용 설치·실행 안내 |
| `AGENTS.md` | 이 저장소에서 작업하는 AI 에이전트용 작업 규칙 |
| `DEVELOPMENT_REPORT.md` | 계속 갱신되는 기술 현황 보고서(rolling doc) |
| `STRUCTURE.md` | 이 문서 — 폴더 구조 설명 |
| `MODEL_RESEARCH_2026-08-26.md`, `OPTIMIZATION_REPORT_2026-08-25.md`, `TRAINING_REPORT_2026-08-25.md` | 특정 시점 작업 스냅샷(날짜가 곧 버전) |
| `reports/phase0.md` | 프로젝트 초기 조사 기록 |
| `reports/performance_log.csv` | 벤치마크 결과를 시점별로 누적하는 성과 추적 파일(Excel에서 바로 열림) |

## 그 외

| 경로 | 용도 |
|---|---|
| `.github/workflows/` | CI(테스트·lint·타입체크 자동 실행) |
| `github_publish/` | GitHub 공개 저장소와 연결된 별도의 중첩 git 저장소(퍼블리시 미러). 이 저장소의 `.gitignore`로 제외되어 있으며, 이번 정리 범위 밖 |

## 알려진 미정리 항목(의도적으로 그대로 둔 것)

- `output/vendor/PaddleOCR` — "변환 결과" 폴더 안에 PaddleOCR 원본 저장소 전체(약 248MB)가 들어있어 위치상 어색하지만, 학습 스크립트가 이 경로를 그대로 참조하고 있어 지금 옮기지 않는다.
- `github_publish/`는 최근 커밋들(editable 렌더러, 읽기 순서 수정 등)과 이미 어긋나 있다. 동기화 여부는 별도 결정 사항이다.
