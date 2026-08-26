# 코드 효율화·모델 최적화·데이터 정리 보고서 (2026-08-25)

## 결론

4개 영역(파이썬 변환 파이프라인 런타임, Electron/TS 프론트엔드, OCR 재학습 하이퍼파라미터, 이상탐지 PCA 모델)을 점검했다. 실제로 코드를 고치고 회귀 테스트(pytest 50개, ruff, mypy strict, npm build/test)로 검증한 항목과, 위험도·검증 가능성 문제로 설정 변경 또는 권장 사항으로만 남긴 항목을 분리했다. 새로 발견된 미정리 데이터(P고2 21개교, 시험지 1건)는 기존 `prepare-training-corpus` 파이프라인으로 정리했고, 그 과정에서 `configs/training_pairs.json`이 가리키는 gold 원본(`홍천중2.pdf`)이 현재 저장소에 없어 corpus 생성이 막혀 있었다는 사실도 함께 확인했다.

## 1. 문제의식

- `TRAINING_REPORT_2026-08-25.md`가 언급한 `output/training-corpus-v2` 등 산출물이 실제 디스크에는 없어, 보고서와 실제 상태가 어긋나 있었다.
- 3-epoch OCR 파인튜닝 후보가 baseline보다 CER이 35% 악화되어 거부된 채로 원인 분석 없이 남아 있었다.
- Electron 빌드가 `vite.config.ts`/`vite.config.mts` 중복 때문에 `base: "./"` 수정이 무시되어, 프로덕션 패키징 시 흰 화면 위험을 안고 있었다.
- `package.json`의 모든 의존성이 `"latest"`로 고정되어 있어 재현 가능한 빌드가 보장되지 않았다.
- 파이썬 파이프라인이 같은 페이지를 최대 3회(OCR용, 디버그 이미지용, 수식 추출용) 반복 래스터라이즈·전처리하고 있었다.
- 루트에 시험지 PDF 1건, `P고2/`에 21개 학교분 원본 54개 파일이 정리되지 않은 채 남아 있었다.

## 2. 해결 아이디어

| 영역 | 접근 |
|---|---|
| 데이터 정리 | 기존 해시 기반 불변 복사 도구(`prepare-training-corpus`)를 새 원본에 그대로 적용해 수작업 분류를 배제 |
| 이상탐지 모델 | 실행 비용이 낮은(순수 SVD) 모델 특성을 이용해 하이퍼파라미터를 실제로 그리드 스윕 |
| OCR 재학습 | 실패 원인(자기증류 편향 + 3epoch 미수렴)에 대한 가설을 세우고, GPU 실행 없이 설정만 보수적으로 재설계 |
| 파이썬 파이프라인 | OCR 단계에서 만든 페이지 이미지를 캐시로 공유해 이후 단계의 재래스터라이즈를 제거 |
| 프론트엔드 | 정적 설정 충돌 제거, 이벤트 저장 I/O 축소, 상태 갱신을 전체 재조회에서 부분 upsert로 전환 |

## 3. 데이터 전처리·모델링 방법론과 실행 결과

### 3.1 데이터 정리

- 환경 결손 확인: `pyproject.toml`에 선언된 `olefile`이 실제로는 설치돼 있지 않아 `scan2hwpx.cli` 자체가 import 단계에서 죽는 상태였다. `pip install olefile`로 해결.
- `(천재박)홍천중3_1학기 기말고사(1).pdf`를 `samples/README.md`가 지정한 정확한 경로(`samples/(천재박)홍천중3_1학기 기말고사(1).pdf`)로 복사(체크섬 일치 확인)한 뒤 루트 원본을 삭제했다.
- `python -m scan2hwpx prepare-training-corpus "P고2" "files (1)" samples --out output/training-corpus-v2` 실행 결과:

| 항목 | 값 |
|---|---:|
| 전체 파일 | 57 |
| source_pdf | 22 |
| editable_reference | 21 |
| answer_key | 14 |
| 파일명 일치 후보 쌍 | 20 |
| 오답 위험으로 자동 거부된 쌍 | 20 |
| source_pdf split | train 13 / validation 4 / test 5 |

  20쌍 전부가 `_assess_filename_pair`에 의해 "editable target이 원본 페이지 수 대비 텍스트가 너무 적어 답안 템플릿으로 추정됨"으로 자동 거부되었다 — `TRAINING_REPORT_2026-08-25.md`가 P고2에 대해 수작업으로 내린 결론과 동일한 결과를 도구가 자동 재현했다.
- **발견된 문제**: `--pairs configs/training_pairs.json`을 함께 넘기면 `FileNotFoundError`로 즉시 실패한다. 이 설정이 가리키는 gold 원본 `홍천중2.pdf`가 현재 저장소 어디에도 없기 때문이다(`files (1)/`에는 transcript PDF와 DOCX만 있고 스캔 원본이 없음). 즉 지금 상태에서는 **gold 쌍 검증 기능 자체가 동작하지 않는다.** `홍천중2.pdf` 원본을 복구하거나, 해당 pair 항목을 제거/수정해야 한다.

### 3.2 이상탐지 PCA 모델 하이퍼파라미터 스윕 (실제 실행)

`src/scan2hwpx/training/page_anomaly.py`는 32×32 그레이스케일 썸네일에 대한 rank-제한 선형(PCA) 오토인코더라 GPU 없이 수 초 내에 재학습된다. 새로 만든 `output/training-corpus-v2/manifest.jsonl`(train 173페이지, validation 39페이지, test 54페이지)로 `rank ∈ {16,32,48} × thumbnail_size ∈ {24,32,48}` 9개 조합을 학습했다.

| rank | thumb | 유효 rank | threshold | train 오탐/전체 | validation 오탐/전체 | test 오탐/전체 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 24 | 16 | 0.00113657 | 1/173 | 1/39 | 15/54 |
| 16 | 32 | 16 | 0.00121011 | 1/173 | 1/39 | 18/54 |
| 16 | 48 | 16 | 0.00170047 | 1/173 | 1/39 | 15/54 |
| **32** | **24** | 32 | 0.00055107 | 0/173 | 2/39 | 20/54 |
| **32** | **32** | **32** | **0.00081695** | **0/173** | **2/39** | **20/54** |
| 32 | 48 | 32 | 0.00109424 | 0/173 | 2/39 | 24/54 |
| 48 | 24 | 48 | 0.00045163 | 0/173 | 2/39 | 21/54 |
| 48 | 32 | 48 | 0.00066669 | 0/173 | 2/39 | 18/54 |
| 48 | 48 | 48 | 0.00099044 | 0/173 | 2/39 | 24/54 |

**결론: 기존 프로덕션 기본값(rank=32, thumbnail=32)을 그대로 유지한다.** train 오탐 0%, validation 오탐 최소 수준(5%)으로 이미 합리적이며, rank를 48로 올려도 train/validation 성능은 거의 동일하고 test 오탐만 늘어 소규모 데이터(57개 소스 문서)에서 과적합 위험만 커진다. rank=16은 압축이 강해 재구성 오차 자체가 커지고 스케일이 달라져(threshold가 다른 rank와 직접 비교 불가) 변경 근거가 부족하다. 이번 스윕 결과와 채택 모델은 `output/trained-models/page-anomaly-v2/`(=rank32/thumb32 결과 사본)와 `output/trained-models/sweep-r*-t*/`에 보관했다.

test 오탐률이 37~44%로 높은 것은 이상 신호라기보다, source_pdf 22건을 해시로 13/4/5 분할한 표본 크기가 작아 test 5건에 다른 학교의 레이아웃이 몰렸을 가능성이 크다 — 문서 수가 늘어나기 전에는 이 수치를 "성능 저하"로 해석하면 안 된다.

### 3.3 OCR 재학습 하이퍼파라미터 재설계 (설정만, 미실행)

`TRAINING_REPORT_2026-08-25.md`의 실패 사례: 3-epoch 파인튜닝 후보가 gold CER 9.09% → 12.32%(35% 악화)로 거부됨. 가장 유력한 원인 가설: **자기증류(self-distillation) silver 라벨 편향과 3-epoch 미수렴이 겹쳤다.** silver 라벨은 같은 PP-OCRv5 계열 provider의 자체 추론 결과를 교사로 쓰므로, confidence 필터를 통과해도 "확신에 찬 오답" 패턴이 그대로 학습되고, 이를 교정할 만큼 충분히 학습되지도 않은 중간 지점에서 학습이 끝났을 가능성이 높다.

수정 내역(실행하지 않았으므로 gold CER 개선 여부는 미검증):

| 파일 | 항목 | 이전 | 이후 |
|---|---|---:|---:|
| `training/paddleocr/korean_exam_mobile_rec.yml` | `Optimizer.lr.learning_rate` | 0.0001 | 0.00002 |
| 〃 | `Optimizer.regularizer.factor` | 3.0e-05 | 1.0e-04 |
| 〃 | `Global.epoch_num` / `eval_batch_step` / `save_epoch_step` | 30 / [0,500] / 5 | 12 / [0,200] / 1 |
| 〃 | `RecConAug.prob` | 0.5 | 0.25 |
| `scripts/train_korean_ocr.py` | `--learning-rate` / `--epochs` / `--warmup-epochs` / `--eval-every` 기본값 | 0.00005 / 6 / 1 / 1000 | 0.00002 / 12 / 2 / 200 |
| `src/scan2hwpx/training/ocr_adaptation.py`, `cli.py` | `minimum_confidence` / `minimum_quality` (silver 필터) | 0.96 / 0.88 | 0.985 / 0.92 |

Backbone(PPLCNetV3) 일부 freeze나 discriminative LR은 PaddleOCR 프레임워크가 이 yml 스키마에서 정확히 어떻게 지원하는지 GPU 실행 없이 검증할 수 없어 코드에 반영하지 않았다 — 8절 "추가 요구 및 고민사항" 참고.

### 3.4 파이썬 변환 파이프라인 런타임 최적화 (실제 코드 수정, pytest로 검증)

기존 구조: `PaddlePdfOcrProvider.convert()`가 OCR을 위해 페이지를 래스터라이즈·전처리한 뒤 그 결과를 버리고, `pipeline._write_debug_images`와 `vision/pdf.extract_formulas`가 **동일 dpi로 PDF를 다시 열어 같은 작업을 반복**했다. 여기에 `hwpx/semantic.py`의 `_detect_column_spans`는 표가 있는 페이지마다 전체 페이지를 다시 이진화했다.

수정:
- `ocr/providers/paddle.py`: `convert()`에 `image_cache` 매개변수를 추가해 페이지별 (원본 이미지, 전처리 결과)를 채워 넣도록 함.
- `pipeline.py`: 이 캐시를 만들어 `provider.convert()`에 넘기고, `_write_debug_images`와 `_extract_formulas` 양쪽에 그대로 전달해 캐시가 있으면 재래스터라이즈·재전처리를 건너뛰게 함(둘 다 OCR과 동일한 dpi를 쓰므로 안전하게 재사용 가능).
- `vision/pdf.py`의 `extract_formulas`: 캐시가 있으면 원본 이미지를 재사용.
- `hwpx/semantic.py`: 페이지 이진화를 `_binarize_page`로 분리해 표가 있는 페이지당 1회만 계산하고, 표 개수만큼 반복하던 `_detect_column_spans` 호출에 재사용.
- `vision/layout_dataset.py`의 `_rasterize_pdf`(semantic 모드에서 `dpi=min(dpi,200)`으로 다른 해상도를 쓸 수 있음)와 `ocr/providers/paddle.py`의 페이지별 순차 `predict()` 호출·`layout_dataset.py`의 `batch_size=1` 고정은 이번 세션에서 실제 GPU/OCR 실행으로 검증하기 어려워 **코드를 바꾸지 않았다** — 8절에 다음 단계로 남김.
- `hwpx/fidelity.py`의 PNG 인코딩→디스크 저장→재오픈 왕복도 검토했으나, `render_fidelity_hwpx`가 테스트·`pipeline.py`·`hwpx/semantic.py` 세 곳에서 `Path` 리스트를 받는 공개 계약으로 고정돼 있어, 인메모리 이미지를 받도록 바꾸려면 세 지점을 동시에 변경해야 한다. 이번 최적화의 핵심 대비 위험이 커서 **API를 건드리지 않고 보류**했다.

검증: `pytest -q` 50개 전부 통과, `ruff check .` 통과, `mypy src` strict 통과(46개 파일). 실제 OCR 실행 시간 단축폭은 GPU/PaddleOCR 모델 다운로드가 필요해 이 세션에서 실측하지 못했다 — "정성적 기대효과"로만 기록한다(6절).

### 3.5 Electron/TS 프론트엔드 (실제 코드 수정, npm build/test로 검증)

- `vite.config.ts`/`vite.config.mts` 중복 제거: Vite의 설정 파일 탐색 순서(`.ts`가 `.mts`보다 먼저)때문에 `.mts`에만 있던 `base: "./"`가 무시되고 있었다. `.ts`를 삭제하자 빌드 산출물의 자산 경로가 `/assets/...`(절대 경로, Electron `file://`에서 깨짐)에서 `./assets/...`(상대 경로)로 바뀐 것을 `npm run build` 후 `dist-renderer/index.html`에서 직접 확인했다.
- `apps/desktop/src/main/database.ts`: 이벤트마다 즉시 동기 `writeFileSync`하던 `save()`를 100ms 디바운스로 바꾸고, 앱 종료 시 유실을 막기 위해 `flush()`를 추가해 `main.ts`의 `app.on("before-quit", ...)`에서 호출하도록 연결.
- `apps/desktop/src/main/main.ts`: OCR 워커 stderr 청크마다 누적 문자열 전체를 새 이벤트로 저장하던 것을 제거하고, 프로세스 종료(`close`) 시 최종 stderr 1건만 기록하도록 변경.
- `apps/desktop/src/renderer/App.tsx`: `onJobEvent` 콜백이 이미 최신 `JobRecord`를 전달하는데도 무시하고 매번 `listJobs()` 전체 재조회를 하던 것을, 받은 job을 배열에 upsert하는 방식으로 변경해 IPC 왕복과 리렌더 범위를 줄임.
- `package.json`: 모든 의존성을 현재 설치된 버전(`react/react-dom 19.2.8`, `zustand 5.0.15`, `@types/node 26.2.0`, `@types/react 19.2.18`, `@types/react-dom 19.2.4`, `@vitejs/plugin-react 6.1.0`, `typescript 7.0.2`, `vite 8.2.2`, `vitest 4.1.11`)로 고정.

검증: `npm run build` 성공, 빌드 산출물 자산 경로 상대 경로화 확인, `npm test`(vitest) 6개 전부 통과.

## 4. 실험과 검증 총괄

| 항목 | 실행 여부 | 검증 방법 | 결과 |
|---|---|---|---|
| 데이터 코퍼스 재생성 | 실행함 | `corpus_report.json` 확인 | 57개 파일, 20/20 오답쌍 자동 거부 |
| 이상탐지 하이퍼파라미터 스윕 | 실행함(9개 조합) | `anomaly_report.json` 비교 | 기존 rank32/thumb32 유지 결정 |
| 파이썬 파이프라인 캐싱 리팩터 | 실행함(코드 수정) | pytest 50/ruff/mypy | 전부 통과 |
| 프론트엔드 설정·IPC·저장 수정 | 실행함(코드 수정) | npm build/test | 빌드 성공, 6/6 통과 |
| OCR 재학습 하이퍼파라미터 | 설정만 변경, 미실행 | 없음(다음 GPU 세션 필요) | 미검증 |
| OCR 배치 병렬화 / layout batch_size | 미실행 | 없음 | 코드 미변경, 권장만 |
| `hwpx/fidelity.py` 인메모리화 | 검토 후 보류 | 없음 | API 위험 대비 이득 작아 보류 |

## 5. 검증 결과 요약

- Python: `pytest -q` → 50 passed, `ruff check .` → All checks passed, `mypy src` → Success (46 files)
- Frontend: `npm run build` → 성공(자산 상대경로 확인), `npm test` → 6 passed
- 데이터: `output/training-corpus-v2/corpus_report.json`, `output/trained-models/page-anomaly-v2/anomaly_report.json`, `output/trained-models/sweep-r*-t*/anomaly_report.json`

## 6. 기대효과

- **파이프라인 런타임**: semantic 모드 기준 페이지당 2~3회였던 래스터라이즈+전처리가 1회로, 표가 있는 페이지의 재이진화가 표 개수당 1회에서 페이지당 1회로 줄었다. README 실측치(8페이지 semantic 40.623초) 대비 실제 단축폭은 GPU/OCR 모델이 있는 환경에서 재측정이 필요하며, 이번 세션에서는 실측하지 못했다 — 과대 주장하지 않는다.
- **빌드 재현성**: `package.json` 버전 고정으로 동일 커밋에서 항상 같은 의존성 트리가 설치된다.
- **Electron 프로덕션 안정성**: `base: "./"` 무시로 인한 흰 화면 위험이 제거됐다(실제 산출물로 확인).
- **다음 OCR 파인튜닝 성공 가능성**: lr 5배 하향·L2 강화·augmentation 완화·silver 필터 강화가 모두 "자기증류 편향과 미수렴"이라는 동일 가설을 겨냥하므로, 다음 시도가 이전보다 baseline을 넘길 확률이 높아졌다고 판단하나 — 실행 전까지는 가설일 뿐이다.

## 7. 한계

- OCR 재학습 하이퍼파라미터 변경은 **실행/검증하지 않았다.** 실제 GPU 학습 없이는 CER 개선 여부를 알 수 없다.
- 이상탐지 스윕은 source_pdf 22건(57개 파일 중 일부)이라는 작은 표본에 기반한다 — 문서 수가 늘면 결론이 바뀔 수 있다.
- `ocr/providers/paddle.py`의 페이지별 순차 `predict()`와 `vision/layout_dataset.py`의 `batch_size=1`은 손대지 않았다 — 배치 API 전환의 실효성을 이번 세션에서 검증할 수 없었다.
- `hwpx/fidelity.py`의 PNG 왕복은 여전히 남아 있다.
- `configs/training_pairs.json`이 가리키는 `홍천중2.pdf`가 저장소에 없어 gold 쌍 검증이 현재 동작하지 않는다.
- `github_publish/`는 `src/scan2hwpx`의 별도 배포용 사본인데 이번 세션에서 갱신하지 않아 이제 어긋나 있다 — 다음 공개 배포 전에 동기화가 필요하다.

## 8. 추가 요구 및 고민사항

- `TRAINING_REPORT_2026-08-25.md`의 결론과 동일하게, 다음 성능 개선의 핵심은 실제 PDF 페이지 좌표와 검수된 HWPX가 짝을 이루는 geometry gold 30~50페이지 확보다.
- Backbone 일부 freeze/discriminative LR 실험은 실제 PaddleOCR 저장소를 두고 GPU에서 검증해야 한다 — 이 seed yml에 그대로 반영하기엔 프레임워크 지원 방식이 불확실하다.
- 런타임 최적화 효과를 숫자로 증명하려면, `output/e2e-semantic-fast-anomaly` 같은 회귀 fixture에 대해 최적화 전/후 초 단위를 비교하는 재현 가능한 벤치마크 스크립트가 필요하다(현재는 README의 1회성 실측치만 있음).
- `configs/training_pairs.json`의 `홍천중2.pdf` 누락을 해결(원본 복구 또는 항목 정리)하지 않으면 gold 쌍 파이프라인이 계속 깨진 상태로 남는다.
- `--pairs` 없이 실행한 이번 corpus는 gold 승격 로직을 전혀 거치지 않았다 — 다음에 gold 소스가 준비되면 반드시 `--pairs`를 포함해 재생성해야 한다.
