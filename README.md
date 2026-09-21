# Exam2HWPX

현재 구현은 한국어 시험지 PDF를 로컬 OCR로 분석해 편집 가능한 HWPX로 변환하는 Windows 데스크톱 앱입니다. 이 코드는 새 제품의 OCR·HWPX 자산으로 유지합니다.

새 제품의 구속 방향은 **한글 문서 재제작 공장형 웹서비스**입니다. 개발 우선순위는 `신뢰 가능한 Model A/B → 완성된 문서 제품 → 안정적 배포 → 서비스 UX`이며, 고정 파이프라인은 `원본 → EvidenceIR → Model A → ContentIR → Model B → HwpDocumentPlan → 결정론적 HWPX compiler → validators → auto-pass/review/block`입니다. 상세 책임과 변경 절차는 [ADR 0001](docs/architecture/0001-model-first-document-factory.md)을 따릅니다.

## 현재 상태

- 앱이 현재 로드하는 OCR 프로토타입: `korean-exam-ppocrv5-20260826-v2`
- 과거 보고값 95.5701%(정규화 편집 유사도 98.1530%)는 이전 train 문서가
  후속 test에 포함된 계보 누출 때문에 독립 성능 근거로 사용할 수 없음
- 2026-08-27 v3 실험: 테스트 지표가 v2보다 낮아 후보로 보존하고 배포하지 않음
- 기존 `blueprint` Model A/B는 규칙 기반 대역이지만, 제품 경계에는 Model A의 정확히
  1페이지 단위 provider-neutral 실행·constrained response 검증·결정론적 document barrier와
  Model B의 설계 전용 단일 실행·응답 검증이 구현됨. 실제 provider·승인 base weight·checkpoint는 없음
- v4 재제작 후보: 41문서·485쪽. 이미지 노드 91건은 파일 SHA·PNG 형식·Evidence 연결은
  일치하지만 전부 `needs_review=true`이고 41문서 모두 권리가 미확인이다. 사람 검수와
  권리 확인 전까지 `training_eligible=false`·`release_eligible=false`
- Model A 표 topology detector v1.3은 선 교차 필터, 엄격 OCR 행렬, 해시 결속 CROP 내부
  빈 격자 억제를 적용한다. 노출된 8문서·87쪽 projection proxy에서 grounded 표 16/24,
  grid exact 14/24, 셀 할당 pair F1 91.37%, unmatched 예측 2건이며 두 건은 proxy가 텍스트
  참조를 잃은 실제 6×10 표다. 이는 사람 gold가 아닌 진단·회귀 결과이고, 현재 풀에는
  학습·튜닝에 노출되지 않은 최종 holdout이 0건이므로 일반화 정확도나 출시 근거로 쓰지 않는다.
- 실제 후보 1건 검수 smoke는 직접 IMAGE 근거가 없는 fallback 그림을 명시적으로 제외한 뒤
  불변 revision을 완료·재열기까지 통과했다. 이는 UI/계약 smoke일 뿐 사람 승인이나 출고 승격이 아니며,
  검수된 v4 Plan과 외부 검증을 통과하기 전 후보를 자동 출고하지 않음
- 현재 확인된 권리 검증 완료 학습 문서와 독립 실스캔 held-out 문서는 모두 0건이므로,
  새 가중치 학습은 데이터 승격 조건을 갖출 때까지 시작하지 않음
- 검수 앱은 typed patch 7종만 받아 원본 밖에 불변 리비전을 만들며, 그림은 검증된
  PAGE_IMAGE/CROP 근거에 재결속하거나 명시적으로 제외할 수 있다. 미저장 이동·동일 필드
  동시 덮어쓰기·선택 후 manifest 교체를 차단함
- 공식 한컴 corpus(10개 원본·2,554 chunks)를 검증 가능한 sidecar로 고정하고,
  완료된 Candidate review → Model B handoff → typed Plan review 완료까지 불변 revision으로 연결함
- Model B는 provider-neutral constrained JSON 요청과 응답 검증 경계까지 구현됨. 내용 순서,
  `render_as`, capability, 공식 근거를 벗어나거나 XML·경로·URL·자동화 payload를 내면 block함
- 유효한 Model B 결과만 request/result/raw response/Plan SHA와 model artifact identity를
  Plan review 1.3에 영속 결속한다. inference binding이 없는 기존 seed-only 1.2 review는 제출 차단
- 완료된 Candidate review와 inference-bound Model B Plan review는 SHA 결속 이미지를 다시
  materialize한 뒤 HWPX compiler v4에 전달되고, 사전 digest 검증 후 atomic no-replace rename으로
  발행된다. 생성물 digest와 10개 품질 항목을 반환하되,
  시각 비교·DVC·한컴 열기/저장/재열기 검증 전에는 QualityReport를 항상 `REVIEW`로 유지함
- 위 Model B 산출물은 여전히 `unverified`·`training_eligible=false`이며,
  인증된 사람 Golden target과 학습 checkpoint는 아직 0건
- Model A v2 평가기는 gold/pred 원시 record에서 공식 8개 지표를 직접 재계산하고
  evidence·image 결속 위반을 별도 hard failure로 남김. v3는 실제 페이지 요청·원시 응답·
  validator 결과·문서 조립을 재생해 high-risk review recall과 10-bin confidence ECE까지
  계산함. 다만 위험 라벨·검수자 신원이 인증되지 않았고 cross-page 관계는 미측정이므로
  report는 항상 `promotable=false`
- `model-first-v2` 승격 정책은 기존 15개 지표 gate를 보존하면서 high-risk review recall
  `>= 0.99`, confidence ECE `<= 0.05`를 추가한 총 17개 지표 gate임. 원시 v3 report를
  실제 승격 경계에 연결하기 전까지 정책 파일만으로 모델을 승격하지 않음
- 결정론적 HWPX compiler v4는 styled text, 빈 셀·병합 셀을 포함한 편집 가능한 표와
  명시적 폭을 가진 inline PNG 그림을 byte-deterministic하게 지원함. 그림 bytes는 경로가
  아닌 EvidenceIR page/crop source SHA에 결속된 immutable bundle로만 받고, 실제 PNG
  decode·정적 RGB/RGBA chunk allowlist·단일/번들/문서 사용 픽셀 한도·manifest/binary/control·
  최종 candidate digest를 검증함. 수식·명시적 column break 또는 비지원 layout/design
  profile은 빈 결과를 내지 않고 compile 전에 차단함
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

실제 v4 후보의 초안 생성 → typed patch → 최신 불변 리비전 재열기 smoke:

```powershell
npm run build
node scripts/smoke-candidate-review.mjs output/hwpx-projection-candidates-20260910-v4
```

## 주요 문서

- [구속력 있는 모델 우선 아키텍처](docs/architecture/0001-model-first-document-factory.md)
- [AI 실험과 제품 승격 흐름](ai/README.md)
- [모델 성과 추이](ai/results/README.md)
- [프로덕션 모델 정책](ai/production/README.md)
- [개발 현황 보고서](docs/DEVELOPMENT_REPORT.md)
- [폴더 구조](STRUCTURE.md)
