# Hancom knowledge pack

이 디렉터리는 문서 설계 모델(Model B)이 참조할 한컴 공식 자료의 출처와 사용 계약을
관리한다. 공식 문서는 OCR 인식 모델이나 문서 분석 모델(Model A)의 학습 데이터가 아니다.

원본 PDF, HTML, 공식 저장소 ZIP은 `output/hancom-knowledge/raw/`에 저장되며 Git에 넣지
않는다. `sources.json`은 다운로드 URL, 고정 revision, 파일 크기, SHA-256, 라이선스 주의사항을
기록한다. 추출한 검색 코퍼스도 `output/hancom-knowledge/processed/`에 생성한다.

검증과 코퍼스 생성을 다시 실행하는 명령은 다음과 같다.

```powershell
python ai/knowledge/build_hancom_knowledge.py
```

명령은 원본 크기·SHA-256·PDF/ZIP/HTML 형식을 먼저 검증하고 `chunks.jsonl`과
`build_report.json`을 원자적으로 작성한다.

모델 B에는 원문 전체가 아니라 질의와 관련된 chunk와 `capability_profile.json`만 전달한다.
HWPX XML, namespace, 스타일·리소스 ID와 package 관계는 모델이 아닌 컴파일러가 책임진다.

공식 문서만으로 파인튜닝이 완성되지는 않는다. 실제 지도 학습에는 다음 pair가 별도로 필요하다.

```text
Content IR + capability profile + retrieved official context
  -> 사람이 검수한 HWPX Plan + spec_refs
```

학습 실험은 `training_contract.json`의 재현 정보와 승격 조건을 기록해야 한다. 원본이나 추출본을
제품과 함께 배포하기 전에는 `sources.json`에 적힌 각 출처의 최신 사용 조건을 다시 검토한다.
