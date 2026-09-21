import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type {
  CandidateBundleSelection,
  CandidateDocumentSummary,
  CandidateJsonObject,
  CandidateJsonValue,
  CandidatePageImageDataUrl,
  CandidateReviewDraftNode,
  CandidateReviewDraftPatchOperation,
  CandidateReviewDraftView,
  CandidateReviewDocument,
  ContentNodeOverlay,
} from "../types/contracts";
import { DraftIssueEditor, DraftNodeEditor } from "./CandidateDraftEditor";
import "./candidate-review-guard.css";

type ReviewTab = "node" | "issues";

interface IssueRow {
  code: string;
  count: number | null;
  description: string;
}

interface CandidateReviewProps {
  onNavigationLockChange?: (navigationLocked: boolean) => void;
}

export interface DraftConflictState {
  currentView: CandidateReviewDraftView;
  localOperations: readonly CandidateReviewDraftPatchOperation[];
  replayableOperations: readonly CandidateReviewDraftPatchOperation[];
  conflictingOperations: readonly CandidateReviewDraftPatchOperation[];
  alreadyAppliedCount: number;
}

const ignoreNavigationLock = () => undefined;

const ISSUE_DESCRIPTIONS: Readonly<Record<string, string>> = {
  empty_paragraph_layout_not_projected: "빈 문단의 간격과 배치가 Plan에 완전히 반영되지 않았습니다.",
  header_footer_gutter_margins_not_representable: "머리말·꼬리말·제본 여백 일부를 현재 Plan 계약으로 표현할 수 없습니다.",
  header_footer_layout_requires_review: "머리말과 꼬리말의 위치를 사람이 확인해야 합니다.",
  header_footer_special_placement_unimplemented: "머리말·꼬리말 전용 배치가 아직 일반 문단으로 평탄화됩니다.",
  human_review_required: "이 문서는 자동 정답이 아닌 사람 검수 후보입니다.",
  hwp_equation_identity_subset_requires_review: "HWP 수식이 제한된 동일 표기 규칙으로 옮겨져 확인이 필요합니다.",
  hwp_equation_script_requires_conversion: "HWP 수식 스크립트를 표준 수식 표현으로 변환해야 합니다.",
  inline_style_runs_flattened: "한 문단 안의 여러 글자 스타일이 하나로 합쳐졌습니다.",
  mixed_content_paragraph_layout_flattened: "텍스트와 개체가 섞인 문단의 배치가 평탄화되었습니다.",
  page_assignment_from_pdf_alignment: "페이지 소속은 PDF 텍스트 정렬 결과로 추정했습니다.",
  page_assignment_requires_pdf_alignment: "HWPX만으로 페이지 소속을 확정할 수 없습니다.",
  page_break_inferred_from_pdf_alignment: "일부 쪽 나눔은 PDF 정렬 결과로 추정했습니다.",
  page_number_control_not_projected: "쪽 번호 제어가 Plan에 직접 반영되지 않았습니다.",
  pdf_observation_confidence_uncalibrated: "PDF text-layer 관측 confidence가 아직 보정되지 않았습니다.",
  rights_manifest_missing: "사용 권리 manifest가 없어 학습·릴리스에 사용할 수 없습니다.",
  section_column_layout_changes_flattened: "문서 중간의 단 수 변경이 첫 레이아웃으로 평탄화되었습니다.",
  shape_layout_flattened: "도형 내부 또는 주변 배치가 일반 흐름으로 평탄화되었습니다.",
  style_facts_not_representable_in_plan: "색상·자간·들여쓰기 등 일부 스타일 사실을 현재 Plan이 표현하지 못합니다.",
  text_alignment_below_threshold: "일부 ContentIR 텍스트와 PDF 근거의 정렬 점수가 기준보다 낮습니다.",
};

export function CandidateReview({
  onNavigationLockChange = ignoreNavigationLock,
}: CandidateReviewProps = {}) {
  const [selection, setSelection] = useState<CandidateBundleSelection | null>(null);
  const [lineageId, setLineageId] = useState<string | null>(null);
  const [document, setDocument] = useState<CandidateReviewDocument | null>(null);
  const [pageNo, setPageNo] = useState(1);
  const [pageImage, setPageImage] = useState<CandidatePageImageDataUrl | null>(null);
  const [pageImageRequestKey, setPageImageRequestKey] = useState<string | null>(null);
  const [readyPageImageRequestKey, setReadyPageImageRequestKey] = useState<string | null>(
    null,
  );
  const [selectedContentRef, setSelectedContentRef] = useState<string | null>(null);
  const [tab, setTab] = useState<ReviewTab>("node");
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviewerLabel, setReviewerLabel] = useState("");
  const [draftOperation, setDraftOperation] =
    useState<"created" | "reopened" | null>(null);
  const [draftView, setDraftView] = useState<CandidateReviewDraftView | null>(null);
  const [draftLoading, setDraftLoading] = useState(false);
  const [draftSaving, setDraftSaving] = useState(false);
  const [draftConflict, setDraftConflict] = useState<DraftConflictState | null>(null);
  const [draftError, setDraftError] = useState<string | null>(null);
  const [activeDirtyEditorKey, setActiveDirtyEditorKey] = useState<string | null>(null);
  const [draftEditorEpoch, setDraftEditorEpoch] = useState(0);
  const draftRequestGeneration = useRef(0);
  const pageImageRequestGeneration = useRef(0);
  const hasUnsavedChanges = activeDirtyEditorKey !== null;
  const navigationLocked = shouldLockCandidateReviewNavigation(
    draftLoading,
    draftSaving,
    Boolean(draftConflict),
    hasUnsavedChanges,
  );
  const completionBlockers = candidateReviewCompletionBlockers(draftView);
  const completionReady = canCompleteCandidateReviewDraft(draftView);

  const handleDraftDirtyChange = useCallback(
    (editorKey: string, dirty: boolean) => {
      setActiveDirtyEditorKey((current) => {
        if (dirty) return current ?? editorKey;
        return current === editorKey ? null : current;
      });
    },
    [],
  );

  const discardUnsavedChanges = useCallback(() => {
    setActiveDirtyEditorKey(null);
    setDraftEditorEpoch((current) => current + 1);
    setDraftError(null);
  }, []);

  useEffect(() => {
    onNavigationLockChange(navigationLocked);
    return () => onNavigationLockChange(false);
  }, [navigationLocked, onNavigationLockChange]);

  useEffect(() => {
    if (!navigationLocked) return;
    const preventAccidentalClose = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", preventAccidentalClose);
    return () => window.removeEventListener("beforeunload", preventAccidentalClose);
  }, [navigationLocked]);

  useEffect(() => {
    if (!lineageId) return;
    let active = true;
    setDocument(null);
    setPageImage(null);
    setPageImageRequestKey(null);
    setReadyPageImageRequestKey(null);
    setSelectedContentRef(null);
    setDraftOperation(null);
    setDraftView(null);
    setDraftConflict(null);
    setDraftError(null);
    setActiveDirtyEditorKey(null);
    setDraftEditorEpoch((current) => current + 1);
    setDraftLoading(false);
    setDraftSaving(false);
    draftRequestGeneration.current += 1;
    setLoading("문서 계약과 근거를 검증하는 중");
    setError(null);
    window.exam2hwpx.loadCandidateDocument(lineageId).then(
      (next) => {
        if (!active) return;
        setDocument(next);
        setPageNo(1);
        setLoading(null);
      },
      (reason: unknown) => {
        if (!active) return;
        setLoading(null);
        setError(errorMessage(reason));
      },
    );
    return () => {
      active = false;
    };
  }, [lineageId]);

  useEffect(() => {
    if (!document) return;
    let active = true;
    const requestKey = `${document.document.lineageId}:${pageNo}:${++pageImageRequestGeneration.current}`;
    setPageImage(null);
    setPageImageRequestKey(null);
    setReadyPageImageRequestKey(null);
    setError(null);
    setLoading(`${pageNo}쪽 이미지를 검증하는 중`);
    window.exam2hwpx.loadCandidatePage(document.document.lineageId, pageNo).then(
      (next) => {
        if (!active) return;
        if (next.pageNo !== pageNo) {
          setLoading(null);
          setError("요청한 쪽과 다른 PDF 근거가 반환되었습니다.");
          return;
        }
        setPageImage(next);
        setPageImageRequestKey(requestKey);
        setLoading(null);
      },
      (reason: unknown) => {
        if (!active) return;
        setLoading(null);
        setError(errorMessage(reason));
      },
    );
    return () => {
      active = false;
    };
  }, [document, pageNo]);

  const pageOverlays = useMemo(
    () => overlaysForPage(document?.overlays ?? [], pageNo),
    [document, pageNo],
  );
  useEffect(() => {
    if (!pageOverlays.some((overlay) => overlay.contentRef === selectedContentRef)) {
      setSelectedContentRef(pageOverlays[0]?.contentRef ?? null);
    }
  }, [pageOverlays, selectedContentRef]);

  const selectedOverlay =
    document?.overlays.find((overlay) => overlay.contentRef === selectedContentRef) ?? null;
  const contentNode =
    document && selectedContentRef
      ? findContentNode(document.contentIr, selectedContentRef)
      : null;
  const planItem =
    document && selectedContentRef
      ? findPlanItem(document.hwpDocumentPlan, selectedContentRef)
      : null;
  const issues = document ? issueRows(document.document, document.report) : [];

  const selectBundle = async () => {
    draftRequestGeneration.current += 1;
    setDraftLoading(false);
    setDraftOperation(null);
    setDraftView(null);
    setDraftConflict(null);
    setDraftError(null);
    setActiveDirtyEditorKey(null);
    setDraftEditorEpoch((current) => current + 1);
    setError(null);
    setLoading("후보 manifest를 검증하는 중");
    try {
      const next = await window.exam2hwpx.selectCandidateBundle();
      if (!next) return;
      setSelection(next);
      setDocument(null);
      setPageImage(null);
      setPageImageRequestKey(null);
      setReadyPageImageRequestKey(null);
      setLineageId(next.documents[0]?.lineageId ?? null);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setLoading(null);
    }
  };

  const startOrReopenDraft = async () => {
    if (!lineageId) return;
    const validationError = reviewerLabelValidationError(reviewerLabel);
    if (validationError) {
      setDraftError(validationError);
      return;
    }
    const generation = ++draftRequestGeneration.current;
    onNavigationLockChange(true);
    setDraftLoading(true);
    setDraftError(null);
    try {
      const next = await window.exam2hwpx.startOrReopenCandidateReviewDraft({
        lineageId,
        reviewerLabel,
      });
      if (draftRequestGeneration.current !== generation) return;
      setDraftOperation(next.operation);
      setDraftView(next.view);
      setDraftConflict(null);
      setActiveDirtyEditorKey(null);
      setDraftEditorEpoch((current) => current + 1);
    } catch (reason) {
      if (draftRequestGeneration.current !== generation) return;
      setDraftOperation(null);
      setDraftView(null);
      setDraftError(errorMessage(reason));
    } finally {
      if (draftRequestGeneration.current === generation) {
        setDraftLoading(false);
        onNavigationLockChange(false);
      }
    }
  };

  const saveDraftOperations = async (
    operations: readonly CandidateReviewDraftPatchOperation[],
    expectedDraftRevision = draftView?.draftRevision,
    baseView: CandidateReviewDraftView | null = draftView,
  ) => {
    if (
      !lineageId ||
      !draftView ||
      !baseView ||
      draftView.status === "complete" ||
      expectedDraftRevision === undefined ||
      operations.length === 0 ||
      draftSaving
    ) {
      return;
    }
    const generation = draftRequestGeneration.current;
    setDraftSaving(true);
    setDraftError(null);
    try {
      const result = await window.exam2hwpx.applyCandidateReviewDraftPatch({
        lineageId,
        expectedDraftRevision,
        operations,
      });
      if (draftRequestGeneration.current !== generation) return;
      if (result.kind === "stale_revision") {
        const conflict = draftConflictForStaleRevision(
          baseView,
          result.currentView,
          operations,
        );
        if (result.currentView.status === "complete") {
          setDraftError(
            "다른 저장에서 검수가 완료되었습니다. 로컬 입력은 유지했으며 직접 폐기하기 전까지 화면을 전환하지 않습니다.",
          );
          setDraftConflict(conflict);
          return;
        }
        if (
          conflict.replayableOperations.length === 0 &&
          conflict.conflictingOperations.length === 0
        ) {
          setDraftView(result.currentView);
          setDraftConflict(null);
          setActiveDirtyEditorKey(null);
          setDraftEditorEpoch((current) => current + 1);
          return;
        }
        setDraftConflict(conflict);
        setDraftError(
          conflict.conflictingOperations.length > 0
            ? `최신 리비전에서 같은 필드 ${conflict.conflictingOperations.length}개가 변경되었습니다. 자동으로 덮어쓰지 않습니다.`
            : "다른 저장으로 리비전이 변경되었습니다. 내 변경은 최신본과 충돌하지 않습니다.",
        );
        return;
      }
      setDraftView(result.view);
      setDraftConflict(null);
      setActiveDirtyEditorKey(null);
      setDraftEditorEpoch((current) => current + 1);
    } catch (reason) {
      if (draftRequestGeneration.current !== generation) return;
      setDraftError(errorMessage(reason));
    } finally {
      if (draftRequestGeneration.current === generation) setDraftSaving(false);
    }
  };

  const completeDraft = async () => {
    if (
      !lineageId ||
      !draftView ||
      !completionReady ||
      navigationLocked ||
      draftSaving
    ) {
      return;
    }
    if (
      !window.confirm(
        "이 검수 단계를 완료하면 새 읽기 전용 리비전이 생성됩니다. " +
          "Golden 승인이나 학습 허가는 아니며, 완료 리비전은 다시 편집할 수 없습니다. 계속할까요?",
      )
    ) {
      return;
    }

    const generation = draftRequestGeneration.current;
    const expectedDraftRevision = draftView.draftRevision;
    onNavigationLockChange(true);
    setDraftSaving(true);
    setDraftError(null);
    try {
      const result = await window.exam2hwpx.completeCandidateReviewDraft({
        lineageId,
        expectedDraftRevision,
      });
      if (draftRequestGeneration.current !== generation) return;
      const nextView = result.kind === "saved" ? result.view : result.currentView;
      setDraftView(nextView);
      setDraftConflict(null);
      setActiveDirtyEditorKey(null);
      setDraftEditorEpoch((current) => current + 1);
      if (result.kind === "stale_revision") {
        setDraftError(
          nextView.status === "complete"
            ? "다른 저장에서 이미 검수 단계가 완료되어 최신 완료본을 열었습니다."
            : "다른 저장으로 리비전이 변경되어 완료하지 않았습니다. 최신본을 확인한 뒤 다시 시도하세요.",
        );
      }
    } catch (reason) {
      if (draftRequestGeneration.current !== generation) return;
      setDraftError(errorMessage(reason));
    } finally {
      if (draftRequestGeneration.current === generation) {
        setDraftSaving(false);
        onNavigationLockChange(false);
      }
    }
  };

  return (
    <main className="candidateReview">
      <header className="reviewHeader">
        <div>
          <h1>한글문서 재제작 후보 검수</h1>
          <p>PDF 근거와 ContentIR·HwpDocumentPlan 후보를 나란히 확인합니다.</p>
        </div>
        <button
          className="secondary"
          disabled={navigationLocked}
          onClick={() => void selectBundle()}
        >
          후보 폴더 선택
        </button>
      </header>
      <div className="reviewOnlyBanner" role="status">
        <strong>
          {draftView?.status === "complete"
            ? "검수 완료본 / 학습 불가"
            : draftView
              ? "검수 초안 편집 / 학습 불가"
              : "읽기 전용 검수 후보 / 학습 불가"}
        </strong>
        <span>
          {draftView?.status === "complete"
            ? "사람 검수 단계만 완료되었습니다. 권리 확인과 Golden·학습 승격은 별도 절차입니다."
            : draftView
              ? "허용된 변경만 별도 불변 리비전으로 저장되며 원본 후보는 바뀌지 않습니다."
              : "수정·승인·Golden 승격 기능은 연결되어 있지 않습니다."}
        </span>
      </div>
      {error && <p className="error reviewError">{error}</p>}
      {!selection ? (
        <section className="card reviewWelcome">
          <h2>검수 후보 번들을 선택하세요</h2>
          <p>manifest에 등록되고 SHA-256 검증을 통과한 문서와 페이지만 표시합니다.</p>
          {loading && <small>{loading}</small>}
        </section>
      ) : (
        <>
          <div className="reviewBundleMeta">
            <span>문서 {selection.summary.documentCount}</span>
            <span>페이지 {selection.summary.pageCount}</span>
            <span>manifest {selection.summary.manifestSha256.slice(0, 12)}…</span>
          </div>
          <section className="draftControls" aria-label="검수 초안 시작 또는 재열기">
            <label>
              <span>검수자 라벨</span>
              <input
                value={reviewerLabel}
                maxLength={64}
                disabled={draftLoading || draftSaving || Boolean(draftView)}
                autoComplete="off"
                spellCheck={false}
                placeholder="reviewer_01"
                aria-invalid={Boolean(reviewerLabelValidationError(reviewerLabel))}
                onChange={(event) => {
                  setReviewerLabel(event.target.value);
                  setDraftError(null);
                }}
              />
            </label>
            <button
              type="button"
              className="primary"
              disabled={!lineageId || navigationLocked}
              onClick={() => void startOrReopenDraft()}
            >
              {draftLoading ? "초안 검증 중" : "검수 초안 시작 / 재열기"}
            </button>
            <div className="draftState" role="status">
              {draftView ? (
                <>
                  <strong>
                    {draftView.status === "complete"
                      ? "검수 단계 완료됨"
                      : draftOperation === "created"
                        ? "새 검수 초안 생성됨"
                        : "기존 검수 초안 재검증됨"}
                  </strong>
                  <span>
                    {draftView.status === "in_progress" ? "진행 중" : "완료·읽기 전용"} · 리비전{" "}
                    {draftView.draftRevision} · 학습 불가
                  </span>
                </>
              ) : (
                <>
                  <strong>초안 미연결</strong>
                  <span>라벨은 신원 확인이나 권리 증명이 아닙니다.</span>
                </>
              )}
            </div>
            {draftView && (
              <div className="draftCompletion" role="status">
                {draftView.status === "complete" ? (
                  <span>
                    이 리비전은 읽기 전용입니다. 권리 확인 전에는 Golden·학습·릴리스에
                    사용할 수 없습니다.
                  </span>
                ) : (
                  <>
                    <span id="draft-completion-readiness">
                      {completionReady
                        ? "모든 검수 항목이 해소되었습니다. 완료 후에는 이 초안을 편집할 수 없습니다."
                        : `남은 검수 노드 ${completionBlockers.needsReviewNodes}개 · 미처리 이슈 ${completionBlockers.pendingIssues}개`}
                    </span>
                    <button
                      type="button"
                      className="primary"
                      aria-describedby="draft-completion-readiness"
                      disabled={!completionReady || navigationLocked}
                      onClick={() => void completeDraft()}
                    >
                      {draftSaving ? "리비전 저장 중" : "검수 단계 완료"}
                    </button>
                  </>
                )}
              </div>
            )}
            {draftError && <p className="draftError">{draftError}</p>}
            {hasUnsavedChanges && !draftConflict && (
              <div className="draftUnsaved" role="status">
                <span>
                  저장 전 변경이 있습니다. 저장하거나 취소하기 전에는 화면을 이동할 수
                  없습니다.
                </span>
                <button
                  type="button"
                  className="secondary"
                  disabled={draftSaving}
                  onClick={discardUnsavedChanges}
                >
                  미저장 변경 취소
                </button>
              </div>
            )}
            {draftConflict && (
              <div className="draftConflict" role="alert">
                <span>
                  {draftConflict.currentView.status === "complete" ? (
                    <>
                      최신본 완료 · 재적용 불가 {draftConflict.conflictingOperations.length}개
                      {" · "}이미 반영 {draftConflict.alreadyAppliedCount}개
                    </>
                  ) : (
                    <>
                      충돌 {draftConflict.conflictingOperations.length}개 · 재적용 가능{" "}
                      {draftConflict.replayableOperations.length}개 · 이미 반영{" "}
                      {draftConflict.alreadyAppliedCount}개
                    </>
                  )}
                </span>
                <button
                  type="button"
                  className="secondary"
                  disabled={draftSaving}
                  onClick={() => {
                    setDraftView(draftConflict.currentView);
                    setDraftConflict(null);
                    setActiveDirtyEditorKey(null);
                    setDraftEditorEpoch((current) => current + 1);
                    setDraftError(null);
                  }}
                >
                  {draftConflict.currentView.status === "complete"
                    ? "완료본으로 전환(로컬 변경 폐기)"
                    : "최신본으로 재설정(로컬 변경 폐기)"}
                </button>
                <button
                  type="button"
                  className="secondary"
                  disabled={!canReplayDraftConflict(draftConflict, draftSaving)}
                  onClick={() =>
                    void saveDraftOperations(
                      draftConflict.replayableOperations,
                      draftConflict.currentView.draftRevision,
                      draftConflict.currentView,
                    )}
                >
                  {draftConflict.conflictingOperations.length > 0
                    ? `충돌 없는 ${draftConflict.replayableOperations.length}개만 저장(나머지 폐기)`
                    : "충돌 없는 변경 다시 적용"}
                </button>
                <DraftConflictRecovery
                  key={draftConflict.currentView.draftRevision}
                  operations={draftConflict.localOperations}
                />
              </div>
            )}
          </section>
          <div className="reviewGrid">
            <DocumentRail
              documents={selection.documents}
              selectedLineageId={lineageId}
              disabled={navigationLocked}
              onSelect={(next) => {
                draftRequestGeneration.current += 1;
                setDraftLoading(false);
                setDraftSaving(false);
                setDraftOperation(null);
                setDraftView(null);
                setDraftConflict(null);
                setDraftError(null);
                setActiveDirtyEditorKey(null);
                setDraftEditorEpoch((current) => current + 1);
                setPageImage(null);
                setPageImageRequestKey(null);
                setReadyPageImageRequestKey(null);
                setLineageId(next);
                setPageNo(1);
              }}
            />
            <PageEvidence
              document={document}
              pageNo={pageNo}
              pageImage={pageImage}
              imageRequestKey={pageImageRequestKey}
              overlays={pageOverlays}
              selectedContentRef={selectedContentRef}
              loading={loading}
              disabled={navigationLocked}
              onPageChange={(nextPageNo) => {
                setPageImage(null);
                setPageImageRequestKey(null);
                setReadyPageImageRequestKey(null);
                setPageNo(nextPageNo);
              }}
              onImageLoaded={setReadyPageImageRequestKey}
              onImageFailed={(requestKey) => {
                setReadyPageImageRequestKey((current) =>
                  current === requestKey ? null : current,
                );
                if (requestKey === pageImageRequestKey) {
                  setError("PDF 페이지 이미지를 표시할 수 없습니다.");
                }
              }}
              onNodeSelect={(contentRef) => {
                setSelectedContentRef(contentRef);
                setTab("node");
              }}
            />
            <DetailPanel
              document={document}
              pageNo={pageNo}
              overlays={pageOverlays}
              selectedOverlay={selectedOverlay}
              selectedContentRef={selectedContentRef}
              contentNode={contentNode}
              planItem={planItem}
              issues={issues}
              draftView={draftView}
              evidenceReady={isCandidatePageEvidenceReady(
                pageImage,
                pageNo,
                pageImageRequestKey,
                readyPageImageRequestKey,
                loading,
              )}
              draftEditorEpoch={draftEditorEpoch}
              activeDirtyEditorKey={activeDirtyEditorKey}
              draftSaving={
                draftSaving ||
                Boolean(draftConflict) ||
                draftView?.status === "complete"
              }
              navigationDisabled={navigationLocked}
              tab={tab}
              onTabChange={setTab}
              onNodeSelect={(contentRef) => {
                setSelectedContentRef(contentRef);
                setTab("node");
              }}
              onDraftDirtyChange={handleDraftDirtyChange}
              onDraftSave={(operations) => void saveDraftOperations(operations)}
            />
          </div>
        </>
      )}
    </main>
  );
}

function DraftConflictRecovery({
  operations,
}: {
  operations: readonly CandidateReviewDraftPatchOperation[];
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="draftConflictRecovery">
      <button
        type="button"
        className="secondary"
        aria-expanded={visible}
        onClick={() => setVisible((current) => !current)}
      >
        {visible ? "로컬 변경 내용 닫기" : "로컬 변경 내용 보기·복사"}
      </button>
      {visible && (
        <>
          <small>폐기하거나 일부만 저장하기 전에 아래 JSON을 복사해 보관하세요.</small>
          <textarea
            aria-label="충돌한 로컬 변경 JSON"
            readOnly
            spellCheck={false}
            rows={8}
            value={JSON.stringify(operations, null, 2)}
            onFocus={(event) => event.currentTarget.select()}
          />
        </>
      )}
    </div>
  );
}

function DocumentRail({
  documents,
  selectedLineageId,
  disabled,
  onSelect,
}: {
  documents: readonly CandidateDocumentSummary[];
  selectedLineageId: string | null;
  disabled: boolean;
  onSelect: (lineageId: string) => void;
}) {
  return (
    <section className="reviewPane documentRail" aria-label="후보 문서 목록">
      <h2>문서 후보 <em>{documents.length}</em></h2>
      <div className="documentList">
        {documents.map((document, index) => (
          <button
            key={document.lineageId}
            disabled={disabled}
            className={document.lineageId === selectedLineageId ? "selected" : ""}
            onClick={() => onSelect(document.lineageId)}
          >
            <b>문서 {String(index + 1).padStart(2, "0")}</b>
            <span>{document.pageCount}쪽 · 이슈 {document.issueCodes.length}</span>
            <small>{document.lineageId.slice(0, 12)}…</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function PageEvidence({
  document,
  pageNo,
  pageImage,
  imageRequestKey,
  overlays,
  selectedContentRef,
  loading,
  disabled,
  onPageChange,
  onImageLoaded,
  onImageFailed,
  onNodeSelect,
}: {
  document: CandidateReviewDocument | null;
  pageNo: number;
  pageImage: CandidatePageImageDataUrl | null;
  imageRequestKey: string | null;
  overlays: readonly ContentNodeOverlay[];
  selectedContentRef: string | null;
  loading: string | null;
  disabled: boolean;
  onPageChange: (pageNo: number) => void;
  onImageLoaded: (requestKey: string) => void;
  onImageFailed: (requestKey: string) => void;
  onNodeSelect: (contentRef: string) => void;
}) {
  const pageCount = document?.document.pageCount ?? 0;
  return (
    <section className="reviewPane evidencePane" aria-label="PDF 페이지 근거">
      <div className="pageToolbar">
        <h2>PDF 근거</h2>
        <div>
          <button
            aria-label="이전 페이지"
            disabled={disabled || pageNo <= 1}
            onClick={() => onPageChange(pageNo - 1)}
          >
            ‹
          </button>
          <span>{pageNo} / {pageCount || "–"}</span>
          <button
            aria-label="다음 페이지"
            disabled={disabled || !pageCount || pageNo >= pageCount}
            onClick={() => onPageChange(pageNo + 1)}
          >
            ›
          </button>
        </div>
      </div>
      <div className="pageViewport">
        {!pageImage || pageImage.pageNo !== pageNo || !imageRequestKey ? (
          <div className="pageLoading">{loading ?? "문서를 선택하세요."}</div>
        ) : (
          <div className="pageCanvas">
            <img
              key={imageRequestKey}
              className="reviewPageImage"
              src={pageImage.dataUrl}
              alt={`PDF ${pageNo}쪽`}
              onLoad={() => onImageLoaded(imageRequestKey)}
              onError={() => onImageFailed(imageRequestKey)}
            />
            {overlays.flatMap((overlay) =>
              overlay.boxes
                .filter((box) => box.pageNo === pageNo)
                .map((box) => (
                  <button
                    type="button"
                    key={`${overlay.contentRef}:${box.observationId}`}
                    className={[
                      "evidenceOverlay",
                      box.kind === "region" ? "region" : "",
                      overlay.contentRef === selectedContentRef ? "selected" : "",
                    ].join(" ")}
                    style={overlayStyle(box.normalized)}
                    title={`${overlay.contentKind} · ${overlay.contentRef}`}
                    aria-label={`${overlay.contentRef} 근거 영역`}
                    disabled={disabled}
                    onClick={() => onNodeSelect(overlay.contentRef)}
                  />
                )),
            )}
          </div>
        )}
      </div>
      <small className="evidenceLegend">
        현재 쪽 ContentIR 노드 {overlays.length}개 · 실선은 선택 노드
      </small>
    </section>
  );
}

function DetailPanel({
  document,
  pageNo,
  overlays,
  selectedOverlay,
  selectedContentRef,
  contentNode,
  planItem,
  issues,
  draftView,
  evidenceReady,
  draftEditorEpoch,
  activeDirtyEditorKey,
  draftSaving,
  navigationDisabled,
  tab,
  onTabChange,
  onNodeSelect,
  onDraftDirtyChange,
  onDraftSave,
}: {
  document: CandidateReviewDocument | null;
  pageNo: number;
  overlays: readonly ContentNodeOverlay[];
  selectedOverlay: ContentNodeOverlay | null;
  selectedContentRef: string | null;
  contentNode: CandidateJsonObject | null;
  planItem: CandidateJsonObject | null;
  issues: readonly IssueRow[];
  draftView: CandidateReviewDraftView | null;
  evidenceReady: boolean;
  draftEditorEpoch: number;
  activeDirtyEditorKey: string | null;
  draftSaving: boolean;
  navigationDisabled: boolean;
  tab: ReviewTab;
  onTabChange: (tab: ReviewTab) => void;
  onNodeSelect: (contentRef: string) => void;
  onDraftDirtyChange: (editorKey: string, dirty: boolean) => void;
  onDraftSave: (operations: readonly CandidateReviewDraftPatchOperation[]) => void;
}) {
  const draftNodesById = new Map(
    (draftView?.nodes ?? []).map((node) => [node.id, node]),
  );
  const draftNode =
    draftView && selectedContentRef
      ? draftNodesById.get(selectedContentRef) ?? null
      : null;
  const pageReviewOperations = buildPageReviewOperations(draftView, overlays);
  const unresolvedNodeIds = new Set(
    (draftView?.nodes ?? [])
      .filter((node) => node.needsReview)
      .map((node) => node.id),
  );
  const excludedPageReviewNodes = overlays.filter(
    (overlay) =>
      unresolvedNodeIds.has(overlay.contentRef) &&
      (overlay.spansMultiplePages || hasPageRegionFallback(overlay)),
  ).length;
  const pageReviewLimitExceeded = pageReviewOperations.length > 1_000;
  return (
    <section className="reviewPane detailPane" aria-label="검수 상세">
      <div className="reviewTabs">
        <button
          className={tab === "node" ? "active" : ""}
          disabled={navigationDisabled && tab !== "node"}
          onClick={() => onTabChange("node")}
        >
          노드 상세
        </button>
        <button
          className={tab === "issues" ? "active" : ""}
          disabled={navigationDisabled && tab !== "issues"}
          onClick={() => onTabChange("issues")}
        >
          문서 이슈 {issues.length}
        </button>
      </div>
      {!document ? (
        <div className="detailEmpty">문서를 불러오는 중입니다.</div>
      ) : tab === "issues" ? (
        <div className="issueList">
          {draftView
            ? draftView.issueDispositions.map((issue) => (
                <DraftIssueEditor
                  key={`${draftEditorEpoch}:${draftView.draftRevision}:${issue.issueCode}`}
                  issue={issue}
                  description={
                    issues.find((row) => row.code === issue.issueCode)?.description ??
                    "세부 원인을 후보 생성 보고서에서 확인해야 합니다."
                  }
                  saving={draftSaving}
                  readOnly={draftView.status === "complete"}
                  activeDirtyEditorKey={activeDirtyEditorKey}
                  onDirtyChange={onDraftDirtyChange}
                  onSave={onDraftSave}
                />
              ))
            : issues.map((issue) => (
                <article key={issue.code}>
                  <div><code>{issue.code}</code>{issue.count !== null && <b>{issue.count}</b>}</div>
                  <p>{issue.description}</p>
                </article>
              ))}
          <MetadataBlock title="정렬 보고" value={document.report.alignment} />
          <MetadataBlock title="안전 플래그" value={document.report.hard_flags} />
        </div>
      ) : (
        <>
          <div className="pageNodeList">
            <div className="pageNodeListHeader">
              <h3>{pageNo}쪽 노드 {overlays.length}</h3>
              {draftView?.status === "in_progress" && (
                <button
                  type="button"
                  className="secondary"
                  title="PDF 원본과 이 페이지의 모든 단순 노드를 대조한 경우에만 사용하세요."
                  disabled={
                    navigationDisabled ||
                    draftSaving ||
                    !evidenceReady ||
                    pageReviewOperations.length === 0 ||
                    pageReviewLimitExceeded
                  }
                  aria-describedby="page-review-safety"
                  onClick={() => {
                    if (
                      window.confirm(
                        `현재 쪽의 ${pageReviewOperations.length}개 노드를 모두 원본 PDF와 직접 대조했습니까? 확인 결과를 새 불변 리비전으로 저장합니다.`,
                      )
                    ) {
                      onDraftSave(pageReviewOperations);
                    }
                  }}
                >
                  현재 쪽 {pageReviewOperations.length}개 확인 처리
                </button>
              )}
            </div>
            {draftView?.status === "in_progress" && excludedPageReviewNodes > 0 && (
              <small className="pageReviewExclusion">
                여러 쪽 또는 전체 영역 근거 {excludedPageReviewNodes}개는 개별 확인해야 합니다.
              </small>
            )}
            {draftView?.status === "in_progress" && (
              <small id="page-review-safety" className="pageReviewSafety">
                {pageReviewLimitExceeded
                  ? "안전한 1회 저장 한도 1,000개를 넘었습니다. 노드를 나누어 개별 확인하세요."
                  : "원본 PDF와 이 쪽의 포함 노드를 모두 직접 대조한 경우에만 일괄 확인하세요."}
              </small>
            )}
            {overlays.map((overlay) => (
              <button
                key={overlay.contentRef}
                disabled={
                  navigationDisabled && overlay.contentRef !== selectedContentRef
                }
                className={overlay.contentRef === selectedContentRef ? "selected" : ""}
                onClick={() => onNodeSelect(overlay.contentRef)}
              >
                <span>{overlay.contentKind}</span>
                <b>
                  {draftNodePreview(draftNodesById.get(overlay.contentRef)) ??
                    nodePreview(findContentNode(document.contentIr, overlay.contentRef))}
                </b>
                <small>근거 {overlay.boxes.length} · {(overlay.confidence * 100).toFixed(1)}%</small>
              </button>
            ))}
          </div>
          {selectedOverlay && (
            <div className="nodeInspection">
              <h3>{draftView ? "원본 후보 ContentIR" : "ContentIR"}</h3>
              <dl>
                <dt>ID</dt><dd>{selectedOverlay.contentRef}</dd>
                <dt>종류</dt><dd>{selectedOverlay.contentKind}</dd>
                <dt>페이지</dt><dd>{selectedOverlay.pageNumbers.join(", ")}</dd>
                <dt>confidence</dt><dd>{selectedOverlay.confidence.toFixed(6)}</dd>
                <dt>{draftNode ? "초안 검수 필요" : "검수 필요"}</dt>
                <dd>
                  {(draftNode?.needsReview ?? selectedOverlay.needsReview) ? "예" : "아니요"}
                </dd>
              </dl>
              {(selectedOverlay.spansMultiplePages || selectedOverlay.usesPageRegionFallback) && (
                <p className="nodeWarning">
                  {selectedOverlay.spansMultiplePages && "여러 페이지 근거 · "}
                  {selectedOverlay.usesPageRegionFallback && "전체 페이지 fallback"}
                </p>
              )}
              <p className="nodeText">{nodePreview(contentNode, 600)}</p>
              <MetadataBlock title="HwpDocumentPlan 항목" value={planItem} />
              {draftView && (
                <section className="draftEditorSection">
                  <h3>검수 초안 · 리비전 {draftView.draftRevision}</h3>
                  <DraftNodeEditor
                    key={`${draftEditorEpoch}:${draftView.draftRevision}:${selectedContentRef ?? ""}`}
                    node={draftNode}
                    imageGroundingCandidates={document.imageGroundingCandidates}
                    primaryPageNo={selectedOverlay.primaryPageNo}
                    saving={draftSaving}
                    readOnly={draftView.status === "complete"}
                    activeDirtyEditorKey={activeDirtyEditorKey}
                    onDirtyChange={onDraftDirtyChange}
                    onSave={onDraftSave}
                  />
                </section>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}

function MetadataBlock({ title, value }: { title: string; value: CandidateJsonValue | undefined }) {
  if (value === undefined) return null;
  return (
    <details className="metadataBlock">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

export function overlaysForPage(
  overlays: readonly ContentNodeOverlay[],
  pageNo: number,
): ContentNodeOverlay[] {
  return overlays.filter((overlay) => overlay.pageNumbers.includes(pageNo));
}

export function findContentNode(
  content: CandidateJsonObject,
  contentRef: string,
): CandidateJsonObject | null {
  const nodes = Array.isArray(content.nodes) ? content.nodes : [];
  return nodes.find((node) => isObject(node) && node.id === contentRef) as
    | CandidateJsonObject
    | undefined ?? null;
}

export function findPlanItem(
  plan: CandidateJsonObject,
  contentRef: string,
): CandidateJsonObject | null {
  const flow = Array.isArray(plan.flow) ? plan.flow : [];
  return flow.find((item) => isObject(item) && item.content_ref === contentRef) as
    | CandidateJsonObject
    | undefined ?? null;
}

export function nodePreview(node: CandidateJsonObject | null, limit = 120): string {
  if (!node) return "내용 없음";
  let value = typeof node.text === "string" ? node.text : "";
  if (!value && typeof node.expression === "string") value = node.expression;
  if (!value && Array.isArray(node.cells)) {
    value = node.cells
      .filter(isObject)
      .map((cell) => typeof cell.text === "string" ? cell.text : "")
      .filter(Boolean)
      .join(" | ");
  }
  if (!value && typeof node.asset_ref === "string") value = `이미지 · ${node.asset_ref}`;
  const normalized = value.replace(/\s+/g, " ").trim() || "내용 없음";
  return normalized.length > limit ? `${normalized.slice(0, limit)}…` : normalized;
}

export function draftNodePreview(
  node: CandidateReviewDraftNode | undefined,
  limit = 120,
): string | null {
  if (!node) return null;
  let value = "";
  if (node.kind === "text") value = node.text;
  if (node.kind === "table") {
    value = node.cells.map((cell) => cell.text).filter(Boolean).join(" | ");
  }
  if (node.kind === "image") value = "이미지";
  if (node.kind === "formula") value = "수식";
  const normalized = value.replace(/\s+/g, " ").trim() || "내용 없음";
  return normalized.length > limit ? `${normalized.slice(0, limit)}…` : normalized;
}

export interface ClassifiedStaleDraftOperations {
  replayableOperations: CandidateReviewDraftPatchOperation[];
  conflictingOperations: CandidateReviewDraftPatchOperation[];
  alreadyAppliedCount: number;
}

export function shouldLockCandidateReviewNavigation(
  draftLoading: boolean,
  draftSaving: boolean,
  hasConflict: boolean,
  hasUnsavedChanges: boolean,
): boolean {
  return draftLoading || draftSaving || hasConflict || hasUnsavedChanges;
}

export function buildPageReviewOperations(
  draftView: CandidateReviewDraftView | null,
  overlays: readonly ContentNodeOverlay[],
): CandidateReviewDraftPatchOperation[] {
  if (!draftView || draftView.status !== "in_progress") return [];
  const nodesById = new Map(draftView.nodes.map((node) => [node.id, node]));
  const selected = new Set<string>();
  const operations: CandidateReviewDraftPatchOperation[] = [];
  for (const overlay of overlays) {
    if (
      selected.has(overlay.contentRef) ||
      overlay.spansMultiplePages ||
      hasPageRegionFallback(overlay)
    ) {
      continue;
    }
    selected.add(overlay.contentRef);
    const node = nodesById.get(overlay.contentRef);
    if (!node?.needsReview) continue;
    operations.push({
      op: "set_needs_review",
      nodeId: node.id,
      needsReview: false,
    });
  }
  return operations;
}

function hasPageRegionFallback(overlay: ContentNodeOverlay): boolean {
  return (
    overlay.usesPageRegionFallback || overlay.boxes.some((box) => box.kind === "region")
  );
}

export function isCandidatePageEvidenceReady(
  pageImage: CandidatePageImageDataUrl | null,
  pageNo: number,
  imageRequestKey: string | null,
  readyImageRequestKey: string | null,
  loading: string | null,
): boolean {
  return (
    loading === null &&
    pageImage !== null &&
    pageImage.pageNo === pageNo &&
    imageRequestKey !== null &&
    readyImageRequestKey === imageRequestKey
  );
}

export interface CandidateReviewCompletionBlockers {
  needsReviewNodes: number;
  pendingIssues: number;
}

export function candidateReviewCompletionBlockers(
  draftView: CandidateReviewDraftView | null,
): CandidateReviewCompletionBlockers {
  if (!draftView) return { needsReviewNodes: 0, pendingIssues: 0 };
  return {
    needsReviewNodes: draftView.nodes.filter((node) => node.needsReview).length,
    pendingIssues: draftView.issueDispositions.filter(
      (issue) => issue.disposition === "pending",
    ).length,
  };
}

export function canCompleteCandidateReviewDraft(
  draftView: CandidateReviewDraftView | null,
): boolean {
  if (!draftView || draftView.status !== "in_progress") return false;
  const blockers = candidateReviewCompletionBlockers(draftView);
  return blockers.needsReviewNodes === 0 && blockers.pendingIssues === 0;
}

export function draftConflictForStaleRevision(
  baseView: CandidateReviewDraftView,
  currentView: CandidateReviewDraftView,
  operations: readonly CandidateReviewDraftPatchOperation[],
): DraftConflictState {
  const classified = classifyStaleDraftOperations(baseView, currentView, operations);
  return {
    currentView,
    localOperations: operations,
    replayableOperations:
      currentView.status === "complete" ? [] : classified.replayableOperations,
    conflictingOperations:
      currentView.status === "complete"
        ? [...classified.replayableOperations, ...classified.conflictingOperations]
        : classified.conflictingOperations,
    alreadyAppliedCount: classified.alreadyAppliedCount,
  };
}

export function canReplayDraftConflict(
  conflict: DraftConflictState,
  saving: boolean,
): boolean {
  return (
    !saving &&
    conflict.currentView.status === "in_progress" &&
    conflict.replayableOperations.length > 0
  );
}

export function classifyStaleDraftOperations(
  baseView: CandidateReviewDraftView,
  currentView: CandidateReviewDraftView,
  operations: readonly CandidateReviewDraftPatchOperation[],
): ClassifiedStaleDraftOperations {
  const replayableOperations: CandidateReviewDraftPatchOperation[] = [];
  const conflictingOperations: CandidateReviewDraftPatchOperation[] = [];
  let alreadyAppliedCount = 0;

  for (const operation of operations) {
    const base = draftOperationFieldValue(baseView, operation);
    const current = draftOperationFieldValue(currentView, operation);
    const desired = draftOperationDesiredValue(operation);
    if (!base.found || !current.found) {
      conflictingOperations.push(operation);
    } else if (current.serialized === desired) {
      alreadyAppliedCount += 1;
    } else if (current.serialized === base.serialized) {
      replayableOperations.push(operation);
    } else {
      conflictingOperations.push(operation);
    }
  }

  return {
    replayableOperations,
    conflictingOperations,
    alreadyAppliedCount,
  };
}

interface DraftFieldValue {
  found: boolean;
  serialized: string;
}

function draftOperationFieldValue(
  view: CandidateReviewDraftView,
  operation: CandidateReviewDraftPatchOperation,
): DraftFieldValue {
  if (operation.op === "set_issue_disposition") {
    const issue = view.issueDispositions.find(
      (candidate) => candidate.issueCode === operation.issueCode,
    );
    return issue
      ? { found: true, serialized: JSON.stringify([issue.disposition, issue.note]) }
      : { found: false, serialized: "" };
  }

  const node = view.nodes.find((candidate) => candidate.id === operation.nodeId);
  if (!node) {
    return operation.op === "drop_image_node"
      ? { found: true, serialized: "absent" }
      : { found: false, serialized: "" };
  }
  if (operation.op === "drop_image_node") {
    return node.kind === "image"
      ? { found: true, serialized: "present" }
      : { found: false, serialized: "" };
  }
  if (operation.op === "set_image_grounding") {
    return { found: false, serialized: "" };
  }
  if (operation.op === "set_needs_review") {
    return { found: true, serialized: JSON.stringify(node.needsReview) };
  }
  if (operation.op === "set_text") {
    return node.kind === "text"
      ? { found: true, serialized: JSON.stringify(node.text) }
      : { found: false, serialized: "" };
  }
  if (operation.op === "set_role") {
    return node.kind === "text"
      ? { found: true, serialized: JSON.stringify(node.role) }
      : { found: false, serialized: "" };
  }
  if (node.kind !== "table") return { found: false, serialized: "" };
  const cell = node.cells.find(
    (candidate) =>
      candidate.row === operation.row &&
      candidate.column === operation.column &&
      candidate.rowSpan === operation.rowSpan &&
      candidate.columnSpan === operation.columnSpan,
  );
  return cell
    ? { found: true, serialized: JSON.stringify(cell.text) }
    : { found: false, serialized: "" };
}

function draftOperationDesiredValue(
  operation: CandidateReviewDraftPatchOperation,
): string {
  switch (operation.op) {
    case "set_text":
    case "set_table_cell_text":
      return JSON.stringify(operation.text);
    case "set_role":
      return JSON.stringify(operation.role);
    case "set_needs_review":
      return JSON.stringify(operation.needsReview);
    case "set_image_grounding":
      return JSON.stringify([operation.assetRef, operation.observationRef]);
    case "drop_image_node":
      return "absent";
    case "set_issue_disposition":
      return JSON.stringify([operation.disposition, operation.note]);
  }
}

export function issueRows(
  document: CandidateDocumentSummary,
  report: CandidateJsonObject,
): IssueRow[] {
  const counts = isObject(report.issue_counts) ? report.issue_counts : {};
  return document.issueCodes.map((code) => ({
    code,
    count: typeof counts[code] === "number" ? counts[code] : null,
    description: ISSUE_DESCRIPTIONS[code] ?? "세부 원인을 후보 생성 보고서에서 확인해야 합니다.",
  }));
}

export function reviewerLabelValidationError(value: string): string | null {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value)) {
    return "영문 또는 숫자로 시작하고 영문, 숫자, 점, 밑줄, 하이픈만 사용해 1~64자로 입력하세요.";
  }
  return null;
}

function overlayStyle(box: readonly [number, number, number, number]): CSSProperties {
  return {
    left: `${box[0] * 100}%`,
    top: `${box[1] * 100}%`,
    width: `${(box[2] - box[0]) * 100}%`,
    height: `${(box[3] - box[1]) * 100}%`,
  };
}

function isObject(value: CandidateJsonValue | undefined): value is CandidateJsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function errorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}
