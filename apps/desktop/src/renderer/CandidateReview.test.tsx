import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type {
  CandidateDocumentSummary,
  CandidateJsonObject,
  CandidateReviewDraftPatchOperation,
  CandidateReviewDraftView,
  ContentNodeOverlay,
} from "../types/contracts";
import {
  buildPageReviewOperations,
  candidateReviewCompletionBlockers,
  canReplayDraftConflict,
  canCompleteCandidateReviewDraft,
  CandidateReview,
  classifyStaleDraftOperations,
  draftConflictForStaleRevision,
  draftNodePreview,
  findContentNode,
  findPlanItem,
  isCandidatePageEvidenceReady,
  issueRows,
  nodePreview,
  overlaysForPage,
  reviewerLabelValidationError,
  shouldLockCandidateReviewNavigation,
} from "./CandidateReview";

describe("candidate review renderer", () => {
  it("starts as an explicitly read-only, non-trainable workspace", () => {
    const markup = renderToStaticMarkup(<CandidateReview />);

    expect(markup).toContain("읽기 전용 검수 후보 / 학습 불가");
    expect(markup).toContain("후보 폴더 선택");
    expect(markup).toContain("수정·승인·Golden 승격 기능은 연결되어 있지 않습니다.");
    expect(markup).not.toMatch(/>저장</);
    expect(markup).not.toMatch(/>승인</);
    expect(markup).not.toMatch(/>학습</);
    expect(markup).not.toMatch(/>릴리스</);
  });

  it("uses the same strict reviewer-label surface constraint as main", () => {
    expect(reviewerLabelValidationError("reviewer_01")).toBeNull();
    expect(reviewerLabelValidationError("")).toMatch(/1~64자/);
    expect(reviewerLabelValidationError(" 검수자")).toMatch(/영문 또는 숫자/);
    expect(reviewerLabelValidationError("reviewer/path")).toMatch(/점, 밑줄, 하이픈/);
    expect(reviewerLabelValidationError("r".repeat(65))).toMatch(/1~64자/);
  });

  it("filters evidence overlays by page without losing multi-page nodes", () => {
    const first = overlay("content-a", [1, 2]);
    const second = overlay("content-b", [2]);

    expect(overlaysForPage([first, second], 1).map((item) => item.contentRef)).toEqual([
      "content-a",
    ]);
    expect(overlaysForPage([first, second], 2).map((item) => item.contentRef)).toEqual([
      "content-a",
      "content-b",
    ]);
  });

  it("reads node, Plan, and issue details from the loaded contracts", () => {
    const content: CandidateJsonObject = {
      nodes: [
        { id: "content-a", kind: "text", text: "  첫 번째   문항  " },
        {
          id: "content-table",
          kind: "table",
          cells: [{ text: "A" }, { text: "B" }],
        },
      ],
    };
    const plan: CandidateJsonObject = {
      flow: [
        {
          id: "flow-a",
          kind: "content",
          content_ref: "content-a",
          render_as: "paragraph",
        },
      ],
    };
    const document: CandidateDocumentSummary = {
      documentId: "candidate-a",
      lineageId: "a".repeat(64),
      split: "train",
      pageCount: 2,
      status: "needs_human_review",
      issueCodes: ["text_alignment_below_threshold", "rights_manifest_missing"],
    };
    const report: CandidateJsonObject = {
      issue_counts: { text_alignment_below_threshold: 3 },
    };

    expect(nodePreview(findContentNode(content, "content-a"))).toBe("첫 번째 문항");
    expect(nodePreview(findContentNode(content, "content-table"))).toBe("A | B");
    expect(findPlanItem(plan, "content-a")).toMatchObject({ render_as: "paragraph" });
    expect(issueRows(document, report)).toEqual([
      expect.objectContaining({ code: "text_alignment_below_threshold", count: 3 }),
      expect.objectContaining({ code: "rights_manifest_missing", count: null }),
    ]);
  });

  it("uses the latest sanitized draft text for node previews", () => {
    expect(
      draftNodePreview({
        id: "text-a",
        kind: "text",
        text: "  검수한   최신 문장  ",
        role: "question",
        needsReview: false,
      }),
    ).toBe("검수한 최신 문장");
    expect(
      draftNodePreview({
        id: "table-a",
        kind: "table",
        needsReview: true,
        cells: [
          { row: 0, column: 0, rowSpan: 1, columnSpan: 1, text: "A" },
          { row: 0, column: 1, rowSpan: 1, columnSpan: 1, text: "B" },
        ],
      }),
    ).toBe("A | B");
  });

  it("never replays a stale operation over a field changed by another revision", () => {
    const base = draftView(1, {
      text: "원문",
      role: "passage",
      needsReview: true,
    });
    const current = draftView(2, {
      text: "다른 검수자의 수정",
      role: "passage",
      needsReview: false,
    });
    const operations: CandidateReviewDraftPatchOperation[] = [
      { op: "set_text", nodeId: "text-a", text: "내 수정" },
      { op: "set_role", nodeId: "text-a", role: "question" },
      { op: "set_needs_review", nodeId: "text-a", needsReview: false },
      {
        op: "set_issue_disposition",
        issueCode: "alignment",
        disposition: "resolved",
        note: "확인",
      },
    ];

    const classified = classifyStaleDraftOperations(base, current, operations);

    expect(classified.conflictingOperations).toEqual([operations[0]]);
    expect(classified.replayableOperations).toEqual([operations[1], operations[3]]);
    expect(classified.alreadyAppliedCount).toBe(1);
  });

  it("replays or recognizes an immutable image drop while regrounding stays fail-closed", () => {
    const text = { id: "text-a", kind: "text" as const, text: "원문", role: "passage" as const, needsReview: false };
    const image = { id: "image-a", kind: "image" as const, needsReview: true };
    const base: CandidateReviewDraftView = {
      ...draftView(1, { text: "원문", role: "passage", needsReview: false }),
      nodes: [text, image],
    };
    const unchanged: CandidateReviewDraftView = { ...base, draftRevision: 2 };
    const dropped: CandidateReviewDraftView = {
      ...base,
      draftRevision: 2,
      nodes: [text],
    };
    const drop: CandidateReviewDraftPatchOperation = {
      op: "drop_image_node",
      nodeId: "image-a",
    };
    const reground: CandidateReviewDraftPatchOperation = {
      op: "set_image_grounding",
      nodeId: "image-a",
      assetRef: "asset-a",
      observationRef: "observation-a",
    };

    expect(classifyStaleDraftOperations(base, unchanged, [drop]).replayableOperations).toEqual([
      drop,
    ]);
    expect(classifyStaleDraftOperations(base, dropped, [drop]).alreadyAppliedCount).toBe(1);
    expect(classifyStaleDraftOperations(base, unchanged, [reground]).conflictingOperations).toEqual([
      reground,
    ]);
  });

  it.each([
    [true, false, false, false],
    [false, true, false, false],
    [false, false, true, false],
    [false, false, false, true],
  ])(
    "locks parent navigation for every active draft guard",
    (draftLoading, draftSaving, hasConflict, hasUnsavedChanges) => {
      expect(
        shouldLockCandidateReviewNavigation(
          draftLoading,
          draftSaving,
          hasConflict,
          hasUnsavedChanges,
        ),
      ).toBe(true);
    },
  );

  it("leaves parent navigation unlocked only when every draft guard is idle", () => {
    expect(shouldLockCandidateReviewNavigation(false, false, false, false)).toBe(false);
  });

  it("keeps local operations recoverable when another writer completed the draft", () => {
    const base = draftView(1, {
      text: "원문",
      role: "passage",
      needsReview: true,
    });
    const complete = {
      ...draftView(2, {
        text: "완료된 다른 수정",
        role: "passage",
        needsReview: true,
      }),
      status: "complete" as const,
    };
    const operations: CandidateReviewDraftPatchOperation[] = [
      { op: "set_text", nodeId: "text-a", text: "내 로컬 수정" },
      { op: "set_role", nodeId: "text-a", role: "question" },
    ];

    const conflict = draftConflictForStaleRevision(base, complete, operations);

    expect(conflict.localOperations).toBe(operations);
    expect(conflict.replayableOperations).toEqual([]);
    expect(conflict.conflictingOperations).toHaveLength(2);
    expect(canReplayDraftConflict(conflict, false)).toBe(false);
  });

  it("allows only the safe subset to be replayed from a mixed conflict", () => {
    const base = draftView(1, {
      text: "원문",
      role: "passage",
      needsReview: true,
    });
    const current = draftView(2, {
      text: "다른 수정",
      role: "passage",
      needsReview: true,
    });
    const operations: CandidateReviewDraftPatchOperation[] = [
      { op: "set_text", nodeId: "text-a", text: "내 수정" },
      { op: "set_role", nodeId: "text-a", role: "question" },
    ];

    const conflict = draftConflictForStaleRevision(base, current, operations);

    expect(conflict.localOperations).toBe(operations);
    expect(conflict.conflictingOperations).toEqual([operations[0]]);
    expect(conflict.replayableOperations).toEqual([operations[1]]);
    expect(canReplayDraftConflict(conflict, false)).toBe(true);
    expect(canReplayDraftConflict(conflict, true)).toBe(false);
  });

  it("builds page-review operations only for unresolved, single-page exact evidence", () => {
    const baseView = draftView(1, {
      text: "원문",
      role: "passage",
      needsReview: true,
    });
    const view: CandidateReviewDraftView = {
      ...baseView,
      nodes: [
        ...baseView.nodes,
        { id: "done", kind: "image", needsReview: false },
        { id: "spanning", kind: "image", needsReview: true },
        { id: "fallback", kind: "formula", needsReview: true },
        { id: "mixed-fallback", kind: "image", needsReview: true },
      ],
    };
    const candidates = [
      overlay("text-a", [1]),
      overlay("text-a", [1]),
      { ...overlay("done", [1]), contentKind: "image" as const },
      { ...overlay("spanning", [1, 2]), contentKind: "image" as const },
      {
        ...overlay("fallback", [1]),
        contentKind: "formula" as const,
        usesPageRegionFallback: true,
      },
      {
        ...overlay("mixed-fallback", [1]),
        contentKind: "image" as const,
        usesPageRegionFallback: false,
        boxes: [
          {
            observationId: "region-a",
            kind: "region",
            pageNo: 1,
            pixel: [0, 0, 100, 100] as const,
            normalized: [0, 0, 1, 1] as const,
          },
          {
            observationId: "line-a",
            kind: "text_line",
            pageNo: 1,
            pixel: [10, 10, 80, 30] as const,
            normalized: [0.1, 0.1, 0.8, 0.3] as const,
          },
        ],
      },
    ];

    expect(buildPageReviewOperations(view, candidates)).toEqual([
      { op: "set_needs_review", nodeId: "text-a", needsReview: false },
    ]);
    expect(buildPageReviewOperations({ ...view, status: "complete" }, candidates)).toEqual([]);
    expect(buildPageReviewOperations(null, candidates)).toEqual([]);
  });

  it("enables page confirmation only after the exact requested image has loaded", () => {
    const image = {
      pageNo: 2,
      mimeType: "image/png" as const,
      sha256: "b".repeat(64),
      dataUrl: "data:image/png;base64,iVBORw0KGgo=",
    };

    expect(isCandidatePageEvidenceReady(image, 2, "request-2", null, null)).toBe(false);
    expect(
      isCandidatePageEvidenceReady(image, 2, "request-2", "request-2", "로딩 중"),
    ).toBe(false);
    expect(isCandidatePageEvidenceReady(image, 1, "request-2", "request-2", null)).toBe(
      false,
    );
    expect(isCandidatePageEvidenceReady(image, 2, "request-3", "request-2", null)).toBe(
      false,
    );
    expect(isCandidatePageEvidenceReady(image, 2, "request-2", "request-2", null)).toBe(
      true,
    );
  });

  it("completes only an in-progress draft with every node and issue resolved", () => {
    const blocked = draftView(3, {
      text: "원문",
      role: "passage",
      needsReview: true,
    });
    expect(candidateReviewCompletionBlockers(blocked)).toEqual({
      needsReviewNodes: 1,
      pendingIssues: 1,
    });
    expect(canCompleteCandidateReviewDraft(blocked)).toBe(false);

    const ready: CandidateReviewDraftView = {
      ...blocked,
      nodes: blocked.nodes.map((node) => ({ ...node, needsReview: false })),
      issueDispositions: [
        { issueCode: "alignment", disposition: "resolved", note: "원본 대조 완료" },
      ],
    };
    expect(candidateReviewCompletionBlockers(ready)).toEqual({
      needsReviewNodes: 0,
      pendingIssues: 0,
    });
    expect(canCompleteCandidateReviewDraft(ready)).toBe(true);
    expect(canCompleteCandidateReviewDraft({ ...ready, status: "complete" })).toBe(false);
    expect(canCompleteCandidateReviewDraft(null)).toBe(false);
  });
});

function draftView(
  revision: number,
  node: { text: string; role: "passage"; needsReview: boolean },
): CandidateReviewDraftView {
  return {
    schemaVersion: "candidate-review-view/1.0",
    documentId: "candidate-a",
    status: "in_progress",
    draftRevision: revision,
    updatedAt: "2026-09-10T00:00:00Z",
    reviewedContentIrContractSha256: "a".repeat(64),
    nodes: [{ id: "text-a", kind: "text", ...node }],
    issueDispositions: [
      { issueCode: "alignment", disposition: "pending", note: "" },
    ],
    rightsStatus: "unverified",
    identityAssurance: "self_asserted_untrusted",
    goldenEligible: false,
    trainingEligible: false,
    releaseEligible: false,
  };
}

function overlay(contentRef: string, pageNumbers: number[]): ContentNodeOverlay {
  return {
    contentRef,
    contentKind: "text",
    confidence: 0.9,
    needsReview: true,
    evidenceRefs: ["observation-a"],
    pageNumbers,
    primaryPageNo: pageNumbers[0],
    spansMultiplePages: pageNumbers.length > 1,
    usesPageRegionFallback: false,
    boxes: [],
  };
}
