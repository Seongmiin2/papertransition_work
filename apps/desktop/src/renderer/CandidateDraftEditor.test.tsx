import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type {
  CandidateImageGroundingCandidate,
  CandidateReviewDraftIssueDisposition,
  CandidateReviewDraftImageNode,
  CandidateReviewDraftTableCell,
  CandidateReviewDraftTableNode,
  CandidateReviewDraftTextNode,
} from "../types/contracts";
import {
  DraftIssueEditor,
  DraftNodeEditor,
  buildDropImageNodePatchOperation,
  buildIssueDispositionPatchOperation,
  buildNeedsReviewPatchOperation,
  buildSetImageGroundingPatchOperation,
  buildTableCellPatchOperation,
  buildTextNodePatchOperations,
  isEditorLocked,
} from "./CandidateDraftEditor";

const textNode: CandidateReviewDraftTextNode = {
  id: "text-1",
  kind: "text",
  text: "기존 문장",
  role: "passage",
  needsReview: true,
};

const issue: CandidateReviewDraftIssueDisposition = {
  issueCode: "text_alignment_below_threshold",
  disposition: "pending",
  note: "",
};

describe("candidate draft editor", () => {
  it("renders only the allowed text, role, and review controls", () => {
    const markup = renderToStaticMarkup(
      <DraftNodeEditor
        node={textNode}
        saving={false}
        readOnly={false}
        activeDirtyEditorKey={null}
        onDirtyChange={() => undefined}
        onSave={() => undefined}
      />,
    );

    expect(markup).toContain("문서 역할");
    expect(markup).toContain("검수 텍스트");
    expect(markup).toContain("추가 검수 필요");
    expect(markup).toContain("변경을 새 리비전으로 저장");
    expect(markup).not.toContain("confidence");
    expect(markup).not.toContain("evidence");
    expect(markup).not.toContain("HwpDocumentPlan");
  });

  it("emits only changed text-node fields", () => {
    expect(
      buildTextNodePatchOperations(textNode, {
        text: "고친 문장",
        role: "question",
        needsReview: false,
      }),
    ).toEqual([
      { op: "set_text", nodeId: "text-1", text: "고친 문장" },
      { op: "set_role", nodeId: "text-1", role: "question" },
      { op: "set_needs_review", nodeId: "text-1", needsReview: false },
    ]);
    expect(
      buildTextNodePatchOperations(textNode, {
        text: textNode.text,
        role: textNode.role,
        needsReview: textNode.needsReview,
      }),
    ).toEqual([]);
    expect(
      buildTextNodePatchOperations(textNode, {
        text: "",
        role: "question",
        needsReview: false,
      }),
    ).toEqual([]);
  });

  it("keeps table topology immutable while editing cell text", () => {
    const cell: CandidateReviewDraftTableCell = {
      row: 2,
      column: 3,
      rowSpan: 2,
      columnSpan: 1,
      text: "원본",
    };

    expect(buildTableCellPatchOperation("table-1", cell, "수정")).toEqual({
      op: "set_table_cell_text",
      nodeId: "table-1",
      row: 2,
      column: 3,
      rowSpan: 2,
      columnSpan: 1,
      text: "수정",
    });
    expect(buildTableCellPatchOperation("table-1", cell, "원본")).toBeNull();
  });

  it("offers only a typed immutable-revision removal for an invalid image", () => {
    const image: CandidateReviewDraftImageNode = {
      id: "image-fallback-1",
      kind: "image",
      needsReview: true,
    };
    const markup = renderToStaticMarkup(
      <DraftNodeEditor
        node={image}
        saving={false}
        readOnly={false}
        activeDirtyEditorKey={null}
        onDirtyChange={() => undefined}
        onSave={() => undefined}
      />,
    );

    expect(markup).toContain("이 이미지 노드 제거");
    expect(markup).toContain("새 리비전");
    expect(buildDropImageNodePatchOperation(image)).toEqual({
      op: "drop_image_node",
      nodeId: "image-fallback-1",
    });
  });

  it("builds image grounding only from a typed candidate", () => {
    const image: CandidateReviewDraftImageNode = {
      id: "image-fallback-1",
      kind: "image",
      needsReview: true,
    };
    const grounding: CandidateImageGroundingCandidate = {
      observationRef: "page-0002-image-00001",
      assetRef: "page-0002-image-00001-crop",
      pageNo: 2,
      sourceKind: "crop",
      sha256: "a".repeat(64),
    };
    const markup = renderToStaticMarkup(
      <DraftNodeEditor
        node={image}
        imageGroundingCandidates={[grounding]}
        primaryPageNo={2}
        saving={false}
        readOnly={false}
        activeDirtyEditorKey={null}
        onDirtyChange={() => undefined}
        onSave={() => undefined}
      />,
    );

    expect(markup).toContain("page-0002-image-00001-crop");
    expect(markup).toContain("aaaaaaaaaaaa");
    expect(buildSetImageGroundingPatchOperation(image, grounding)).toEqual({
      op: "set_image_grounding",
      nodeId: "image-fallback-1",
      assetRef: "page-0002-image-00001-crop",
      observationRef: "page-0002-image-00001",
    });
  });

  it("emits review flags and issue dispositions without promotion fields", () => {
    expect(buildNeedsReviewPatchOperation(textNode, false)).toEqual({
      op: "set_needs_review",
      nodeId: "text-1",
      needsReview: false,
    });
    expect(buildNeedsReviewPatchOperation(textNode, true)).toBeNull();
    expect(
      buildIssueDispositionPatchOperation(
        issue,
        "accepted_limitation",
        "원본 문서에도 식별 불가능한 영역",
      ),
    ).toEqual({
      op: "set_issue_disposition",
      issueCode: "text_alignment_below_threshold",
      disposition: "accepted_limitation",
      note: "원본 문서에도 식별 불가능한 영역",
    });
    expect(
      buildIssueDispositionPatchOperation(issue, "accepted_limitation", "   "),
    ).toBeNull();
  });

  it("requires a reason when an issue limitation is accepted", () => {
    const markup = renderToStaticMarkup(
      <DraftIssueEditor
        issue={{ ...issue, disposition: "accepted_limitation" }}
        description="정렬 근거를 사람이 확인해야 합니다."
        saving={false}
        readOnly={false}
        activeDirtyEditorKey={null}
        onDirtyChange={() => undefined}
        onSave={() => undefined}
      />,
    );

    expect(markup).toContain("한계를 수용한 이유(필수)");
    expect(markup).toContain("aria-invalid=\"true\"");
    expect(markup).toContain("disabled");
  });

  it("locks every editor except the one holding an unsaved change", () => {
    expect(isEditorLocked(null, "node-text:text-1")).toBe(false);
    expect(isEditorLocked("node-text:text-1", "node-text:text-1")).toBe(false);
    expect(isEditorLocked("node-text:text-1", "issue:alignment")).toBe(true);
  });

  it("keeps completed tables pageable while every mutation stays read-only", () => {
    const table: CandidateReviewDraftTableNode = {
      id: "table-1",
      kind: "table",
      needsReview: false,
      cells: Array.from({ length: 41 }, (_, index) => ({
        row: index,
        column: 0,
        rowSpan: 1,
        columnSpan: 1,
        text: `셀 ${index + 1}`,
      })),
    };
    const markup = renderToStaticMarkup(
      <DraftNodeEditor
        node={table}
        saving={false}
        readOnly
        activeDirtyEditorKey={null}
        onDirtyChange={() => undefined}
        onSave={() => undefined}
      />,
    );

    expect(markup).toContain("읽기 전용");
    expect(markup).not.toContain("새 리비전 저장 중");
    expect(markup).toMatch(/<button[^>]*>다음<\/button>/);
    expect(markup).not.toMatch(/<button[^>]*disabled[^>]*>다음<\/button>/);
  });
});
