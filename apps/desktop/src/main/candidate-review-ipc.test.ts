import { describe, expect, it } from "vitest";
import {
  parseCandidateReviewDraftCompletionRequest,
  parseCandidateReviewDraftPatchRequest,
} from "./candidate-review-ipc.js";

const lineageId = "a".repeat(64);

describe("candidate review completion IPC boundary", () => {
  it("rebuilds the exact lineage and revision request", () => {
    const source = { lineageId, expectedDraftRevision: 2 };

    const parsed = parseCandidateReviewDraftCompletionRequest(source);

    expect(parsed).toEqual(source);
    expect(parsed).not.toBe(source);
  });

  it.each([
    null,
    {},
    { lineageId, expectedDraftRevision: 1, localPath: "C:/secret" },
    { lineageId: "not-a-sha", expectedDraftRevision: 1 },
    { lineageId, expectedDraftRevision: "1" },
    { lineageId, expectedDraftRevision: 0 },
    { lineageId, expectedDraftRevision: 1_000_001 },
    { lineageId, expectedDraftRevision: 2 ** 53 },
  ])("rejects malformed or expanded completion input: %#", (value) => {
    expect(() => parseCandidateReviewDraftCompletionRequest(value)).toThrow(
      "검수 완료 요청 형식이 올바르지 않습니다.",
    );
  });
});

describe("candidate review patch IPC boundary", () => {
  it("rebuilds all seven allowed operations into a new strict object", () => {
    const source = {
      lineageId,
      expectedDraftRevision: 2,
      operations: [
        { op: "set_text", nodeId: "text-1", text: "수정" },
        { op: "set_role", nodeId: "text-1", role: "question" },
        {
          op: "set_table_cell_text",
          nodeId: "table-1",
          row: 0,
          column: 1,
          rowSpan: 2,
          columnSpan: 1,
          text: "",
        },
        { op: "set_needs_review", nodeId: "image-1", needsReview: false },
        {
          op: "set_image_grounding",
          nodeId: "image-1",
          assetRef: "page-0001-image-0001-crop",
          observationRef: "page-0001-image-0001",
        },
        { op: "drop_image_node", nodeId: "image-2" },
        {
          op: "set_issue_disposition",
          issueCode: "rights_manifest_missing",
          disposition: "accepted_limitation",
          note: "연구 전용으로만 유지",
        },
      ],
    };

    const parsed = parseCandidateReviewDraftPatchRequest(source);

    expect(parsed).toEqual(source);
    expect(parsed).not.toBe(source);
    expect(parsed.operations[0]).not.toBe(source.operations[0]);
  });

  it.each([
    null,
    {},
    { lineageId, expectedDraftRevision: 1, operations: [], localPath: "C:/secret" },
    { lineageId: "not-a-sha", expectedDraftRevision: 1, operations: [{}] },
    {
      lineageId,
      expectedDraftRevision: "1",
      operations: [{ op: "set_text", nodeId: "text-1", text: "valid" }],
    },
    {
      lineageId,
      expectedDraftRevision: 1,
      operations: [{ op: "set_text", nodeId: "text-1", text: "", plan: {} }],
    },
    {
      lineageId,
      expectedDraftRevision: 1,
      operations: [{ op: "set_role", nodeId: "text-1", role: "table" }],
    },
    {
      lineageId,
      expectedDraftRevision: 1,
      operations: [
        {
          op: "set_image_grounding",
          nodeId: "image-1",
          assetRef: "asset-a",
          observationRef: "observation-a",
          localPath: "C:/secret.png",
        },
      ],
    },
    {
      lineageId,
      expectedDraftRevision: 1,
      operations: [{ op: "drop_image_node", nodeId: "image-1", reason: "free text" }],
    },
    {
      lineageId,
      expectedDraftRevision: 1,
      operations: [
        {
          op: "set_issue_disposition",
          issueCode: "rights_manifest_missing",
          disposition: "accepted_limitation",
          note: "   ",
        },
      ],
    },
  ])("rejects malformed or expanded input without echoing it: %#", (value) => {
    expect(() => parseCandidateReviewDraftPatchRequest(value)).toThrow(
      "검수 패치 요청 형식이 올바르지 않습니다.",
    );
  });

  it("bounds operation count and individual fields", () => {
    const operation = {
      op: "set_needs_review",
      nodeId: "text-1",
      needsReview: true,
    };
    expect(() =>
      parseCandidateReviewDraftPatchRequest({
        lineageId,
        expectedDraftRevision: 1,
        operations: Array.from({ length: 1_001 }, () => operation),
      }),
    ).toThrow();
    expect(() =>
      parseCandidateReviewDraftPatchRequest({
        lineageId,
        expectedDraftRevision: 1,
        operations: [{ op: "set_text", nodeId: "x".repeat(1_025), text: "ok" }],
      }),
    ).toThrow();
  });

  it("counts emoji as UTF-16 code units at the field boundary", () => {
    const atLimit = "😀".repeat(500_000);
    expect(
      parseCandidateReviewDraftPatchRequest({
        lineageId,
        expectedDraftRevision: 1,
        operations: [{ op: "set_text", nodeId: "text-1", text: atLimit }],
      }).operations[0],
    ).toMatchObject({ text: atLimit });

    expect(() =>
      parseCandidateReviewDraftPatchRequest({
        lineageId,
        expectedDraftRevision: 1,
        operations: [{ op: "set_text", nodeId: "text-1", text: `${atLimit}a` }],
      }),
    ).toThrow();
  });

  it("accepts the JS safe-integer boundary and rejects 2^53", () => {
    const operation = {
      op: "set_table_cell_text",
      nodeId: "table-1",
      row: Number.MAX_SAFE_INTEGER,
      column: 0,
      rowSpan: 1,
      columnSpan: 1,
      text: "",
    };
    expect(
      parseCandidateReviewDraftPatchRequest({
        lineageId,
        expectedDraftRevision: 1,
        operations: [operation],
      }).operations[0],
    ).toMatchObject({ row: Number.MAX_SAFE_INTEGER });

    expect(() =>
      parseCandidateReviewDraftPatchRequest({
        lineageId,
        expectedDraftRevision: 1,
        operations: [{ ...operation, row: 2 ** 53 }],
      }),
    ).toThrow();
  });

  it("uses the same draft revision upper bound as the bridge", () => {
    const request = {
      lineageId,
      expectedDraftRevision: 1_000_000,
      operations: [{ op: "set_needs_review", nodeId: "text-1", needsReview: true }],
    };
    expect(parseCandidateReviewDraftPatchRequest(request).expectedDraftRevision).toBe(
      1_000_000,
    );

    expect(() =>
      parseCandidateReviewDraftPatchRequest({
        ...request,
        expectedDraftRevision: 1_000_001,
      }),
    ).toThrow();
  });

  it.each(["\ud800", "\udc00"])(
    "rejects an unpaired UTF-16 surrogate before crossing IPC: %s",
    (text) => {
      expect(() =>
        parseCandidateReviewDraftPatchRequest({
          lineageId,
          expectedDraftRevision: 1,
          operations: [{ op: "set_text", nodeId: "text-1", text }],
        }),
      ).toThrow();
    },
  );
});
