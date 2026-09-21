import {
  candidateReviewTextRoles,
  type CandidateReviewDraftCompletionRequest,
  type CandidateReviewDraftPatchOperation,
  type CandidateReviewDraftPatchRequest,
  type CandidateReviewIssueDisposition,
  type CandidateReviewTextRole,
} from "../types/contracts.js";

const SHA256 = /^[0-9a-f]{64}$/;
const ISSUE_CODE = /^[a-z][a-z0-9_.-]*$/;
const MAX_OPERATIONS = 1_000;
const MAX_NODE_ID_LENGTH = 1_024;
const MAX_ISSUE_CODE_LENGTH = 256;
const MAX_FIELD_CHARS = 1_000_000;
const MAX_TOTAL_CHARS = 10_000_000;
const MAX_DRAFT_REVISION = 1_000_000;
const INVALID_REQUEST = "검수 패치 요청 형식이 올바르지 않습니다.";
const INVALID_COMPLETION_REQUEST = "검수 완료 요청 형식이 올바르지 않습니다.";

export function parseCandidateReviewDraftCompletionRequest(
  value: unknown,
): CandidateReviewDraftCompletionRequest {
  if (!isRecord(value)) invalidCompletionRequest();
  const actualKeys = Object.keys(value).sort();
  const expectedKeys = ["expectedDraftRevision", "lineageId"];
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index]) ||
    typeof value.lineageId !== "string" ||
    !SHA256.test(value.lineageId) ||
    !isPositiveSafeInteger(value.expectedDraftRevision) ||
    value.expectedDraftRevision > MAX_DRAFT_REVISION
  ) {
    invalidCompletionRequest();
  }
  return {
    lineageId: value.lineageId,
    expectedDraftRevision: value.expectedDraftRevision,
  };
}

export function parseCandidateReviewDraftPatchRequest(
  value: unknown,
): CandidateReviewDraftPatchRequest {
  const request = recordWithKeys(value, [
    "expectedDraftRevision",
    "lineageId",
    "operations",
  ]);
  if (
    typeof request.lineageId !== "string" ||
    !SHA256.test(request.lineageId) ||
    !isPositiveSafeInteger(request.expectedDraftRevision) ||
    request.expectedDraftRevision > MAX_DRAFT_REVISION ||
    !Array.isArray(request.operations) ||
    request.operations.length < 1 ||
    request.operations.length > MAX_OPERATIONS
  ) {
    invalidRequest();
  }

  let totalChars = 0;
  const operations = request.operations.map((operation) => {
    const parsed = parseOperation(operation);
    totalChars += operationTextLength(parsed);
    if (totalChars > MAX_TOTAL_CHARS) invalidRequest();
    return parsed;
  });
  return {
    lineageId: request.lineageId,
    expectedDraftRevision: request.expectedDraftRevision,
    operations,
  };
}

function parseOperation(value: unknown): CandidateReviewDraftPatchOperation {
  if (!isRecord(value) || typeof value.op !== "string") invalidRequest();
  switch (value.op) {
    case "set_text": {
      const operation = recordWithKeys(value, ["nodeId", "op", "text"]);
      return {
        op: "set_text",
        nodeId: nodeId(operation.nodeId),
        text: boundedText(operation.text, false),
      };
    }
    case "set_role": {
      const operation = recordWithKeys(value, ["nodeId", "op", "role"]);
      if (
        typeof operation.role !== "string" ||
        !(candidateReviewTextRoles as readonly string[]).includes(operation.role)
      ) {
        invalidRequest();
      }
      return {
        op: "set_role",
        nodeId: nodeId(operation.nodeId),
        role: operation.role as CandidateReviewTextRole,
      };
    }
    case "set_table_cell_text": {
      const operation = recordWithKeys(value, [
        "column",
        "columnSpan",
        "nodeId",
        "op",
        "row",
        "rowSpan",
        "text",
      ]);
      if (
        !isNonNegativeSafeInteger(operation.row) ||
        !isNonNegativeSafeInteger(operation.column) ||
        !isPositiveSafeInteger(operation.rowSpan) ||
        !isPositiveSafeInteger(operation.columnSpan)
      ) {
        invalidRequest();
      }
      return {
        op: "set_table_cell_text",
        nodeId: nodeId(operation.nodeId),
        row: operation.row,
        column: operation.column,
        rowSpan: operation.rowSpan,
        columnSpan: operation.columnSpan,
        text: boundedText(operation.text, true),
      };
    }
    case "set_needs_review": {
      const operation = recordWithKeys(value, ["needsReview", "nodeId", "op"]);
      if (typeof operation.needsReview !== "boolean") invalidRequest();
      return {
        op: "set_needs_review",
        nodeId: nodeId(operation.nodeId),
        needsReview: operation.needsReview,
      };
    }
    case "set_image_grounding": {
      const operation = recordWithKeys(value, [
        "assetRef",
        "nodeId",
        "observationRef",
        "op",
      ]);
      return {
        op: "set_image_grounding",
        nodeId: nodeId(operation.nodeId),
        assetRef: nodeId(operation.assetRef),
        observationRef: nodeId(operation.observationRef),
      };
    }
    case "drop_image_node": {
      const operation = recordWithKeys(value, ["nodeId", "op"]);
      return {
        op: "drop_image_node",
        nodeId: nodeId(operation.nodeId),
      };
    }
    case "set_issue_disposition": {
      const operation = recordWithKeys(value, [
        "disposition",
        "issueCode",
        "note",
        "op",
      ]);
      if (
        typeof operation.issueCode !== "string" ||
        operation.issueCode.length > MAX_ISSUE_CODE_LENGTH ||
        !ISSUE_CODE.test(operation.issueCode) ||
        !isDisposition(operation.disposition)
      ) {
        invalidRequest();
      }
      const note = boundedText(operation.note, true);
      if (operation.disposition === "accepted_limitation" && !note.trim()) {
        invalidRequest();
      }
      return {
        op: "set_issue_disposition",
        issueCode: operation.issueCode,
        disposition: operation.disposition,
        note,
      };
    }
    default:
      return invalidRequest();
  }
}

function operationTextLength(operation: CandidateReviewDraftPatchOperation): number {
  switch (operation.op) {
    case "set_text":
    case "set_table_cell_text":
      return operation.nodeId.length + operation.text.length;
    case "set_role":
    case "set_needs_review":
    case "drop_image_node":
      return operation.nodeId.length;
    case "set_image_grounding":
      return (
        operation.nodeId.length +
        operation.assetRef.length +
        operation.observationRef.length
      );
    case "set_issue_disposition":
      return operation.issueCode.length + operation.note.length;
  }
}

function nodeId(value: unknown): string {
  if (
    typeof value !== "string" ||
    value.length < 1 ||
    value.length > MAX_NODE_ID_LENGTH ||
    hasUnpairedSurrogate(value)
  ) {
    invalidRequest();
  }
  return value;
}

function boundedText(value: unknown, allowEmpty: boolean): string {
  if (
    typeof value !== "string" ||
    (!allowEmpty && value.length === 0) ||
    value.length > MAX_FIELD_CHARS ||
    hasUnpairedSurrogate(value)
  ) {
    invalidRequest();
  }
  return value;
}

function hasUnpairedSurrogate(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true;
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function isDisposition(value: unknown): value is CandidateReviewIssueDisposition {
  return (
    value === "pending" ||
    value === "resolved" ||
    value === "accepted_limitation"
  );
}

function isPositiveSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 1;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function recordWithKeys(
  value: unknown,
  expectedKeys: readonly string[],
): Record<string, unknown> {
  if (!isRecord(value)) invalidRequest();
  const actualKeys = Object.keys(value).sort();
  const sortedExpected = [...expectedKeys].sort();
  if (
    actualKeys.length !== sortedExpected.length ||
    actualKeys.some((key, index) => key !== sortedExpected[index])
  ) {
    invalidRequest();
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function invalidRequest(): never {
  throw new TypeError(INVALID_REQUEST);
}

function invalidCompletionRequest(): never {
  throw new TypeError(INVALID_COMPLETION_REQUEST);
}
