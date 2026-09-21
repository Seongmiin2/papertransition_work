export const jobStates = ["QUEUED", "STAGING", "PREPROCESSING", "OCR", "STRUCTURING", "REVIEW_READY", "EXPORTING", "VALIDATING", "COMPLETED", "FAILED", "RETRYING", "CANCELLED"] as const;
export type JobState = typeof jobStates[number];

export interface JobRecord {
  id: string;
  sourcePath: string;
  sourceName: string;
  sourceSha256: string;
  status: JobState;
  stage: JobState;
  progress: number;
  pageCount: number | null;
  outputPath: string | null;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

export type CandidateJsonPrimitive = string | number | boolean | null;
export type CandidateJsonValue =
  | CandidateJsonPrimitive
  | CandidateJsonObject
  | CandidateJsonValue[];
export interface CandidateJsonObject {
  [key: string]: CandidateJsonValue;
}

export interface CandidateBundleSummary {
  schemaVersion: "hwp-human-review-candidates/1.0";
  manifestSha256: string;
  documentCount: number;
  pageCount: number;
  issueCounts: Readonly<Record<string, number>>;
}

export interface CandidateDocumentSummary {
  documentId: string;
  lineageId: string;
  split: string;
  pageCount: number;
  status: "needs_human_review";
  issueCodes: readonly string[];
}

export interface CandidateBundleSelection {
  summary: CandidateBundleSummary;
  documents: readonly CandidateDocumentSummary[];
}

export interface CandidatePageSummary {
  pageNo: number;
  width: number;
  height: number;
  rotation: number;
}

export interface EvidenceOverlayBox {
  observationId: string;
  kind: string;
  pageNo: number;
  pixel: readonly [number, number, number, number];
  normalized: readonly [number, number, number, number];
}

export interface ContentNodeOverlay {
  contentRef: string;
  contentKind: string;
  confidence: number;
  needsReview: boolean;
  evidenceRefs: readonly string[];
  pageNumbers: readonly number[];
  primaryPageNo: number;
  spansMultiplePages: boolean;
  usesPageRegionFallback: boolean;
  boxes: readonly EvidenceOverlayBox[];
}

export interface CandidateImageGroundingCandidate {
  observationRef: string;
  assetRef: string;
  pageNo: number;
  sourceKind: "page_image" | "crop";
  sha256: string;
}

export interface CandidateReviewDocument {
  document: CandidateDocumentSummary;
  artifactDigests: {
    evidenceIrSha256: string;
    contentIrSha256: string;
    hwpDocumentPlanSha256: string;
  };
  pages: readonly CandidatePageSummary[];
  overlays: readonly ContentNodeOverlay[];
  imageGroundingCandidates: readonly CandidateImageGroundingCandidate[];
  evidenceIr: CandidateJsonObject;
  contentIr: CandidateJsonObject;
  hwpDocumentPlan: CandidateJsonObject;
  report: CandidateJsonObject;
}

export interface CandidatePageImageDataUrl {
  pageNo: number;
  mimeType: "image/png";
  sha256: string;
  dataUrl: string;
}

export interface CandidateReviewDraftRequest {
  lineageId: string;
  reviewerLabel: string;
}

export const candidateReviewTextRoles = [
  "title",
  "instruction",
  "passage",
  "question",
  "choice",
  "caption",
  "header",
  "footer",
  "other",
] as const;
export type CandidateReviewTextRole = typeof candidateReviewTextRoles[number];
export type CandidateReviewContentRole =
  | CandidateReviewTextRole
  | "table"
  | "image"
  | "formula";

export interface CandidateReviewDraftTextNode {
  id: string;
  kind: "text";
  text: string;
  role: CandidateReviewContentRole;
  needsReview: boolean;
}

export interface CandidateReviewDraftTableCell {
  row: number;
  column: number;
  rowSpan: number;
  columnSpan: number;
  text: string;
}

export interface CandidateReviewDraftTableNode {
  id: string;
  kind: "table";
  needsReview: boolean;
  cells: readonly CandidateReviewDraftTableCell[];
}

export interface CandidateReviewDraftImageNode {
  id: string;
  kind: "image";
  needsReview: boolean;
}

export interface CandidateReviewDraftFormulaNode {
  id: string;
  kind: "formula";
  needsReview: boolean;
}

export type CandidateReviewDraftNode =
  | CandidateReviewDraftTextNode
  | CandidateReviewDraftTableNode
  | CandidateReviewDraftImageNode
  | CandidateReviewDraftFormulaNode;

export type CandidateReviewIssueDisposition =
  | "pending"
  | "resolved"
  | "accepted_limitation";

export interface CandidateReviewDraftIssueDisposition {
  issueCode: string;
  disposition: CandidateReviewIssueDisposition;
  note: string;
}

export interface CandidateReviewDraftView {
  schemaVersion: "candidate-review-view/1.0";
  documentId: string;
  status: "in_progress" | "complete";
  draftRevision: number;
  updatedAt: string;
  reviewedContentIrContractSha256: string;
  nodes: readonly CandidateReviewDraftNode[];
  issueDispositions: readonly CandidateReviewDraftIssueDisposition[];
  rightsStatus: "unverified";
  identityAssurance: "self_asserted_untrusted";
  goldenEligible: false;
  trainingEligible: false;
  releaseEligible: false;
}

export type CandidateReviewDraftPatchOperation =
  | { op: "set_text"; nodeId: string; text: string }
  | { op: "set_role"; nodeId: string; role: CandidateReviewTextRole }
  | {
      op: "set_table_cell_text";
      nodeId: string;
      row: number;
      column: number;
      rowSpan: number;
      columnSpan: number;
      text: string;
    }
  | { op: "set_needs_review"; nodeId: string; needsReview: boolean }
  | {
      op: "set_image_grounding";
      nodeId: string;
      assetRef: string;
      observationRef: string;
    }
  | { op: "drop_image_node"; nodeId: string }
  | {
      op: "set_issue_disposition";
      issueCode: string;
      disposition: CandidateReviewIssueDisposition;
      note: string;
    };

export interface CandidateReviewDraftPatchRequest {
  lineageId: string;
  expectedDraftRevision: number;
  operations: readonly CandidateReviewDraftPatchOperation[];
}

export interface CandidateReviewDraftCompletionRequest {
  lineageId: string;
  expectedDraftRevision: number;
}

export interface CandidateReviewDraftOpenResult {
  operation: "created" | "reopened";
  view: CandidateReviewDraftView;
}

export type CandidateReviewDraftPatchResult =
  | { kind: "saved"; view: CandidateReviewDraftView }
  | { kind: "stale_revision"; currentView: CandidateReviewDraftView };

export type CandidateReviewDraftCompletionResult =
  | { kind: "saved"; view: CandidateReviewDraftView }
  | { kind: "stale_revision"; currentView: CandidateReviewDraftView };

export interface Exam2HwpxApi {
  getPathForFile(file: File): string;
  selectPdfs(): Promise<string[]>;
  enqueue(paths: string[]): Promise<JobRecord[]>;
  listJobs(): Promise<JobRecord[]>;
  cancel(jobId: string): Promise<void>;
  openPath(path: string): Promise<void>;
  selectCandidateBundle(): Promise<CandidateBundleSelection | null>;
  loadCandidateDocument(lineageId: string): Promise<CandidateReviewDocument>;
  loadCandidatePage(
    lineageId: string,
    pageNo: number,
  ): Promise<CandidatePageImageDataUrl>;
  startOrReopenCandidateReviewDraft(
    request: CandidateReviewDraftRequest,
  ): Promise<CandidateReviewDraftOpenResult>;
  applyCandidateReviewDraftPatch(
    request: CandidateReviewDraftPatchRequest,
  ): Promise<CandidateReviewDraftPatchResult>;
  completeCandidateReviewDraft(
    request: CandidateReviewDraftCompletionRequest,
  ): Promise<CandidateReviewDraftCompletionResult>;
  onJobEvent(callback: (job: JobRecord) => void): () => void;
}
