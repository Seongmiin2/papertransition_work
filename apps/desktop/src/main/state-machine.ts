import type { JobState } from "../types/contracts.js";

export const transitions: Readonly<Record<JobState, readonly JobState[]>> = {
  QUEUED: ["STAGING", "CANCELLED"], STAGING: ["PREPROCESSING", "FAILED", "CANCELLED"],
  PREPROCESSING: ["OCR", "FAILED", "CANCELLED"], OCR: ["STRUCTURING", "FAILED", "CANCELLED"],
  STRUCTURING: ["REVIEW_READY", "EXPORTING", "FAILED", "CANCELLED"],
  REVIEW_READY: ["EXPORTING", "CANCELLED"], EXPORTING: ["VALIDATING", "FAILED", "CANCELLED"],
  VALIDATING: ["COMPLETED", "FAILED", "CANCELLED"], COMPLETED: [], FAILED: ["RETRYING"],
  RETRYING: ["STAGING", "PREPROCESSING", "OCR", "STRUCTURING", "EXPORTING", "CANCELLED"], CANCELLED: ["RETRYING"]
};

export function assertTransition(from: JobState, to: JobState): void {
  if (!transitions[from].includes(to)) throw new Error(`Invalid job transition: ${from} -> ${to}`);
}
