import type { JobState } from "../types/contracts.js";

export const conversionStages = ["OCR", "STRUCTURING", "EXPORTING", "VALIDATING", "COMPLETED"] as const satisfies readonly JobState[];
export type ConversionStage = (typeof conversionStages)[number];

export interface ProgressUpdate {
  stage: ConversionStage;
  progress: number;
  pageCount: number | null;
}

/**
 * Map one conversion-worker progress message to its stage and progress.
 * Unknown messages return null so a job never falls back to an earlier stage.
 */
export function progressFromMessage(message: string): ProgressUpdate | null {
  const page = /page\s+(\d+)\/(\d+)/i.exec(message);
  if (page) {
    const current = Number(page[1]);
    const total = Number(page[2]);
    return { stage: "OCR", progress: Math.min(0.7, 0.1 + (current / total) * 0.6), pageCount: total };
  }
  if (message.includes("detecting tables")) return { stage: "STRUCTURING", progress: 0.74, pageCount: null };
  if (message.includes("writing editable HWPX")) return { stage: "EXPORTING", progress: 0.8, pageCount: null };
  if (message.includes("verifying in Hancom")) return { stage: "VALIDATING", progress: 0.9, pageCount: null };
  return null;
}

/** Stages still to pass, in order, to move a job from `current` to `target`. */
export function stagesBetween(current: JobState, target: ConversionStage): ConversionStage[] {
  const from = conversionStages.indexOf(current as ConversionStage);
  const to = conversionStages.indexOf(target);
  return from < 0 || to <= from ? [] : conversionStages.slice(from + 1, to + 1);
}
