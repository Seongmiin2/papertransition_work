import { describe, expect, it } from "vitest";
import { progressFromMessage, stagesBetween } from "./progress.js";

describe("conversion progress", () => {
  it("spreads OCR pages across the OCR share of the bar", () => {
    expect(progressFromMessage("exam.pdf - page 4/8: 페이지 OCR")).toEqual({ stage: "OCR", progress: 0.4, pageCount: 8 });
    expect(progressFromMessage("exam.pdf - page 8/8: 페이지 OCR")?.progress).toBe(0.7);
  });

  it("maps the later pipeline steps to later stages with more progress", () => {
    const steps = [
      "exam.pdf - page 8/8: 페이지 OCR",
      "exam.pdf - detecting tables, boxes, and pictures",
      "exam.pdf - writing editable HWPX",
      "exam.pdf - verifying in Hancom",
    ].map((message) => progressFromMessage(message));
    expect(steps.map((step) => step?.stage)).toEqual(["OCR", "STRUCTURING", "EXPORTING", "VALIDATING"]);
    const progress = steps.map((step) => step?.progress ?? 0);
    expect(progress).toEqual([...progress].sort((a, b) => a - b));
  });

  it("ignores messages it does not know instead of resetting to the start", () => {
    expect(progressFromMessage("exam.pdf - something new")).toBeNull();
  });

  it("lists only the stages ahead of the current one", () => {
    expect(stagesBetween("OCR", "COMPLETED")).toEqual(["STRUCTURING", "EXPORTING", "VALIDATING", "COMPLETED"]);
    expect(stagesBetween("EXPORTING", "COMPLETED")).toEqual(["VALIDATING", "COMPLETED"]);
    expect(stagesBetween("VALIDATING", "EXPORTING")).toEqual([]);
  });
});
