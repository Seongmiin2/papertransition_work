import { describe, expect, it } from "vitest";
import { assertTransition } from "./state-machine.js";

describe("job state machine", () => {
  it("accepts the production path", () => expect(() => assertTransition("OCR", "STRUCTURING")).not.toThrow());
  it.each(["QUEUED", "OCR", "VALIDATING", "RETRYING"] as const)("allows cancelling %s jobs", (state) => {
    expect(() => assertTransition(state, "CANCELLED")).not.toThrow();
  });
  it("rejects impossible completion", () => expect(() => assertTransition("QUEUED", "COMPLETED")).toThrow());
});
