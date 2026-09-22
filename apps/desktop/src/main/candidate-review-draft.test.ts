import {
  mkdirSync,
  mkdtempSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  startOrReopenCandidateReviewDraft,
  validateReviewerLabel,
  type CandidateReviewDraftCommandRunner,
} from "./candidate-review-draft.js";

const roots: string[] = [];
const manifestSha256 = "a".repeat(64);
const lineageId = "b".repeat(64);
const documentId = "candidate-01";

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("candidate review draft bridge", () => {
  it("strictly validates reviewer labels", () => {
    expect(validateReviewerLabel("reviewer_01")).toBe("reviewer_01");
    for (const invalid of ["", " leading", "한글", "a/b", "a".repeat(65)]) {
      expect(() => validateReviewerLabel(invalid)).toThrow(/검수자 라벨/);
    }
  });

  it("creates only the derived userData draft and returns a path-free summary", async () => {
    const fixture = makeFixture();
    let destination = "";
    const runner: CandidateReviewDraftCommandRunner = vi.fn(async (command) => {
      expect(command.shell).toBe(false);
      expect(command.windowsHide).toBe(true);
      expect(command.args[1]).toBe("start");
      expect(command.args).toContain("--reviewer-label");
      expect(command.args).toContain("--expected-manifest-sha256");
      expect(command.args).toContain(manifestSha256);
      expect(command.args).toContain("--expected-lineage-id");
      expect(command.args).toContain(lineageId);
      destination = command.args[command.args.indexOf("--out") + 1];
      expect(destination).toBe(
        join(
          realpathSync.native(fixture.userDataRoot),
          "candidate-reviews",
          manifestSha256,
          lineageId,
          "draft.json",
        ),
      );
      writeFileSync(destination, "{}", "utf8");
      return { code: 0, stderr: "", stdout: startSummary(destination) };
    });

    const status = await startOrReopenCandidateReviewDraft({
      ...fixture,
      manifestSha256,
      lineageId,
      documentId,
      reviewerLabel: "reviewer_01",
      commandRunner: runner,
    });

    expect(status).toMatchObject({ operation: "created", status: "in_progress", draftRevision: 1 });
    expect(JSON.stringify(status)).not.toContain(fixture.userDataRoot);
    expect(JSON.stringify(status)).not.toContain(fixture.candidateRoot);
    expect(JSON.stringify(status)).not.toContain("output");
  });

  it("revalidates an existing draft before reporting it as reopened", async () => {
    const fixture = makeFixture();
    const destination = join(
      fixture.userDataRoot,
      "candidate-reviews",
      manifestSha256,
      lineageId,
      "draft.json",
    );
    mkdirSync(join(destination, ".."), { recursive: true });
    writeFileSync(destination, "untrusted until CLI verification", "utf8");
    const runner: CandidateReviewDraftCommandRunner = vi.fn(async (command) => {
      const canonicalDestination = join(
        realpathSync.native(fixture.userDataRoot),
        "candidate-reviews",
        manifestSha256,
        lineageId,
        "draft.json",
      );
      expect(command.args).toEqual([
        realpathSync.native(fixture.verifyScript),
        "verify",
        realpathSync.native(fixture.candidateRoot),
        canonicalDestination,
        "--expected-manifest-sha256",
        manifestSha256,
        "--expected-lineage-id",
        lineageId,
      ]);
      return { code: 0, stderr: "", stdout: verifySummary() };
    });

    const status = await startOrReopenCandidateReviewDraft({
      ...fixture,
      manifestSha256,
      lineageId,
      documentId,
      reviewerLabel: "reviewer-02",
      commandRunner: runner,
    });

    expect(status.operation).toBe("reopened");
    expect(status.trainingEligible).toBe(false);
    expect(runner).toHaveBeenCalledOnce();
  });

  it("reopens and verifies the latest immutable revision", async () => {
    const fixture = makeFixture();
    const directory = join(
      fixture.userDataRoot,
      "candidate-reviews",
      manifestSha256,
      lineageId,
    );
    mkdirSync(directory, { recursive: true });
    writeFileSync(join(directory, "draft.json"), "revision one", "utf8");
    writeFileSync(join(directory, "draft.r2.json"), "revision two", "utf8");
    const runner: CandidateReviewDraftCommandRunner = vi.fn(async (command) => {
      expect(command.args[3]).toBe(
        realpathSync.native(join(directory, "draft.r2.json")),
      );
      return { code: 0, stderr: "", stdout: verifySummary(2) };
    });

    const status = await startOrReopenCandidateReviewDraft({
      ...fixture,
      manifestSha256,
      lineageId,
      documentId,
      reviewerLabel: "reviewer-latest",
      commandRunner: runner,
    });

    expect(status).toMatchObject({ operation: "reopened", draftRevision: 2 });
  });

  it("serializes concurrent create and reopen calls for one document", async () => {
    const fixture = makeFixture();
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let active = 0;
    let maximumActive = 0;
    const runner: CandidateReviewDraftCommandRunner = vi.fn(async (command) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      const outputIndex = command.args.indexOf("--out");
      if (outputIndex >= 0) {
        await firstGate;
        const destination = command.args[outputIndex + 1];
        writeFileSync(destination, "revision one", "utf8");
        active -= 1;
        return { code: 0, stderr: "", stdout: startSummary(destination) };
      }
      active -= 1;
      return { code: 0, stderr: "", stdout: verifySummary() };
    });
    const options = {
      ...fixture,
      manifestSha256,
      lineageId,
      documentId,
      reviewerLabel: "reviewer-queue",
      commandRunner: runner,
    };

    const first = startOrReopenCandidateReviewDraft(options);
    await vi.waitFor(() => expect(runner).toHaveBeenCalledTimes(1));
    const second = startOrReopenCandidateReviewDraft(options);
    await Promise.resolve();
    expect(runner).toHaveBeenCalledTimes(1);
    releaseFirst();

    await expect(first).resolves.toMatchObject({ operation: "created" });
    await expect(second).resolves.toMatchObject({ operation: "reopened" });
    expect(maximumActive).toBe(1);
  });

  it("rejects non-whitelisted or unsafe CLI summaries", async () => {
    const fixture = makeFixture();
    const runner: CandidateReviewDraftCommandRunner = async (command) => {
      const destination = command.args[command.args.indexOf("--out") + 1];
      writeFileSync(destination, "{}", "utf8");
      const parsed = JSON.parse(startSummary(destination));
      parsed.local_path = fixture.candidateRoot;
      return { code: 0, stderr: "", stdout: JSON.stringify(parsed) };
    };

    await expect(startOrReopenCandidateReviewDraft({
      ...fixture,
      manifestSha256,
      lineageId,
      documentId,
      reviewerLabel: "reviewer",
      commandRunner: runner,
    })).rejects.toThrow(/허용되지 않은 필드/);
  });

  it("rejects a CLI summary bound to a different candidate snapshot", async () => {
    const fixture = makeFixture();
    const runner: CandidateReviewDraftCommandRunner = async (command) => {
      const destination = command.args[command.args.indexOf("--out") + 1];
      writeFileSync(destination, "{}", "utf8");
      const parsed = JSON.parse(startSummary(destination));
      parsed.candidate_manifest_sha256 = "f".repeat(64);
      return { code: 0, stderr: "", stdout: JSON.stringify(parsed) };
    };

    await expect(
      startOrReopenCandidateReviewDraft({
        ...fixture,
        manifestSha256,
        lineageId,
        documentId,
        reviewerLabel: "reviewer",
        commandRunner: runner,
      }),
    ).rejects.toThrow(/안전 정책/);
  });

  it("does not include a missing local CLI path in its error", async () => {
    const fixture = makeFixture();
    rmSync(fixture.startScript);

    let error: unknown;
    try {
      await startOrReopenCandidateReviewDraft({
        ...fixture,
        manifestSha256,
        lineageId,
        documentId,
        reviewerLabel: "reviewer",
      });
    } catch (reason) {
      error = reason;
    }

    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toBe("review draft start CLI is unavailable");
    expect((error as Error).message).not.toContain(fixture.projectRoot);
  });
});

function makeFixture() {
  const root = mkdtempSync(join(tmpdir(), "exam2hwpx-review-draft-"));
  roots.push(root);
  const projectRoot = join(root, "project");
  const candidateRoot = join(root, "candidate");
  const userDataRoot = join(root, "user-data");
  const scripts = join(projectRoot, "ai", "datasets");
  mkdirSync(scripts, { recursive: true });
  mkdirSync(candidateRoot);
  mkdirSync(userDataRoot);
  const script = join(scripts, "candidate_review_draft.py");
  writeFileSync(script, "# fixture", "utf8");
  const pythonRuntimeSelector = async () => "python-test";
  return {
    projectRoot,
    candidateRoot,
    userDataRoot,
    startScript: script,
    verifyScript: script,
    pythonRuntimeSelector,
  };
}

function startSummary(output: string): string {
  return JSON.stringify({ ...summaryFields(), output });
}

function verifySummary(revision = 1): string {
  return JSON.stringify(summaryFields(revision));
}

function summaryFields(revision = 1) {
  return {
    schema_version: "candidate-review-draft/1.0",
    candidate_manifest_sha256: manifestSha256,
    document_id: documentId,
    lineage_id: lineageId,
    status: "in_progress",
    draft_revision: revision,
    rights_status: "unverified",
    identity_assurance: "self_asserted_untrusted",
    golden_eligible: false,
    training_eligible: false,
    release_eligible: false,
  };
}
