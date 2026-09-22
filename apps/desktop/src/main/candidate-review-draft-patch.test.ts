import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  applyCandidateReviewDraftPatch,
  CandidateReviewDraftStaleError,
  completeCandidateReviewDraft,
  isCandidateReviewDraftStaleError,
  loadCandidateReviewDraftView,
  type CandidateReviewDraftBridgeCommand,
  type CandidateReviewDraftBridgeCommandRunner,
  type CandidateReviewDraftPatchRequest,
} from "./candidate-review-draft-patch.js";
import { pythonRuntimeEnvironment } from "./python-runtime.js";

const manifestSha256 = "a".repeat(64);
const lineageId = "b".repeat(64);
const contentSha256 = "c".repeat(64);
const documentId = "candidate-01";
const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("candidate review draft patch bridge", () => {
  it("translates all seven operations and creates an immutable sanitized revision", async () => {
    const fixture = makeFixture();
    const original = readFileSync(fixture.draftPath);
    const commands: CandidateReviewDraftBridgeCommand[] = [];
    const runtimeSelector = vi.fn(async () => "python-test");
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      commands.push(command);
      if (command.args[1] === "patch") {
        const destination = command.args[command.args.indexOf("--out") + 1];
        writeFileSync(destination, "immutable revision 2", "utf8");
        return commandResult(patchSummary(2));
      }
      return commandResult(reviewView(2));
    });
    const request: CandidateReviewDraftPatchRequest = {
      expectedDraftRevision: 1,
      operations: [
        { op: "set_text", nodeId: "text-1", text: "교정된 문장" },
        { op: "set_role", nodeId: "text-1", role: "question" },
        {
          op: "set_table_cell_text",
          nodeId: "table-1",
          row: 0,
          column: 0,
          rowSpan: 1,
          columnSpan: 1,
          text: "교정된 셀",
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
          note: "별도 권리 확인 필요",
        },
      ],
    };

    const view = await applyCandidateReviewDraftPatch({
      ...fixture.bridgeOptions,
      request,
      commandRunner: runner,
      pythonRuntimeSelector: runtimeSelector,
    });

    expect(commands).toHaveLength(2);
    const [patchCommand, viewCommand] = commands;
    expect(patchCommand.args).toEqual([
      fixture.script,
      "patch",
      fixture.candidateRoot,
      fixture.draftPath,
      "--expected-manifest-sha256",
      manifestSha256,
      "--expected-lineage-id",
      lineageId,
      "--out",
      fixture.revisionPath(2),
    ]);
    expect(JSON.parse(patchCommand.stdin ?? "")).toEqual({
      schema_version: "candidate-review-patch/1.1",
      expected_draft_revision: 1,
      operations: [
        { op: "set_text", node_id: "text-1", text: "교정된 문장" },
        { op: "set_role", node_id: "text-1", role: "question" },
        {
          op: "set_table_cell_text",
          node_id: "table-1",
          row: 0,
          column: 0,
          row_span: 1,
          column_span: 1,
          text: "교정된 셀",
        },
        { op: "set_needs_review", node_id: "image-1", needs_review: false },
        {
          op: "set_image_grounding",
          node_id: "image-1",
          asset_ref: "page-0001-image-0001-crop",
          observation_ref: "page-0001-image-0001",
        },
        { op: "drop_image_node", node_id: "image-2" },
        {
          op: "set_issue_disposition",
          issue_code: "rights_manifest_missing",
          disposition: "accepted_limitation",
          note: "별도 권리 확인 필요",
        },
      ],
    });
    expect(viewCommand.args).toEqual([
      fixture.script,
      "view",
      fixture.candidateRoot,
      fixture.revisionPath(2),
      "--expected-manifest-sha256",
      manifestSha256,
      "--expected-lineage-id",
      lineageId,
    ]);
    expect(viewCommand.stdin).toBeNull();
    for (const command of commands) {
      expect(command.executable).toBe("python-test");
      expect(command.shell).toBe(false);
      expect(command.windowsHide).toBe(true);
      expect(command.cwd).toBe(fixture.projectRoot);
      expect(command.env).toEqual(pythonRuntimeEnvironment(fixture.projectRoot));
    }
    expect(runtimeSelector).toHaveBeenCalledOnce();
    expect(runtimeSelector).toHaveBeenCalledWith({
      projectRoot: fixture.projectRoot,
      requiredImports: ["scan2hwpx", "pydantic"],
    });

    expect(readFileSync(fixture.draftPath)).toEqual(original);
    expect(readFileSync(fixture.revisionPath(2), "utf8")).toBe("immutable revision 2");
    expect(readdirSync(fixture.candidateRoot)).toEqual([]);
    expect(view).toEqual(camelReviewView(2));
    const serializedView = JSON.stringify(view);
    expect(serializedView).not.toContain(fixture.projectRoot);
    expect(serializedView).not.toContain(fixture.userDataRoot);
    expect(serializedView).not.toContain(fixture.candidateRoot);
    expect(serializedView).not.toContain("evidence_refs");
    expect(serializedView).not.toContain("confidence");
  });

  it("completes one exact revision as a new immutable noneligible revision", async () => {
    const fixture = makeFixture();
    const original = readFileSync(fixture.draftPath);
    const commands: CandidateReviewDraftBridgeCommand[] = [];
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      commands.push(command);
      if (command.args[1] === "complete") {
        const destination = command.args[command.args.indexOf("--out") + 1];
        writeFileSync(destination, "immutable complete revision 2", "utf8");
        return commandResult(completionSummary(2));
      }
      return commandResult(reviewView(2, "complete"));
    });

    const view = await completeCandidateReviewDraft({
      ...fixture.bridgeOptions,
      request: { expectedDraftRevision: 1 },
      commandRunner: runner,
      pythonRuntimeSelector: async () => "python-test",
    });

    expect(commands).toHaveLength(2);
    expect(commands[0].args).toEqual([
      fixture.script,
      "complete",
      fixture.candidateRoot,
      fixture.draftPath,
      "--expected-manifest-sha256",
      manifestSha256,
      "--expected-lineage-id",
      lineageId,
      "--out",
      fixture.revisionPath(2),
    ]);
    expect(JSON.parse(commands[0].stdin ?? "")).toEqual({
      schema_version: "candidate-review-completion/1.0",
      expected_draft_revision: 1,
    });
    expect(commands[1].args).toEqual([
      fixture.script,
      "view",
      fixture.candidateRoot,
      fixture.revisionPath(2),
      "--expected-manifest-sha256",
      manifestSha256,
      "--expected-lineage-id",
      lineageId,
    ]);
    expect(readFileSync(fixture.draftPath)).toEqual(original);
    expect(readFileSync(fixture.revisionPath(2), "utf8")).toBe(
      "immutable complete revision 2",
    );
    expect(view).toEqual(camelReviewView(2, "complete"));
    expect(view).toMatchObject({
      status: "complete",
      rightsStatus: "unverified",
      identityAssurance: "self_asserted_untrusted",
      goldenEligible: false,
      trainingEligible: false,
      releaseEligible: false,
    });
  });

  it("maps a completion atomic-create loss to the stale recovery path", async () => {
    const fixture = makeFixture();
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      if (command.args[1] === "complete") {
        writeFileSync(fixture.revisionPath(2), "other process revision", "utf8");
        return { code: 1, stdout: "", stderr: "destination already exists" };
      }
      return commandResult(reviewView(2, "complete"));
    });

    await expect(
      completeCandidateReviewDraft({
        ...fixture.bridgeOptions,
        request: { expectedDraftRevision: 1 },
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      }),
    ).rejects.toThrow("Review draft revision is stale. Reopen it and retry.");

    expect(readFileSync(fixture.revisionPath(2), "utf8")).toBe(
      "other process revision",
    );
  });

  it("serializes completion behind an in-flight patch for the same document", async () => {
    const fixture = makeFixture();
    let releasePatch!: () => void;
    let markPatchEntered!: () => void;
    const patchGate = new Promise<void>((resolve) => {
      releasePatch = resolve;
    });
    const patchEntered = new Promise<void>((resolve) => {
      markPatchEntered = resolve;
    });
    const commands: CandidateReviewDraftBridgeCommand[] = [];
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      commands.push(command);
      if (command.args[1] === "patch") {
        markPatchEntered();
        await patchGate;
        writeFileSync(fixture.revisionPath(2), "immutable revision 2", "utf8");
        return commandResult(patchSummary(2));
      }
      if (command.args[1] === "complete") {
        writeFileSync(fixture.revisionPath(3), "immutable complete revision 3", "utf8");
        return commandResult(completionSummary(3));
      }
      const isComplete = command.args[3] === fixture.revisionPath(3);
      return commandResult(reviewView(isComplete ? 3 : 2, isComplete ? "complete" : "in_progress"));
    });
    const sharedOptions = {
      ...fixture.bridgeOptions,
      commandRunner: runner,
      pythonRuntimeSelector: async () => "python-test",
    };

    const patchPromise = applyCandidateReviewDraftPatch({
      ...sharedOptions,
      request: needsReviewRequest(1),
    });
    await patchEntered;
    const completionPromise = completeCandidateReviewDraft({
      ...sharedOptions,
      request: { expectedDraftRevision: 2 },
    });
    await Promise.resolve();
    expect(commands).toHaveLength(1);
    releasePatch();

    const [patched, completed] = await Promise.all([
      patchPromise,
      completionPromise,
    ]);
    expect(patched).toMatchObject({ status: "in_progress", draftRevision: 2 });
    expect(completed).toMatchObject({ status: "complete", draftRevision: 3 });
    expect(commands.map((command) => command.args[1])).toEqual([
      "patch",
      "view",
      "complete",
      "view",
    ]);
  });

  it("finds revision ten as the latest stored draft", async () => {
    const fixture = makeFixture();
    for (let revision = 2; revision <= 10; revision += 1) {
      writeFileSync(fixture.revisionPath(revision), `revision ${revision}`, "utf8");
    }
    const commands: CandidateReviewDraftBridgeCommand[] = [];
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      commands.push(command);
      return commandResult(reviewView(10));
    });

    const view = await loadCandidateReviewDraftView({
      ...fixture.bridgeOptions,
      commandRunner: runner,
      pythonRuntimeSelector: async () => "python-test",
    });

    expect(view.draftRevision).toBe(10);
    expect(commands).toHaveLength(1);
    expect(commands[0].args).toEqual([
      fixture.script,
      "view",
      fixture.candidateRoot,
      fixture.revisionPath(10),
      "--expected-manifest-sha256",
      manifestSha256,
      "--expected-lineage-id",
      lineageId,
    ]);
  });

  it("rejects a stale revision without running a command or creating output", async () => {
    const fixture = makeFixture();
    writeFileSync(fixture.revisionPath(2), "immutable revision 2", "utf8");
    const originalRevisionOne = readFileSync(fixture.draftPath);
    const originalRevisionTwo = readFileSync(fixture.revisionPath(2));
    const runner = vi.fn<CandidateReviewDraftBridgeCommandRunner>();
    const runtimeSelector = vi.fn(async () => "python-test");

    const result = applyCandidateReviewDraftPatch({
      ...fixture.bridgeOptions,
      request: needsReviewRequest(1),
      commandRunner: runner,
      pythonRuntimeSelector: runtimeSelector,
    });
    let failure: unknown;
    try {
      await result;
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(CandidateReviewDraftStaleError);
    expect(isCandidateReviewDraftStaleError(failure)).toBe(true);
    expect((failure as Error).message).toBe(
      "Review draft revision is stale. Reopen it and retry.",
    );

    expect(runner).not.toHaveBeenCalled();
    expect(runtimeSelector).not.toHaveBeenCalled();
    expect(existsSync(fixture.revisionPath(3))).toBe(false);
    expect(readFileSync(fixture.draftPath)).toEqual(originalRevisionOne);
    expect(readFileSync(fixture.revisionPath(2))).toEqual(originalRevisionTwo);
  });

  it("maps a cross-process atomic-create loss to the stale recovery path", async () => {
    const fixture = makeFixture();
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      if (command.args[1] === "patch") {
        writeFileSync(fixture.revisionPath(2), "other process revision", "utf8");
        return { code: 1, stdout: "", stderr: "destination already exists" };
      }
      return commandResult(reviewView(2));
    });

    await expect(
      applyCandidateReviewDraftPatch({
        ...fixture.bridgeOptions,
        request: needsReviewRequest(1),
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      }),
    ).rejects.toThrow("Review draft revision is stale. Reopen it and retry.");

    expect(readFileSync(fixture.revisionPath(2), "utf8")).toBe(
      "other process revision",
    );
  });

  it("serializes patch and view commands for the same document", async () => {
    const fixture = makeFixture();
    let releasePatch!: () => void;
    let markPatchEntered!: () => void;
    const patchGate = new Promise<void>((resolve) => {
      releasePatch = resolve;
    });
    const patchEntered = new Promise<void>((resolve) => {
      markPatchEntered = resolve;
    });
    let activeCommands = 0;
    let maximumActiveCommands = 0;
    const commands: CandidateReviewDraftBridgeCommand[] = [];
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      commands.push(command);
      activeCommands += 1;
      maximumActiveCommands = Math.max(maximumActiveCommands, activeCommands);
      try {
        if (command.args[1] === "patch") {
          markPatchEntered();
          await patchGate;
          writeFileSync(fixture.revisionPath(2), "immutable revision 2", "utf8");
          return commandResult(patchSummary(2));
        }
        const revision = command.args[3] === fixture.draftPath ? 1 : 2;
        return commandResult(reviewView(revision));
      } finally {
        activeCommands -= 1;
      }
    });
    const sharedOptions = {
      ...fixture.bridgeOptions,
      commandRunner: runner,
      pythonRuntimeSelector: async () => "python-test",
    };

    const patchPromise = applyCandidateReviewDraftPatch({
      ...sharedOptions,
      request: needsReviewRequest(1),
    });
    await patchEntered;
    const loadPromise = loadCandidateReviewDraftView(sharedOptions);
    try {
      await Promise.resolve();
      expect(commands).toHaveLength(1);
    } finally {
      releasePatch();
    }

    const [patched, loaded] = await Promise.all([patchPromise, loadPromise]);
    expect(patched.draftRevision).toBe(2);
    expect(loaded.draftRevision).toBe(2);
    expect(commands.map((command) => command.args[1])).toEqual([
      "patch",
      "view",
      "view",
    ]);
    expect(maximumActiveCommands).toBe(1);
  });

  it("serializes concurrent patches and rejects the queued stale writer", async () => {
    const fixture = makeFixture();
    let releasePatch!: () => void;
    let markPatchEntered!: () => void;
    const patchGate = new Promise<void>((resolve) => {
      releasePatch = resolve;
    });
    const patchEntered = new Promise<void>((resolve) => {
      markPatchEntered = resolve;
    });
    const commands: CandidateReviewDraftBridgeCommand[] = [];
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async (command) => {
      commands.push(command);
      if (command.args[1] === "patch") {
        markPatchEntered();
        await patchGate;
        writeFileSync(fixture.revisionPath(2), "immutable revision 2", "utf8");
        return commandResult(patchSummary(2));
      }
      return commandResult(reviewView(2));
    });
    const options = {
      ...fixture.bridgeOptions,
      request: needsReviewRequest(1),
      commandRunner: runner,
      pythonRuntimeSelector: async () => "python-test",
    };

    const first = applyCandidateReviewDraftPatch(options);
    await patchEntered;
    const second = applyCandidateReviewDraftPatch(options);
    void second.catch(() => undefined);
    await Promise.resolve();
    expect(commands).toHaveLength(1);
    releasePatch();

    await expect(first).resolves.toMatchObject({ draftRevision: 2 });
    await expect(second).rejects.toThrow(
      "Review draft revision is stale. Reopen it and retry.",
    );
    expect(commands.map((command) => command.args[1])).toEqual([
      "patch",
      "view",
    ]);
    expect(existsSync(fixture.revisionPath(3))).toBe(false);
  });

  it("rejects an unlisted mutation before invoking Python", async () => {
    const fixture = makeFixture();
    const runner = vi.fn<CandidateReviewDraftBridgeCommandRunner>();

    await expect(
      applyCandidateReviewDraftPatch({
        ...fixture.bridgeOptions,
        request: {
          expectedDraftRevision: 1,
          operations: [
            { op: "set_confidence", nodeId: "text-1", confidence: 1 },
          ],
        } as unknown as CandidateReviewDraftPatchRequest,
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      }),
    ).rejects.toThrow("Review draft patch request is invalid.");

    expect(runner).not.toHaveBeenCalled();
    expect(existsSync(fixture.revisionPath(2))).toBe(false);
  });

  it.each(["\ud800", "\udc00"])(
    "rejects an unpaired UTF-16 surrogate before invoking Python: %s",
    async (text) => {
      const fixture = makeFixture();
      const runner = vi.fn<CandidateReviewDraftBridgeCommandRunner>();

      await expect(
        applyCandidateReviewDraftPatch({
          ...fixture.bridgeOptions,
          request: {
            expectedDraftRevision: 1,
            operations: [{ op: "set_text", nodeId: "text-1", text }],
          },
          commandRunner: runner,
          pythonRuntimeSelector: async () => "python-test",
        }),
      ).rejects.toThrow("Review draft patch request is invalid.");

      expect(runner).not.toHaveBeenCalled();
      expect(existsSync(fixture.revisionPath(2))).toBe(false);
    },
  );

  it("accepts the full Python ContentRole set in a sanitized text-node view", async () => {
    const fixture = makeFixture();
    const payload = reviewView(1);
    (payload.nodes[0] as Record<string, unknown>).role = "formula";
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async () =>
      commandResult(payload),
    );

    const view = await loadCandidateReviewDraftView({
      ...fixture.bridgeOptions,
      commandRunner: runner,
      pythonRuntimeSelector: async () => "python-test",
    });

    expect(view.nodes[0]).toMatchObject({ kind: "text", role: "formula" });
  });

  it("returns a generic redacted error for a failed Python command", async () => {
    const fixture = makeFixture();
    const privateText = "renderer-private-text";
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async () => ({
      code: 1,
      stdout: "",
      stderr: `${fixture.projectRoot} ${fixture.candidateRoot} ${fixture.draftPath} ${privateText}`,
    }));
    let failure: unknown;

    try {
      await applyCandidateReviewDraftPatch({
        ...fixture.bridgeOptions,
        request: {
          expectedDraftRevision: 1,
          operations: [{ op: "set_text", nodeId: "text-1", text: privateText }],
        },
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      });
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(Error);
    expect((failure as Error).message).toBe("Review draft update failed.");
    expect((failure as Error).message).not.toContain(fixture.projectRoot);
    expect((failure as Error).message).not.toContain(fixture.candidateRoot);
    expect((failure as Error).message).not.toContain(fixture.draftPath);
    expect((failure as Error).message).not.toContain(privateText);
    expect(existsSync(fixture.revisionPath(2))).toBe(false);
  });

  it("rejects extra fields nested inside a review view", async () => {
    const fixture = makeFixture();
    const payload = reviewView(1);
    const firstNode = (payload.nodes as Array<Record<string, unknown>>)[0];
    firstNode.confidence = 0.99;
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async () =>
      commandResult(payload),
    );

    await expect(
      loadCandidateReviewDraftView({
        ...fixture.bridgeOptions,
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      }),
    ).rejects.toThrow("Review draft view is unavailable.");
  });

  it("rejects a complete view with unresolved review state", async () => {
    const fixture = makeFixture();
    const payload = reviewView(1, "complete");
    (payload.nodes[0] as Record<string, unknown>).needs_review = true;
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(async () =>
      commandResult(payload),
    );

    await expect(
      loadCandidateReviewDraftView({
        ...fixture.bridgeOptions,
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      }),
    ).rejects.toThrow("Review draft view is unavailable.");
  });

  it("rejects a patch summary bound to a different manifest", async () => {
    const fixture = makeFixture();
    const runner: CandidateReviewDraftBridgeCommandRunner = vi.fn(
      async (command) => {
        if (command.args[1] === "patch") {
          const destination = command.args[command.args.indexOf("--out") + 1];
          writeFileSync(destination, "immutable revision 2", "utf8");
          return commandResult({
            ...patchSummary(2),
            candidate_manifest_sha256: "f".repeat(64),
          });
        }
        return commandResult(reviewView(2));
      },
    );

    await expect(
      applyCandidateReviewDraftPatch({
        ...fixture.bridgeOptions,
        request: needsReviewRequest(1),
        commandRunner: runner,
        pythonRuntimeSelector: async () => "python-test",
      }),
    ).rejects.toThrow("Review draft update failed.");
  });
});

function makeFixture() {
  const root = mkdtempSync(join(tmpdir(), "exam2hwpx-review-patch-"));
  roots.push(root);
  const projectRoot = join(root, "project");
  const candidateRoot = join(root, "candidate");
  const userDataRoot = join(root, "user-data");
  const scripts = join(projectRoot, "ai", "datasets");
  const draftDirectory = join(
    userDataRoot,
    "candidate-reviews",
    manifestSha256,
    lineageId,
  );
  mkdirSync(join(projectRoot, "src"), { recursive: true });
  mkdirSync(scripts, { recursive: true });
  mkdirSync(candidateRoot);
  mkdirSync(draftDirectory, { recursive: true });
  const script = join(scripts, "candidate_review_draft.py");
  const draftPath = join(draftDirectory, "draft.json");
  writeFileSync(script, "# fixture", "utf8");
  writeFileSync(draftPath, "immutable revision 1", "utf8");

  const canonicalProjectRoot = realpathSync.native(projectRoot);
  const canonicalCandidateRoot = realpathSync.native(candidateRoot);
  const canonicalUserDataRoot = realpathSync.native(userDataRoot);
  const canonicalDraftDirectory = realpathSync.native(draftDirectory);
  return {
    projectRoot: canonicalProjectRoot,
    candidateRoot: canonicalCandidateRoot,
    userDataRoot: canonicalUserDataRoot,
    draftPath: realpathSync.native(draftPath),
    script: realpathSync.native(script),
    revisionPath: (revision: number) =>
      join(canonicalDraftDirectory, `draft.r${revision}.json`),
    bridgeOptions: {
      projectRoot: canonicalProjectRoot,
      userDataRoot: canonicalUserDataRoot,
      candidateRoot: canonicalCandidateRoot,
      manifestSha256,
      lineageId,
      documentId,
    },
  };
}

function needsReviewRequest(expectedDraftRevision: number): CandidateReviewDraftPatchRequest {
  return {
    expectedDraftRevision,
    operations: [{ op: "set_needs_review", nodeId: "image-1", needsReview: false }],
  };
}

function commandResult(payload: unknown) {
  return { code: 0, stderr: "", stdout: JSON.stringify(payload) };
}

function patchSummary(revision: number) {
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

function completionSummary(revision: number) {
  return { ...patchSummary(revision), status: "complete" };
}

function reviewView(
  revision: number,
  status: "in_progress" | "complete" = "in_progress",
) {
  return {
    schema_version: "candidate-review-view/1.0",
    document_id: documentId,
    status,
    draft_revision: revision,
    updated_at: "2026-09-10T15:00:00+09:00",
    reviewed_content_ir_contract_sha256: contentSha256,
    nodes: [
      {
        id: "text-1",
        kind: "text",
        text: "교정된 문장",
        role: "question",
        needs_review: false,
      },
      {
        id: "table-1",
        kind: "table",
        needs_review: false,
        cells: [
          {
            row: 0,
            column: 0,
            row_span: 1,
            column_span: 1,
            text: "교정된 셀",
          },
        ],
      },
      { id: "image-1", kind: "image", needs_review: false },
      {
        id: "formula-1",
        kind: "formula",
        needs_review: status === "in_progress",
      },
    ],
    issue_dispositions: [
      {
        issue_code: "rights_manifest_missing",
        disposition: "accepted_limitation",
        note: "별도 권리 확인 필요",
      },
    ],
    rights_status: "unverified",
    identity_assurance: "self_asserted_untrusted",
    golden_eligible: false,
    training_eligible: false,
    release_eligible: false,
  };
}

function camelReviewView(
  revision: number,
  status: "in_progress" | "complete" = "in_progress",
) {
  return {
    schemaVersion: "candidate-review-view/1.0",
    documentId,
    status,
    draftRevision: revision,
    updatedAt: "2026-09-10T15:00:00+09:00",
    rightsStatus: "unverified",
    identityAssurance: "self_asserted_untrusted",
    goldenEligible: false,
    trainingEligible: false,
    releaseEligible: false,
    reviewedContentIrContractSha256: contentSha256,
    nodes: [
      {
        id: "text-1",
        kind: "text",
        text: "교정된 문장",
        role: "question",
        needsReview: false,
      },
      {
        id: "table-1",
        kind: "table",
        needsReview: false,
        cells: [
          {
            row: 0,
            column: 0,
            rowSpan: 1,
            columnSpan: 1,
            text: "교정된 셀",
          },
        ],
      },
      { id: "image-1", kind: "image", needsReview: false },
      {
        id: "formula-1",
        kind: "formula",
        needsReview: status === "in_progress",
      },
    ],
    issueDispositions: [
      {
        issueCode: "rights_manifest_missing",
        disposition: "accepted_limitation",
        note: "별도 권리 확인 필요",
      },
    ],
  };
}
