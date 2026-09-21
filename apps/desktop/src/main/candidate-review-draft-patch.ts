import { spawn } from "node:child_process";
import {
  existsSync,
  lstatSync,
  readdirSync,
  realpathSync,
} from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  pythonRuntimeEnvironment,
  selectPythonRuntime,
} from "./python-runtime.js";

const DRAFT_SCHEMA = "candidate-review-draft/1.0";
const PATCH_SCHEMA = "candidate-review-patch/1.1";
const COMPLETION_SCHEMA = "candidate-review-completion/1.0";
const VIEW_SCHEMA = "candidate-review-view/1.0";
const SHA256 = /^[0-9a-f]{64}$/;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const ISSUE_CODE = /^[a-z][a-z0-9_.-]*$/;
const REVISION_FILE = /^draft\.r([1-9][0-9]*)\.json$/;
const MAX_DRAFT_REVISION = 1_000_000;
const MAX_PATCH_OPERATIONS = 1_000;
const MAX_NODE_ID_CHARS = 1_024;
const MAX_DOCUMENT_ID_CHARS = 256;
const MAX_ISSUE_CODE_CHARS = 256;
const MAX_VIEW_NODES = 5_000;
const MAX_VIEW_CELLS = 100_000;
const MAX_VIEW_ISSUES = 1_000;
const MAX_FIELD_CHARS = 1_000_000;
const MAX_TOTAL_VIEW_CHARS = 10_000_000;
const MAX_STDIN_BYTES = 16 * 1024 * 1024;
const MAX_STDOUT_BYTES = 64 * 1024 * 1024;
const MAX_STDERR_BYTES = 16 * 1024;
const PROCESS_TIMEOUT_MS = 120_000;
const STALE_MESSAGE = "Review draft revision is stale. Reopen it and retry.";
const PATCH_FAILED_MESSAGE = "Review draft update failed.";
const COMPLETION_FAILED_MESSAGE = "Review draft completion failed.";
const VIEW_FAILED_MESSAGE = "Review draft view is unavailable.";
const STORAGE_FAILED_MESSAGE = "Review draft storage is unavailable.";
const SERVICE_FAILED_MESSAGE = "Review draft service is unavailable.";
const REQUEST_FAILED_MESSAGE = "Review draft patch request is invalid.";
const COMPLETION_REQUEST_FAILED_MESSAGE =
  "Review draft completion request is invalid.";
const TEXT_ROLES = [
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
const CONTENT_ROLES = [...TEXT_ROLES, "table", "image", "formula"] as const;
const DISPOSITIONS = ["pending", "resolved", "accepted_limitation"] as const;

export class CandidateReviewDraftStaleError extends Error {
  constructor() {
    super(STALE_MESSAGE);
    this.name = "CandidateReviewDraftStaleError";
  }
}

export function isCandidateReviewDraftStaleError(
  value: unknown,
): value is CandidateReviewDraftStaleError {
  return value instanceof CandidateReviewDraftStaleError;
}

export type CandidateReviewTextRole =
  | "title"
  | "instruction"
  | "passage"
  | "question"
  | "choice"
  | "caption"
  | "header"
  | "footer"
  | "other";

export type CandidateReviewContentRole =
  | CandidateReviewTextRole
  | "table"
  | "image"
  | "formula";

export type CandidateReviewDraftPatchOperation =
  | {
      readonly op: "set_text";
      readonly nodeId: string;
      readonly text: string;
    }
  | {
      readonly op: "set_role";
      readonly nodeId: string;
      readonly role: CandidateReviewTextRole;
    }
  | {
      readonly op: "set_table_cell_text";
      readonly nodeId: string;
      readonly row: number;
      readonly column: number;
      readonly rowSpan: number;
      readonly columnSpan: number;
      readonly text: string;
    }
  | {
      readonly op: "set_needs_review";
      readonly nodeId: string;
      readonly needsReview: boolean;
    }
  | {
      readonly op: "set_image_grounding";
      readonly nodeId: string;
      readonly assetRef: string;
      readonly observationRef: string;
    }
  | {
      readonly op: "drop_image_node";
      readonly nodeId: string;
    }
  | {
      readonly op: "set_issue_disposition";
      readonly issueCode: string;
      readonly disposition: "pending" | "resolved" | "accepted_limitation";
      readonly note: string;
    };

export interface CandidateReviewDraftPatchRequest {
  readonly expectedDraftRevision: number;
  readonly operations: readonly CandidateReviewDraftPatchOperation[];
}

export interface CandidateReviewDraftCompletionRequest {
  readonly expectedDraftRevision: number;
}

export interface CandidateReviewDraftTextNodeView {
  readonly id: string;
  readonly kind: "text";
  readonly text: string;
  readonly role: CandidateReviewContentRole;
  readonly needsReview: boolean;
}

export interface CandidateReviewDraftTableCellView {
  readonly row: number;
  readonly column: number;
  readonly rowSpan: number;
  readonly columnSpan: number;
  readonly text: string;
}

export interface CandidateReviewDraftTableNodeView {
  readonly id: string;
  readonly kind: "table";
  readonly needsReview: boolean;
  readonly cells: readonly CandidateReviewDraftTableCellView[];
}

export interface CandidateReviewDraftImageNodeView {
  readonly id: string;
  readonly kind: "image";
  readonly needsReview: boolean;
}

export interface CandidateReviewDraftFormulaNodeView {
  readonly id: string;
  readonly kind: "formula";
  readonly needsReview: boolean;
}

export type CandidateReviewDraftNodeView =
  | CandidateReviewDraftTextNodeView
  | CandidateReviewDraftTableNodeView
  | CandidateReviewDraftImageNodeView
  | CandidateReviewDraftFormulaNodeView;

export interface CandidateReviewDraftIssueDispositionView {
  readonly issueCode: string;
  readonly disposition: "pending" | "resolved" | "accepted_limitation";
  readonly note: string;
}

export interface CandidateReviewDraftView {
  readonly schemaVersion: typeof VIEW_SCHEMA;
  readonly documentId: string;
  readonly status: "in_progress" | "complete";
  readonly draftRevision: number;
  readonly updatedAt: string;
  readonly rightsStatus: "unverified";
  readonly identityAssurance: "self_asserted_untrusted";
  readonly goldenEligible: false;
  readonly trainingEligible: false;
  readonly releaseEligible: false;
  readonly reviewedContentIrContractSha256: string;
  readonly nodes: readonly CandidateReviewDraftNodeView[];
  readonly issueDispositions: readonly CandidateReviewDraftIssueDispositionView[];
}

export interface CandidateReviewDraftBridgeCommand {
  readonly executable: string;
  readonly args: readonly string[];
  readonly cwd: string;
  readonly env: NodeJS.ProcessEnv;
  readonly shell: false;
  readonly windowsHide: true;
  readonly stdin: string | null;
}

export interface CandidateReviewDraftBridgeCommandResult {
  readonly code: number | null;
  readonly stdout: string;
  readonly stderr: string;
}

export type CandidateReviewDraftBridgeCommandRunner = (
  command: CandidateReviewDraftBridgeCommand,
) => Promise<CandidateReviewDraftBridgeCommandResult>;

export interface CandidateReviewDraftBridgeOptions {
  readonly projectRoot: string;
  readonly userDataRoot: string;
  readonly candidateRoot: string;
  readonly manifestSha256: string;
  readonly lineageId: string;
  readonly documentId: string;
  readonly commandRunner?: CandidateReviewDraftBridgeCommandRunner;
  readonly pythonRuntimeSelector?: typeof selectPythonRuntime;
  readonly patchScript?: string;
  readonly completionScript?: string;
  readonly viewScript?: string;
}

export interface ApplyCandidateReviewDraftPatchOptions
  extends CandidateReviewDraftBridgeOptions {
  readonly request: CandidateReviewDraftPatchRequest;
}

export interface CompleteCandidateReviewDraftOptions
  extends CandidateReviewDraftBridgeOptions {
  readonly request: CandidateReviewDraftCompletionRequest;
}

interface PreparedContext {
  readonly projectRoot: string;
  readonly candidateRoot: string;
  readonly draftDirectory: string;
  readonly documentId: string;
  readonly manifestSha256: string;
  readonly lineageId: string;
  readonly patchScript: string | null;
  readonly completionScript: string | null;
  readonly viewScript: string;
  readonly commandRunner: CandidateReviewDraftBridgeCommandRunner;
  readonly pythonRuntimeSelector: typeof selectPythonRuntime;
}

export interface StoredCandidateReviewDraft {
  readonly revision: number;
  readonly path: string;
}

interface NormalizedPatchRequest {
  readonly expectedDraftRevision: number;
  readonly stdin: string;
}

interface NormalizedCompletionRequest {
  readonly expectedDraftRevision: number;
  readonly stdin: string;
}

const documentQueues = new Map<string, Promise<void>>();

export async function applyCandidateReviewDraftPatch(
  options: ApplyCandidateReviewDraftPatchOptions,
): Promise<CandidateReviewDraftView> {
  const request = normalizePatchRequest(options.request);
  const context = prepareContext(options, true);
  return await serializeCandidateReviewDraftDocument(
    context.draftDirectory,
    async () => {
      const current = latestStoredCandidateReviewDraft(context.draftDirectory);
      if (current.revision !== request.expectedDraftRevision) {
        throw new CandidateReviewDraftStaleError();
      }
      const nextRevision = current.revision + 1;
      if (nextRevision > MAX_DRAFT_REVISION) {
        throw new CandidateReviewDraftStaleError();
      }
      const destination = revisionPath(context.draftDirectory, nextRevision);
      if (existsSync(destination)) throw new CandidateReviewDraftStaleError();
      const executable = await selectRuntime(context);
      const patchScript = context.patchScript;
      if (!patchScript) throw new Error(SERVICE_FAILED_MESSAGE);
      let patchStdout: string;
      try {
        patchStdout = await executeCommand(
          context,
          {
            executable,
            args: [
              patchScript,
              context.candidateRoot,
              current.path,
              "--expected-manifest-sha256",
              context.manifestSha256,
              "--expected-lineage-id",
              context.lineageId,
              "--out",
              destination,
            ],
            cwd: context.projectRoot,
            env: pythonRuntimeEnvironment(context.projectRoot),
            shell: false,
            windowsHide: true,
            stdin: request.stdin,
          },
          PATCH_FAILED_MESSAGE,
        );
      } catch (error) {
        if (existsSync(destination)) throw new CandidateReviewDraftStaleError();
        throw error;
      }
      parsePatchSummary(
        patchStdout,
        context.documentId,
        context.manifestSha256,
        context.lineageId,
        nextRevision,
      );
      requiredRegularFile(destination, STORAGE_FAILED_MESSAGE);
      const view = await executeView(
        context,
        executable,
        destination,
        nextRevision,
      );
      if (view.status !== "in_progress") throw new Error(VIEW_FAILED_MESSAGE);
      return view;
    },
  );
}

export async function completeCandidateReviewDraft(
  options: CompleteCandidateReviewDraftOptions,
): Promise<CandidateReviewDraftView> {
  const request = normalizeCompletionRequest(options.request);
  const context = prepareContext(options, false, true);
  return await serializeCandidateReviewDraftDocument(
    context.draftDirectory,
    async () => {
      const current = latestStoredCandidateReviewDraft(context.draftDirectory);
      if (current.revision !== request.expectedDraftRevision) {
        throw new CandidateReviewDraftStaleError();
      }
      const nextRevision = current.revision + 1;
      if (nextRevision > MAX_DRAFT_REVISION) {
        throw new CandidateReviewDraftStaleError();
      }
      const destination = revisionPath(context.draftDirectory, nextRevision);
      if (existsSync(destination)) throw new CandidateReviewDraftStaleError();
      const executable = await selectRuntime(context);
      const completionScript = context.completionScript;
      if (!completionScript) throw new Error(SERVICE_FAILED_MESSAGE);
      let completionStdout: string;
      try {
        completionStdout = await executeCommand(
          context,
          {
            executable,
            args: [
              completionScript,
              context.candidateRoot,
              current.path,
              "--expected-manifest-sha256",
              context.manifestSha256,
              "--expected-lineage-id",
              context.lineageId,
              "--out",
              destination,
            ],
            cwd: context.projectRoot,
            env: pythonRuntimeEnvironment(context.projectRoot),
            shell: false,
            windowsHide: true,
            stdin: request.stdin,
          },
          COMPLETION_FAILED_MESSAGE,
        );
      } catch (error) {
        if (existsSync(destination)) throw new CandidateReviewDraftStaleError();
        throw error;
      }
      parseCompletionSummary(
        completionStdout,
        context.documentId,
        context.manifestSha256,
        context.lineageId,
        nextRevision,
      );
      requiredRegularFile(destination, STORAGE_FAILED_MESSAGE);
      const view = await executeView(
        context,
        executable,
        destination,
        nextRevision,
      );
      if (view.status !== "complete") throw new Error(COMPLETION_FAILED_MESSAGE);
      return view;
    },
  );
}

export async function loadCandidateReviewDraftView(
  options: CandidateReviewDraftBridgeOptions,
): Promise<CandidateReviewDraftView> {
  const context = prepareContext(options, false);
  return await serializeCandidateReviewDraftDocument(
    context.draftDirectory,
    async () => {
      const current = latestStoredCandidateReviewDraft(context.draftDirectory);
      const executable = await selectRuntime(context);
      return await executeView(
        context,
        executable,
        current.path,
        current.revision,
      );
    },
  );
}

async function executeView(
  context: PreparedContext,
  executable: string,
  draftPath: string,
  expectedRevision: number,
): Promise<CandidateReviewDraftView> {
  const stdout = await executeCommand(
    context,
    {
      executable,
      args: [
        context.viewScript,
        context.candidateRoot,
        draftPath,
        "--expected-manifest-sha256",
        context.manifestSha256,
        "--expected-lineage-id",
        context.lineageId,
      ],
      cwd: context.projectRoot,
      env: pythonRuntimeEnvironment(context.projectRoot),
      shell: false,
      windowsHide: true,
      stdin: null,
    },
    VIEW_FAILED_MESSAGE,
  );
  return parseView(stdout, context.documentId, expectedRevision);
}

async function selectRuntime(context: PreparedContext): Promise<string> {
  try {
    return await context.pythonRuntimeSelector({
      projectRoot: context.projectRoot,
      requiredImports: ["scan2hwpx", "pydantic"],
    });
  } catch {
    throw new Error(SERVICE_FAILED_MESSAGE);
  }
}

async function executeCommand(
  context: PreparedContext,
  command: CandidateReviewDraftBridgeCommand,
  failureMessage: string,
): Promise<string> {
  let result: CandidateReviewDraftBridgeCommandResult;
  try {
    result = await context.commandRunner(command);
  } catch {
    throw new Error(failureMessage);
  }
  if (result.code !== 0) throw new Error(failureMessage);
  return result.stdout;
}

export async function serializeCandidateReviewDraftDocument<T>(
  draftDirectory: string,
  task: () => Promise<T>,
): Promise<T> {
  const key = comparablePath(draftDirectory);
  const previous = documentQueues.get(key) ?? Promise.resolve();
  const result = previous.catch(() => undefined).then(task);
  const tail = result.then(
    () => undefined,
    () => undefined,
  );
  documentQueues.set(key, tail);
  try {
    return await result;
  } finally {
    if (documentQueues.get(key) === tail) documentQueues.delete(key);
  }
}

function prepareContext(
  options: CandidateReviewDraftBridgeOptions,
  requirePatchScript: boolean,
  requireCompletionScript = false,
): PreparedContext {
  const projectRoot = requiredDirectory(options.projectRoot, SERVICE_FAILED_MESSAGE);
  const candidateRoot = requiredDirectory(options.candidateRoot, STORAGE_FAILED_MESSAGE);
  const userDataRoot = requiredDirectory(options.userDataRoot, STORAGE_FAILED_MESSAGE);
  const manifestSha256 = requiredSha256(options.manifestSha256, REQUEST_FAILED_MESSAGE);
  const lineageId = requiredSha256(options.lineageId, REQUEST_FAILED_MESSAGE);
  const documentId = requiredIdentifier(options.documentId, REQUEST_FAILED_MESSAGE);
  const requestedDraftDirectory = join(
    userDataRoot,
    "candidate-reviews",
    manifestSha256,
    lineageId,
  );
  requireDirectoryTree(
    userDataRoot,
    requestedDraftDirectory,
    STORAGE_FAILED_MESSAGE,
  );
  const draftDirectory = requiredDirectory(
    requestedDraftDirectory,
    STORAGE_FAILED_MESSAGE,
  );
  if (
    !isInsideOrEqual(userDataRoot, draftDirectory) ||
    isInsideOrEqual(candidateRoot, draftDirectory)
  ) {
    throw new Error(STORAGE_FAILED_MESSAGE);
  }
  const defaultPatchScript = join(
    projectRoot,
    "ai",
    "datasets",
    "patch_candidate_review_draft.py",
  );
  const defaultViewScript = join(
    projectRoot,
    "ai",
    "datasets",
    "view_candidate_review_draft.py",
  );
  const defaultCompletionScript = join(
    projectRoot,
    "ai",
    "datasets",
    "complete_candidate_review_draft.py",
  );
  return {
    projectRoot,
    candidateRoot,
    draftDirectory,
    documentId,
    manifestSha256,
    lineageId,
    patchScript: requirePatchScript
      ? requiredRegularFile(
          options.patchScript ?? defaultPatchScript,
          SERVICE_FAILED_MESSAGE,
        )
      : null,
    completionScript: requireCompletionScript
      ? requiredRegularFile(
          options.completionScript ?? defaultCompletionScript,
          SERVICE_FAILED_MESSAGE,
        )
      : null,
    viewScript: requiredRegularFile(
      options.viewScript ?? defaultViewScript,
      SERVICE_FAILED_MESSAGE,
    ),
    commandRunner: options.commandRunner ?? runBridgeCommand,
    pythonRuntimeSelector: options.pythonRuntimeSelector ?? selectPythonRuntime,
  };
}

export function latestStoredCandidateReviewDraft(
  draftDirectory: string,
): StoredCandidateReviewDraft {
  const revisions = new Map<number, string>();
  revisions.set(
    1,
    requiredRegularFile(
      join(draftDirectory, "draft.json"),
      STORAGE_FAILED_MESSAGE,
    ),
  );
  let entries;
  try {
    entries = readdirSync(draftDirectory, { withFileTypes: true });
  } catch {
    throw new Error(STORAGE_FAILED_MESSAGE);
  }
  for (const entry of entries) {
    const match = REVISION_FILE.exec(entry.name);
    if (!match) continue;
    const revision = Number(match[1]);
    if (
      !Number.isSafeInteger(revision) ||
      revision < 2 ||
      revision > MAX_DRAFT_REVISION ||
      entry.isSymbolicLink() ||
      !entry.isFile()
    ) {
      throw new Error(STORAGE_FAILED_MESSAGE);
    }
    revisions.set(
      revision,
      requiredRegularFile(
        join(draftDirectory, entry.name),
        STORAGE_FAILED_MESSAGE,
      ),
    );
  }
  const latestRevision = Math.max(...revisions.keys());
  for (let revision = 1; revision <= latestRevision; revision += 1) {
    if (!revisions.has(revision)) throw new Error(STORAGE_FAILED_MESSAGE);
  }
  return {
    revision: latestRevision,
    path: revisions.get(latestRevision) as string,
  };
}

function revisionPath(draftDirectory: string, revision: number): string {
  return revision === 1
    ? join(draftDirectory, "draft.json")
    : join(draftDirectory, `draft.r${revision}.json`);
}

async function runBridgeCommand(
  command: CandidateReviewDraftBridgeCommand,
): Promise<CandidateReviewDraftBridgeCommandResult> {
  return await new Promise((resolvePromise, rejectPromise) => {
    let child;
    try {
      child = spawn(command.executable, [...command.args], {
        cwd: command.cwd,
        env: command.env,
        shell: command.shell,
        windowsHide: command.windowsHide,
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch {
      rejectPromise(new Error(SERVICE_FAILED_MESSAGE));
      return;
    }
    let stdout = "";
    let stderr = "";
    let failure: Error | null = null;
    const timer = setTimeout(() => {
      failure = new Error(SERVICE_FAILED_MESSAGE);
      child.kill();
    }, PROCESS_TIMEOUT_MS);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      if (Buffer.byteLength(stdout, "utf8") > MAX_STDOUT_BYTES && !failure) {
        failure = new Error(SERVICE_FAILED_MESSAGE);
        child.kill();
      }
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      if (Buffer.byteLength(stderr, "utf8") > MAX_STDERR_BYTES) {
        stderr = Buffer.from(stderr, "utf8")
          .subarray(-MAX_STDERR_BYTES)
          .toString("utf8");
      }
    });
    child.stdin.once("error", () => {
      if (!failure) failure = new Error(SERVICE_FAILED_MESSAGE);
    });
    child.once("error", () => {
      if (!failure) failure = new Error(SERVICE_FAILED_MESSAGE);
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (failure) rejectPromise(failure);
      else resolvePromise({ code, stdout, stderr });
    });
    child.stdin.end(command.stdin ?? "", "utf8");
  });
}

function requiredDirectory(path: unknown, message: string): string {
  if (typeof path !== "string" || !isAbsolute(path)) throw new Error(message);
  const absolute = resolve(path);
  try {
    const stats = lstatSync(absolute);
    if (stats.isSymbolicLink() || !stats.isDirectory()) throw new Error(message);
    return realpathSync.native(absolute);
  } catch {
    throw new Error(message);
  }
}

function requiredRegularFile(path: unknown, message: string): string {
  if (typeof path !== "string" || !isAbsolute(path)) throw new Error(message);
  const absolute = resolve(path);
  try {
    const stats = lstatSync(absolute);
    if (stats.isSymbolicLink() || !stats.isFile()) throw new Error(message);
    return realpathSync.native(absolute);
  } catch {
    throw new Error(message);
  }
}

function requireDirectoryTree(
  parent: string,
  child: string,
  message: string,
): void {
  if (!isInsideOrEqual(parent, child)) throw new Error(message);
  const offset = relative(parent, child);
  let current = parent;
  for (const segment of offset.split(sep).filter(Boolean)) {
    current = join(current, segment);
    try {
      const stats = lstatSync(current);
      if (stats.isSymbolicLink() || !stats.isDirectory()) {
        throw new Error(message);
      }
    } catch {
      throw new Error(message);
    }
  }
}

function requiredSha256(value: unknown, message: string): string {
  if (typeof value !== "string" || !SHA256.test(value)) throw new Error(message);
  return value;
}

function requiredIdentifier(value: unknown, message: string): string {
  if (
    typeof value !== "string" ||
    value.length > MAX_DOCUMENT_ID_CHARS ||
    !IDENTIFIER.test(value)
  ) {
    throw new Error(message);
  }
  return value;
}

function isInsideOrEqual(parent: string, child: string): boolean {
  const offset = relative(parent, child);
  return (
    offset === "" ||
    (offset !== ".." &&
      !offset.startsWith(`..${sep}`) &&
      !isAbsolute(offset))
  );
}

function comparablePath(path: string): string {
  const absolute = resolve(path);
  return process.platform === "win32" ? absolute.toLowerCase() : absolute;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function sameKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const allowed = [...expected].sort();
  return (
    actual.length === allowed.length &&
    actual.every((key, index) => key === allowed[index])
  );
}

function parseJsonObject(payload: string, message: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(payload);
  } catch {
    throw new Error(message);
  }
  if (!isRecord(value)) throw new Error(message);
  return value;
}

function requiredString(
  value: unknown,
  maxLength: number,
  message: string,
  allowEmpty = false,
): string {
  if (
    typeof value !== "string" ||
    value.length > maxLength ||
    (!allowEmpty && value.length === 0) ||
    hasUnpairedSurrogate(value)
  ) {
    throw new Error(message);
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

function requiredInteger(
  value: unknown,
  minimum: number,
  maximum: number,
  message: string,
): number {
  if (
    !Number.isSafeInteger(value) ||
    Number(value) < minimum ||
    Number(value) > maximum
  ) {
    throw new Error(message);
  }
  return Number(value);
}

function normalizePatchRequest(value: unknown): NormalizedPatchRequest {
  if (
    !isRecord(value) ||
    !sameKeys(value, ["expectedDraftRevision", "operations"])
  ) {
    throw new Error(REQUEST_FAILED_MESSAGE);
  }
  const expectedDraftRevision = requiredInteger(
    value.expectedDraftRevision,
    1,
    MAX_DRAFT_REVISION,
    REQUEST_FAILED_MESSAGE,
  );
  if (
    !Array.isArray(value.operations) ||
    value.operations.length < 1 ||
    value.operations.length > MAX_PATCH_OPERATIONS
  ) {
    throw new Error(REQUEST_FAILED_MESSAGE);
  }
  const operations = value.operations.map(normalizePatchOperation);
  const totalTextChars = operations.reduce(
    (total, operation) =>
      total +
      recordStringLength(operation, "node_id") +
      recordStringLength(operation, "asset_ref") +
      recordStringLength(operation, "observation_ref") +
      recordStringLength(operation, "issue_code") +
      recordStringLength(operation, "text") +
      recordStringLength(operation, "note"),
    0,
  );
  if (totalTextChars > MAX_TOTAL_VIEW_CHARS) {
    throw new Error(REQUEST_FAILED_MESSAGE);
  }
  const stdin = JSON.stringify({
    schema_version: PATCH_SCHEMA,
    expected_draft_revision: expectedDraftRevision,
    operations,
  });
  if (Buffer.byteLength(stdin, "utf8") > MAX_STDIN_BYTES) {
    throw new Error(REQUEST_FAILED_MESSAGE);
  }
  return { expectedDraftRevision, stdin };
}

function normalizeCompletionRequest(value: unknown): NormalizedCompletionRequest {
  if (
    !isRecord(value) ||
    !sameKeys(value, ["expectedDraftRevision"])
  ) {
    throw new Error(COMPLETION_REQUEST_FAILED_MESSAGE);
  }
  const expectedDraftRevision = requiredInteger(
    value.expectedDraftRevision,
    1,
    MAX_DRAFT_REVISION,
    COMPLETION_REQUEST_FAILED_MESSAGE,
  );
  const stdin = JSON.stringify({
    schema_version: COMPLETION_SCHEMA,
    expected_draft_revision: expectedDraftRevision,
  });
  if (Buffer.byteLength(stdin, "utf8") > MAX_STDIN_BYTES) {
    throw new Error(COMPLETION_REQUEST_FAILED_MESSAGE);
  }
  return { expectedDraftRevision, stdin };
}

function normalizePatchOperation(value: unknown): Record<string, unknown> {
  if (!isRecord(value) || typeof value.op !== "string") {
    throw new Error(REQUEST_FAILED_MESSAGE);
  }
  if (value.op === "set_text") {
    requireExactKeys(value, ["nodeId", "op", "text"], REQUEST_FAILED_MESSAGE);
    return {
      op: value.op,
      node_id: requiredString(
        value.nodeId,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
      text: requiredString(
        value.text,
        MAX_FIELD_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
    };
  }
  if (value.op === "set_role") {
    requireExactKeys(value, ["nodeId", "op", "role"], REQUEST_FAILED_MESSAGE);
    const role = requiredString(value.role, 32, REQUEST_FAILED_MESSAGE);
    if (!TEXT_ROLES.includes(role as CandidateReviewTextRole)) {
      throw new Error(REQUEST_FAILED_MESSAGE);
    }
    return {
      op: value.op,
      node_id: requiredString(
        value.nodeId,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
      role,
    };
  }
  if (value.op === "set_table_cell_text") {
    requireExactKeys(
      value,
      ["column", "columnSpan", "nodeId", "op", "row", "rowSpan", "text"],
      REQUEST_FAILED_MESSAGE,
    );
    return {
      op: value.op,
      node_id: requiredString(
        value.nodeId,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
      row: requiredInteger(
        value.row,
        0,
        Number.MAX_SAFE_INTEGER,
        REQUEST_FAILED_MESSAGE,
      ),
      column: requiredInteger(
        value.column,
        0,
        Number.MAX_SAFE_INTEGER,
        REQUEST_FAILED_MESSAGE,
      ),
      row_span: requiredInteger(
        value.rowSpan,
        1,
        Number.MAX_SAFE_INTEGER,
        REQUEST_FAILED_MESSAGE,
      ),
      column_span: requiredInteger(
        value.columnSpan,
        1,
        Number.MAX_SAFE_INTEGER,
        REQUEST_FAILED_MESSAGE,
      ),
      text: requiredString(
        value.text,
        MAX_FIELD_CHARS,
        REQUEST_FAILED_MESSAGE,
        true,
      ),
    };
  }
  if (value.op === "set_needs_review") {
    requireExactKeys(
      value,
      ["needsReview", "nodeId", "op"],
      REQUEST_FAILED_MESSAGE,
    );
    if (typeof value.needsReview !== "boolean") {
      throw new Error(REQUEST_FAILED_MESSAGE);
    }
    return {
      op: value.op,
      node_id: requiredString(
        value.nodeId,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
      needs_review: value.needsReview,
    };
  }
  if (value.op === "set_image_grounding") {
    requireExactKeys(
      value,
      ["assetRef", "nodeId", "observationRef", "op"],
      REQUEST_FAILED_MESSAGE,
    );
    return {
      op: value.op,
      node_id: requiredString(
        value.nodeId,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
      asset_ref: requiredString(
        value.assetRef,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
      observation_ref: requiredString(
        value.observationRef,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
    };
  }
  if (value.op === "drop_image_node") {
    requireExactKeys(value, ["nodeId", "op"], REQUEST_FAILED_MESSAGE);
    return {
      op: value.op,
      node_id: requiredString(
        value.nodeId,
        MAX_NODE_ID_CHARS,
        REQUEST_FAILED_MESSAGE,
      ),
    };
  }
  if (value.op === "set_issue_disposition") {
    requireExactKeys(
      value,
      ["disposition", "issueCode", "note", "op"],
      REQUEST_FAILED_MESSAGE,
    );
    const issueCode = requiredString(
      value.issueCode,
      MAX_ISSUE_CODE_CHARS,
      REQUEST_FAILED_MESSAGE,
    );
    if (!ISSUE_CODE.test(issueCode)) throw new Error(REQUEST_FAILED_MESSAGE);
    const disposition = requiredString(
      value.disposition,
      32,
      REQUEST_FAILED_MESSAGE,
    );
    if (!DISPOSITIONS.includes(disposition as (typeof DISPOSITIONS)[number])) {
      throw new Error(REQUEST_FAILED_MESSAGE);
    }
    const note = requiredString(
      value.note,
      MAX_FIELD_CHARS,
      REQUEST_FAILED_MESSAGE,
      true,
    );
    if (disposition === "accepted_limitation" && !note.trim()) {
      throw new Error(REQUEST_FAILED_MESSAGE);
    }
    return {
      op: value.op,
      issue_code: issueCode,
      disposition,
      note,
    };
  }
  throw new Error(REQUEST_FAILED_MESSAGE);
}

function recordStringLength(
  value: Record<string, unknown>,
  key: string,
): number {
  return typeof value[key] === "string" ? value[key].length : 0;
}

function requireExactKeys(
  value: Record<string, unknown>,
  keys: readonly string[],
  message: string,
): void {
  if (!sameKeys(value, keys)) throw new Error(message);
}

function parsePatchSummary(
  stdout: string,
  expectedDocumentId: string,
  expectedManifestSha256: string,
  expectedLineageId: string,
  expectedRevision: number,
): void {
  const value = parseJsonObject(stdout, PATCH_FAILED_MESSAGE);
  if (
    !sameKeys(value, [
      "candidate_manifest_sha256",
      "document_id",
      "draft_revision",
      "golden_eligible",
      "identity_assurance",
      "lineage_id",
      "release_eligible",
      "rights_status",
      "schema_version",
      "status",
      "training_eligible",
    ]) ||
    value.schema_version !== DRAFT_SCHEMA ||
    value.candidate_manifest_sha256 !== expectedManifestSha256 ||
    value.document_id !== expectedDocumentId ||
    value.lineage_id !== expectedLineageId ||
    value.status !== "in_progress" ||
    value.draft_revision !== expectedRevision ||
    value.rights_status !== "unverified" ||
    value.identity_assurance !== "self_asserted_untrusted" ||
    value.golden_eligible !== false ||
    value.training_eligible !== false ||
    value.release_eligible !== false
  ) {
    throw new Error(PATCH_FAILED_MESSAGE);
  }
}

function parseCompletionSummary(
  stdout: string,
  expectedDocumentId: string,
  expectedManifestSha256: string,
  expectedLineageId: string,
  expectedRevision: number,
): void {
  const value = parseJsonObject(stdout, COMPLETION_FAILED_MESSAGE);
  if (
    !sameKeys(value, [
      "candidate_manifest_sha256",
      "document_id",
      "draft_revision",
      "golden_eligible",
      "identity_assurance",
      "lineage_id",
      "release_eligible",
      "rights_status",
      "schema_version",
      "status",
      "training_eligible",
    ]) ||
    value.schema_version !== DRAFT_SCHEMA ||
    value.candidate_manifest_sha256 !== expectedManifestSha256 ||
    value.document_id !== expectedDocumentId ||
    value.lineage_id !== expectedLineageId ||
    value.status !== "complete" ||
    value.draft_revision !== expectedRevision ||
    value.rights_status !== "unverified" ||
    value.identity_assurance !== "self_asserted_untrusted" ||
    value.golden_eligible !== false ||
    value.training_eligible !== false ||
    value.release_eligible !== false
  ) {
    throw new Error(COMPLETION_FAILED_MESSAGE);
  }
}

function parseView(
  stdout: string,
  expectedDocumentId: string,
  expectedRevision: number,
): CandidateReviewDraftView {
  const value = parseJsonObject(stdout, VIEW_FAILED_MESSAGE);
  if (
    !sameKeys(value, [
      "document_id",
      "draft_revision",
      "golden_eligible",
      "identity_assurance",
      "issue_dispositions",
      "nodes",
      "release_eligible",
      "reviewed_content_ir_contract_sha256",
      "rights_status",
      "schema_version",
      "status",
      "training_eligible",
      "updated_at",
    ]) ||
    value.schema_version !== VIEW_SCHEMA ||
    value.document_id !== expectedDocumentId ||
    value.draft_revision !== expectedRevision ||
    (value.status !== "in_progress" && value.status !== "complete") ||
    value.rights_status !== "unverified" ||
    value.identity_assurance !== "self_asserted_untrusted" ||
    value.golden_eligible !== false ||
    value.training_eligible !== false ||
    value.release_eligible !== false ||
    typeof value.reviewed_content_ir_contract_sha256 !== "string" ||
    !SHA256.test(value.reviewed_content_ir_contract_sha256)
  ) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  const updatedAt = requiredTimestamp(value.updated_at);
  if (
    !Array.isArray(value.nodes) ||
    value.nodes.length < 1 ||
    value.nodes.length > MAX_VIEW_NODES ||
    !Array.isArray(value.issue_dispositions) ||
    value.issue_dispositions.length > MAX_VIEW_ISSUES
  ) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  const nodes = value.nodes.map(parseViewNode);
  rejectDuplicates(
    nodes.map((node) => node.id),
    VIEW_FAILED_MESSAGE,
  );
  const issueDispositions = value.issue_dispositions.map(parseIssueDisposition);
  rejectDuplicates(
    issueDispositions.map((item) => item.issueCode),
    VIEW_FAILED_MESSAGE,
  );
  const cellCount = nodes.reduce(
    (count, node) => count + (node.kind === "table" ? node.cells.length : 0),
    0,
  );
  if (cellCount > MAX_VIEW_CELLS) throw new Error(VIEW_FAILED_MESSAGE);
  const textChars =
    nodes.reduce((count, node) => {
      if (node.kind === "text") return count + node.text.length;
      if (node.kind === "table") {
        return count + node.cells.reduce((total, cell) => total + cell.text.length, 0);
      }
      return count;
    }, 0) +
    issueDispositions.reduce((count, item) => count + item.note.length, 0);
  if (textChars > MAX_TOTAL_VIEW_CHARS) throw new Error(VIEW_FAILED_MESSAGE);
  if (
    value.status === "complete" &&
    (nodes.some((node) => node.needsReview) ||
      issueDispositions.some((item) => item.disposition === "pending"))
  ) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  return {
    schemaVersion: VIEW_SCHEMA,
    documentId: expectedDocumentId,
    status: value.status,
    draftRevision: expectedRevision,
    updatedAt,
    rightsStatus: "unverified",
    identityAssurance: "self_asserted_untrusted",
    goldenEligible: false,
    trainingEligible: false,
    releaseEligible: false,
    reviewedContentIrContractSha256: value.reviewed_content_ir_contract_sha256,
    nodes,
    issueDispositions,
  };
}

function parseViewNode(value: unknown): CandidateReviewDraftNodeView {
  if (!isRecord(value) || typeof value.kind !== "string") {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  const id = requiredString(
    value.id,
    MAX_NODE_ID_CHARS,
    VIEW_FAILED_MESSAGE,
  );
  if (typeof value.needs_review !== "boolean") {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  if (value.kind === "text") {
    requireExactKeys(
      value,
      ["id", "kind", "needs_review", "role", "text"],
      VIEW_FAILED_MESSAGE,
    );
    const role = requiredString(value.role, 32, VIEW_FAILED_MESSAGE);
    if (!CONTENT_ROLES.includes(role as CandidateReviewContentRole)) {
      throw new Error(VIEW_FAILED_MESSAGE);
    }
    return {
      id,
      kind: "text",
      text: requiredString(value.text, MAX_FIELD_CHARS, VIEW_FAILED_MESSAGE),
      role: role as CandidateReviewContentRole,
      needsReview: value.needs_review,
    };
  }
  if (value.kind === "table") {
    requireExactKeys(
      value,
      ["cells", "id", "kind", "needs_review"],
      VIEW_FAILED_MESSAGE,
    );
    if (
      !Array.isArray(value.cells) ||
      value.cells.length < 1 ||
      value.cells.length > MAX_VIEW_CELLS
    ) {
      throw new Error(VIEW_FAILED_MESSAGE);
    }
    const cells = value.cells.map(parseViewCell);
    rejectDuplicates(
      cells.map((cell) =>
        [cell.row, cell.column, cell.rowSpan, cell.columnSpan].join(":"),
      ),
      VIEW_FAILED_MESSAGE,
    );
    return { id, kind: "table", needsReview: value.needs_review, cells };
  }
  if (value.kind === "image" || value.kind === "formula") {
    requireExactKeys(
      value,
      ["id", "kind", "needs_review"],
      VIEW_FAILED_MESSAGE,
    );
    return { id, kind: value.kind, needsReview: value.needs_review };
  }
  throw new Error(VIEW_FAILED_MESSAGE);
}

function parseViewCell(value: unknown): CandidateReviewDraftTableCellView {
  if (
    !isRecord(value) ||
    !sameKeys(value, [
      "column",
      "column_span",
      "row",
      "row_span",
      "text",
    ])
  ) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  return {
    row: requiredInteger(
      value.row,
      0,
      Number.MAX_SAFE_INTEGER,
      VIEW_FAILED_MESSAGE,
    ),
    column: requiredInteger(
      value.column,
      0,
      Number.MAX_SAFE_INTEGER,
      VIEW_FAILED_MESSAGE,
    ),
    rowSpan: requiredInteger(
      value.row_span,
      1,
      Number.MAX_SAFE_INTEGER,
      VIEW_FAILED_MESSAGE,
    ),
    columnSpan: requiredInteger(
      value.column_span,
      1,
      Number.MAX_SAFE_INTEGER,
      VIEW_FAILED_MESSAGE,
    ),
    text: requiredString(
      value.text,
      MAX_FIELD_CHARS,
      VIEW_FAILED_MESSAGE,
      true,
    ),
  };
}

function parseIssueDisposition(
  value: unknown,
): CandidateReviewDraftIssueDispositionView {
  if (
    !isRecord(value) ||
    !sameKeys(value, ["disposition", "issue_code", "note"])
  ) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  const issueCode = requiredString(
    value.issue_code,
    MAX_ISSUE_CODE_CHARS,
    VIEW_FAILED_MESSAGE,
  );
  if (!ISSUE_CODE.test(issueCode)) throw new Error(VIEW_FAILED_MESSAGE);
  const disposition = requiredString(
    value.disposition,
    32,
    VIEW_FAILED_MESSAGE,
  );
  if (!DISPOSITIONS.includes(disposition as (typeof DISPOSITIONS)[number])) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  const note = requiredString(
    value.note,
    MAX_FIELD_CHARS,
    VIEW_FAILED_MESSAGE,
    true,
  );
  if (disposition === "accepted_limitation" && !note.trim()) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  return {
    issueCode,
    disposition: disposition as CandidateReviewDraftIssueDispositionView["disposition"],
    note,
  };
}

function requiredTimestamp(value: unknown): string {
  const timestamp = requiredString(value, 64, VIEW_FAILED_MESSAGE);
  if (
    !/(?:Z|[+-][0-9]{2}:[0-9]{2})$/.test(timestamp) ||
    !Number.isFinite(Date.parse(timestamp))
  ) {
    throw new Error(VIEW_FAILED_MESSAGE);
  }
  return timestamp;
}

function rejectDuplicates(values: readonly string[], message: string): void {
  if (new Set(values).size !== values.length) throw new Error(message);
}
