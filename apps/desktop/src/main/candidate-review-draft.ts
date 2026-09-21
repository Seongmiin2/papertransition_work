import { spawn } from "node:child_process";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  realpathSync,
} from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  latestStoredCandidateReviewDraft,
  serializeCandidateReviewDraftDocument,
} from "./candidate-review-draft-patch.js";
import {
  pythonRuntimeEnvironment,
  selectPythonRuntime,
} from "./python-runtime.js";

const DRAFT_SCHEMA = "candidate-review-draft/1.0";
const SHA256 = /^[0-9a-f]{64}$/;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const MAX_REVIEWER_LABEL_LENGTH = 64;
const MAX_STDOUT_BYTES = 64 * 1024;
const MAX_STDERR_BYTES = 16 * 1024;
const PROCESS_TIMEOUT_MS = 120_000;

export interface CandidateReviewDraftStatus {
  operation: "created" | "reopened";
  schemaVersion: typeof DRAFT_SCHEMA;
  documentId: string;
  status: "in_progress" | "complete";
  draftRevision: number;
  rightsStatus: "unverified";
  identityAssurance: "self_asserted_untrusted";
  goldenEligible: false;
  trainingEligible: false;
  releaseEligible: false;
}

export interface CandidateReviewDraftCommand {
  executable: string;
  args: readonly string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  shell: false;
  windowsHide: true;
}

export interface CandidateReviewDraftCommandResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

export type CandidateReviewDraftCommandRunner = (
  command: CandidateReviewDraftCommand,
) => Promise<CandidateReviewDraftCommandResult>;

export interface StartOrReopenCandidateReviewDraftOptions {
  projectRoot: string;
  userDataRoot: string;
  candidateRoot: string;
  manifestSha256: string;
  lineageId: string;
  documentId: string;
  reviewerLabel: string;
  commandRunner?: CandidateReviewDraftCommandRunner;
  pythonRuntimeSelector?: typeof selectPythonRuntime;
  startScript?: string;
  verifyScript?: string;
}

export function validateReviewerLabel(value: unknown): string {
  if (
    typeof value !== "string" ||
    value.length > MAX_REVIEWER_LABEL_LENGTH ||
    !IDENTIFIER.test(value)
  ) {
    throw new TypeError(
      "검수자 라벨은 영문 또는 숫자로 시작하고 영문, 숫자, 점, 밑줄, 하이픈만 포함한 1~64자여야 합니다.",
    );
  }
  return value;
}

export async function startOrReopenCandidateReviewDraft(
  options: StartOrReopenCandidateReviewDraftOptions,
): Promise<CandidateReviewDraftStatus> {
  const manifestSha256 = requiredSha256(options.manifestSha256, "candidate manifest");
  const lineageId = requiredSha256(options.lineageId, "candidate lineage");
  const documentId = requiredIdentifier(options.documentId, "candidate document id");
  const reviewerLabel = validateReviewerLabel(options.reviewerLabel);
  const projectRoot = requiredDirectory(options.projectRoot, "project root");
  const candidateRoot = requiredDirectory(options.candidateRoot, "candidate root");
  const userDataRoot = ensureUserDataRoot(options.userDataRoot);
  const destination = prepareDraftDestination(userDataRoot, manifestSha256, lineageId);
  if (isWithin(candidateRoot, destination)) {
    throw new Error("검수 초안 저장 위치는 후보 원본 밖이어야 합니다.");
  }
  return await serializeCandidateReviewDraftDocument(
    dirname(destination),
    async () => {
      const existing = existingRegularFile(destination, "review draft");
      const operation = existing ? "reopened" : "created";
      const stored = existing
        ? latestStoredCandidateReviewDraft(dirname(destination))
        : { revision: 1, path: destination };
      const defaultStartScript = join(
        projectRoot,
        "ai",
        "datasets",
        "start_candidate_review_draft.py",
      );
      const defaultVerifyScript = join(
        projectRoot,
        "ai",
        "datasets",
        "verify_candidate_review_draft.py",
      );
      const script = requiredRegularFile(
        operation === "created"
          ? options.startScript ?? defaultStartScript
          : options.verifyScript ?? defaultVerifyScript,
        operation === "created"
          ? "review draft start CLI"
          : "review draft verify CLI",
      );
      const executable = await (
        options.pythonRuntimeSelector ?? selectPythonRuntime
      )({
        projectRoot,
        requiredImports: ["scan2hwpx", "pydantic"],
      });
      const args =
        operation === "created"
          ? [
              script,
              candidateRoot,
              documentId,
              "--reviewer-label",
              reviewerLabel,
              "--expected-manifest-sha256",
              manifestSha256,
              "--expected-lineage-id",
              lineageId,
              "--out",
              stored.path,
            ]
          : [
              script,
              candidateRoot,
              stored.path,
              "--expected-manifest-sha256",
              manifestSha256,
              "--expected-lineage-id",
              lineageId,
            ];
      const command: CandidateReviewDraftCommand = {
        executable,
        args,
        cwd: projectRoot,
        env: pythonRuntimeEnvironment(projectRoot),
        shell: false,
        windowsHide: true,
      };

      const result = await (options.commandRunner ?? runDraftCommand)(command);
      if (result.code !== 0) {
        throw new Error("검수 초안 검증 프로세스가 실패했습니다.");
      }
      const summary = parseSummary(
        result.stdout,
        operation,
        documentId,
        manifestSha256,
        lineageId,
        stored.path,
        stored.revision,
      );
      existingRegularFile(stored.path, "review draft", true);
      return summary;
    },
  );
}

async function runDraftCommand(
  command: CandidateReviewDraftCommand,
): Promise<CandidateReviewDraftCommandResult> {
  return await new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command.executable, [...command.args], {
      cwd: command.cwd,
      env: command.env,
      shell: command.shell,
      windowsHide: command.windowsHide,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let failure: Error | null = null;
    const timer = setTimeout(() => {
      failure = new Error("검수 초안 검증 시간이 초과되었습니다.");
      child.kill();
    }, PROCESS_TIMEOUT_MS);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
      if (Buffer.byteLength(stdout, "utf8") > MAX_STDOUT_BYTES && !failure) {
        failure = new Error("검수 초안 검증 응답이 허용 크기를 초과했습니다.");
        child.kill();
      }
    });
    child.stderr.on("data", (chunk: string) => {
      stderr = `${stderr}${chunk}`;
      if (Buffer.byteLength(stderr, "utf8") > MAX_STDERR_BYTES) {
        stderr = Buffer.from(stderr, "utf8").subarray(-MAX_STDERR_BYTES).toString("utf8");
      }
    });
    child.once("error", () => {
      if (!failure) failure = new Error("검수 초안 검증 프로세스를 시작하지 못했습니다.");
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (failure) rejectPromise(failure);
      else resolvePromise({ code, stdout, stderr });
    });
  });
}

function parseSummary(
  stdout: string,
  operation: "created" | "reopened",
  expectedDocumentId: string,
  expectedManifestSha256: string,
  expectedLineageId: string,
  destination: string,
  expectedRevision: number,
): CandidateReviewDraftStatus {
  let value: unknown;
  try {
    value = JSON.parse(stdout);
  } catch {
    throw new Error("검수 초안 검증 응답이 올바른 JSON이 아닙니다.");
  }
  if (!isRecord(value)) throw new Error("검수 초안 검증 응답 형식이 올바르지 않습니다.");
  const commonKeys = [
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
  ];
  const allowedKeys = operation === "created" ? [...commonKeys, "output"] : commonKeys;
  if (!sameKeys(value, allowedKeys)) {
    throw new Error("검수 초안 검증 응답에 허용되지 않은 필드가 있습니다.");
  }
  if (
    value.schema_version !== DRAFT_SCHEMA ||
    value.candidate_manifest_sha256 !== expectedManifestSha256 ||
    value.document_id !== expectedDocumentId ||
    value.lineage_id !== expectedLineageId ||
    (value.status !== "in_progress" && value.status !== "complete") ||
    !Number.isSafeInteger(value.draft_revision) ||
    Number(value.draft_revision) < 1 ||
    value.draft_revision !== expectedRevision ||
    value.rights_status !== "unverified" ||
    value.identity_assurance !== "self_asserted_untrusted" ||
    value.golden_eligible !== false ||
    value.training_eligible !== false ||
    value.release_eligible !== false
  ) {
    throw new Error("검수 초안 검증 응답이 안전 정책과 일치하지 않습니다.");
  }
  if (
    operation === "created" &&
    (value.status !== "in_progress" ||
      value.draft_revision !== 1 ||
      typeof value.output !== "string" ||
      comparablePath(value.output) !== comparablePath(destination))
  ) {
    throw new Error("새 검수 초안 응답이 요청한 저장 대상과 일치하지 않습니다.");
  }
  return {
    operation,
    schemaVersion: DRAFT_SCHEMA,
    documentId: expectedDocumentId,
    status: value.status as "in_progress" | "complete",
    draftRevision: Number(value.draft_revision),
    rightsStatus: "unverified",
    identityAssurance: "self_asserted_untrusted",
    goldenEligible: false,
    trainingEligible: false,
    releaseEligible: false,
  };
}

function ensureUserDataRoot(path: string): string {
  if (typeof path !== "string" || !path || !isAbsolute(path)) {
    throw new TypeError("app userData path must be absolute");
  }
  try {
    mkdirSync(path, { recursive: true });
  } catch {
    throw new Error("app userData directory is unavailable");
  }
  return requiredDirectory(path, "app userData");
}

function prepareDraftDestination(
  userDataRoot: string,
  manifestSha256: string,
  lineageId: string,
): string {
  let parent = ensureChildDirectory(userDataRoot, "candidate-reviews");
  parent = ensureChildDirectory(parent, manifestSha256);
  parent = ensureChildDirectory(parent, lineageId);
  const destination = join(parent, "draft.json");
  if (!isWithin(userDataRoot, destination)) {
    throw new Error("검수 초안 저장 위치가 app userData 밖입니다.");
  }
  return destination;
}

function ensureChildDirectory(parent: string, name: string): string {
  const path = join(parent, name);
  try {
    if (!existsSync(path)) mkdirSync(path);
  } catch {
    throw new Error("review draft directory is unavailable");
  }
  return requiredDirectory(path, "review draft directory");
}

function requiredDirectory(path: string, label: string): string {
  if (typeof path !== "string" || !path || !isAbsolute(path)) {
    throw new TypeError(`${label} path is invalid`);
  }
  const absolute = resolve(path);
  let stats;
  try {
    stats = lstatSync(absolute);
  } catch {
    throw new Error(`${label} is unavailable`);
  }
  if (stats.isSymbolicLink() || !stats.isDirectory()) {
    throw new Error(`${label} must be a regular directory`);
  }
  try {
    return realpathSync.native(absolute);
  } catch {
    throw new Error(`${label} is unavailable`);
  }
}

function requiredRegularFile(path: string, label: string): string {
  if (typeof path !== "string" || !path || !isAbsolute(path)) {
    throw new TypeError(`${label} path is invalid`);
  }
  const absolute = resolve(path);
  let stats;
  try {
    stats = lstatSync(absolute);
  } catch {
    throw new Error(`${label} is unavailable`);
  }
  if (stats.isSymbolicLink() || !stats.isFile()) {
    throw new Error(`${label} must be a regular file`);
  }
  try {
    return realpathSync.native(absolute);
  } catch {
    throw new Error(`${label} is unavailable`);
  }
}

function existingRegularFile(path: string, label: string, required = false): boolean {
  if (!existsSync(path)) {
    if (required) throw new Error(`${label} was not created`);
    return false;
  }
  let stats;
  try {
    stats = lstatSync(path);
  } catch {
    throw new Error(`${label} is unavailable`);
  }
  if (stats.isSymbolicLink() || !stats.isFile()) {
    throw new Error(`${label} must be a regular file`);
  }
  return true;
}

function requiredSha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !SHA256.test(value)) {
    throw new TypeError(`${label} must be a lowercase SHA-256`);
  }
  return value;
}

function requiredIdentifier(value: unknown, label: string): string {
  if (typeof value !== "string" || !IDENTIFIER.test(value)) {
    throw new TypeError(`${label} is invalid`);
  }
  return value;
}

function isWithin(parent: string, child: string): boolean {
  const offset = relative(parent, child);
  return offset !== "" && offset !== ".." && !offset.startsWith(`..${sep}`) && !isAbsolute(offset);
}

function comparablePath(path: string): string {
  const absolute = resolve(path);
  return process.platform === "win32" ? absolute.toLowerCase() : absolute;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function sameKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const allowed = [...expected].sort();
  return actual.length === allowed.length && actual.every((key, index) => key === allowed[index]);
}
