import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { createHash, randomUUID } from "node:crypto";
import {
  copyFileSync,
  createReadStream,
  existsSync,
  mkdirSync,
  rmSync,
} from "node:fs";
import { basename, join, resolve } from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { BrowserWindow as BrowserWindowType, IpcMainInvokeEvent } from "electron";
import {
  openCandidateReviewBundle,
  type CandidateReviewBundle,
} from "./candidate-review.js";
import {
  startOrReopenCandidateReviewDraft,
  validateReviewerLabel,
} from "./candidate-review-draft.js";
import {
  applyCandidateReviewDraftPatch,
  completeCandidateReviewDraft,
  isCandidateReviewDraftStaleError,
  loadCandidateReviewDraftView,
} from "./candidate-review-draft-patch.js";
import {
  parseCandidateReviewDraftCompletionRequest,
  parseCandidateReviewDraftPatchRequest,
} from "./candidate-review-ipc.js";
import {
  pythonRuntimeEnvironment,
  selectPythonRuntime,
} from "./python-runtime.js";
import { JobDatabase } from "./database.js";
import type { JobRecord } from "../types/contracts.js";

let window: BrowserWindowType | null = null;
let database: JobDatabase;
let candidateReviewBundle: CandidateReviewBundle | null = null;
let candidateReviewRoot: string | null = null;
const MAX_BATCH_FILES = 10;
const queue: string[] = [];
let worker: ChildProcessWithoutNullStreams | null = null;
let activeJobId: string | null = null;
let workerBuffer = "";
let workerStderr = "";
let quitting = false;

function isTerminal(status: JobRecord["status"]): boolean {
  return status === "COMPLETED" || status === "FAILED" || status === "CANCELLED";
}

function selectedCandidateBundle(): CandidateReviewBundle {
  if (!candidateReviewBundle) throw new Error("먼저 검수 후보 폴더를 선택하세요.");
  candidateReviewBundle.assertCurrentSnapshot();
  return candidateReviewBundle;
}

function ipcLineageId(value: unknown): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new TypeError("잘못된 후보 문서 식별자입니다.");
  }
  return value;
}

function ipcPageNo(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
    throw new TypeError("잘못된 페이지 번호입니다.");
  }
  return value;
}

function ipcDraftRequest(value: unknown): { lineageId: string; reviewerLabel: string } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("검수 초안 요청 형식이 올바르지 않습니다.");
  }
  const request = value as Record<string, unknown>;
  const keys = Object.keys(request).sort();
  if (keys.length !== 2 || keys[0] !== "lineageId" || keys[1] !== "reviewerLabel") {
    throw new TypeError("검수 초안 요청 필드가 올바르지 않습니다.");
  }
  return {
    lineageId: ipcLineageId(request.lineageId),
    reviewerLabel: validateReviewerLabel(request.reviewerLabel),
  };
}

async function sha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

function publish(job: JobRecord): JobRecord { window?.webContents.send("jobs:event", job); return job; }

function progressFromMessage(message: string): { progress: number; pageCount: number | null } {
  const page = /page\s+(\d+)\/(\d+)/i.exec(message);
  if (page) {
    const current = Number(page[1]);
    const total = Number(page[2]);
    return { progress: Math.min(0.84, 0.1 + (current / total) * 0.74), pageCount: total };
  }
  if (message.includes("writing editable HWPX")) return { progress: 0.84, pageCount: null };
  return { progress: 0.1, pageCount: null };
}

function terminateWorker(child: ChildProcessWithoutNullStreams): void {
  if (process.platform === "win32" && child.pid) {
    spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    return;
  }
  child.kill("SIGTERM");
}

function workerFailureMessage(code: number | null, stderr: string): string {
  const detail = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .at(-1);
  if (stderr.includes("No module named 'scan2hwpx'")) {
    return "변환 엔진을 찾지 못했습니다. 프로젝트 설치가 완료되었는지 확인하세요.";
  }
  if (stderr.includes("No module named 'paddleocr'")) {
    return "로컬 OCR 엔진이 설치되지 않았습니다. 프로그램 설치를 완료한 뒤 다시 시도하세요.";
  }
  return detail
    ? `변환 엔진이 종료되었습니다 (${code ?? "unknown"}): ${detail}`
    : `변환 엔진이 종료되었습니다 (${code ?? "unknown"}).`;
}

function finishActiveJob(id: string): void {
  if (activeJobId !== id) return;
  if (workerStderr) database.event(id, "worker_stderr", { message: workerStderr });
  rmSync(join(app.getPath("userData"), "jobs", id, "input"), { recursive: true, force: true });
  activeJobId = null;
  workerStderr = "";
  void processQueue();
}

function consumeWorkerOutput(data: Buffer): void {
  workerBuffer += String(data);
  const lines = workerBuffer.split("\n");
  workerBuffer = lines.pop() ?? "";
  for (const line of lines) {
    if (!line.trim()) continue;
    const id = activeJobId;
    if (!id) continue;
    let event: { event: string; data: Record<string, unknown> };
    try {
      event = JSON.parse(line) as { event: string; data: Record<string, unknown> };
    } catch {
      database.event(id, "worker_stdout", { message: line.slice(-4000) });
      continue;
    }
    database.event(id, event.event, event.data);
    if (isTerminal(database.get(id).status)) continue;
    if (event.event === "progress") {
      const next = progressFromMessage(String(event.data.message ?? ""));
      publish(database.setProgress(id, next.progress, next.pageCount));
    }
    if (event.event === "completed") {
      publish(database.transition(id, "STRUCTURING", { progress: 0.86 }));
      publish(database.transition(id, "EXPORTING", { progress: 0.92 }));
      publish(database.transition(id, "VALIDATING", { progress: 0.97 }));
      publish(database.transition(id, "COMPLETED", { progress: 1, outputPath: String(event.data.output_path) }));
      finishActiveJob(id);
    }
    if (event.event === "failed") {
      publish(database.transition(id, "FAILED", { errorMessage: String(event.data.message) }));
      finishActiveJob(id);
    }
  }
}

async function ensureWorker(): Promise<ChildProcessWithoutNullStreams> {
  if (worker && worker.exitCode === null && !worker.killed) return worker;
  const projectRoot = app.getAppPath();
  const pythonExecutable = await selectPythonRuntime({
    projectRoot,
    requiredImports: ["scan2hwpx", "pydantic", "paddleocr"],
  });
  if (quitting) throw new Error("프로그램을 종료하고 있습니다.");
  if (worker && worker.exitCode === null && !worker.killed) return worker;
  const child = spawn(pythonExecutable, ["-m", "scan2hwpx.worker"], {
    cwd: projectRoot,
    env: pythonRuntimeEnvironment(projectRoot),
    shell: false,
    windowsHide: true,
    stdio: ["pipe", "pipe", "pipe"],
  });
  worker = child;
  workerBuffer = "";
  child.stdout.on("data", consumeWorkerOutput);
  child.stderr.on("data", (data) => {
    workerStderr = `${workerStderr}${String(data)}`.slice(-4000);
  });
  child.on("error", (error) => {
    workerStderr = `${workerStderr}\n${error.message}`.slice(-4000);
  });
  child.on("close", (code) => {
    if (worker !== child) return;
    worker = null;
    workerBuffer = "";
    const id = activeJobId;
    activeJobId = null;
    if (id) {
      if (workerStderr) database.event(id, "worker_stderr", { message: workerStderr });
      if (!isTerminal(database.get(id).status)) {
        publish(database.transition(id, "FAILED", { errorMessage: workerFailureMessage(code, workerStderr) }));
      }
      rmSync(join(app.getPath("userData"), "jobs", id, "input"), { recursive: true, force: true });
    }
    workerStderr = "";
    if (!quitting) void processQueue();
  });
  return child;
}

async function processQueue(): Promise<void> {
  if (activeJobId || !queue.length || quitting) return;
  const id = queue.shift()!;
  let job = database.get(id);
  const root = join(app.getPath("userData"), "jobs", id);
  const inputDir = join(root, "input");
  mkdirSync(inputDir, { recursive: true });
  activeJobId = id;
  try {
    job = publish(database.transition(id, "STAGING", { progress: 0.03 }));
    const staged = join(inputDir, job.sourceName);
    copyFileSync(job.sourcePath, staged);
    job = publish(database.transition(id, "PREPROCESSING", { progress: 0.08 }));
    job = publish(database.transition(id, "OCR", { progress: 0.1 }));
    const projectRoot = app.getAppPath();
    const deployedModels = join(projectRoot, "ai", "production", "deployed");
    const pageAnomalyModel = join(deployedModels, "page-anomaly-v1", "page_anomaly_linear_autoencoder.npz");
    const recognitionModel = join(deployedModels, "korean-exam-ppocrv5");
    const child = await ensureWorker();
    if (quitting || isTerminal(database.get(id).status)) {
      if (activeJobId === id) activeJobId = null;
      rmSync(inputDir, { recursive: true, force: true });
      if (!quitting) void processQueue();
      return;
    }
    workerStderr = "";
    child.stdin.write(JSON.stringify({ protocol_version: "1.0", request_id: id, method: "convert", params: { job_id: id, input_pdf: staged, work_dir: root, options: { mode: "fast", dpi: 120, device: "auto", renderer: "editable", write_diagnostics: false, verify_hancom: true, recognition_model_dir: existsSync(join(recognitionModel, "inference.yml")) ? recognitionModel : null, page_anomaly_model: existsSync(pageAnomalyModel) ? pageAnomalyModel : null } } }) + "\n");
  } catch (error) {
    if (activeJobId === id) activeJobId = null;
    rmSync(inputDir, { recursive: true, force: true });
    if (!isTerminal(database.get(id).status)) {
      publish(database.transition(id, "FAILED", {
        errorMessage: error instanceof Error ? error.message : String(error),
      }));
    }
    void processQueue();
  }
}

function createWindow(): void {
  window = new BrowserWindow({ width: 1440, height: 900, minWidth: 1180, minHeight: 720,
    webPreferences: { preload: join(__dirname, "../preload/index.js"), contextIsolation: true, nodeIntegration: false, sandbox: true } });
  const dev = process.env.VITE_DEV_SERVER_URL;
  if (dev) void window.loadURL(dev); else void window.loadFile(join(__dirname, "../../dist-renderer/index.html"));
}

app.whenReady().then(() => {
  mkdirSync(app.getPath("userData"), { recursive: true });
  database = new JobDatabase(join(app.getPath("userData"), "exam2hwpx.sqlite"));
  queue.push(...database.recoverInterrupted());
  ipcMain.handle("files:select", async () => (await dialog.showOpenDialog({ properties: ["openFile", "multiSelections"], filters: [{ name: "PDF", extensions: ["pdf"] }] })).filePaths);
  ipcMain.handle("jobs:list", () => database.list());
  ipcMain.handle("jobs:enqueue", async (_event, paths: string[]) => {
    const uniquePaths = [...new Set(paths)];
    if (uniquePaths.length > MAX_BATCH_FILES) throw new Error(`한 번에 최대 ${MAX_BATCH_FILES}개까지 변환할 수 있습니다.`);
    const jobs: JobRecord[] = [];
    for (const sourcePath of uniquePaths) {
      if (resolve(sourcePath) !== sourcePath || !sourcePath.toLowerCase().endsWith(".pdf")) throw new Error("유효한 절대 PDF 경로가 아닙니다.");
      const now = new Date().toISOString();
      const job: JobRecord = { id: randomUUID(), sourcePath, sourceName: basename(sourcePath), sourceSha256: await sha256(sourcePath), status: "QUEUED", stage: "QUEUED", progress: 0, pageCount: null, outputPath: null, errorMessage: null, createdAt: now, updatedAt: now };
      database.insert(job); queue.push(job.id); jobs.push(job);
    }
    void processQueue(); return jobs;
  });
  ipcMain.handle("jobs:cancel", (_event: IpcMainInvokeEvent, id: string) => {
    const queuedIndex = queue.indexOf(id);
    if (queuedIndex >= 0) queue.splice(queuedIndex, 1);
    const current = database.get(id);
    if (!isTerminal(current.status)) publish(database.transition(id, "CANCELLED"));
    if (activeJobId === id && worker) terminateWorker(worker);
  });
  ipcMain.handle("review:select-bundle", async () => {
    const selection = await dialog.showOpenDialog({ properties: ["openDirectory"] });
    if (selection.canceled || !selection.filePaths.length) return null;
    const nextBundle = openCandidateReviewBundle(selection.filePaths[0]);
    candidateReviewBundle = nextBundle;
    candidateReviewRoot = nextBundle.canonicalRootPath;
    return { summary: nextBundle.summary, documents: nextBundle.documents };
  });
  ipcMain.handle("review:load-document", (_event, lineageId: unknown) =>
    selectedCandidateBundle().loadDocument(ipcLineageId(lineageId)),
  );
  ipcMain.handle("review:load-page", (_event, lineageId: unknown, pageNo: unknown) =>
    selectedCandidateBundle().readPageImageDataUrl(
      ipcLineageId(lineageId),
      ipcPageNo(pageNo),
    ),
  );
  ipcMain.handle("review:start-or-reopen-draft", async (_event, value: unknown) => {
    const request = ipcDraftRequest(value);
    const bundle = selectedCandidateBundle();
    const root = candidateReviewRoot;
    if (!root) throw new Error("먼저 검수 후보 폴더를 선택하세요.");
    const document = bundle.documents.find(
      (candidate) => candidate.lineageId === request.lineageId,
    );
    if (!document) throw new Error("등록되지 않은 검수 후보 문서입니다.");
    const bridgeOptions = {
      projectRoot: app.getAppPath(),
      userDataRoot: app.getPath("userData"),
      candidateRoot: root,
      manifestSha256: bundle.summary.manifestSha256,
      lineageId: document.lineageId,
      documentId: document.documentId,
    };
    const status = await startOrReopenCandidateReviewDraft({
      ...bridgeOptions,
      reviewerLabel: request.reviewerLabel,
    });
    const view = await loadCandidateReviewDraftView(bridgeOptions);
    return { operation: status.operation, view };
  });
  ipcMain.handle("review:apply-draft-patch", async (_event, value: unknown) => {
    const request = parseCandidateReviewDraftPatchRequest(value);
    const bundle = selectedCandidateBundle();
    const root = candidateReviewRoot;
    if (!root) throw new Error("먼저 검수 후보 폴더를 선택하세요.");
    const document = bundle.documents.find(
      (candidate) => candidate.lineageId === request.lineageId,
    );
    if (!document) throw new Error("등록되지 않은 검수 후보 문서입니다.");
    const bridgeOptions = {
      projectRoot: app.getAppPath(),
      userDataRoot: app.getPath("userData"),
      candidateRoot: root,
      manifestSha256: bundle.summary.manifestSha256,
      lineageId: document.lineageId,
      documentId: document.documentId,
    };
    try {
      const view = await applyCandidateReviewDraftPatch({
        ...bridgeOptions,
        request: {
          expectedDraftRevision: request.expectedDraftRevision,
          operations: request.operations,
        },
      });
      return { kind: "saved" as const, view };
    } catch (error) {
      if (!isCandidateReviewDraftStaleError(error)) throw error;
      const currentView = await loadCandidateReviewDraftView(bridgeOptions);
      return { kind: "stale_revision" as const, currentView };
    }
  });
  ipcMain.handle("review:complete-draft", async (_event, value: unknown) => {
    const request = parseCandidateReviewDraftCompletionRequest(value);
    const bundle = selectedCandidateBundle();
    const root = candidateReviewRoot;
    if (!root) throw new Error("먼저 검수 후보 폴더를 선택하세요.");
    const document = bundle.documents.find(
      (candidate) => candidate.lineageId === request.lineageId,
    );
    if (!document) throw new Error("등록되지 않은 검수 후보 문서입니다.");
    const bridgeOptions = {
      projectRoot: app.getAppPath(),
      userDataRoot: app.getPath("userData"),
      candidateRoot: root,
      manifestSha256: bundle.summary.manifestSha256,
      lineageId: document.lineageId,
      documentId: document.documentId,
    };
    try {
      const view = await completeCandidateReviewDraft({
        ...bridgeOptions,
        request: {
          expectedDraftRevision: request.expectedDraftRevision,
        },
      });
      return { kind: "saved" as const, view };
    } catch (error) {
      if (!isCandidateReviewDraftStaleError(error)) throw error;
      const currentView = await loadCandidateReviewDraftView(bridgeOptions);
      return { kind: "stale_revision" as const, currentView };
    }
  });
  ipcMain.handle("files:open", async (_event: IpcMainInvokeEvent, path: string) => { const error = await shell.openPath(path); if (error) throw new Error(error); });
  createWindow();
  void processQueue();
});

app.on("before-quit", () => {
  quitting = true;
  database?.flush();
  if (worker) terminateWorker(worker);
});
