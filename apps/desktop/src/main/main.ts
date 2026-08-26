import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { createHash, randomUUID } from "node:crypto";
import { copyFileSync, createReadStream, existsSync, mkdirSync } from "node:fs";
import { basename, delimiter, join, resolve } from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { BrowserWindow as BrowserWindowType, IpcMainInvokeEvent } from "electron";
import { JobDatabase } from "./database.js";
import type { JobRecord } from "../types/contracts.js";

let window: BrowserWindowType | null = null;
let database: JobDatabase;
const queue: string[] = [];
const running = new Map<string, ChildProcessWithoutNullStreams>();

function isTerminal(status: JobRecord["status"]): boolean {
  return status === "COMPLETED" || status === "FAILED" || status === "CANCELLED";
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

async function processQueue(): Promise<void> {
  if (running.size || !queue.length) return;
  const id = queue.shift()!;
  let job = database.get(id);
  const root = join(app.getPath("userData"), "jobs", id);
  const inputDir = join(root, "input");
  mkdirSync(inputDir, { recursive: true });
  try {
    job = publish(database.transition(id, "STAGING", { progress: 0.03 }));
    const staged = join(inputDir, job.sourceName);
    copyFileSync(job.sourcePath, staged);
    job = publish(database.transition(id, "PREPROCESSING", { progress: 0.08 }));
    job = publish(database.transition(id, "OCR", { progress: 0.1 }));
    const projectRoot = app.getAppPath();
    const venvPython = join(projectRoot, ".venv", "Scripts", "python.exe");
    if (!existsSync(venvPython)) {
      throw new Error("변환 엔진이 준비되지 않았습니다. .venv\\Scripts\\python.exe를 찾지 못했습니다.");
    }
    const workerPath = join(projectRoot, "src");
    const pageAnomalyModel = join(projectRoot, "output", "trained-models", "page-anomaly-v1", "page_anomaly_linear_autoencoder.npz");
    const workerEnv = {
      ...process.env,
      PYTHONPATH: [workerPath, process.env.PYTHONPATH].filter(Boolean).join(delimiter),
      PYTHONUTF8: "1"
    };
    let workerStderr = "";
    const child = spawn(venvPython, ["-m", "scan2hwpx.worker"], {
      cwd: projectRoot,
      env: workerEnv,
      stdio: ["pipe", "pipe", "pipe"]
    });
    running.set(id, child);
    child.stderr.on("data", (data) => {
      workerStderr = `${workerStderr}${String(data)}`.slice(-4000);
    });
    let buffer = "";
    child.stdout.on("data", (data) => {
      buffer += String(data);
      const lines = buffer.split("\n"); buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
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
        }
        if (event.event === "failed") publish(database.transition(id, "FAILED", { errorMessage: String(event.data.message) }));
      }
    });
    child.on("error", (error) => {
      running.delete(id);
      if (!isTerminal(database.get(id).status)) {
        publish(database.transition(id, "FAILED", { errorMessage: `변환 엔진을 시작하지 못했습니다: ${error.message}` }));
      }
      void processQueue();
    });
    child.on("close", (code) => {
      running.delete(id);
      if (workerStderr) database.event(id, "worker_stderr", { message: workerStderr });
      if (!isTerminal(database.get(id).status)) {
        publish(database.transition(id, "FAILED", { errorMessage: workerFailureMessage(code, workerStderr) }));
      }
      void processQueue();
    });
    child.stdin.write(JSON.stringify({ protocol_version: "1.0", request_id: randomUUID(), method: "convert", params: { job_id: id, input_pdf: staged, work_dir: root, options: { remove_red_marks: true, ocr_provider: "local", layout: "exam_auto", mode: "fast", dpi: 240, device: "auto", renderer: "semantic", page_anomaly_model: existsSync(pageAnomalyModel) ? pageAnomalyModel : null } } }) + "\n");
    child.stdin.end();
  } catch (error) {
    publish(database.transition(id, "FAILED", { errorMessage: error instanceof Error ? error.message : String(error) }));
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
    const jobs: JobRecord[] = [];
    for (const sourcePath of paths) {
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
    const child = running.get(id);
    if (child) terminateWorker(child);
  });
  ipcMain.handle("files:open", async (_event: IpcMainInvokeEvent, path: string) => { const error = await shell.openPath(path); if (error) throw new Error(error); });
  createWindow();
  void processQueue();
});

app.on("before-quit", () => database?.flush());
