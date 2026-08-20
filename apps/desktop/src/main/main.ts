import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream, mkdirSync, copyFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { JobDatabase } from "./database.js";
import type { JobRecord } from "../types/contracts.js";

let window: BrowserWindow | null = null;
let database: JobDatabase;
const queue: string[] = [];
const running = new Map<string, ChildProcessWithoutNullStreams>();

async function sha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

function publish(job: JobRecord): JobRecord { window?.webContents.send("jobs:event", job); return job; }

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
    const child = spawn("python", ["-m", "scan2hwpx.worker"], { cwd: resolve("."), stdio: ["pipe", "pipe", "pipe"] });
    running.set(id, child);
    child.stderr.on("data", (data) => database.event(id, "worker_stderr", { message: String(data).slice(-4000) }));
    let buffer = "";
    child.stdout.on("data", (data) => {
      buffer += String(data);
      const lines = buffer.split("\n"); buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line) as { event: string; data: Record<string, unknown> };
        database.event(id, event.event, event.data);
        if (event.event === "progress") publish(database.setProgress(id, Math.min(0.84, database.get(id).progress + 0.01)));
        if (event.event === "completed") {
          publish(database.transition(id, "STRUCTURING", { progress: 0.86 }));
          publish(database.transition(id, "EXPORTING", { progress: 0.92 }));
          publish(database.transition(id, "VALIDATING", { progress: 0.97 }));
          publish(database.transition(id, "COMPLETED", { progress: 1, outputPath: String(event.data.output_path) }));
        }
        if (event.event === "failed") publish(database.transition(id, "FAILED", { errorMessage: String(event.data.message) }));
      }
    });
    child.on("close", (code) => {
      running.delete(id);
      if (code && database.get(id).status !== "FAILED") publish(database.transition(id, "FAILED", { errorMessage: `worker exited ${code}` }));
      void processQueue();
    });
    child.stdin.write(JSON.stringify({ protocol_version: "1.0", request_id: randomUUID(), method: "convert", params: { job_id: id, input_pdf: staged, work_dir: root, options: { remove_red_marks: true, ocr_provider: "local", layout: "exam_auto" } } }) + "\n");
    child.stdin.end();
  } catch (error) {
    publish(database.transition(id, "FAILED", { errorMessage: error instanceof Error ? error.message : String(error) }));
    void processQueue();
  }
}

function createWindow(): void {
  window = new BrowserWindow({ width: 1440, height: 900, minWidth: 1180, minHeight: 720,
    webPreferences: { preload: join(import.meta.dirname, "../preload/index.js"), contextIsolation: true, nodeIntegration: false, sandbox: true } });
  const dev = process.env.VITE_DEV_SERVER_URL;
  if (dev) void window.loadURL(dev); else void window.loadFile(join(import.meta.dirname, "../../dist-renderer/index.html"));
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
  ipcMain.handle("jobs:cancel", (_event, id: string) => {
    const queuedIndex = queue.indexOf(id);
    if (queuedIndex >= 0) queue.splice(queuedIndex, 1);
    running.get(id)?.kill();
    const current = database.get(id);
    if (current.status !== "CANCELLED") publish(database.transition(id, "CANCELLED"));
  });
  ipcMain.handle("files:open", async (_event, path: string) => { const error = await shell.openPath(path); if (error) throw new Error(error); });
  createWindow();
  void processQueue();
});
