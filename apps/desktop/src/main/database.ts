import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { JobRecord, JobState } from "../types/contracts.js";
import { assertTransition } from "./state-machine.js";

interface StoreFile {
  jobs: JobRecord[];
  events: Array<{ jobId: string; event: string; payload: unknown; createdAt: string }>;
}

export class JobDatabase {
  private readonly path: string;
  private store: StoreFile;
  private saveTimer: NodeJS.Timeout | null = null;
  private dirty = false;

  constructor(path: string) {
    this.path = path.replace(/\.sqlite$/i, ".json");
    mkdirSync(dirname(this.path), { recursive: true });
    this.store = this.load();
  }

  insert(job: JobRecord): void {
    if (this.store.jobs.some((item) => item.id === job.id)) throw new Error(`Duplicate job: ${job.id}`);
    this.store.jobs.push(job);
    this.save();
  }

  list(): JobRecord[] {
    return [...this.store.jobs].sort((left, right) => right.createdAt.localeCompare(left.createdAt));
  }

  get(id: string): JobRecord {
    const job = this.store.jobs.find((item) => item.id === id);
    if (!job) throw new Error(`Unknown job: ${id}`);
    return job;
  }

  recoverInterrupted(): string[] {
    const now = new Date().toISOString();
    for (const job of this.store.jobs) {
      if (!["COMPLETED", "FAILED", "CANCELLED", "QUEUED"].includes(job.status)) {
        job.status = "RETRYING";
        job.stage = "RETRYING";
        job.updatedAt = now;
        this.event(job.id, "recovered", {});
      }
    }
    this.save();
    return this.store.jobs
      .filter((job) => job.status === "QUEUED" || job.status === "RETRYING")
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt))
      .map((job) => job.id);
  }

  setProgress(id: string, progress: number, pageCount: number | null = null): JobRecord {
    const job = this.get(id);
    job.progress = progress;
    if (pageCount !== null) job.pageCount = pageCount;
    job.updatedAt = new Date().toISOString();
    this.save();
    return job;
  }

  transition(id: string, status: JobState, patch: Partial<JobRecord> = {}): JobRecord {
    const current = this.get(id);
    assertTransition(current.status, status);
    const next: JobRecord = { ...current, ...patch, status, stage: status, updatedAt: new Date().toISOString() };
    this.store.jobs = this.store.jobs.map((job) => (job.id === id ? next : job));
    this.event(id, "transition", { from: current.status, to: status });
    this.save();
    return next;
  }

  event(jobId: string, event: string, payload: unknown): void {
    this.store.events.push({ jobId, event, payload, createdAt: new Date().toISOString() });
    this.store.events = this.store.events.slice(-1000);
    this.save();
  }

  private load(): StoreFile {
    if (!existsSync(this.path)) return { jobs: [], events: [] };
    const parsed = JSON.parse(readFileSync(this.path, "utf8")) as Partial<StoreFile>;
    return {
      jobs: Array.isArray(parsed.jobs) ? parsed.jobs : [],
      events: Array.isArray(parsed.events) ? parsed.events : [],
    };
  }

  private save(): void {
    this.dirty = true;
    if (this.saveTimer) return;
    this.saveTimer = setTimeout(() => this.flush(), 100);
  }

  flush(): void {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    if (!this.dirty) return;
    this.dirty = false;
    const tempPath = `${this.path}.${process.pid}.tmp`;
    writeFileSync(tempPath, JSON.stringify(this.store, null, 2), "utf8");
    renameSync(tempPath, this.path);
  }
}
