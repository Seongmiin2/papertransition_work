import Database from "better-sqlite3";
import type { JobRecord, JobState } from "../types/contracts.js";
import { assertTransition } from "./state-machine.js";

export class JobDatabase {
  private readonly db: Database.Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.pragma("journal_mode = WAL");
    this.db.exec(`CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY, source_path TEXT NOT NULL, source_name TEXT NOT NULL,
      source_sha256 TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL,
      progress REAL NOT NULL DEFAULT 0, page_count INTEGER, output_path TEXT,
      error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    ); CREATE TABLE IF NOT EXISTS job_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, event TEXT NOT NULL,
      payload_json TEXT NOT NULL, created_at TEXT NOT NULL
    );`);
  }

  insert(job: JobRecord): void {
    this.db.prepare(`INSERT INTO jobs VALUES
      (@id,@sourcePath,@sourceName,@sourceSha256,@status,@stage,@progress,@pageCount,@outputPath,@errorMessage,@createdAt,@updatedAt)`)
      .run(job);
  }

  list(): JobRecord[] {
    return this.db.prepare("SELECT * FROM jobs ORDER BY created_at DESC").all().map(mapRow);
  }

  get(id: string): JobRecord {
    const row = this.db.prepare("SELECT * FROM jobs WHERE id = ?").get(id);
    if (!row) throw new Error(`Unknown job: ${id}`);
    return mapRow(row);
  }

  recoverInterrupted(): string[] {
    const active = this.db.prepare("SELECT id FROM jobs WHERE status NOT IN ('COMPLETED','FAILED','CANCELLED','QUEUED')").all() as Array<{ id: string }>;
    const now = new Date().toISOString();
    const update = this.db.prepare("UPDATE jobs SET status='RETRYING', stage='RETRYING', updated_at=? WHERE id=?");
    for (const row of active) {
      update.run(now, row.id);
      this.event(row.id, "recovered", {});
    }
    const queued = this.db.prepare("SELECT id FROM jobs WHERE status IN ('QUEUED','RETRYING') ORDER BY created_at").all() as Array<{ id: string }>;
    return queued.map((row) => row.id);
  }

  setProgress(id: string, progress: number): JobRecord {
    this.db.prepare("UPDATE jobs SET progress=?, updated_at=? WHERE id=?").run(progress, new Date().toISOString(), id);
    return this.get(id);
  }

  transition(id: string, status: JobState, patch: Partial<JobRecord> = {}): JobRecord {
    const current = this.get(id);
    assertTransition(current.status, status);
    const next = { ...current, ...patch, status, stage: status, updatedAt: new Date().toISOString() };
    this.db.prepare(`UPDATE jobs SET status=@status, stage=@stage, progress=@progress,
      page_count=@pageCount, output_path=@outputPath, error_message=@errorMessage,
      updated_at=@updatedAt WHERE id=@id`).run(next);
    this.event(id, "transition", { from: current.status, to: status });
    return next;
  }

  event(jobId: string, event: string, payload: unknown): void {
    this.db.prepare("INSERT INTO job_events(job_id,event,payload_json,created_at) VALUES(?,?,?,?)")
      .run(jobId, event, JSON.stringify(payload), new Date().toISOString());
  }
}

function mapRow(value: unknown): JobRecord {
  const row = value as Record<string, unknown>;
  return {
    id: String(row.id), sourcePath: String(row.source_path), sourceName: String(row.source_name),
    sourceSha256: String(row.source_sha256), status: row.status as JobState, stage: row.stage as JobState,
    progress: Number(row.progress), pageCount: row.page_count === null ? null : Number(row.page_count),
    outputPath: row.output_path === null ? null : String(row.output_path),
    errorMessage: row.error_message === null ? null : String(row.error_message),
    createdAt: String(row.created_at), updatedAt: String(row.updated_at)
  };
}
