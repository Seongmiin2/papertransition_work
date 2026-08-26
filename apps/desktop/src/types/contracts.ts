export const jobStates = ["QUEUED", "STAGING", "PREPROCESSING", "OCR", "STRUCTURING", "REVIEW_READY", "EXPORTING", "VALIDATING", "COMPLETED", "FAILED", "RETRYING", "CANCELLED"] as const;
export type JobState = typeof jobStates[number];

export interface JobRecord {
  id: string;
  sourcePath: string;
  sourceName: string;
  sourceSha256: string;
  status: JobState;
  stage: JobState;
  progress: number;
  pageCount: number | null;
  outputPath: string | null;
  errorMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface Exam2HwpxApi {
  getPathForFile(file: File): string;
  selectPdfs(): Promise<string[]>;
  enqueue(paths: string[]): Promise<JobRecord[]>;
  listJobs(): Promise<JobRecord[]>;
  cancel(jobId: string): Promise<void>;
  openPath(path: string): Promise<void>;
  onJobEvent(callback: (job: JobRecord) => void): () => void;
}
