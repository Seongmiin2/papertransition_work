import { contextBridge, ipcRenderer, webUtils, type IpcRendererEvent } from "electron";
import type { Exam2HwpxApi, JobRecord } from "../types/contracts.js";

const api: Exam2HwpxApi = {
  getPathForFile: (file) => webUtils.getPathForFile(file),
  selectPdfs: () => ipcRenderer.invoke("files:select"),
  enqueue: (paths) => ipcRenderer.invoke("jobs:enqueue", paths),
  listJobs: () => ipcRenderer.invoke("jobs:list"),
  cancel: (jobId) => ipcRenderer.invoke("jobs:cancel", jobId),
  openPath: (path) => ipcRenderer.invoke("files:open", path),
  onJobEvent: (callback) => {
    const listener = (_event: IpcRendererEvent, job: JobRecord) => callback(job);
    ipcRenderer.on("jobs:event", listener);
    return () => ipcRenderer.removeListener("jobs:event", listener);
  }
};
contextBridge.exposeInMainWorld("exam2hwpx", api);
