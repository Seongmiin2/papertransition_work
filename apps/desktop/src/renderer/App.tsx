import { useEffect, useState } from "react";
import type { JobRecord } from "../types/contracts";

const MAX_BATCH_FILES = 10;

export function App() {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [pending, setPending] = useState<string[]>([]);
  const refresh = () => window.exam2hwpx.listJobs().then(setJobs);
  const upsertJob = (job: JobRecord) => setJobs((current) => {
    const index = current.findIndex((item) => item.id === job.id);
    if (index === -1) return [job, ...current];
    const next = [...current];
    next[index] = job;
    return next;
  });
  useEffect(() => { void refresh(); return window.exam2hwpx.onJobEvent(upsertJob); }, []);
  const add = (paths: string[]) => setPending((old) => [...new Set([...old, ...paths.filter((p) => p.toLowerCase().endsWith(".pdf"))])].slice(0, MAX_BATCH_FILES));
  const start = async () => { if (!pending.length) return; await window.exam2hwpx.enqueue(pending); setPending([]); await refresh(); };
  return <div className="shell">
    <aside><div className="brand"><span>문</span> Exam2HWPX</div><nav><button className="active">＋ 새 변환</button><button>작업 기록</button><button>모델 관리</button><button>설정</button></nav><small>로컬 처리 · 개인정보 보호</small></aside>
    <main>
      <header><div><h1>시험지 PDF를 편집 가능한 한글로</h1><p>파일은 이 컴퓨터에서 처리되며 외부로 전송되지 않습니다.</p></div><div className="local">● 로컬 OCR</div></header>
      <section className="card drop" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); add([...e.dataTransfer.files].map((f) => window.exam2hwpx.getPathForFile(f))); }}>
        <div className="fileIcon">PDF</div><h2>PDF를 여기에 끌어다 놓으세요</h2><p>한 번에 최대 10개까지 추가할 수 있습니다.</p>
        <button className="secondary" onClick={async () => add(await window.exam2hwpx.selectPdfs())}>파일 선택</button>
      </section>
      {pending.length > 0 && <section className="card"><h2>변환할 파일 <em>{pending.length}</em></h2>{pending.map((p) => <div className="pending" key={p}><span>PDF</span><div><b>{p.split(/[\\/]/).pop()}</b><small>{p}</small></div><button onClick={() => setPending(pending.filter((x) => x !== p))}>삭제</button></div>)}<div className="actions"><label><input type="checkbox" defaultChecked /> 빨간 채점 표시 제거</label><button className="primary" onClick={() => void start()}>HWPX 변환 시작</button></div></section>}
      <section className="card"><div className="sectionTitle"><h2>작업 대기열</h2><button onClick={() => void refresh()}>새로고침</button></div>{jobs.length === 0 ? <div className="empty">아직 변환 작업이 없습니다.</div> : jobs.map((job) => <article className="job" key={job.id}><div className="jobTop"><div><b>{job.sourceName}</b><small>{label(job.status)}</small></div><strong className={`state ${job.status.toLowerCase()}`}>{label(job.status)}</strong></div><div className="progress"><i style={{width: `${job.progress * 100}%`}} /></div><div className="jobBottom"><span>{Math.round(job.progress * 100)}%</span><div>{job.outputPath && <button onClick={() => void window.exam2hwpx.openPath(job.outputPath!)}>HWPX 열기</button>}{!["COMPLETED","FAILED","CANCELLED"].includes(job.status) && <button onClick={() => void window.exam2hwpx.cancel(job.id)}>취소</button>}</div></div>{job.errorMessage && <p className="error">{job.errorMessage}</p>}</article>)}</section>
    </main>
  </div>;
}

const names: Record<string,string> = { QUEUED:"대기", STAGING:"파일 준비", PREPROCESSING:"전처리", OCR:"문자 인식", STRUCTURING:"문항 구성", REVIEW_READY:"검토 필요", EXPORTING:"한글 생성", VALIDATING:"결과 검증", COMPLETED:"완료", FAILED:"실패", RETRYING:"재시도", CANCELLED:"취소됨" };
function label(value: string) { return names[value] ?? value; }
