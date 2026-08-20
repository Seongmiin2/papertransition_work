from __future__ import annotations

import json
import os
import shutil
import threading
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore[import-untyped]

from scan2hwpx.hwpx import render_hwpx, validate_hwpx
from scan2hwpx.ocr.providers import FixtureOcrProvider
from scan2hwpx.pipeline import convert_pdf


@dataclass(frozen=True)
class ExchangePaths:
    root: Path
    inbox: Path
    outbox: Path
    requests: Path


class ExchangeStore:
    """Local hand-off folders shared by the desktop UI and Codex workspace."""

    def __init__(self, workspace: Path) -> None:
        root = workspace / "exchange"
        self.paths = ExchangePaths(
            root=root,
            inbox=root / "inbox",
            outbox=root / "outbox",
            requests=root / "requests",
        )
        for path in (self.paths.inbox, self.paths.outbox, self.paths.requests):
            path.mkdir(parents=True, exist_ok=True)
        self._path_lock = threading.Lock()

    def available_path(self, folder: Path, name: str) -> Path:
        with self._path_lock:
            candidate = folder / Path(name).name
            counter = 1
            while candidate.exists():
                candidate = folder / f"{Path(name).stem}_{counter}{Path(name).suffix}"
                counter += 1
            candidate.touch(exist_ok=False)
            return candidate

    def receive_files(self, sources: list[Path]) -> list[Path]:
        copied: list[Path] = []
        for source in sources:
            if not source.is_file():
                raise ValueError(f"파일이 아닙니다: {source}")
            target = self.available_path(self.paths.inbox, source.name)
            shutil.copy2(source, target)
            copied.append(target)
        return copied

    def save_request(self, text: str, files: list[Path]) -> Path:
        if not text.strip() and not files:
            raise ValueError("요청 내용이나 첨부 파일이 필요합니다.")
        now = datetime.now().astimezone()
        stamp = now.strftime("%Y%m%d-%H%M%S-%f")
        payload = {
            "created_at": now.isoformat(),
            "request": text.strip(),
            "files": [path.relative_to(self.paths.root).as_posix() for path in files],
            "status": "ready_for_codex",
        }
        target = self.paths.requests / f"request-{stamp}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def list_outputs(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.paths.outbox.rglob("*.hwpx")
                if path.is_file() and not path.name.startswith(".")
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def create_job_directory(self, stem: str) -> Path:
        with self._path_lock:
            candidate = self.paths.outbox / stem
            counter = 1
            while candidate.exists():
                candidate = self.paths.outbox / f"{stem}_{counter}"
                counter += 1
            candidate.mkdir(parents=True)
            return candidate


class BatchConverter:
    """Bounded local queue that accepts any number of fixture jobs."""

    def __init__(self, workspace: Path, store: ExchangeStore, max_workers: int = 4) -> None:
        self.workspace = workspace
        self.store = store
        self.max_workers = max(1, max_workers)
        self._pdf_lock = threading.Lock()

    def convert_one(self, source: Path, progress: Callable[[str], None] | None = None) -> Path:
        if source.suffix.lower() == ".pdf":
            job_dir = self.store.create_job_directory(source.stem)
            target = job_dir / "result.hwpx"
            try:
                # Hancom COM and the local OCR model are intentionally serialized.
                # The queue itself accepts 20+ files without exhausting RAM or VRAM.
                with self._pdf_lock:
                    convert_pdf(source, target, dpi=300, progress=progress)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return target

        if source.suffix.lower() != ".json":
            raise ValueError(f"변환할 수 없는 형식입니다: {source.suffix}")
        document = FixtureOcrProvider().convert(source)
        target = self.store.available_path(self.store.paths.outbox, f"{source.stem}.hwpx")
        try:
            render_hwpx(document, self.workspace / "templates" / "exam_base.hwpx", target)
            result = validate_hwpx(target)
            if not result.valid:
                raise ValueError("HWPX 검증 실패: " + "; ".join(result.errors))
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def convert_many(
        self, sources: list[Path], progress: Callable[[str], None] | None = None
    ) -> tuple[list[Path], list[tuple[Path, str]]]:
        completed: list[Path] = []
        failed: list[tuple[Path, str]] = []
        worker_count = min(self.max_workers, max(1, len(sources)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            jobs = {pool.submit(self.convert_one, source, progress): source for source in sources}
            for future in as_completed(jobs):
                source = jobs[future]
                try:
                    completed.append(future.result())
                except (OSError, ValueError, KeyError) as exc:
                    failed.append((source, str(exc)))
        return completed, failed


class DesktopApp(TkinterDnD.Tk):  # type: ignore[misc]
    def __init__(self, workspace: Path | None = None) -> None:
        super().__init__()
        self.workspace = (workspace or Path.cwd()).resolve()
        self.store = ExchangeStore(self.workspace)
        self.converter = BatchConverter(self.workspace, self.store)
        self.selected_files: list[Path] = []
        self.title("Scan2HWPX 파일 작업함")
        self.geometry("900x650")
        self.minsize(760, 540)
        self._configure_style()
        self._build_ui()
        self.refresh_outputs()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Malgun Gothic", 17, "bold"))
        style.configure("Section.TLabel", font=("Malgun Gothic", 12, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        outer = ttk.Frame(self, padding=16)
        outer.grid(sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        outer.rowconfigure(5, weight=1)

        ttk.Label(outer, text="로컬 파일 작업함", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text="파일을 아래 영역에 끌어놓고 작업 내용을 적어 주세요.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))
        file_bar = ttk.Frame(outer)
        file_bar.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(file_bar, text="파일 선택", command=self.choose_files).pack(side="left")
        ttk.Button(file_bar, text="선택 제거", command=self.clear_files).pack(side="left", padx=6)
        self.file_count = ttk.Label(file_bar, text="0개 파일")
        self.file_count.pack(side="right")

        drop_frame = ttk.LabelFrame(
            outer, text="여기에 PDF, HWPX, DOCX, 이미지 등을 끌어놓으세요", padding=8
        )
        drop_frame.grid(row=3, column=0, sticky="nsew")
        drop_frame.columnconfigure(0, weight=1)
        drop_frame.rowconfigure(0, weight=1)
        self.file_list = tk.Listbox(
            drop_frame, height=8, selectmode=tk.EXTENDED, borderwidth=0, highlightthickness=0
        )
        self.file_list.grid(row=0, column=0, sticky="nsew")
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._on_drop)
        ttk.Label(outer, text="작업 내용", style="Section.TLabel").grid(
            row=4, column=0, sticky="w", pady=(12, 4)
        )
        self.request_text = tk.Text(outer, height=6, wrap="word")
        self.request_text.grid(row=5, column=0, sticky="ew")

        action_bar = ttk.Frame(outer)
        action_bar.grid(row=6, column=0, sticky="new", pady=(10, 0))
        self.send_button = ttk.Button(
            action_bar, text="작업 요청 보내기", command=self.submit_request
        )
        self.send_button.pack(side="left")
        self.convert_button = ttk.Button(
            action_bar, text="PDF를 HWPX로 변환", command=self.convert_selected
        )
        self.convert_button.pack(side="left", padx=6)
        ttk.Button(
            action_bar, text="작업 폴더 열기", command=lambda: self.open_path(self.store.paths.root)
        ).pack(side="left", padx=6)
        self.status = ttk.Label(action_bar, text="준비됨")
        self.status.pack(side="right")

        ttk.Separator(outer).grid(row=7, column=0, sticky="ew", pady=14)
        output_header = ttk.Frame(outer)
        output_header.grid(row=8, column=0, sticky="ew")
        ttk.Label(output_header, text="완료된 결과", style="Section.TLabel").pack(side="left")
        ttk.Button(output_header, text="새로고침", command=self.refresh_outputs).pack(side="right")
        ttk.Button(output_header, text="선택 파일 열기", command=self.open_selected_output).pack(
            side="right", padx=6
        )
        self.output_list = tk.Listbox(outer, height=8)
        self.output_list.grid(row=9, column=0, sticky="nsew", pady=(6, 0))
        outer.rowconfigure(3, weight=1)
        outer.rowconfigure(9, weight=1)

    def choose_files(self) -> None:
        names = filedialog.askopenfilenames(title="Codex에 보낼 파일 선택")
        self._add_files(Path(name) for name in names)

    def _on_drop(self, event: Any) -> str:
        names = self.tk.splitlist(event.data)
        self._add_files(Path(name) for name in names)
        return "break"

    def _add_files(self, paths: Any) -> None:
        for path in paths:
            if path not in self.selected_files and path.is_file():
                self.selected_files.append(path)
                self.file_list.insert(tk.END, str(path))
        self._update_file_count()

    def _update_file_count(self) -> None:
        self.file_count.config(text=f"{len(self.selected_files)}개 파일")

    def clear_files(self) -> None:
        selected = list(self.file_list.curselection())  # type: ignore[no-untyped-call]
        if not selected:
            self.selected_files.clear()
            self.file_list.delete(0, tk.END)
            self._update_file_count()
            return
        for index in reversed(selected):
            del self.selected_files[index]
            self.file_list.delete(index)
        self._update_file_count()

    def submit_request(self) -> None:
        try:
            copied = self.store.receive_files(self.selected_files)
            request = self.store.save_request(self.request_text.get("1.0", tk.END), copied)
        except (OSError, ValueError) as exc:
            messagebox.showerror("전송 실패", str(exc))
            return
        self.status.config(text=f"저장됨: {request.name}")
        self.selected_files.clear()
        self.file_list.delete(0, tk.END)
        self._update_file_count()
        self.request_text.delete("1.0", tk.END)
        messagebox.showinfo(
            "작업함 저장 완료",
            "파일과 요청을 로컬 작업함에 저장했습니다.\n현재 Codex 대화에서 '작업함 확인해'라고 알려주세요.",
        )

    def convert_selected(self) -> None:
        sources = [path for path in self.selected_files if path.suffix.lower() in {".pdf", ".json"}]
        if not sources or len(sources) != len(self.selected_files):
            messagebox.showwarning("입력 확인", "PDF 파일만 선택하세요.")
            return
        self.status.config(text=f"{len(sources)}건 변환 중…")
        self.send_button.config(state="disabled")
        self.convert_button.config(state="disabled")
        threading.Thread(target=self._convert_worker, args=(sources,), daemon=True).start()

    def _convert_worker(self, sources: list[Path]) -> None:
        completed, failed = self.converter.convert_many(
            sources,
            lambda message: self.after(0, lambda: self.status.config(text=message)),
        )
        self.after(0, lambda: self._conversion_done(completed, failed))

    def _conversion_done(self, completed: list[Path], failed: list[tuple[Path, str]]) -> None:
        self.send_button.config(state="normal")
        self.convert_button.config(state="normal")
        self.status.config(text=f"완료 {len(completed)}건 / 실패 {len(failed)}건")
        self.refresh_outputs()
        if failed:
            details = "\n".join(f"{path.name}: {error}" for path, error in failed[:10])
            messagebox.showwarning("일괄 변환 결과", details)

    def refresh_outputs(self) -> None:
        self.outputs = self.store.list_outputs()
        self.output_list.delete(0, tk.END)
        for path in self.outputs:
            timestamp = (
                datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M")
            )
            relative = path.relative_to(self.store.paths.outbox)
            self.output_list.insert(tk.END, f"{timestamp}  {relative}")

    def open_selected_output(self) -> None:
        selection = self.output_list.curselection()  # type: ignore[no-untyped-call]
        if selection:
            self.open_path(self.outputs[selection[0]])

    @staticmethod
    def open_path(path: Path) -> None:
        if os.name == "nt":
            os.startfile(path)
        else:
            messagebox.showinfo("경로", str(path))


def main() -> int:
    app = DesktopApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
