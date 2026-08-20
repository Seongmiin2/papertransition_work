import json
from pathlib import Path

from scan2hwpx.desktop_app import BatchConverter, ExchangeStore
from scan2hwpx.hwpx import validate_hwpx


def test_exchange_store_copies_files_and_records_request(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    store = ExchangeStore(tmp_path / "workspace")
    copied = store.receive_files([source])
    request = store.save_request("이 파일을 확인해", copied)
    payload = json.loads(request.read_text(encoding="utf-8"))
    assert copied[0].read_text(encoding="utf-8") == "hello"
    assert payload["request"] == "이 파일을 확인해"
    assert payload["files"] == ["inbox/source.txt"]
    assert payload["status"] == "ready_for_codex"


def test_exchange_store_avoids_overwriting_same_name(tmp_path: Path) -> None:
    source = tmp_path / "same.txt"
    source.write_text("one", encoding="utf-8")
    store = ExchangeStore(tmp_path / "workspace")
    first = store.receive_files([source])[0]
    source.write_text("two", encoding="utf-8")
    second = store.receive_files([source])[0]
    assert first.name == "same.txt"
    assert second.name == "same_1.txt"
    assert first.read_text(encoding="utf-8") == "one"


def test_batch_converter_handles_more_than_twenty_jobs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "templates").mkdir(parents=True)
    (workspace / "templates" / "exam_base.hwpx").write_bytes(
        Path("templates/exam_base.hwpx").read_bytes()
    )
    fixture = json.loads(Path("tests/fixtures/ocr_page_1.json").read_text(encoding="utf-8"))
    sources: list[Path] = []
    for index in range(25):
        fixture["document_id"] = f"batch-{index}"
        source = tmp_path / f"fixture-{index}.json"
        source.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
        sources.append(source)
    completed, failed = BatchConverter(
        workspace, ExchangeStore(workspace), max_workers=4
    ).convert_many(sources)
    assert len(completed) == 25
    assert failed == []
    assert all(validate_hwpx(path).valid for path in completed)
