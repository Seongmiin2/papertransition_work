from __future__ import annotations

import ctypes
import errno
import os
import subprocess
from pathlib import Path

import pytest

import scan2hwpx.evaluation.safe_artifact_io as safe_io


def test_windows_directory_publish_does_not_replace_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".model.partial"
    destination = tmp_path / "model"
    staging.mkdir()
    (staging / "weights").write_bytes(b"staged")

    def racing_rename(source: Path, target: Path) -> None:
        assert Path(source) == staging
        Path(target).mkdir()
        raise FileExistsError(str(target))

    monkeypatch.setattr(safe_io.sys, "platform", "win32")
    monkeypatch.setattr(safe_io.os, "rename", racing_rename)

    with pytest.raises(FileExistsError, match="appeared"):
        safe_io.publish_directory_create_only(staging, destination)

    assert (staging / "weights").read_bytes() == b"staged"
    assert not list(destination.iterdir())


def test_linux_directory_publish_uses_renameat2_noreplace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".model.partial"
    destination = tmp_path / "model"
    staging.mkdir()
    calls: list[tuple[object, ...]] = []

    class FakeRenameAt2:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            destination.mkdir()
            ctypes.set_errno(errno.EEXIST)
            return -1

    class FakeLibC:
        renameat2 = FakeRenameAt2()

    monkeypatch.setattr(safe_io.sys, "platform", "linux")
    monkeypatch.setattr(safe_io.ctypes, "CDLL", lambda *args, **kwargs: FakeLibC())

    with pytest.raises(FileExistsError, match="appeared"):
        safe_io.publish_directory_create_only(staging, destination)

    assert calls and calls[0][-1] == 1
    assert staging.is_dir()
    assert not list(destination.iterdir())


def test_directory_publish_fails_closed_on_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".model.partial"
    staging.mkdir()
    monkeypatch.setattr(safe_io.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="unavailable"):
        safe_io.publish_directory_create_only(staging, tmp_path / "model")


def test_directory_publish_runs_no_fallible_validation_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".model.partial"
    destination = tmp_path / "model"
    staging.mkdir()
    (staging / "weights").write_bytes(b"staged")
    original_rename = safe_io.os.rename
    original_snapshot_check = safe_io._require_same_snapshot
    committed = False

    def observed_rename(source: Path, target: Path) -> None:
        nonlocal committed
        original_rename(source, target)
        committed = True

    def reject_post_commit_validation(
        before: os.stat_result,
        after: os.stat_result,
        label: str,
    ) -> None:
        if committed:
            raise AssertionError("validation ran after the directory publish commit")
        original_snapshot_check(before, after, label)

    monkeypatch.setattr(safe_io.sys, "platform", "win32")
    monkeypatch.setattr(safe_io.os, "rename", observed_rename)
    monkeypatch.setattr(safe_io, "_require_same_snapshot", reject_post_commit_validation)

    safe_io.publish_directory_create_only(staging, destination)

    assert (destination / "weights").read_bytes() == b"staged"
    assert not staging.exists()


def test_file_publish_does_not_replace_existing_destination(tmp_path: Path) -> None:
    staging = tmp_path / ".weights.partial"
    destination = tmp_path / "weights"
    staging.write_bytes(b"downloaded")
    destination.write_bytes(b"racer")

    with pytest.raises(FileExistsError, match="already exists"):
        safe_io.publish_file_create_only(staging, destination)

    assert destination.read_bytes() == b"racer"
    assert staging.read_bytes() == b"downloaded"


def test_file_publish_moves_valid_source_as_final_commit(tmp_path: Path) -> None:
    staging = tmp_path / ".weights.partial"
    destination = tmp_path / "weights"
    staging.write_bytes(b"downloaded")

    safe_io.publish_file_create_only(staging, destination)

    assert destination.read_bytes() == b"downloaded"
    assert not staging.exists()


def test_windows_file_publish_does_not_replace_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".weights.partial"
    destination = tmp_path / "weights"
    staging.write_bytes(b"downloaded")

    def racing_rename(source: Path, target: Path) -> None:
        assert Path(source) == staging
        Path(target).write_bytes(b"racer")
        raise FileExistsError(str(target))

    monkeypatch.setattr(safe_io.sys, "platform", "win32")
    monkeypatch.setattr(safe_io.os, "rename", racing_rename)

    with pytest.raises(FileExistsError, match="appeared"):
        safe_io.publish_file_create_only(staging, destination)

    assert destination.read_bytes() == b"racer"
    assert staging.read_bytes() == b"downloaded"


def test_file_publish_rejects_source_changed_during_precommit_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".weights.partial"
    destination = tmp_path / "weights"
    staging.write_bytes(b"original")
    real_sha256 = safe_io.sha256_bounded_regular_file

    def changing_digest(path: Path, *, max_bytes: int, label: str) -> str:
        digest = real_sha256(path, max_bytes=max_bytes, label=label)
        Path(path).write_bytes(b"modified-after-validation")
        return digest

    monkeypatch.setattr(safe_io, "sha256_bounded_regular_file", changing_digest)

    with pytest.raises(safe_io.SafeArtifactIOError, match="changed"):
        safe_io.publish_file_create_only(staging, destination)

    assert not destination.exists()
    assert staging.read_bytes() == b"modified-after-validation"


def test_file_publish_runs_no_fallible_validation_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".weights.partial"
    destination = tmp_path / "weights"
    staging.write_bytes(b"downloaded")
    original_rename = safe_io.os.rename
    original_snapshot_check = safe_io._require_same_snapshot
    committed = False

    def observed_rename(source: Path, target: Path) -> None:
        nonlocal committed
        original_rename(source, target)
        committed = True

    def reject_post_commit_validation(
        before: os.stat_result,
        after: os.stat_result,
        label: str,
    ) -> None:
        if committed:
            raise AssertionError("validation ran after the file publish commit")
        original_snapshot_check(before, after, label)

    monkeypatch.setattr(safe_io.sys, "platform", "win32")
    monkeypatch.setattr(safe_io.os, "rename", observed_rename)
    monkeypatch.setattr(safe_io, "_require_same_snapshot", reject_post_commit_validation)

    safe_io.publish_file_create_only(staging, destination)

    assert destination.read_bytes() == b"downloaded"
    assert not staging.exists()


def test_file_publish_fails_closed_on_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".weights.partial"
    destination = tmp_path / "weights"
    staging.write_bytes(b"downloaded")
    monkeypatch.setattr(safe_io.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="unavailable"):
        safe_io.publish_file_create_only(staging, destination)

    assert not destination.exists()
    assert staging.read_bytes() == b"downloaded"


def test_bounded_regular_read_rejects_oversized_and_symlink_inputs(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"1234")
    with pytest.raises(safe_io.SafeArtifactIOError, match="size limit"):
        safe_io.read_bounded_regular_file(oversized, max_bytes=3, label="snapshot")

    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(oversized)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(safe_io.SafeArtifactIOError, match="non-symlink"):
        safe_io.read_bounded_regular_file(link, max_bytes=8, label="snapshot")


def test_staging_cleanup_is_limited_to_expected_temporary_child(tmp_path: Path) -> None:
    staging = tmp_path / ".risk.partial"
    staging.mkdir()
    (staging / "weights").write_bytes(b"partial")

    safe_io.remove_staging_directory(staging, parent=tmp_path)
    assert not staging.exists()

    outside = tmp_path.parent / ".outside.partial"
    with pytest.raises(safe_io.SafeArtifactIOError, match="unexpected staging path"):
        safe_io.remove_staging_directory(outside, parent=tmp_path)


def test_staging_cleanup_does_not_follow_nested_directory_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "protected.txt"
    protected.write_bytes(b"keep")
    staging = tmp_path / ".risk.partial"
    staging.mkdir()
    nested_link = staging / "nested-link"
    try:
        nested_link.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    safe_io.remove_staging_directory(staging, parent=tmp_path)

    assert protected.read_bytes() == b"keep"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_staging_cleanup_does_not_follow_nested_windows_junction(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "protected.txt"
    protected.write_bytes(b"keep")
    staging = tmp_path / ".risk.partial"
    staging.mkdir()
    junction = staging / "nested-junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode:
        pytest.skip(f"junction creation unavailable: {created.stderr.strip()}")

    safe_io.remove_staging_directory(staging, parent=tmp_path)

    assert protected.read_bytes() == b"keep"
