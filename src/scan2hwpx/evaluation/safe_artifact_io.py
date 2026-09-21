from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import BinaryIO


class SafeArtifactIOError(ValueError):
    """Raised when an artifact cannot be read or published safely."""


def read_bounded_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    source = Path(path)
    descriptor: int | None = None
    try:
        path_metadata = source.lstat()
        _require_regular_non_reparse(path_metadata, label)
        if path_metadata.st_size > max_bytes:
            raise SafeArtifactIOError(f"{label} exceeds the size limit")
        flags = os.O_RDONLY
        for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(source, flags)
        opened_metadata = os.fstat(descriptor)
        _require_regular_non_reparse(opened_metadata, label)
        _require_same_snapshot(path_metadata, opened_metadata, label)
        if opened_metadata.st_size > max_bytes:
            raise SafeArtifactIOError(f"{label} exceeds the size limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = _read_stream_bounded(stream, max_bytes=max_bytes, label=label)
            read_metadata = os.fstat(stream.fileno())
        _require_same_snapshot(opened_metadata, read_metadata, label)
        final_metadata = source.lstat()
        _require_regular_non_reparse(final_metadata, label)
        _require_same_snapshot(read_metadata, final_metadata, label)
        return payload
    except SafeArtifactIOError:
        raise
    except OSError as exc:
        raise SafeArtifactIOError(f"{label} is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def sha256_bounded_regular_file(path: Path, *, max_bytes: int, label: str) -> str:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    source = Path(path)
    descriptor: int | None = None
    try:
        path_metadata = source.lstat()
        _require_regular_non_reparse(path_metadata, label)
        if path_metadata.st_size > max_bytes:
            raise SafeArtifactIOError(f"{label} exceeds the size limit")
        flags = os.O_RDONLY
        for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(source, flags)
        opened_metadata = os.fstat(descriptor)
        _require_regular_non_reparse(opened_metadata, label)
        _require_same_snapshot(path_metadata, opened_metadata, label)
        if opened_metadata.st_size > max_bytes:
            raise SafeArtifactIOError(f"{label} exceeds the size limit")
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            while chunk := stream.read(min(1024 * 1024, max_bytes + 1 - total)):
                total += len(chunk)
                if total > max_bytes:
                    raise SafeArtifactIOError(f"{label} exceeds the size limit")
                digest.update(chunk)
            read_metadata = os.fstat(stream.fileno())
        _require_same_snapshot(opened_metadata, read_metadata, label)
        final_metadata = source.lstat()
        _require_regular_non_reparse(final_metadata, label)
        _require_same_snapshot(read_metadata, final_metadata, label)
        return digest.hexdigest()
    except SafeArtifactIOError:
        raise
    except OSError as exc:
        raise SafeArtifactIOError(f"{label} is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def publish_directory_create_only(source: Path, destination: Path) -> None:
    """Validate then atomically publish a directory as the final operation."""
    source_metadata = source.lstat()
    _require_directory_non_reparse(source_metadata, "staged artifact")
    validated_metadata = source.lstat()
    _require_directory_non_reparse(validated_metadata, "staged artifact")
    _require_same_snapshot(source_metadata, validated_metadata, "staged artifact")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    _rename_create_only(source, destination, artifact_kind="directory")


def publish_file_create_only(source: Path, destination: Path) -> None:
    """Publish a validated staging file with one atomic create-only rename.

    The staging path must remain private to the caller until this function returns.
    Every fallible integrity check precedes the rename, which is the final commit;
    there is no path-based rollback that could unlink a concurrently replaced file.
    """
    metadata = source.lstat()
    _require_regular_non_reparse(metadata, "staged artifact")
    sha256_bounded_regular_file(
        source,
        max_bytes=metadata.st_size,
        label="staged artifact",
    )
    validated_metadata = source.lstat()
    _require_regular_non_reparse(validated_metadata, "staged artifact")
    _require_same_snapshot(metadata, validated_metadata, "staged artifact")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    _rename_create_only(source, destination, artifact_kind="file")


def _rename_create_only(
    source: Path,
    destination: Path,
    *,
    artifact_kind: str,
) -> None:
    if sys.platform == "win32":
        try:
            os.rename(source, destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"output appeared during build: {destination}") from exc
            raise
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(f"atomic create-only {artifact_kind} publish is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"output appeared during build: {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))
    raise RuntimeError(f"atomic create-only {artifact_kind} publish is unavailable")


def remove_staging_directory(staging: Path, *, parent: Path) -> None:
    """Remove only the expected non-reparse temporary child directory."""
    parent_path = parent.absolute()
    staging_path = staging.absolute()
    if (
        staging_path.parent != parent_path
        or not staging_path.name.startswith(".")
        or not staging_path.name.endswith(".partial")
    ):
        raise SafeArtifactIOError("refusing to remove an unexpected staging path")
    try:
        metadata = staging_path.lstat()
    except FileNotFoundError:
        return
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise SafeArtifactIOError("refusing to remove a replaced staging directory")
    shutil.rmtree(staging_path)


def _read_stream_bounded(stream: BinaryIO, *, max_bytes: int, label: str) -> bytes:
    payload = bytearray()
    while True:
        remaining = max_bytes + 1 - len(payload)
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise SafeArtifactIOError(f"{label} exceeds the size limit")


def _require_regular_non_reparse(metadata: os.stat_result, label: str) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SafeArtifactIOError(f"{label} must be a regular non-symlink file")


def _require_directory_non_reparse(metadata: os.stat_result, label: str) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise SafeArtifactIOError(f"{label} must be a non-symlink directory")


def _require_same_snapshot(
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    if not (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    ):
        raise SafeArtifactIOError(f"{label} changed while it was read")
