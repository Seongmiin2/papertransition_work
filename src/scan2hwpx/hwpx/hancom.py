from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Decline every message box (OK, cancel, abort, cancel, no, cancel) so a
# "damaged file" prompt fails the open instead of blocking hidden automation.
_DECLINE_MESSAGE_BOXES = 0x224121
_VERIFY_OPEN_ARGUMENTS = "suspendpassword:true;versionwarning:false"


class HancomRoundTripError(RuntimeError):
    """Hancom ran but rejected a generated HWPX."""


def verify_hancom_roundtrip(path: Path, expected_pages: int) -> bool:
    """Open, save, and reopen `path` in Hancom. Returns False if Hancom is unavailable.

    Raises HancomRoundTripError when Hancom rejects the file or opens it with a
    page count other than `expected_pages`.
    """
    try:
        from pyhwpx import Hwp  # type: ignore[import-untyped]

        _ensure_hancom_security_module()
        hwp: Any = Hwp(new=True, visible=False)
    except Exception:  # noqa: BLE001 - missing/broken Hancom means "not verified"
        return False
    try:
        hwp.SetMessageBoxMode(_DECLINE_MESSAGE_BOXES)
        with tempfile.TemporaryDirectory(
            prefix="scan2hwpx-roundtrip-", ignore_cleanup_errors=True
        ) as directory:
            saved = Path(directory) / "roundtrip.hwpx"
            _open_and_check_pages(hwp, path.resolve(), expected_pages, "열지")
            resaved = hwp.save_as(str(saved), format="HWPX")
            hwp.close(is_dirty=False)
            if not resaved:
                raise HancomRoundTripError("한글에서 HWPX로 다시 저장하지 못했습니다.")
            _open_and_check_pages(hwp, saved, expected_pages, "다시 열지")
            hwp.close(is_dirty=False)
    finally:
        hwp.quit()
    return True


def _open_and_check_pages(hwp: Any, path: Path, expected_pages: int, action: str) -> None:
    if not hwp.open(str(path), arg=_VERIFY_OPEN_ARGUMENTS):
        raise HancomRoundTripError(f"한글이 생성된 HWPX를 {action} 못했습니다.")
    pages = hwp.PageCount
    if pages != expected_pages:
        raise HancomRoundTripError(f"한글에서 {pages}쪽으로 열렸습니다(원본 {expected_pages}쪽).")


def _ensure_hancom_security_module() -> None:
    """Register pyhwpx's local file checker without relying on the global pip command."""
    if os.name != "nt":
        return
    import winreg

    import pyhwpx

    dll = Path(pyhwpx.__file__).with_name("FilePathCheckerModule.dll").resolve()
    if not dll.is_file():
        raise RuntimeError(f"한글 자동화 보안 모듈을 찾을 수 없습니다: {dll}")
    key_path = r"Software\HNC\HwpAutomation\Modules"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        winreg.SetValueEx(key, "FilePathCheckerModule", 0, winreg.REG_SZ, str(dll))
