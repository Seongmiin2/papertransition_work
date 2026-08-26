from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scan2hwpx.clean_layout import CleanItem, CleanKind, build_clean_items
from scan2hwpx.ir.models import Document


def render_clean_with_hancom(document: Document, output: Path) -> None:
    from pyhwpx import Hwp  # type: ignore[import-untyped]

    _ensure_hancom_security_module()
    output.parent.mkdir(parents=True, exist_ok=True)
    items = build_clean_items(document)
    hwp: Any = Hwp(new=True, visible=False)
    try:
        _set_two_columns(hwp)
        current_page = 1
        current_column = 0
        for item in items:
            if item.page_no != current_page:
                hwp.BreakPage()
                current_page = item.page_no
                current_column = 0
            if item.column != current_column:
                hwp.BreakColumn()
                current_column = item.column
            _insert_clean_item(hwp, item)
        if not hwp.save_as(str(output.resolve()), format="HWPX"):
            raise RuntimeError("정리된 HWPX 저장에 실패했습니다.")
    finally:
        hwp.quit()


def _set_two_columns(hwp: Any) -> None:
    coldef = hwp.HParameterSet.HColDef
    hwp.HAction.GetDefault("MultiColumn", coldef.HSet)
    coldef.Count = 2
    coldef.SameSize = 1
    coldef.SameGap = hwp.MiliToHwpUnit(8.0)
    coldef.HSet.SetItem("ApplyClass", 832)
    coldef.HSet.SetItem("ApplyTo", 6)
    if not hwp.HAction.Execute("MultiColumn", coldef.HSet):
        raise RuntimeError("한글 2단 설정에 실패했습니다.")


def _insert_clean_item(hwp: Any, item: CleanItem) -> None:
    bold = item.kind in {CleanKind.QUESTION, CleanKind.INSTRUCTION}
    size = 9.5 if item.kind == CleanKind.QUESTION else 9
    hwp.set_font(FaceName="함초롬바탕", Height=size, Bold=bold)
    para = hwp.HParameterSet.HParaShape
    hwp.HAction.GetDefault("ParagraphShape", para.HSet)
    para.AlignType = 0
    para.LineSpacingType = 0
    para.LineSpacing = 155
    para.PrevSpacing = hwp.MiliToHwpUnit(2.2 if item.kind == CleanKind.QUESTION else 0.4)
    para.NextSpacing = hwp.MiliToHwpUnit(1.0)
    para.LeftMargin = hwp.MiliToHwpUnit(4.0 if item.kind == CleanKind.CHOICE else 0.0)
    para.Indentation = hwp.MiliToHwpUnit(-2.5 if item.kind == CleanKind.CHOICE else 0.0)
    hwp.HAction.Execute("ParagraphShape", para.HSet)
    prefix = "[검토 필요] " if item.confidence < 0.65 else ""
    hwp.insert_text(prefix + item.text)
    hwp.BreakPara()


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
