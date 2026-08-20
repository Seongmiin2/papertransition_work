import zipfile
from pathlib import Path

from scan2hwpx.hwpx import ensure_minimal_template, validate_hwpx


def test_minimal_template_validates(tmp_path: Path) -> None:
    template = tmp_path / "exam_base.hwpx"
    ensure_minimal_template(template)
    assert validate_hwpx(template).valid


def test_validator_rejects_missing_entries(tmp_path: Path) -> None:
    broken = tmp_path / "broken.hwpx"
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr("mimetype", "wrong")
    result = validate_hwpx(broken)
    assert not result.valid
    assert any("missing required entry" in error for error in result.errors)
