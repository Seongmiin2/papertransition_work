from pathlib import Path

from scan2hwpx.ocr.providers import FixtureOcrProvider


def test_fixture_preserves_pixel_and_normalized_coordinates() -> None:
    document = FixtureOcrProvider().convert(Path("tests/fixtures/ocr_page_1.json"))
    block = document.pages[0].blocks[0]
    assert block.bbox.pixel == (180.0, 140.0, 2300.0, 260.0)
    assert block.bbox.normalized[2] == 2300 / 2480
    assert block.source_provider == "fixture"
    assert "b-mark" in document.qa.low_confidence_blocks
