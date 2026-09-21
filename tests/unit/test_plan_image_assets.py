from __future__ import annotations

import binascii
import hashlib
import json
import struct
import zlib
from collections.abc import Mapping
from io import BytesIO
from typing import cast

import pytest
from PIL import Image
from pydantic import ValidationError

import scan2hwpx.hwpx.assets as assets_module
from scan2hwpx.contracts import (
    BBox,
    ContentIR,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    ImageContentNode,
    ObservationKind,
    contract_sha256,
)
from scan2hwpx.hwpx.assets import (
    IMAGE_ASSET_BUNDLE_VERSION,
    MAX_IMAGE_ASSET_BYTES,
    MAX_IMAGE_ASSET_DECLARED_PIXELS,
    MAX_IMAGE_ASSET_UNIQUE_PIXELS,
    MAX_IMAGE_ASSETS,
    MAX_IMAGE_DIMENSION_PX,
    MAX_IMAGE_PIXELS,
    ImageAssetBundle,
    ImageAssetBundleError,
    PngImageAsset,
    build_image_asset_bundle,
    image_asset_bundle_sha256,
    validate_image_asset_bundle,
)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png(
    width: int = 1,
    height: int = 1,
    *,
    pixel: bytes = b"\x00\x00\x00\xff",
    idat_payload: bytes | None = None,
    before_idat: tuple[bytes, ...] = (),
    after_idat: tuple[bytes, ...] = (),
    trailer: bytes = b"",
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    scanlines = b"".join(b"\x00" + (pixel * width) for _ in range(height))
    compressed = zlib.compress(scanlines) if idat_payload is None else idat_payload
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + b"".join(before_idat)
        + _chunk(b"IDAT", compressed)
        + b"".join(after_idat)
        + _chunk(b"IEND", b"")
        + trailer
    )


def _png_with_header_dimensions(width: int, height: int) -> bytes:
    payload = bytearray(_png())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    payload[16:29] = ihdr
    payload[29:33] = struct.pack(">I", binascii.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    return bytes(payload)


def _animated_png() -> bytes:
    buffer = BytesIO()
    frames = (
        Image.new("RGBA", (1, 1), (0, 0, 0, 255)),
        Image.new("RGBA", (1, 1), (255, 255, 255, 255)),
    )
    frames[0].save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[frames[1]],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


def _one_frame_apng() -> bytes:
    animation_control = _chunk(b"acTL", struct.pack(">II", 1, 0))
    frame_control = _chunk(
        b"fcTL",
        struct.pack(">IIIIIHHBB", 0, 1, 1, 0, 0, 1, 100, 0, 0),
    )
    return _png(before_idat=(animation_control, frame_control))


def _asset(asset_ref: str, payload: bytes | None = None) -> PngImageAsset:
    actual_payload = payload or _png()
    return PngImageAsset(
        asset_ref=asset_ref,
        media_type="image/png",
        sha256=hashlib.sha256(actual_payload).hexdigest(),
        payload=actual_payload,
    )


def _contracts(
    payloads: tuple[tuple[str, bytes], ...],
    *,
    node_refs: tuple[str, ...] | None = None,
    source_kinds: Mapping[str, EvidenceSourceKind] | None = None,
    source_sha256: Mapping[str, str | None] | None = None,
    observation_source_refs: tuple[str, ...] | None = None,
    observation_kinds: tuple[ObservationKind, ...] | None = None,
) -> tuple[EvidenceIR, ContentIR]:
    payload_by_ref = dict(payloads)
    actual_node_refs = node_refs or tuple(payload_by_ref)
    sources = tuple(
        EvidenceSource(
            id=asset_ref,
            kind=(
                source_kinds.get(asset_ref, EvidenceSourceKind.PAGE_IMAGE)
                if source_kinds is not None
                else EvidenceSourceKind.PAGE_IMAGE
            ),
            artifact_ref=f"blob://{asset_ref}",
            producer="fixture",
            sha256=(
                source_sha256[asset_ref]
                if source_sha256 is not None and asset_ref in source_sha256
                else hashlib.sha256(payload).hexdigest()
            ),
        )
        for asset_ref, payload in payloads
    )
    observations = tuple(
        EvidenceObservation(
            id=f"observation-{index}",
            kind=(
                observation_kinds[index - 1]
                if observation_kinds is not None
                else ObservationKind.IMAGE
            ),
            bbox=BBox(
                pixel=(0.0, 0.0, 1.0, 1.0),
                normalized=(0.0, 0.0, 1.0, 1.0),
            ),
            confidence=0.99,
            source_refs=(
                (
                    observation_source_refs[index - 1]
                    if observation_source_refs is not None
                    else asset_ref
                ),
            ),
        )
        for index, asset_ref in enumerate(actual_node_refs, start=1)
    )
    evidence = EvidenceIR(
        id="evidence-1",
        source_document_sha256="f" * 64,
        sources=sources,
        pages=(
            EvidencePage(
                id="page-1",
                page_no=1,
                width=1.0,
                height=1.0,
                image_source_ref=payloads[0][0],
                observations=observations,
            ),
        ),
    )
    nodes = tuple(
        ImageContentNode(
            id=f"image-{index}",
            role=ContentRole.IMAGE,
            asset_ref=asset_ref,
            evidence_refs=(f"observation-{index}",),
            confidence=0.99,
        )
        for index, asset_ref in enumerate(actual_node_refs, start=1)
    )
    content = ContentIR(
        id="content-1",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        revision=3,
        nodes=nodes,
        reading_order=tuple(node.id for node in nodes),
    )
    return evidence, content


def test_builds_evidence_and_content_bound_bundle_with_deterministic_digest() -> None:
    first_payload = _png(2, 3)
    second_payload = _png(4, 5)
    evidence, content = _contracts((("asset-a", first_payload), ("asset-b", second_payload)))
    first_asset = _asset("asset-a", first_payload)
    second_asset = _asset("asset-b", second_payload)

    bundle = build_image_asset_bundle(
        evidence,
        content,
        (first_asset, second_asset),
    )

    assert bundle.schema_version == IMAGE_ASSET_BUNDLE_VERSION
    assert bundle.evidence_ir_id == evidence.id
    assert bundle.evidence_ir_contract_sha256 == contract_sha256(evidence)
    assert bundle.content_ir_id == content.id
    assert bundle.content_ir_revision == content.revision
    assert bundle.content_ir_contract_sha256 == contract_sha256(content)
    assert first_asset.byte_length == len(first_asset.payload)
    assert (first_asset.width_px, first_asset.height_px) == (2, 3)

    expected_metadata = {
        "assets": [
            {
                "asset_ref": "asset-a",
                "byte_length": len(first_asset.payload),
                "height_px": 3,
                "media_type": "image/png",
                "sha256": first_asset.sha256,
                "width_px": 2,
            },
            {
                "asset_ref": "asset-b",
                "byte_length": len(second_asset.payload),
                "height_px": 5,
                "media_type": "image/png",
                "sha256": second_asset.sha256,
                "width_px": 4,
            },
        ],
        "content_ir_contract_sha256": contract_sha256(content),
        "content_ir_id": content.id,
        "content_ir_revision": content.revision,
        "evidence_ir_contract_sha256": contract_sha256(evidence),
        "evidence_ir_id": evidence.id,
        "schema_version": IMAGE_ASSET_BUNDLE_VERSION,
    }
    canonical = json.dumps(
        expected_metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert image_asset_bundle_sha256(bundle) == hashlib.sha256(canonical).hexdigest()
    assert image_asset_bundle_sha256(bundle) == image_asset_bundle_sha256(
        build_image_asset_bundle(evidence, content, (first_asset, second_asset))
    )


def test_asset_digest_and_binary_type_are_validated_immediately() -> None:
    payload = _png()
    with pytest.raises(ValidationError, match="sha256"):
        PngImageAsset(
            asset_ref="asset-a",
            media_type="image/png",
            sha256="0" * 64,
            payload=payload,
        )
    with pytest.raises(ValidationError, match="immutable bytes"):
        PngImageAsset(
            asset_ref="asset-a",
            media_type="image/png",
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=cast(bytes, bytearray(payload)),
        )


def test_asset_requires_literal_png_media_type_and_lowercase_sha256() -> None:
    payload = _png()
    with pytest.raises(ValidationError, match="image/png"):
        PngImageAsset(
            asset_ref="asset-a",
            media_type="image/jpeg",  # type: ignore[arg-type]
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=payload,
        )
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        PngImageAsset(
            asset_ref="asset-a",
            media_type="image/png",
            sha256=hashlib.sha256(payload).hexdigest().upper(),
            payload=payload,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not a png", "PNG signature"),
        (_png_with_header_dimensions(0, 1), "non-zero"),
        (_png_with_header_dimensions(1, 0), "non-zero"),
        (_png_with_header_dimensions(MAX_IMAGE_DIMENSION_PX + 1, 1), "dimension"),
        (_png_with_header_dimensions(10_000, 10_000), "pixel count"),
    ],
)
def test_rejects_invalid_or_resource_bomb_png_headers(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _asset("asset-a", payload)


def test_single_asset_pixel_cap_limits_rgba_decode_to_about_64_mib() -> None:
    assert MAX_IMAGE_PIXELS == 16_000_000
    with pytest.raises(ValidationError, match="pixel count"):
        _asset("asset-a", _png_with_header_dimensions(5_000, 4_000))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_png(idat_payload=b"not-deflate"), "zlib stream is invalid"),
        (
            _png(before_idat=(_chunk(b"ABCD", b"unknown-critical"),)),
            "unknown critical",
        ),
        (_png(trailer=b"polyglot"), "IEND must be final"),
        (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + _chunk(b"IEND", b""),
            "requires at least one IDAT",
        ),
    ],
)
def test_rejects_malformed_png_chunk_streams(payload: bytes, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _asset("asset-a", payload)


def test_rejects_duplicate_ihdr_corrupt_crc_and_animated_png() -> None:
    valid = _png()
    duplicate_ihdr = valid[:33] + valid[8:33] + valid[33:]
    with pytest.raises(ValidationError, match="exactly one first IHDR"):
        _asset("asset-a", duplicate_ihdr)

    corrupt_crc = bytearray(valid)
    corrupt_crc[-5] ^= 0x01
    with pytest.raises(ValidationError, match="checksum"):
        _asset("asset-a", bytes(corrupt_crc))

    with pytest.raises(ValidationError, match="APNG chunks"):
        _asset("asset-a", _animated_png())


def test_rejects_one_frame_apng_even_when_pillow_reports_static() -> None:
    payload = _one_frame_apng()
    with Image.open(BytesIO(payload)) as opened:
        assert getattr(opened, "n_frames", 1) == 1
        assert not getattr(opened, "is_animated", False)
    with pytest.raises(ValidationError, match="APNG chunks"):
        _asset("asset-a", payload)


def test_v1_ancillary_allowlist_accepts_only_one_valid_phys_before_idat() -> None:
    physical_dimensions = _chunk(
        b"pHYs",
        struct.pack(">IIB", 3_780, 3_780, 1),
    )
    assert (
        _asset(
            "asset-a",
            _png(before_idat=(physical_dimensions,)),
        ).width_px
        == 1
    )

    invalid_payloads = (
        _png(before_idat=(physical_dimensions, physical_dimensions)),
        _png(before_idat=(_chunk(b"pHYs", b"short"),)),
        _png(before_idat=(_chunk(b"pHYs", struct.pack(">IIB", 1, 1, 2)),)),
        _png(after_idat=(physical_dimensions,)),
    )
    for payload in invalid_payloads:
        with pytest.raises(ValidationError, match="pHYs chunk"):
            _asset("asset-a", payload)


def test_v1_profile_rejects_plte_trns_and_unknown_ancillary_chunks() -> None:
    rejected = (
        (_chunk(b"PLTE", b"\x00"), "PLTE"),
        (_chunk(b"tRNS", b"\x00\x00"), "unsupported ancillary"),
        (_chunk(b"tEXt", b"key\x00value"), "unsupported ancillary"),
    )
    for chunk, message in rejected:
        with pytest.raises(ValidationError, match=message):
            _asset("asset-a", _png(before_idat=(chunk,)))


def test_rejects_zlib_trailer_truncation_wrong_size_and_invalid_filter() -> None:
    scanline = b"\x00\x00\x00\x00\xff"
    compressed = zlib.compress(scanline)
    invalid_streams = (
        compressed + b"zlib-polyglot",
        compressed[:-1],
        zlib.compress(scanline + b"extra"),
        zlib.compress(b"\x05\x00\x00\x00\xff"),
    )
    for stream in invalid_streams:
        with pytest.raises(ValidationError, match="zlib stream|filter byte"):
            _asset("asset-a", _png(idat_payload=stream))


def test_rejects_png_outside_static_noninterlaced_rgb_rgba_profile() -> None:
    grayscale = bytearray(_png())
    grayscale[25] = 0
    grayscale[29:33] = struct.pack(
        ">I",
        binascii.crc32(bytes(grayscale[12:29])) & 0xFFFFFFFF,
    )
    with pytest.raises(ValidationError, match="8-bit RGB or RGBA"):
        _asset("asset-a", bytes(grayscale))

    interlaced = bytearray(_png())
    interlaced[28] = 1
    interlaced[29:33] = struct.pack(
        ">I",
        binascii.crc32(bytes(interlaced[12:29])) & 0xFFFFFFFF,
    )
    with pytest.raises(ValidationError, match="methods are unsupported"):
        _asset("asset-a", bytes(interlaced))


def test_rejects_invalid_ihdr_and_truncated_png() -> None:
    payload = bytearray(_png())
    payload[12:16] = b"JHDR"
    with pytest.raises(ValidationError, match="IHDR"):
        _asset("asset-a", bytes(payload))
    with pytest.raises(ValidationError, match="chunk"):
        _asset("asset-a", _png()[:40])


def test_rejects_per_asset_and_total_payload_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    oversized = _png(trailer=b"x" * MAX_IMAGE_ASSET_BYTES)
    with pytest.raises(ValidationError, match="per-asset"):
        _asset("asset-a", oversized)

    first_payload = _png(pixel=b"\x00\x00\x00\xff")
    second_payload = _png(pixel=b"\xff\xff\xff\xff")
    evidence, content = _contracts((("asset-a", first_payload), ("asset-b", second_payload)))
    first = _asset("asset-a", first_payload)
    second = _asset("asset-b", second_payload)
    monkeypatch.setattr(
        assets_module,
        "MAX_IMAGE_ASSET_TOTAL_BYTES",
        len(first.payload) + len(second.payload) - 1,
    )
    with pytest.raises(ImageAssetBundleError, match="total byte"):
        build_image_asset_bundle(evidence, content, (first, second))


def test_bundle_unique_payload_pixel_budget_and_digest_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_IMAGE_ASSET_UNIQUE_PIXELS == 64_000_000
    assert MAX_IMAGE_ASSET_DECLARED_PIXELS == 64_000_000
    first_payload = _png(2, 2, pixel=b"\x00\x00\x00\xff")
    second_payload = _png(2, 2, pixel=b"\xff\xff\xff\xff")
    evidence, content = _contracts((("asset-a", first_payload), ("asset-b", second_payload)))
    monkeypatch.setattr(assets_module, "MAX_IMAGE_ASSET_UNIQUE_PIXELS", 7)
    with pytest.raises(ImageAssetBundleError, match="unique pixel"):
        build_image_asset_bundle(
            evidence,
            content,
            (_asset("asset-a", first_payload), _asset("asset-b", second_payload)),
        )

    shared_payload = _png(2, 2)
    shared_evidence, shared_content = _contracts(
        (("asset-a", shared_payload), ("asset-b", shared_payload))
    )
    monkeypatch.setattr(assets_module, "MAX_IMAGE_ASSET_UNIQUE_PIXELS", 4)
    shared_assets = (
        _asset("asset-a", shared_payload),
        _asset("asset-b", shared_payload),
    )
    bundle = build_image_asset_bundle(
        shared_evidence,
        shared_content,
        shared_assets,
    )
    assert len(bundle.assets) == 2

    def fail_asset_model_dump(
        self: PngImageAsset,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        raise AssertionError("asset model_dump must not run before declared-pixel preflight")

    monkeypatch.setattr(assets_module, "MAX_IMAGE_ASSET_DECLARED_PIXELS", 7)
    monkeypatch.setattr(PngImageAsset, "model_dump", fail_asset_model_dump)
    with pytest.raises(ImageAssetBundleError, match="declared pixel"):
        build_image_asset_bundle(
            shared_evidence,
            shared_content,
            shared_assets,
        )


def test_rejects_asset_count_limit_before_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_a = _png(pixel=b"\x00\x00\x00\xff")
    payload_b = _png(pixel=b"\xff\xff\xff\xff")
    evidence, content = _contracts((("asset-a", payload_a), ("asset-b", payload_b)))
    monkeypatch.setattr(assets_module, "MAX_IMAGE_ASSETS", 1)
    with pytest.raises(ValidationError, match="asset count"):
        ImageAssetBundle(
            evidence_ir_id=evidence.id,
            evidence_ir_contract_sha256=contract_sha256(evidence),
            content_ir_id=content.id,
            content_ir_revision=content.revision,
            content_ir_contract_sha256=contract_sha256(content),
            assets=(_asset("asset-a", payload_a), _asset("asset-b", payload_b)),
        )


def test_forged_oversize_bundle_is_rejected_before_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _png()
    evidence, content = _contracts((("asset-a", payload),))
    valid_asset = _asset("asset-a", payload)
    forged = ImageAssetBundle.model_construct(
        evidence_ir_id=evidence.id,
        evidence_ir_contract_sha256=contract_sha256(evidence),
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_contract_sha256=contract_sha256(content),
        assets=(valid_asset,) * (MAX_IMAGE_ASSETS + 1),
    )

    def fail_model_dump(
        self: ImageAssetBundle,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        raise AssertionError("model_dump must not run before bundle preflight")

    monkeypatch.setattr(ImageAssetBundle, "model_dump", fail_model_dump)
    with pytest.raises(ImageAssetBundleError, match="asset count"):
        validate_image_asset_bundle(evidence, content, forged)
    with pytest.raises(ImageAssetBundleError, match="asset count"):
        image_asset_bundle_sha256(forged)


def test_exact_coverage_rejects_duplicate_extra_missing_and_noncanonical_order() -> None:
    payload_a = _png(pixel=b"\x00\x00\x00\xff")
    payload_b = _png(pixel=b"\xff\xff\xff\xff")
    payload_c = _png(pixel=b"\xff\x00\x00\xff")
    evidence, content = _contracts(
        (("asset-a", payload_a), ("asset-b", payload_b), ("asset-c", payload_c)),
        node_refs=("asset-a", "asset-b"),
    )
    first = _asset("asset-a", payload_a)
    second = _asset("asset-b", payload_b)
    with pytest.raises(ValidationError, match="duplicate asset_ref"):
        ImageAssetBundle(
            evidence_ir_id=evidence.id,
            evidence_ir_contract_sha256=contract_sha256(evidence),
            content_ir_id=content.id,
            content_ir_revision=content.revision,
            content_ir_contract_sha256=contract_sha256(content),
            assets=(first, first),
        )
    with pytest.raises(ImageAssetBundleError, match="noncanonical asset order"):
        build_image_asset_bundle(evidence, content, (second, first))
    with pytest.raises(ImageAssetBundleError, match="missing image asset refs: asset-b"):
        build_image_asset_bundle(evidence, content, (first,))
    with pytest.raises(ImageAssetBundleError, match="extra image asset refs: asset-c"):
        build_image_asset_bundle(
            evidence,
            content,
            (first, second, _asset("asset-c", payload_c)),
        )


def test_reused_content_asset_ref_requires_one_bundle_entry() -> None:
    payload = _png()
    evidence, content = _contracts(
        (("asset-a", payload),),
        node_refs=("asset-a", "asset-a"),
    )
    bundle = build_image_asset_bundle(evidence, content, (_asset("asset-a", payload),))
    assert tuple(asset.asset_ref for asset in bundle.assets) == ("asset-a",)
    assert validate_image_asset_bundle(evidence, content, bundle) == bundle


def test_rejects_source_digest_kind_and_direct_grounding_bypasses() -> None:
    payload_a = _png(pixel=b"\x00\x00\x00\xff")
    payload_b = _png(pixel=b"\xff\xff\xff\xff")
    mismatched_evidence, mismatched_content = _contracts(
        (("asset-a", payload_a),),
        source_sha256={"asset-a": hashlib.sha256(payload_b).hexdigest()},
    )
    with pytest.raises(ImageAssetBundleError, match="EvidenceSource sha256"):
        build_image_asset_bundle(
            mismatched_evidence,
            mismatched_content,
            (_asset("asset-a", payload_a),),
        )

    missing_sha_evidence, missing_sha_content = _contracts(
        (("asset-a", payload_a),),
        source_sha256={"asset-a": None},
    )
    with pytest.raises(ImageAssetBundleError, match="requires sha256"):
        build_image_asset_bundle(
            missing_sha_evidence,
            missing_sha_content,
            (_asset("asset-a", payload_a),),
        )

    wrong_kind_evidence, wrong_kind_content = _contracts(
        (("asset-a", payload_a),),
        source_kinds={"asset-a": EvidenceSourceKind.ORIGINAL_DOCUMENT},
    )
    with pytest.raises(ImageAssetBundleError, match="PAGE_IMAGE or CROP"):
        build_image_asset_bundle(
            wrong_kind_evidence,
            wrong_kind_content,
            (_asset("asset-a", payload_a),),
        )

    ungrounded_evidence, ungrounded_content = _contracts(
        (("asset-a", payload_a), ("asset-b", payload_b)),
        node_refs=("asset-a",),
        observation_source_refs=("asset-b",),
    )
    with pytest.raises(ImageAssetBundleError, match="not directly grounded"):
        build_image_asset_bundle(
            ungrounded_evidence,
            ungrounded_content,
            (_asset("asset-a", payload_a),),
        )

    region_evidence, region_content = _contracts(
        (("asset-a", payload_a),),
        observation_kinds=(ObservationKind.REGION,),
    )
    with pytest.raises(
        ImageAssetBundleError,
        match="exactly one IMAGE observation",
    ):
        build_image_asset_bundle(
            region_evidence,
            region_content,
            (_asset("asset-a", payload_a),),
        )


def test_exact_coverage_reports_missing_evidence_source_as_contract_error() -> None:
    payload = _png()
    evidence, content = _contracts(
        (("asset-a", payload),),
        node_refs=("asset-missing",),
        observation_source_refs=("asset-a",),
    )
    bundle = ImageAssetBundle(
        evidence_ir_id=evidence.id,
        evidence_ir_contract_sha256=contract_sha256(evidence),
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_contract_sha256=contract_sha256(content),
        assets=(_asset("asset-missing", payload),),
    )

    with pytest.raises(ImageAssetBundleError, match="not an EvidenceIR source"):
        assets_module._require_exact_coverage(evidence, content, bundle)


def test_crop_source_with_matching_digest_and_observation_is_allowed() -> None:
    payload = _png()
    evidence, content = _contracts(
        (("asset-a", payload),),
        source_kinds={"asset-a": EvidenceSourceKind.CROP},
    )
    bundle = build_image_asset_bundle(evidence, content, (_asset("asset-a", payload),))
    assert validate_image_asset_bundle(evidence, content, bundle) == bundle


def test_validation_rejects_evidence_and_content_lineage_mismatch() -> None:
    payload = _png()
    evidence, content = _contracts((("asset-a", payload),))
    bundle = ImageAssetBundle(
        evidence_ir_id="other-evidence",
        evidence_ir_contract_sha256=contract_sha256(evidence),
        content_ir_id="other-content",
        content_ir_revision=content.revision,
        content_ir_contract_sha256=contract_sha256(content),
        assets=(_asset("asset-a", payload),),
    )
    with pytest.raises(
        ImageAssetBundleError,
        match="evidence_ir_id.*content_ir_id",
    ):
        validate_image_asset_bundle(evidence, content, bundle)

    stale_content = content.model_copy(
        update={"evidence_ir_sha256": "0" * 64},
    )
    with pytest.raises(ImageAssetBundleError, match="EvidenceIR lineage"):
        build_image_asset_bundle(evidence, stale_content, (_asset("asset-a", payload),))


def test_rejects_evidence_content_asset_and_bundle_subclasses() -> None:
    class EvidenceSubclass(EvidenceIR):
        pass

    class ContentSubclass(ContentIR):
        pass

    class AssetSubclass(PngImageAsset):
        pass

    class BundleSubclass(ImageAssetBundle):
        pass

    payload = _png()
    evidence, content = _contracts((("asset-a", payload),))
    subclass_evidence = EvidenceSubclass.model_validate(evidence.model_dump(mode="python"))
    with pytest.raises(TypeError, match="EvidenceIR"):
        build_image_asset_bundle(
            subclass_evidence,
            content,
            (_asset("asset-a", payload),),
        )

    subclass_content = ContentSubclass.model_validate(content.model_dump(mode="python"))
    with pytest.raises(TypeError, match="ContentIR"):
        build_image_asset_bundle(
            evidence,
            subclass_content,
            (_asset("asset-a", payload),),
        )

    subclass_asset = AssetSubclass(**_asset("asset-a", payload).model_dump(mode="python"))
    with pytest.raises(TypeError, match="PngImageAsset"):
        build_image_asset_bundle(evidence, content, (subclass_asset,))

    bundle = build_image_asset_bundle(
        evidence,
        content,
        (_asset("asset-a", payload),),
    )
    subclass_bundle = BundleSubclass(**bundle.model_dump(mode="python"))
    with pytest.raises(TypeError, match="ImageAssetBundle"):
        validate_image_asset_bundle(evidence, content, subclass_bundle)
    with pytest.raises(TypeError, match="ImageAssetBundle"):
        image_asset_bundle_sha256(subclass_bundle)


def test_revalidation_catches_forged_and_mutated_models() -> None:
    payload = _png()
    evidence, content = _contracts((("asset-a", payload),))
    forged_asset = PngImageAsset.model_construct(
        asset_ref="asset-a",
        media_type="image/png",
        sha256="0" * 64,
        payload=payload,
    )
    with pytest.raises(ImageAssetBundleError, match="valid strict PngImageAsset"):
        build_image_asset_bundle(evidence, content, (forged_asset,))

    bundle = build_image_asset_bundle(
        evidence,
        content,
        (_asset("asset-a", payload),),
    )
    object.__setattr__(bundle.assets[0], "sha256", "0" * 64)
    with pytest.raises(ImageAssetBundleError, match="valid strict ImageAssetBundle"):
        validate_image_asset_bundle(evidence, content, bundle)

    forged_content = content.model_copy()
    object.__setattr__(forged_content, "revision", True)
    with pytest.raises(ImageAssetBundleError, match="valid strict ContentIR"):
        build_image_asset_bundle(
            evidence,
            forged_content,
            (_asset("asset-a", payload),),
        )

    forged_evidence = evidence.model_copy()
    object.__setattr__(forged_evidence, "source_document_sha256", "INVALID")
    with pytest.raises(ImageAssetBundleError, match="valid strict EvidenceIR"):
        build_image_asset_bundle(
            forged_evidence,
            content,
            (_asset("asset-a", payload),),
        )
