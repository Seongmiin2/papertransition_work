from __future__ import annotations

import binascii
import hashlib
import json
import struct
import warnings
import zlib
from collections import Counter
from collections.abc import Mapping
from io import BytesIO
from typing import Annotated, Literal, Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from scan2hwpx.contracts import (
    ContentIR,
    EvidenceIR,
    EvidenceSourceKind,
    ImageContentNode,
    ObservationKind,
    contract_sha256,
)

IMAGE_ASSET_BUNDLE_VERSION: Literal["image-asset-bundle/1.0"] = "image-asset-bundle/1.0"
MAX_IMAGE_ASSET_BYTES = 16 * 1024 * 1024
MAX_IMAGE_ASSETS = 1_000
MAX_IMAGE_ASSET_TOTAL_BYTES = 128 * 1024 * 1024
MAX_IMAGE_DIMENSION_PX = 32_768
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_ASSET_UNIQUE_PIXELS = 64_000_000
MAX_IMAGE_ASSET_DECLARED_PIXELS = 64_000_000

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_LENGTH = 13
_MAX_PNG_CHUNKS = 10_000
_KNOWN_CRITICAL_CHUNKS = frozenset((b"IHDR", b"PLTE", b"IDAT", b"IEND"))
_APNG_CHUNKS = frozenset((b"acTL", b"fcTL", b"fdAT"))
_ALLOWED_ANCILLARY_CHUNKS = frozenset((b"pHYs",))
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_NonEmpty = Annotated[str, Field(min_length=1)]
_Sha256 = Annotated[str, Field(pattern=_SHA256_PATTERN)]


class ImageAssetBundleError(ValueError):
    """The runtime image assets are not an exact, safe match for a ContentIR."""


class _StrictRuntimeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class PngImageAsset(_StrictRuntimeModel):
    asset_ref: _NonEmpty
    media_type: Literal["image/png"]
    sha256: _Sha256
    payload: bytes = Field(min_length=1, repr=False)

    @field_validator("payload", mode="before")
    @classmethod
    def require_immutable_bytes(cls, value: object) -> object:
        if type(value) is not bytes:
            raise ValueError("image payload must be immutable bytes")
        return value

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if len(self.payload) > MAX_IMAGE_ASSET_BYTES:
            raise ValueError("image payload exceeds the per-asset byte limit")
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("image payload does not match sha256")
        _validate_png(self.payload)
        return self

    @property
    def byte_length(self) -> int:
        return len(self.payload)

    @property
    def width_px(self) -> int:
        return _read_png_dimensions(self.payload)[0]

    @property
    def height_px(self) -> int:
        return _read_png_dimensions(self.payload)[1]


class ImageAssetBundle(_StrictRuntimeModel):
    schema_version: Literal["image-asset-bundle/1.0"] = IMAGE_ASSET_BUNDLE_VERSION
    evidence_ir_id: _NonEmpty
    evidence_ir_contract_sha256: _Sha256
    content_ir_id: _NonEmpty
    content_ir_revision: int = Field(ge=1)
    content_ir_contract_sha256: _Sha256
    assets: tuple[PngImageAsset, ...]

    @model_validator(mode="before")
    @classmethod
    def preflight_asset_count(cls, value: object) -> object:
        if isinstance(value, Mapping):
            assets = value.get("assets")
            if isinstance(assets, (list, tuple)) and len(assets) > MAX_IMAGE_ASSETS:
                raise ValueError("image asset count exceeds the bundle limit")
            if isinstance(assets, (list, tuple)):
                total_bytes = 0
                declared_pixels = 0
                unique_pixels = 0
                seen_sha256: set[str] = set()
                for asset in assets:
                    if type(asset) is PngImageAsset:
                        payload: object = asset.payload
                        sha256: object = asset.sha256
                    elif isinstance(asset, Mapping):
                        payload = asset.get("payload")
                        sha256 = asset.get("sha256")
                    else:
                        continue
                    if type(payload) is not bytes:
                        continue
                    if len(payload) > MAX_IMAGE_ASSET_BYTES:
                        raise ValueError("image payload exceeds the per-asset byte limit")
                    total_bytes += len(payload)
                    if total_bytes > MAX_IMAGE_ASSET_TOTAL_BYTES:
                        raise ValueError("image assets exceed the bundle total byte limit")
                    width, height = _read_png_dimensions(payload)
                    pixels = width * height
                    declared_pixels += pixels
                    if declared_pixels > MAX_IMAGE_ASSET_DECLARED_PIXELS:
                        raise ValueError("image assets exceed the bundle declared pixel limit")
                    if type(sha256) is str and sha256 not in seen_sha256:
                        seen_sha256.add(sha256)
                        unique_pixels += pixels
                        if unique_pixels > MAX_IMAGE_ASSET_UNIQUE_PIXELS:
                            raise ValueError("image assets exceed the bundle unique pixel limit")
        return value

    @model_validator(mode="after")
    def validate_assets(self) -> Self:
        if len(self.assets) > MAX_IMAGE_ASSETS:
            raise ValueError("image asset count exceeds the bundle limit")
        if any(type(asset) is not PngImageAsset for asset in self.assets):
            raise ValueError("bundle assets must be exact PngImageAsset instances")

        asset_refs = tuple(asset.asset_ref for asset in self.assets)
        duplicates = sorted(
            asset_ref for asset_ref, count in Counter(asset_refs).items() if count > 1
        )
        if duplicates:
            raise ValueError("duplicate asset_ref: " + ", ".join(duplicates))
        if asset_refs != tuple(sorted(asset_refs)):
            raise ValueError("noncanonical asset order")

        total_bytes = sum(asset.byte_length for asset in self.assets)
        if total_bytes > MAX_IMAGE_ASSET_TOTAL_BYTES:
            raise ValueError("image assets exceed the bundle total byte limit")
        declared_pixels = _declared_pixel_count(self.assets)
        if declared_pixels > MAX_IMAGE_ASSET_DECLARED_PIXELS:
            raise ValueError("image assets exceed the bundle declared pixel limit")
        unique_pixels = _unique_payload_pixel_count(self.assets)
        if unique_pixels > MAX_IMAGE_ASSET_UNIQUE_PIXELS:
            raise ValueError("image assets exceed the bundle unique pixel limit")
        return self


def build_image_asset_bundle(
    evidence_ir: EvidenceIR,
    content_ir: ContentIR,
    assets: tuple[PngImageAsset, ...],
) -> ImageAssetBundle:
    """Bind already-materialized PNG bytes to exactly one canonical ContentIR."""
    evidence = _revalidate_evidence_ir(evidence_ir)
    content = _revalidate_content_ir(content_ir)
    _require_content_evidence_integrity(evidence, content)
    validated_assets = _revalidate_asset_tuple(assets)
    try:
        bundle = ImageAssetBundle(
            evidence_ir_id=evidence.id,
            evidence_ir_contract_sha256=contract_sha256(evidence),
            content_ir_id=content.id,
            content_ir_revision=content.revision,
            content_ir_contract_sha256=contract_sha256(content),
            assets=validated_assets,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ImageAssetBundleError(f"invalid image asset bundle: {exc}") from exc
    _require_exact_coverage(evidence, content, bundle)
    return bundle


def validate_image_asset_bundle(
    evidence_ir: EvidenceIR,
    content_ir: ContentIR,
    bundle: ImageAssetBundle,
) -> ImageAssetBundle:
    """Strictly revalidate a bundle and its exact ContentIR lineage and coverage."""
    evidence = _revalidate_evidence_ir(evidence_ir)
    content = _revalidate_content_ir(content_ir)
    _require_content_evidence_integrity(evidence, content)
    validated_bundle = _revalidate_bundle(bundle)
    expected_lineage: tuple[tuple[str, object, object], ...] = (
        ("evidence_ir_id", validated_bundle.evidence_ir_id, evidence.id),
        (
            "evidence_ir_contract_sha256",
            validated_bundle.evidence_ir_contract_sha256,
            contract_sha256(evidence),
        ),
        ("content_ir_id", validated_bundle.content_ir_id, content.id),
        (
            "content_ir_revision",
            validated_bundle.content_ir_revision,
            content.revision,
        ),
        (
            "content_ir_contract_sha256",
            validated_bundle.content_ir_contract_sha256,
            contract_sha256(content),
        ),
    )
    mismatches = [field for field, actual, expected in expected_lineage if actual != expected]
    if mismatches:
        raise ImageAssetBundleError("image asset bundle lineage mismatch: " + ", ".join(mismatches))
    _require_exact_coverage(evidence, content, validated_bundle)
    return validated_bundle


def image_asset_bundle_sha256(bundle: ImageAssetBundle) -> str:
    """Return the deterministic metadata digest for a strictly valid bundle."""
    validated = _revalidate_bundle(bundle)
    metadata = {
        "schema_version": validated.schema_version,
        "evidence_ir_id": validated.evidence_ir_id,
        "evidence_ir_contract_sha256": validated.evidence_ir_contract_sha256,
        "content_ir_id": validated.content_ir_id,
        "content_ir_revision": validated.content_ir_revision,
        "content_ir_contract_sha256": validated.content_ir_contract_sha256,
        "assets": [
            {
                "asset_ref": asset.asset_ref,
                "media_type": asset.media_type,
                "sha256": asset.sha256,
                "byte_length": asset.byte_length,
                "width_px": asset.width_px,
                "height_px": asset.height_px,
            }
            for asset in validated.assets
        ],
    }
    canonical = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _revalidate_evidence_ir(evidence_ir: EvidenceIR) -> EvidenceIR:
    if type(evidence_ir) is not EvidenceIR:
        raise TypeError("evidence_ir must be an exact EvidenceIR")
    try:
        payload = evidence_ir.model_dump(mode="python", round_trip=True, warnings=False)
        return EvidenceIR.model_validate(payload, strict=True)
    except (AttributeError, TypeError, ValueError, ValidationError, OverflowError) as exc:
        raise ImageAssetBundleError("evidence_ir is not a valid strict EvidenceIR") from exc


def _revalidate_content_ir(content_ir: ContentIR) -> ContentIR:
    if type(content_ir) is not ContentIR:
        raise TypeError("content_ir must be an exact ContentIR")
    try:
        payload = content_ir.model_dump(mode="python", round_trip=True, warnings=False)
        return ContentIR.model_validate(payload, strict=True)
    except (AttributeError, TypeError, ValueError, ValidationError, OverflowError) as exc:
        raise ImageAssetBundleError("content_ir is not a valid strict ContentIR") from exc


def _revalidate_asset_tuple(
    assets: tuple[PngImageAsset, ...],
) -> tuple[PngImageAsset, ...]:
    if type(assets) is not tuple:
        raise TypeError("assets must be a tuple")
    if len(assets) > MAX_IMAGE_ASSETS:
        raise ImageAssetBundleError("image asset count exceeds the bundle limit")

    _preflight_asset_tuple(assets)
    validated: list[PngImageAsset] = []
    for asset in assets:
        try:
            payload = asset.model_dump(mode="python", round_trip=True, warnings=False)
            restored = PngImageAsset.model_validate(payload, strict=True)
        except (AttributeError, TypeError, ValueError, ValidationError, OverflowError) as exc:
            raise ImageAssetBundleError("asset is not a valid strict PngImageAsset") from exc
        validated.append(restored)
    return tuple(validated)


def _preflight_asset_tuple(assets: tuple[PngImageAsset, ...]) -> None:
    total_bytes = 0
    declared_pixels = 0
    unique_pixels = 0
    seen_sha256: set[str] = set()
    for asset in assets:
        if type(asset) is not PngImageAsset:
            raise TypeError("each asset must be an exact PngImageAsset")
        raw_payload: object = getattr(asset, "payload", None)
        if type(raw_payload) is not bytes:
            raise ImageAssetBundleError("asset payload must be immutable bytes")
        if len(raw_payload) > MAX_IMAGE_ASSET_BYTES:
            raise ImageAssetBundleError("image payload exceeds the per-asset byte limit")
        total_bytes += len(raw_payload)
        if total_bytes > MAX_IMAGE_ASSET_TOTAL_BYTES:
            raise ImageAssetBundleError("image assets exceed the bundle total byte limit")
        try:
            width, height = _read_png_dimensions(raw_payload)
        except ValueError as exc:
            raise ImageAssetBundleError("asset has an invalid PNG header") from exc
        pixels = width * height
        declared_pixels += pixels
        if declared_pixels > MAX_IMAGE_ASSET_DECLARED_PIXELS:
            raise ImageAssetBundleError("image assets exceed the bundle declared pixel limit")
        raw_sha256: object = getattr(asset, "sha256", None)
        if type(raw_sha256) is str and raw_sha256 not in seen_sha256:
            seen_sha256.add(raw_sha256)
            unique_pixels += pixels
            if unique_pixels > MAX_IMAGE_ASSET_UNIQUE_PIXELS:
                raise ImageAssetBundleError("image assets exceed the bundle unique pixel limit")


def _revalidate_bundle(bundle: ImageAssetBundle) -> ImageAssetBundle:
    if type(bundle) is not ImageAssetBundle:
        raise TypeError("bundle must be an exact ImageAssetBundle")
    _preflight_bundle_instance(bundle)
    try:
        payload = bundle.model_dump(mode="python", round_trip=True, warnings=False)
        return ImageAssetBundle.model_validate(payload, strict=True)
    except (AttributeError, TypeError, ValueError, ValidationError, OverflowError) as exc:
        raise ImageAssetBundleError("bundle is not a valid strict ImageAssetBundle") from exc


def _preflight_bundle_instance(bundle: ImageAssetBundle) -> None:
    raw_assets: object = getattr(bundle, "assets", None)
    if type(raw_assets) is not tuple:
        raise ImageAssetBundleError("bundle assets must be a tuple")
    if len(raw_assets) > MAX_IMAGE_ASSETS:
        raise ImageAssetBundleError("image asset count exceeds the bundle limit")
    total_bytes = 0
    declared_pixels = 0
    unique_pixels = 0
    seen_sha256: set[str] = set()
    for asset in raw_assets:
        if type(asset) is not PngImageAsset:
            raise ImageAssetBundleError("bundle assets must be exact PngImageAsset instances")
        raw_payload: object = getattr(asset, "payload", None)
        raw_sha256: object = getattr(asset, "sha256", None)
        if type(raw_payload) is not bytes:
            raise ImageAssetBundleError("asset payload must be immutable bytes")
        if len(raw_payload) > MAX_IMAGE_ASSET_BYTES:
            raise ImageAssetBundleError("image payload exceeds the per-asset byte limit")
        total_bytes += len(raw_payload)
        if total_bytes > MAX_IMAGE_ASSET_TOTAL_BYTES:
            raise ImageAssetBundleError("image assets exceed the bundle total byte limit")
        try:
            width, height = _read_png_dimensions(raw_payload)
        except ValueError as exc:
            raise ImageAssetBundleError("asset has an invalid PNG header") from exc
        pixels = width * height
        declared_pixels += pixels
        if declared_pixels > MAX_IMAGE_ASSET_DECLARED_PIXELS:
            raise ImageAssetBundleError("image assets exceed the bundle declared pixel limit")
        if type(raw_sha256) is str and raw_sha256 not in seen_sha256:
            seen_sha256.add(raw_sha256)
            unique_pixels += pixels
            if unique_pixels > MAX_IMAGE_ASSET_UNIQUE_PIXELS:
                raise ImageAssetBundleError("image assets exceed the bundle unique pixel limit")


def _require_content_evidence_integrity(
    evidence: EvidenceIR,
    content: ContentIR,
) -> None:
    try:
        content.assert_evidence_integrity(evidence)
    except ValueError as exc:
        raise ImageAssetBundleError(f"ContentIR EvidenceIR lineage is invalid: {exc}") from exc


def _require_exact_coverage(
    evidence: EvidenceIR,
    content: ContentIR,
    bundle: ImageAssetBundle,
) -> None:
    expected_refs = {node.asset_ref for node in content.nodes if isinstance(node, ImageContentNode)}
    actual_refs = {asset.asset_ref for asset in bundle.assets}
    missing = sorted(expected_refs - actual_refs)
    extra = sorted(actual_refs - expected_refs)
    if missing:
        raise ImageAssetBundleError("missing image asset refs: " + ", ".join(missing))
    if extra:
        raise ImageAssetBundleError("extra image asset refs: " + ", ".join(extra))
    assets_by_ref = {asset.asset_ref: asset for asset in bundle.assets}
    sources_by_id = {source.id: source for source in evidence.sources}
    observations_by_id = {
        observation.id: observation for page in evidence.pages for observation in page.observations
    }
    for node in content.nodes:
        if not isinstance(node, ImageContentNode):
            continue
        source = sources_by_id.get(node.asset_ref)
        if source is None:
            raise ImageAssetBundleError(
                f"image asset_ref {node.asset_ref!r} is not an EvidenceIR source"
            )
        if source.kind not in (EvidenceSourceKind.PAGE_IMAGE, EvidenceSourceKind.CROP):
            raise ImageAssetBundleError(
                f"image asset source {source.id} must be PAGE_IMAGE or CROP"
            )
        if source.sha256 is None:
            raise ImageAssetBundleError(f"image asset source {source.id} requires sha256")
        asset = assets_by_ref[node.asset_ref]
        if asset.sha256 != source.sha256:
            raise ImageAssetBundleError(
                f"image asset {node.asset_ref} does not match its EvidenceSource sha256"
            )
        image_groundings = tuple(
            observations_by_id[observation_ref]
            for observation_ref in node.evidence_refs
            if (
                observations_by_id[observation_ref].kind == ObservationKind.IMAGE
                and node.asset_ref in observations_by_id[observation_ref].source_refs
            )
        )
        if len(image_groundings) != 1:
            raise ImageAssetBundleError(
                f"image node {node.id} is not directly grounded by exactly one IMAGE "
                f"observation for source {node.asset_ref}"
            )


def _unique_payload_pixel_count(assets: tuple[PngImageAsset, ...]) -> int:
    pixels_by_sha: dict[str, int] = {}
    for asset in assets:
        pixels_by_sha.setdefault(asset.sha256, asset.width_px * asset.height_px)
    return sum(pixels_by_sha.values())


def _declared_pixel_count(assets: tuple[PngImageAsset, ...]) -> int:
    return sum(asset.width_px * asset.height_px for asset in assets)


def _read_png_dimensions(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(_PNG_SIGNATURE):
        raise ValueError("image payload has an invalid PNG signature")
    if len(payload) < 33:
        raise ValueError("image payload is too short for a PNG IHDR")
    ihdr_length = struct.unpack(">I", payload[8:12])[0]
    if ihdr_length != _PNG_IHDR_LENGTH or payload[12:16] != b"IHDR":
        raise ValueError("PNG must start with a 13-byte IHDR chunk")
    expected_crc = struct.unpack(">I", payload[29:33])[0]
    actual_crc = binascii.crc32(payload[12:29]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError("PNG IHDR checksum is invalid")
    width, height = struct.unpack(">II", payload[16:24])
    return width, height


def _validate_png(payload: bytes) -> None:
    width, height, expected_mode, compressed_pixels = _parse_png_chunks(payload)
    if width == 0 or height == 0:
        raise ValueError("PNG width and height must be non-zero")
    if width > MAX_IMAGE_DIMENSION_PX or height > MAX_IMAGE_DIMENSION_PX:
        raise ValueError("PNG dimension exceeds the image limit")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("PNG pixel count exceeds the image limit")
    _validate_png_pixel_stream(
        compressed_pixels,
        width=width,
        height=height,
        channels=3 if expected_mode == "RGB" else 4,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format != "PNG":
                    raise ValueError("image payload is not a PNG")
                if image.size != (width, height):
                    raise ValueError("PNG decoder dimensions disagree with IHDR")
                if image.mode != expected_mode:
                    raise ValueError("PNG decoder mode disagrees with IHDR")
                if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("animated PNG assets are not supported")
                image.verify()
            with Image.open(BytesIO(payload)) as decoded:
                if decoded.format != "PNG":
                    raise ValueError("decoded image payload is not a PNG")
                if decoded.size != (width, height) or decoded.mode != expected_mode:
                    raise ValueError("decoded PNG metadata disagrees with IHDR")
                if getattr(decoded, "is_animated", False) or getattr(decoded, "n_frames", 1) != 1:
                    raise ValueError("animated PNG assets are not supported")
                decoded.load()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise ValueError("image payload is not a valid PNG") from exc


def _parse_png_chunks(payload: bytes) -> tuple[int, int, str, bytes]:
    width, height = _read_png_dimensions(payload)
    bit_depth = payload[24]
    color_type = payload[25]
    compression_method = payload[26]
    filter_method = payload[27]
    interlace_method = payload[28]
    expected_mode = _expected_png_mode(bit_depth, color_type)
    if compression_method != 0 or filter_method != 0 or interlace_method != 0:
        raise ValueError("PNG IHDR methods are unsupported")

    offset = len(_PNG_SIGNATURE)
    chunk_count = 0
    seen_ihdr = False
    seen_phys = False
    seen_idat = False
    idat_closed = False
    seen_iend = False
    idat_parts: list[bytes] = []
    while offset < len(payload):
        chunk_count += 1
        if chunk_count > _MAX_PNG_CHUNKS:
            raise ValueError("PNG chunk count exceeds the image limit")
        if len(payload) - offset < 12:
            raise ValueError("PNG chunk stream is truncated")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(payload):
            raise ValueError("PNG chunk length exceeds the payload")
        if len(chunk_type) != 4 or not all(
            65 <= character <= 90 or 97 <= character <= 122 for character in chunk_type
        ):
            raise ValueError("PNG chunk type must contain four ASCII letters")
        if chunk_type[2] & 0x20:
            raise ValueError("PNG chunk type uses the reserved lowercase bit")
        chunk_data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : chunk_end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("PNG chunk checksum is invalid")

        if chunk_count == 1 and chunk_type != b"IHDR":
            raise ValueError("PNG IHDR must be the first chunk")
        if chunk_type == b"IHDR":
            if seen_ihdr or chunk_count != 1 or length != _PNG_IHDR_LENGTH:
                raise ValueError("PNG must contain exactly one first IHDR chunk")
            seen_ihdr = True
        elif not seen_ihdr:
            raise ValueError("PNG IHDR must be the first chunk")
        elif not chunk_type[0] & 0x20 and chunk_type not in _KNOWN_CRITICAL_CHUNKS:
            raise ValueError("PNG contains an unknown critical chunk")
        elif chunk_type == b"PLTE":
            raise ValueError("PNG PLTE chunks are unsupported by the v1 profile")
        elif chunk_type == b"IDAT":
            if idat_closed:
                raise ValueError("PNG IDAT chunks must be consecutive")
            seen_idat = True
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            if seen_iend or length != 0:
                raise ValueError("PNG must contain exactly one empty IEND chunk")
            if not seen_idat:
                raise ValueError("PNG requires at least one IDAT chunk")
            seen_iend = True
            if chunk_end != len(payload):
                raise ValueError("PNG IEND must be final with no trailing payload")
        elif chunk_type in _APNG_CHUNKS:
            raise ValueError("APNG chunks are unsupported by the static PNG profile")
        elif chunk_type == b"pHYs":
            if seen_phys or seen_idat or length != 9 or chunk_data[8] not in (0, 1):
                raise ValueError("PNG pHYs chunk is invalid or out of order")
            seen_phys = True
        elif chunk_type[0] & 0x20:
            if chunk_type not in _ALLOWED_ANCILLARY_CHUNKS:
                raise ValueError("PNG contains an unsupported ancillary chunk")
        elif seen_idat:
            idat_closed = True

        offset = chunk_end
        if seen_iend:
            break

    if not seen_ihdr or not seen_idat or not seen_iend or offset != len(payload):
        raise ValueError("PNG chunk stream is incomplete")
    return width, height, expected_mode, b"".join(idat_parts)


def _expected_png_mode(bit_depth: int, color_type: int) -> str:
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError("PNG must use the static 8-bit RGB or RGBA profile")
    return "RGB" if color_type == 2 else "RGBA"


def _validate_png_pixel_stream(
    compressed: bytes,
    *,
    width: int,
    height: int,
    channels: int,
) -> None:
    row_bytes = 1 + width * channels
    expected_bytes = height * row_bytes
    inflater = zlib.decompressobj()
    pending = compressed
    decoded_bytes = 0
    next_filter_offset = 0

    while pending:
        remaining = expected_bytes + 1 - decoded_bytes
        if remaining <= 0:
            raise ValueError("PNG decoded pixel stream exceeds the expected size")
        before = len(pending)
        try:
            decoded = inflater.decompress(pending, min(64 * 1024, remaining))
        except zlib.error as exc:
            raise ValueError("PNG IDAT zlib stream is invalid") from exc
        pending = inflater.unconsumed_tail
        chunk_end = decoded_bytes + len(decoded)
        while next_filter_offset < chunk_end:
            if decoded[next_filter_offset - decoded_bytes] > 4:
                raise ValueError("PNG scanline filter byte is invalid")
            next_filter_offset += row_bytes
        decoded_bytes = chunk_end
        if not decoded and len(pending) == before:
            raise ValueError("PNG IDAT zlib stream made no progress")

    if (
        not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
        or decoded_bytes != expected_bytes
        or next_filter_offset != expected_bytes
    ):
        raise ValueError("PNG IDAT zlib stream has invalid length or trailing data")
