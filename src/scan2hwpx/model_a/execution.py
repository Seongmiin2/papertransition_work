from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from scan2hwpx.contracts import EvidenceIR

from .inference import (
    DEFAULT_MAX_RESPONSE_BYTES,
    BuiltModelADocumentAssembly,
    BuiltModelAInferenceRequest,
    BuiltModelAInferenceResult,
    ModelAModelArtifact,
    ModelAPageImage,
    assemble_model_a_document,
    build_model_a_inference_request,
    validate_model_a_inference_response,
)


class ModelAResponseProvider(Protocol):
    """Provider-neutral boundary that returns one raw response per page request."""

    def generate(self, request: BuiltModelAInferenceRequest, /) -> bytes: ...


def execute_model_a_document(
    evidence_ir: EvidenceIR,
    page_images: Sequence[ModelAPageImage],
    model: ModelAModelArtifact,
    provider: ModelAResponseProvider,
    *,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> BuiltModelADocumentAssembly:
    """Execute exactly one provider call per page, then run the document barrier.

    Every page image and request is validated before the first provider call. Provider
    responses are never repaired or retried; a blocked page result deterministically
    blocks the assembled document. The returned assembly retains every canonical
    request, raw response, validation result, and their digests in ``page_results``.
    """

    response_byte_limit = _require_response_byte_limit(max_response_bytes)
    images = tuple(page_images)
    requests = _preflight_requests(evidence_ir, images, model)
    page_results: list[BuiltModelAInferenceResult] = []
    for request in requests:
        raw_response = provider.generate(request)
        if not isinstance(raw_response, bytes):
            raise TypeError("Model A provider response must be bytes")
        page_results.append(
            validate_model_a_inference_response(
                request,
                raw_response,
                max_response_bytes=response_byte_limit,
            )
        )

    return assemble_model_a_document(evidence_ir, tuple(page_results))


def _require_response_byte_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_response_bytes must be a positive integer")
    if value > DEFAULT_MAX_RESPONSE_BYTES:
        raise ValueError("max_response_bytes cannot exceed the hard response byte limit")
    return value


def _preflight_requests(
    evidence_ir: EvidenceIR,
    page_images: tuple[ModelAPageImage, ...],
    model: ModelAModelArtifact,
) -> tuple[BuiltModelAInferenceRequest, ...]:
    for image in page_images:
        if type(image) is not ModelAPageImage:
            raise TypeError("page_images must contain only ModelAPageImage values")
        ModelAPageImage.__post_init__(image)

    image_ids = tuple(image.page_id for image in page_images)
    duplicates = sorted(page_id for page_id, count in Counter(image_ids).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate page image ids: " + ", ".join(duplicates))

    expected_ids = {page.id for page in evidence_ir.pages}
    actual_ids = set(image_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            "page images must cover every EvidenceIR page exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )

    image_by_page_id = {image.page_id: image for image in page_images}
    pages = tuple(sorted(evidence_ir.pages, key=lambda page: (page.page_no, page.id)))
    return tuple(
        build_model_a_inference_request(
            evidence_ir,
            page_image=image_by_page_id[page.id],
            model=model,
        )
        for page in pages
    )
