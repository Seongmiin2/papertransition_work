from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from scan2hwpx.contracts import ContentIR, HwpDocumentPlan, contract_sha256
from scan2hwpx.evaluation.model_b_plan_review import (
    ModelBOfficialSpecRefEvidence,
    ModelBPlanGroundingEvidence,
)
from scan2hwpx.knowledge.hancom import (
    HWPX_PLAN_RETRIEVAL_ROUTE,
    HWPX_PLAN_RETRIEVAL_ROUTE_ID,
    HancomChunk,
    load_chunks_bytes,
    retrieve_chunks,
    retrieve_hwpx_plan_chunks,
)

CANDIDATE_SCHEMA = "hwp-human-review-candidates/1.0"
SIDECAR_SCHEMA = "model-b-grounding-sidecars/1.0"
RETRIEVAL_ARTIFACT_SCHEMA = "model-b-plan-retrieval-artifact/1.0"
CHUNK_DIGEST_ALGORITHM = "sha256"
CHUNK_CANONICALIZATION = (
    "UTF-8 JSON of the eight record fields; ensure_ascii=false; sort_keys=true; "
    "separators=(',',':'); allow_nan=false; no BOM or trailing newline"
)
CHUNK_RECORD_FIELDS = (
    "id",
    "section",
    "source_id",
    "source_sha256",
    "source_url",
    "tags",
    "text",
    "title",
)

_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_CONTRACT_BYTES = 64 * 1024 * 1024
_MAX_CAPABILITY_BYTES = 1024 * 1024
_MAX_CORPUS_BYTES = 256 * 1024 * 1024
_MAX_RETRIEVAL_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_GROUNDING_EVIDENCE_BYTES = 4 * 1024 * 1024
_MAX_DOCUMENTS = 10_000
_MAX_PAGES_PER_DOCUMENT = 10_000
_MAX_IMAGE_CROPS_PER_DOCUMENT = 100_000
_MAX_ISSUES = 1_000
_MAX_TEXT = 1_024
_MAX_WORKERS = 64

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedText = Annotated[str, Field(min_length=1, max_length=_MAX_TEXT)]
SafeCount = Annotated[int, Field(ge=0, le=(1 << 53) - 1)]
IssueCode = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
Split = Literal["train", "validation", "test"]


class ModelBGroundingSidecarBuildError(ValueError):
    """Raised when immutable Model B grounding sidecars cannot be built safely."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class _ArtifactBinding(_StrictModel):
    path: str = Field(min_length=1, max_length=4_096)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        parts = value.split("/")
        if (
            "\\" in value
            or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or any(
                not part[0].isalnum()
                or any(not (character.isalnum() or character in "._-") for character in part)
                for part in parts
            )
        ):
            raise ValueError("candidate artifact path must be a safe relative path")
        return value


class _ContractArtifactBinding(_ArtifactBinding):
    contract_sha256: Sha256


class _SourceArtifact(_StrictModel):
    sha256: Sha256


class _SourceArtifacts(_StrictModel):
    hwp: _SourceArtifact
    hwpx: _SourceArtifact
    pdf: _SourceArtifact


class _CandidateArtifacts(_StrictModel):
    projection_source: _ArtifactBinding
    evidence_ir: _ContractArtifactBinding
    content_ir_candidate: _ContractArtifactBinding
    hwp_document_plan_candidate: _ContractArtifactBinding
    report: _ArtifactBinding
    page_images: tuple[_ArtifactBinding, ...] = Field(max_length=_MAX_PAGES_PER_DOCUMENT)
    image_crops: tuple[_ArtifactBinding, ...] = Field(max_length=_MAX_IMAGE_CROPS_PER_DOCUMENT)

    def bindings(self) -> Iterator[_ArtifactBinding]:
        yield self.projection_source
        yield self.evidence_ir
        yield self.content_ir_candidate
        yield self.hwp_document_plan_candidate
        yield self.report
        yield from self.page_images
        yield from self.image_crops


class _CandidateDocument(_StrictModel):
    document_id: BoundedText
    lineage_id: Sha256
    split: Split
    page_count: int = Field(gt=0, le=_MAX_PAGES_PER_DOCUMENT)
    status: Literal["needs_human_review"]
    source_artifacts: _SourceArtifacts
    artifacts: _CandidateArtifacts
    issue_codes: tuple[IssueCode, ...] = Field(max_length=_MAX_ISSUES)

    @model_validator(mode="after")
    def validate_document_bindings(self) -> Self:
        if self.document_id != f"candidate-{self.lineage_id[:24]}":
            raise ValueError("candidate document_id does not match lineage_id")
        if self.source_artifacts.hwp.sha256 != self.lineage_id:
            raise ValueError("candidate HWP source digest does not match lineage_id")
        expected_contract_paths = {
            self.artifacts.content_ir_candidate.path: (
                f"{self.lineage_id}/content_ir.candidate.json"
            ),
            self.artifacts.hwp_document_plan_candidate.path: (
                f"{self.lineage_id}/hwp_document_plan.candidate.json"
            ),
        }
        if any(actual != expected for actual, expected in expected_contract_paths.items()):
            raise ValueError("candidate ContentIR/Plan path is not canonical")
        paths = [binding.path for binding in self.artifacts.bindings()]
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("candidate document contains duplicate artifact paths")
        if any(PurePosixPath(path).parts[0] != self.lineage_id for path in paths):
            raise ValueError("candidate artifact path is outside its lineage directory")
        if len(set(self.issue_codes)) != len(self.issue_codes):
            raise ValueError("candidate document contains duplicate issue codes")
        return self


class _RightsManifest(_StrictModel):
    status: Literal["missing"]


class _NamedProducer(_StrictModel):
    name: BoundedText
    version: BoundedText


class _SourceBundle(_StrictModel):
    schema_version: Literal["hwp-hwpx-projection-pairs/1.0"]
    manifest_sha256: Sha256
    source_dataset_report_sha256: Sha256
    source_inventory_sha256: Sha256
    producer: _NamedProducer


class _CandidateProducer(_StrictModel):
    name: BoundedText
    version: BoundedText
    pymupdf_version: BoundedText
    dpi: int = Field(ge=72, le=600)
    pdf_text_order: Literal["native"]


class _CapabilityBinding(_StrictModel):
    id: BoundedText
    sha256: Sha256


class _ContractVersions(_StrictModel):
    projection_source: Literal["hwpx-projection-source/1.0"]
    evidence_ir: Literal["evidence-ir/1.0"]
    content_ir: Literal["content-ir/1.0"]
    hwp_document_plan: Literal["hwp-document-plan/1.0"]


class _SplitCount(_StrictModel):
    documents: SafeCount
    pages: SafeCount


class _CandidateManifest(_StrictModel):
    schema_version: Literal["hwp-human-review-candidates/1.0"]
    artifact_role: Literal["human_review_candidate"]
    research_only: Literal[True]
    human_review_required: Literal[True]
    golden_eligible: Literal[False]
    training_eligible: Literal[False]
    release_eligible: Literal[False]
    rights_manifest: _RightsManifest
    source_bundle: _SourceBundle
    producer: _CandidateProducer
    capability_profile: _CapabilityBinding
    knowledge_corpus_sha256: Sha256
    contract_versions: _ContractVersions
    document_count: int = Field(gt=0, le=_MAX_DOCUMENTS)
    page_count: int = Field(gt=0, le=_MAX_DOCUMENTS * _MAX_PAGES_PER_DOCUMENT)
    splits: dict[Split, _SplitCount]
    issue_counts: dict[IssueCode, SafeCount] = Field(max_length=_MAX_ISSUES)
    documents: tuple[_CandidateDocument, ...] = Field(
        min_length=1,
        max_length=_MAX_DOCUMENTS,
    )

    @model_validator(mode="after")
    def validate_counts_and_uniqueness(self) -> Self:
        if self.document_count != len(self.documents):
            raise ValueError("candidate document_count does not match documents")
        if self.page_count != sum(document.page_count for document in self.documents):
            raise ValueError("candidate page_count does not match documents")
        if set(self.splits) != {"train", "validation", "test"}:
            raise ValueError("candidate splits must contain train, validation, and test")
        for split in ("train", "validation", "test"):
            selected = [document for document in self.documents if document.split == split]
            expected = _SplitCount(
                documents=len(selected),
                pages=sum(document.page_count for document in selected),
            )
            if self.splits[split] != expected:
                raise ValueError(f"candidate split counts do not match for {split}")
        expected_issues = Counter(
            issue for document in self.documents for issue in document.issue_codes
        )
        if self.issue_counts != dict(expected_issues):
            raise ValueError("candidate issue_counts do not match documents")
        document_ids = [document.document_id.casefold() for document in self.documents]
        lineages = [document.lineage_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("candidate manifest contains duplicate document ids")
        if len(set(lineages)) != len(lineages):
            raise ValueError("candidate manifest contains duplicate lineages")
        paths = [
            binding.path.casefold()
            for document in self.documents
            for binding in document.artifacts.bindings()
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("candidate manifest contains duplicate artifact paths")
        return self


class _CapabilityProfile(_StrictModel):
    schema_version: Literal["1.0"]
    profile_id: BoundedText
    target_format: Literal["hwpx"]
    knowledge_manifest: BoundedText
    content_policy: Literal["immutable"]
    layout_policy: Literal["clean_reauthor"]
    planner_may_emit: tuple[BoundedText, ...] = Field(min_length=1, max_length=32)
    planner_must_reference_content: Literal[True]
    compiler_owns: tuple[BoundedText, ...] = Field(min_length=1, max_length=128)
    planner_forbidden: tuple[BoundedText, ...] = Field(min_length=1, max_length=128)
    requires_review: tuple[BoundedText, ...] = Field(max_length=128)
    retrieval_routes: dict[BoundedText, tuple[BoundedText, ...]] = Field(
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_policy_sets(self) -> Self:
        for label, values in (
            ("planner_may_emit", self.planner_may_emit),
            ("compiler_owns", self.compiler_owns),
            ("planner_forbidden", self.planner_forbidden),
            ("requires_review", self.requires_review),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"capability profile {label} contains duplicates")
        required_forbidden = {"raw_xml", "local_file_path", "external_url"}
        if not required_forbidden.issubset(self.planner_forbidden):
            raise ValueError("capability profile lacks required planner prohibitions")
        if any(not tags or len(set(tags)) != len(tags) for tags in self.retrieval_routes.values()):
            raise ValueError("capability profile retrieval routes must be nonempty and unique")
        return self


class _CandidateBundleBinding(_StrictModel):
    schema_version: Literal["hwp-human-review-candidates/1.0"]
    manifest_sha256: Sha256


class _CandidateContractArtifacts(_StrictModel):
    content_ir_candidate: _ContractArtifactBinding
    hwp_document_plan_candidate: _ContractArtifactBinding


class _SidecarArtifacts(_StrictModel):
    retrieval: _ArtifactBinding
    grounding_evidence: _ContractArtifactBinding


class _RetrievedChunkDigest(_StrictModel):
    algorithm: Literal["sha256"]
    canonical_record_fields: tuple[BoundedText, ...]
    canonicalization: BoundedText


class _RetrievalSummary(_StrictModel):
    route_id: Literal["hwpx-projection-candidate-official-refs/1.0"]
    algorithm: Literal["scan2hwpx.knowledge.hancom.retrieve_chunks/1.0"]
    deduplication: Literal["first_occurrence_by_chunk_id"]
    retrieved_chunk_digest: _RetrievedChunkDigest
    retrieved_chunk_count: int = Field(gt=0, le=1_000)


class _SidecarDocument(_StrictModel):
    document_id: BoundedText
    lineage_id: Sha256
    split: Split
    status: Literal["unverified_requires_human_review"]
    research_only: Literal[True]
    human_review_required: Literal[True]
    golden_eligible: Literal[False]
    training_eligible: Literal[False]
    release_eligible: Literal[False]
    rights_status: Literal["unverified"]
    identity_assurance: Literal["self_asserted_untrusted"]
    candidate_artifacts: _CandidateContractArtifacts
    artifacts: _SidecarArtifacts

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.document_id != f"candidate-{self.lineage_id[:24]}":
            raise ValueError("sidecar document_id does not match lineage_id")
        if self.artifacts.retrieval.path != f"{self.lineage_id}/retrieval.json":
            raise ValueError("sidecar retrieval path is not canonical")
        if self.artifacts.grounding_evidence.path != f"{self.lineage_id}/grounding_evidence.json":
            raise ValueError("sidecar grounding evidence path is not canonical")
        return self


class _SidecarManifest(_StrictModel):
    schema_version: Literal["model-b-grounding-sidecars/1.0"]
    artifact_role: Literal["model_b_plan_review_grounding"]
    research_only: Literal[True]
    human_review_required: Literal[True]
    golden_eligible: Literal[False]
    training_eligible: Literal[False]
    release_eligible: Literal[False]
    rights_status: Literal["unverified"]
    identity_assurance: Literal["self_asserted_untrusted"]
    candidate_bundle: _CandidateBundleBinding
    knowledge_corpus_sha256: Sha256
    capability_profile: _CapabilityBinding
    retrieval: _RetrievalSummary
    document_count: int = Field(gt=0, le=_MAX_DOCUMENTS)
    documents: tuple[_SidecarDocument, ...] = Field(
        min_length=1,
        max_length=_MAX_DOCUMENTS,
    )

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        if self.document_count != len(self.documents):
            raise ValueError("sidecar document_count does not match documents")
        lineages = [document.lineage_id for document in self.documents]
        document_ids = [document.document_id.casefold() for document in self.documents]
        if len(set(lineages)) != len(lineages):
            raise ValueError("sidecar manifest contains duplicate lineages")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("sidecar manifest contains duplicate document ids")
        paths = [
            binding.path.casefold()
            for document in self.documents
            for binding in (
                document.artifacts.retrieval,
                document.artifacts.grounding_evidence,
            )
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("sidecar manifest contains duplicate artifact paths")
        return self


@dataclass(frozen=True, slots=True)
class _RetrievalContext:
    selected: tuple[HancomChunk, ...]
    route_results: tuple[dict[str, object], ...]
    serialized_chunks: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _InputSnapshot:
    label: str
    path: Path
    maximum_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _DocumentBuild:
    lineage_id: str
    manifest_record: dict[str, object]
    input_snapshots: tuple[_InputSnapshot, ...]


@dataclass(frozen=True, slots=True)
class VerifiedModelBGroundingSidecar:
    """Typed contracts returned only after full candidate and sidecar rebinding."""

    document_id: str
    lineage_id: str
    candidate_manifest_sha256: str
    content_ir: ContentIR
    hwp_document_plan: HwpDocumentPlan
    grounding_evidence: ModelBPlanGroundingEvidence


def canonical_hancom_chunk_record_bytes(chunk: HancomChunk) -> bytes:
    """Return the exact canonical bytes hashed as a retrieved-chunk digest.

    The digest excludes every sidecar wrapper field. It covers only the eight
    validated HancomChunk fields represented by CHUNK_RECORD_FIELDS.
    """
    if not isinstance(chunk, HancomChunk):
        raise TypeError("chunk must be a HancomChunk")
    record = _chunk_record(chunk)
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def retrieved_chunk_sha256(chunk: HancomChunk) -> str:
    return hashlib.sha256(canonical_hancom_chunk_record_bytes(chunk)).hexdigest()


def _chunk_record(chunk: HancomChunk) -> dict[str, object]:
    return {
        "id": chunk.id,
        "source_id": chunk.source_id,
        "title": chunk.title,
        "section": chunk.section,
        "text": chunk.text,
        "tags": list(chunk.tags),
        "source_url": chunk.source_url,
        "source_sha256": chunk.source_sha256,
    }


def _build_retrieval_context(chunks: tuple[HancomChunk, ...]) -> _RetrievalContext:
    route_results: list[dict[str, object]] = []
    for query, tags, limit in HWPX_PLAN_RETRIEVAL_ROUTE:
        matches = retrieve_chunks(query, tags=tags, limit=limit, chunks=chunks)
        route_results.append(
            {
                "query": query,
                "tags": list(tags),
                "limit": limit,
                "retrieved_chunk_ids": [chunk.id for chunk in matches],
            }
        )
    selected = retrieve_hwpx_plan_chunks(chunks=chunks)
    serialized: tuple[dict[str, object], ...] = tuple(
        {
            "record": _chunk_record(chunk),
            "retrieved_chunk_sha256": retrieved_chunk_sha256(chunk),
        }
        for chunk in selected
    )
    return _RetrievalContext(
        selected=selected,
        route_results=tuple(route_results),
        serialized_chunks=serialized,
    )


def load_verified_model_b_grounding_sidecar(
    candidate_bundle_root: Path | str,
    sidecar_bundle_root: Path | str,
    knowledge_corpus_path: Path | str,
    capability_profile_path: Path | str,
    *,
    lineage_id: str,
) -> VerifiedModelBGroundingSidecar:
    """Return typed handoff contracts only after rebinding every pinned input.

    Callers must use this verifier instead of loading grounding_evidence.json
    directly. The returned evidence is tied back to the current candidate
    manifest, registered ContentIR/Plan, exact corpus, exact capability profile,
    deterministic retrieval bytes, and complete sidecar inventory.
    """
    if (
        not isinstance(lineage_id, str)
        or len(lineage_id) != 64
        or any(character not in "0123456789abcdef" for character in lineage_id)
    ):
        raise ValueError("lineage_id must be a lowercase SHA-256 digest")

    candidate_root = _required_directory(Path(candidate_bundle_root), "candidate bundle")
    candidate_manifest_path = _resolve_registered_file(
        candidate_root,
        "manifest.json",
        "candidate manifest",
    )
    corpus_path = _required_regular_file(
        Path(knowledge_corpus_path),
        "knowledge corpus",
    )
    capability_path = _required_regular_file(
        Path(capability_profile_path),
        "capability profile",
    )
    candidate_manifest_bytes = _read_bounded_file(
        candidate_manifest_path,
        _MAX_MANIFEST_BYTES,
        "candidate manifest",
    )
    corpus_bytes = _read_bounded_file(
        corpus_path,
        _MAX_CORPUS_BYTES,
        "knowledge corpus",
    )
    capability_bytes = _read_bounded_file(
        capability_path,
        _MAX_CAPABILITY_BYTES,
        "capability profile",
    )
    candidate_manifest = _parse_candidate_manifest(candidate_manifest_bytes)
    capability = _parse_capability_profile(capability_bytes)
    candidate_manifest_sha256 = _sha256_bytes(candidate_manifest_bytes)
    corpus_sha256 = _sha256_bytes(corpus_bytes)
    capability_sha256 = _sha256_bytes(capability_bytes)
    _assert_candidate_reference_bindings(
        candidate_manifest,
        capability,
        corpus_sha256,
        capability_sha256,
    )
    chunks = load_chunks_bytes(corpus_bytes)
    retrieval = _build_retrieval_context(chunks)

    sidecar_root = _required_directory(
        Path(sidecar_bundle_root),
        "grounding sidecar bundle",
    )
    sidecar_manifest_path = _resolve_registered_file(
        sidecar_root,
        "manifest.json",
        "grounding sidecar manifest",
    )
    sidecar_manifest_bytes = _read_bounded_file(
        sidecar_manifest_path,
        _MAX_MANIFEST_BYTES,
        "grounding sidecar manifest",
    )
    _assert_strict_json(sidecar_manifest_bytes, "grounding sidecar manifest")
    try:
        sidecar_manifest = _SidecarManifest.model_validate_json(
            sidecar_manifest_bytes,
            strict=True,
        )
    except ValidationError as exc:
        raise ModelBGroundingSidecarBuildError(
            "grounding sidecar manifest is not valid strict JSON"
        ) from exc
    if sidecar_manifest.candidate_bundle.manifest_sha256 != candidate_manifest_sha256:
        raise ModelBGroundingSidecarBuildError(
            "grounding sidecar candidate manifest binding does not match"
        )
    if sidecar_manifest.knowledge_corpus_sha256 != corpus_sha256:
        raise ModelBGroundingSidecarBuildError(
            "grounding sidecar knowledge corpus binding does not match"
        )
    if (
        sidecar_manifest.capability_profile.id != capability.profile_id
        or sidecar_manifest.capability_profile.sha256 != capability_sha256
    ):
        raise ModelBGroundingSidecarBuildError(
            "grounding sidecar capability profile binding does not match"
        )
    digest_metadata = sidecar_manifest.retrieval.retrieved_chunk_digest
    if (
        sidecar_manifest.retrieval.retrieved_chunk_count != len(retrieval.selected)
        or digest_metadata.algorithm != CHUNK_DIGEST_ALGORITHM
        or tuple(digest_metadata.canonical_record_fields) != CHUNK_RECORD_FIELDS
        or digest_metadata.canonicalization != CHUNK_CANONICALIZATION
    ):
        raise ModelBGroundingSidecarBuildError(
            "grounding sidecar retrieval metadata does not match implementation"
        )

    candidate_by_lineage = {
        document.lineage_id: document for document in candidate_manifest.documents
    }
    sidecar_by_lineage = {document.lineage_id: document for document in sidecar_manifest.documents}
    if set(candidate_by_lineage) != set(sidecar_by_lineage):
        raise ModelBGroundingSidecarBuildError(
            "grounding sidecar document inventory does not match candidate manifest"
        )
    for current_lineage, registered_candidate in candidate_by_lineage.items():
        registered_sidecar = sidecar_by_lineage[current_lineage]
        if (
            registered_sidecar.document_id != registered_candidate.document_id
            or registered_sidecar.split != registered_candidate.split
            or registered_sidecar.candidate_artifacts.content_ir_candidate
            != registered_candidate.artifacts.content_ir_candidate
            or registered_sidecar.candidate_artifacts.hwp_document_plan_candidate
            != registered_candidate.artifacts.hwp_document_plan_candidate
        ):
            raise ModelBGroundingSidecarBuildError(
                "grounding sidecar candidate artifact binding does not match"
            )
    _assert_existing_sidecar_inventory(sidecar_root, sidecar_manifest)

    candidate_document = candidate_by_lineage.get(lineage_id)
    sidecar_document = sidecar_by_lineage.get(lineage_id)
    if candidate_document is None or sidecar_document is None:
        raise ModelBGroundingSidecarBuildError(
            "lineage_id is not registered in candidate and sidecar manifests"
        )
    content, plan, candidate_snapshots = _load_verified_candidate_contracts(
        candidate_document,
        candidate_root,
        capability.profile_id,
        tuple(chunk.id for chunk in retrieval.selected),
    )
    retrieval_path = _resolve_registered_file(
        sidecar_root,
        sidecar_document.artifacts.retrieval.path,
        "retrieval artifact",
    )
    evidence_path = _resolve_registered_file(
        sidecar_root,
        sidecar_document.artifacts.grounding_evidence.path,
        "grounding evidence",
    )
    retrieval_bytes = _read_bounded_file(
        retrieval_path,
        _MAX_RETRIEVAL_ARTIFACT_BYTES,
        "retrieval artifact",
    )
    evidence_bytes = _read_bounded_file(
        evidence_path,
        _MAX_GROUNDING_EVIDENCE_BYTES,
        "grounding evidence",
    )
    retrieval_sha256 = _sha256_bytes(retrieval_bytes)
    if retrieval_sha256 != sidecar_document.artifacts.retrieval.sha256:
        raise ModelBGroundingSidecarBuildError(
            "retrieval artifact SHA-256 does not match sidecar manifest"
        )
    if _sha256_bytes(evidence_bytes) != sidecar_document.artifacts.grounding_evidence.sha256:
        raise ModelBGroundingSidecarBuildError(
            "grounding evidence SHA-256 does not match sidecar manifest"
        )
    _assert_strict_json(retrieval_bytes, "retrieval artifact")
    _assert_strict_json(evidence_bytes, "grounding evidence")
    actual_retrieval = json.loads(retrieval_bytes.decode("utf-8"))
    expected_retrieval = _retrieval_artifact_value(
        candidate_document,
        candidate_manifest_sha256,
        corpus_sha256,
        capability.profile_id,
        capability_sha256,
        retrieval,
    )
    if actual_retrieval != expected_retrieval:
        raise ModelBGroundingSidecarBuildError(
            "retrieval artifact cannot be reproduced from pinned inputs"
        )
    try:
        evidence = ModelBPlanGroundingEvidence.model_validate_json(
            evidence_bytes,
            strict=True,
        )
    except ValidationError as exc:
        raise ModelBGroundingSidecarBuildError(
            "grounding evidence is not valid strict JSON"
        ) from exc
    if contract_sha256(evidence) != sidecar_document.artifacts.grounding_evidence.contract_sha256:
        raise ModelBGroundingSidecarBuildError(
            "grounding evidence contract SHA-256 does not match sidecar manifest"
        )
    expected_evidence = _grounding_evidence(
        corpus_sha256,
        retrieval_sha256,
        capability.profile_id,
        capability_sha256,
        retrieval,
    )
    if evidence != expected_evidence:
        raise ModelBGroundingSidecarBuildError(
            "grounding evidence cannot be reproduced from pinned inputs"
        )

    snapshots = [
        _InputSnapshot(
            "candidate manifest",
            candidate_manifest_path,
            _MAX_MANIFEST_BYTES,
            candidate_manifest_sha256,
        ),
        _InputSnapshot(
            "knowledge corpus",
            corpus_path,
            _MAX_CORPUS_BYTES,
            corpus_sha256,
        ),
        _InputSnapshot(
            "capability profile",
            capability_path,
            _MAX_CAPABILITY_BYTES,
            capability_sha256,
        ),
        _InputSnapshot(
            "grounding sidecar manifest",
            sidecar_manifest_path,
            _MAX_MANIFEST_BYTES,
            _sha256_bytes(sidecar_manifest_bytes),
        ),
        _InputSnapshot(
            "retrieval artifact",
            retrieval_path,
            _MAX_RETRIEVAL_ARTIFACT_BYTES,
            retrieval_sha256,
        ),
        _InputSnapshot(
            "grounding evidence",
            evidence_path,
            _MAX_GROUNDING_EVIDENCE_BYTES,
            _sha256_bytes(evidence_bytes),
        ),
        *candidate_snapshots,
    ]
    _assert_inputs_unchanged(snapshots)
    return VerifiedModelBGroundingSidecar(
        document_id=candidate_document.document_id,
        lineage_id=lineage_id,
        candidate_manifest_sha256=candidate_manifest_sha256,
        content_ir=content,
        hwp_document_plan=plan,
        grounding_evidence=evidence,
    )


def build_model_b_grounding_sidecars(
    candidate_bundle_root: Path | str,
    knowledge_corpus_path: Path | str,
    capability_profile_path: Path | str,
    output_root: Path | str,
    *,
    max_workers: int,
) -> dict[str, Any]:
    """Build a create-only grounding bundle without mutating candidate inputs."""
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= _MAX_WORKERS
    ):
        raise ValueError(f"max_workers must be an integer between 1 and {_MAX_WORKERS}")

    candidate_root = _required_directory(
        Path(candidate_bundle_root),
        "candidate bundle",
    )
    candidate_manifest_path = _resolve_registered_file(
        candidate_root,
        "manifest.json",
        "candidate manifest",
    )
    corpus_path = _required_regular_file(
        Path(knowledge_corpus_path),
        "knowledge corpus",
    )
    capability_path = _required_regular_file(
        Path(capability_profile_path),
        "capability profile",
    )
    candidate_manifest_bytes = _read_bounded_file(
        candidate_manifest_path,
        _MAX_MANIFEST_BYTES,
        "candidate manifest",
    )
    corpus_bytes = _read_bounded_file(
        corpus_path,
        _MAX_CORPUS_BYTES,
        "knowledge corpus",
    )
    capability_bytes = _read_bounded_file(
        capability_path,
        _MAX_CAPABILITY_BYTES,
        "capability profile",
    )
    candidate_manifest = _parse_candidate_manifest(candidate_manifest_bytes)
    capability = _parse_capability_profile(capability_bytes)

    candidate_manifest_sha256 = _sha256_bytes(candidate_manifest_bytes)
    corpus_sha256 = _sha256_bytes(corpus_bytes)
    capability_sha256 = _sha256_bytes(capability_bytes)
    _assert_candidate_reference_bindings(
        candidate_manifest,
        capability,
        corpus_sha256,
        capability_sha256,
    )

    chunks = load_chunks_bytes(corpus_bytes)
    retrieval = _build_retrieval_context(chunks)
    destination = Path(output_root).resolve()
    _assert_output_is_separate(
        destination,
        (candidate_root, corpus_path.parent, capability_path.parent),
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
        )
    )
    published = False
    try:
        builds: dict[str, _DocumentBuild] = {}
        failures: dict[str, Exception] = {}
        documents = tuple(
            sorted(candidate_manifest.documents, key=lambda document: document.lineage_id)
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_lineage = {
                executor.submit(
                    _build_document_sidecars,
                    document,
                    candidate_root,
                    candidate_manifest_sha256,
                    corpus_sha256,
                    capability,
                    capability_sha256,
                    retrieval,
                    staging,
                ): document.lineage_id
                for document in documents
            }
            for future in as_completed(future_to_lineage):
                lineage_id = future_to_lineage[future]
                try:
                    builds[lineage_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures[lineage_id] = exc
        if failures:
            failed = ", ".join(sorted(failures))
            first = failures[min(failures)]
            raise ModelBGroundingSidecarBuildError(
                f"grounding sidecar build failed for candidate lineages: {failed}"
            ) from first

        ordered = [builds[document.lineage_id] for document in documents]
        sidecar_manifest: dict[str, Any] = {
            "schema_version": SIDECAR_SCHEMA,
            "artifact_role": "model_b_plan_review_grounding",
            "research_only": True,
            "human_review_required": True,
            "golden_eligible": False,
            "training_eligible": False,
            "release_eligible": False,
            "rights_status": "unverified",
            "identity_assurance": "self_asserted_untrusted",
            "candidate_bundle": {
                "schema_version": CANDIDATE_SCHEMA,
                "manifest_sha256": candidate_manifest_sha256,
            },
            "knowledge_corpus_sha256": corpus_sha256,
            "capability_profile": {
                "id": capability.profile_id,
                "sha256": capability_sha256,
            },
            "retrieval": {
                "route_id": HWPX_PLAN_RETRIEVAL_ROUTE_ID,
                "algorithm": "scan2hwpx.knowledge.hancom.retrieve_chunks/1.0",
                "deduplication": "first_occurrence_by_chunk_id",
                "retrieved_chunk_digest": {
                    "algorithm": CHUNK_DIGEST_ALGORITHM,
                    "canonical_record_fields": list(CHUNK_RECORD_FIELDS),
                    "canonicalization": CHUNK_CANONICALIZATION,
                },
                "retrieved_chunk_count": len(retrieval.selected),
            },
            "document_count": len(ordered),
            "documents": [item.manifest_record for item in ordered],
        }
        manifest_payload = _json_bytes(sidecar_manifest)
        if len(manifest_payload) > _MAX_MANIFEST_BYTES:
            raise ModelBGroundingSidecarBuildError(
                "grounding sidecar manifest exceeds the size limit"
            )
        _write_bytes(staging / "manifest.json", manifest_payload)
        _assert_sidecar_inventory(staging, ordered)
        snapshots = [
            _InputSnapshot(
                "candidate manifest",
                candidate_manifest_path,
                _MAX_MANIFEST_BYTES,
                candidate_manifest_sha256,
            ),
            _InputSnapshot(
                "knowledge corpus",
                corpus_path,
                _MAX_CORPUS_BYTES,
                corpus_sha256,
            ),
            _InputSnapshot(
                "capability profile",
                capability_path,
                _MAX_CAPABILITY_BYTES,
                capability_sha256,
            ),
        ]
        snapshots.extend(snapshot for item in ordered for snapshot in item.input_snapshots)
        _assert_inputs_unchanged(snapshots)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output appeared during build: {destination}")
        _publish_directory_create_only(staging, destination)
        published = True
        return sidecar_manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _load_verified_candidate_contracts(
    document: _CandidateDocument,
    candidate_root: Path,
    capability_profile_id: str,
    retrieved_ids: tuple[str, ...],
) -> tuple[ContentIR, HwpDocumentPlan, tuple[_InputSnapshot, ...]]:
    content_binding = document.artifacts.content_ir_candidate
    plan_binding = document.artifacts.hwp_document_plan_candidate
    content_path = _resolve_registered_file(
        candidate_root,
        content_binding.path,
        "ContentIR candidate",
    )
    plan_path = _resolve_registered_file(
        candidate_root,
        plan_binding.path,
        "HwpDocumentPlan candidate",
    )
    content_bytes = _read_bounded_file(
        content_path,
        _MAX_CONTRACT_BYTES,
        "ContentIR candidate",
    )
    plan_bytes = _read_bounded_file(
        plan_path,
        _MAX_CONTRACT_BYTES,
        "HwpDocumentPlan candidate",
    )
    if _sha256_bytes(content_bytes) != content_binding.sha256:
        raise ModelBGroundingSidecarBuildError("registered ContentIR artifact SHA-256 mismatch")
    if _sha256_bytes(plan_bytes) != plan_binding.sha256:
        raise ModelBGroundingSidecarBuildError(
            "registered HwpDocumentPlan artifact SHA-256 mismatch"
        )
    _assert_strict_json(content_bytes, "ContentIR candidate")
    _assert_strict_json(plan_bytes, "HwpDocumentPlan candidate")
    try:
        content = ContentIR.model_validate_json(content_bytes, strict=True)
        plan = HwpDocumentPlan.model_validate_json(plan_bytes, strict=True)
    except ValidationError as exc:
        raise ModelBGroundingSidecarBuildError(
            "registered ContentIR/Plan artifact is not valid strict JSON"
        ) from exc
    if contract_sha256(content) != content_binding.contract_sha256:
        raise ModelBGroundingSidecarBuildError("registered ContentIR contract SHA-256 mismatch")
    if contract_sha256(plan) != plan_binding.contract_sha256:
        raise ModelBGroundingSidecarBuildError(
            "registered HwpDocumentPlan contract SHA-256 mismatch"
        )
    try:
        plan.assert_content_integrity(content)
    except ValueError as exc:
        raise ModelBGroundingSidecarBuildError(
            "registered ContentIR/Plan contract lineage mismatch"
        ) from exc
    if plan.capability_profile_id != capability_profile_id:
        raise ModelBGroundingSidecarBuildError(
            "plan capability profile does not match the pinned input"
        )
    if tuple(plan.official_spec_refs) != retrieved_ids:
        raise ModelBGroundingSidecarBuildError(
            "plan official_spec_refs do not exactly match the pinned retrieval route"
        )
    snapshots = (
        _InputSnapshot(
            f"{document.lineage_id} ContentIR candidate",
            content_path,
            _MAX_CONTRACT_BYTES,
            content_binding.sha256,
        ),
        _InputSnapshot(
            f"{document.lineage_id} HwpDocumentPlan candidate",
            plan_path,
            _MAX_CONTRACT_BYTES,
            plan_binding.sha256,
        ),
    )
    return content, plan, snapshots


def _retrieval_artifact_value(
    document: _CandidateDocument,
    candidate_manifest_sha256: str,
    corpus_sha256: str,
    capability_profile_id: str,
    capability_sha256: str,
    retrieval: _RetrievalContext,
) -> dict[str, object]:
    return {
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA,
        "artifact_role": "model_b_plan_retrieval_context",
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "rights_status": "unverified",
        "identity_assurance": "self_asserted_untrusted",
        "document_id": document.document_id,
        "lineage_id": document.lineage_id,
        "candidate_bundle": {
            "schema_version": CANDIDATE_SCHEMA,
            "manifest_sha256": candidate_manifest_sha256,
        },
        "candidate_artifacts": {
            "content_ir_candidate": (
                document.artifacts.content_ir_candidate.model_dump(mode="json")
            ),
            "hwp_document_plan_candidate": (
                document.artifacts.hwp_document_plan_candidate.model_dump(mode="json")
            ),
        },
        "knowledge_corpus_sha256": corpus_sha256,
        "capability_profile": {
            "id": capability_profile_id,
            "sha256": capability_sha256,
        },
        "retrieval_route": {
            "id": HWPX_PLAN_RETRIEVAL_ROUTE_ID,
            "algorithm": "scan2hwpx.knowledge.hancom.retrieve_chunks/1.0",
            "deduplication": "first_occurrence_by_chunk_id",
            "queries": list(retrieval.route_results),
        },
        "retrieved_chunk_digest": {
            "algorithm": CHUNK_DIGEST_ALGORITHM,
            "canonical_record_fields": list(CHUNK_RECORD_FIELDS),
            "canonicalization": CHUNK_CANONICALIZATION,
        },
        "retrieved_chunks": list(retrieval.serialized_chunks),
    }


def _grounding_evidence(
    corpus_sha256: str,
    retrieval_artifact_sha256: str,
    capability_profile_id: str,
    capability_sha256: str,
    retrieval: _RetrievalContext,
) -> ModelBPlanGroundingEvidence:
    return ModelBPlanGroundingEvidence(
        knowledge_corpus_sha256=corpus_sha256,
        retrieval_artifact_sha256=retrieval_artifact_sha256,
        capability_profile_id=capability_profile_id,
        capability_profile_sha256=capability_sha256,
        official_spec_refs=tuple(
            ModelBOfficialSpecRefEvidence(
                official_spec_ref=chunk.id,
                source_document_sha256=chunk.source_sha256,
                retrieved_chunk_sha256=retrieved_chunk_sha256(chunk),
            )
            for chunk in retrieval.selected
        ),
    )


def _build_document_sidecars(
    document: _CandidateDocument,
    candidate_root: Path,
    candidate_manifest_sha256: str,
    corpus_sha256: str,
    capability: _CapabilityProfile,
    capability_sha256: str,
    retrieval: _RetrievalContext,
    staging: Path,
) -> _DocumentBuild:
    content_binding = document.artifacts.content_ir_candidate
    plan_binding = document.artifacts.hwp_document_plan_candidate
    _, _, input_snapshots = _load_verified_candidate_contracts(
        document,
        candidate_root,
        capability.profile_id,
        tuple(chunk.id for chunk in retrieval.selected),
    )

    document_dir = staging / document.lineage_id
    document_dir.mkdir()
    retrieval_value = _retrieval_artifact_value(
        document,
        candidate_manifest_sha256,
        corpus_sha256,
        capability.profile_id,
        capability_sha256,
        retrieval,
    )
    retrieval_binding = _write_json_artifact(
        document_dir / "retrieval.json",
        retrieval_value,
        staging,
        maximum_bytes=_MAX_RETRIEVAL_ARTIFACT_BYTES,
        label="retrieval artifact",
    )
    evidence = _grounding_evidence(
        corpus_sha256,
        retrieval_binding["sha256"],
        capability.profile_id,
        capability_sha256,
        retrieval,
    )
    evidence_binding = _write_contract_artifact(
        document_dir / "grounding_evidence.json",
        evidence,
        staging,
        maximum_bytes=_MAX_GROUNDING_EVIDENCE_BYTES,
        label="grounding evidence",
    )
    manifest_record = {
        "document_id": document.document_id,
        "lineage_id": document.lineage_id,
        "split": document.split,
        "status": "unverified_requires_human_review",
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "rights_status": "unverified",
        "identity_assurance": "self_asserted_untrusted",
        "candidate_artifacts": {
            "content_ir_candidate": content_binding.model_dump(mode="json"),
            "hwp_document_plan_candidate": plan_binding.model_dump(mode="json"),
        },
        "artifacts": {
            "retrieval": retrieval_binding,
            "grounding_evidence": evidence_binding,
        },
    }
    return _DocumentBuild(
        lineage_id=document.lineage_id,
        manifest_record=manifest_record,
        input_snapshots=input_snapshots,
    )


def _write_json_artifact(
    path: Path,
    value: dict[str, object],
    root: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> dict[str, str]:
    payload = _json_bytes(value)
    if len(payload) > maximum_bytes:
        raise ModelBGroundingSidecarBuildError(f"{label} exceeds the size limit")
    _write_bytes(path, payload)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_bytes(payload),
    }


def _write_contract_artifact(
    path: Path,
    contract: ModelBPlanGroundingEvidence,
    root: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> dict[str, str]:
    binding = _write_json_artifact(
        path,
        contract.model_dump(mode="json"),
        root,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    binding["contract_sha256"] = contract_sha256(contract)
    return binding


def _write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"sidecar artifact already exists: {path.name}")
    with path.open("xb") as stream:
        stream.write(payload)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _assert_sidecar_inventory(
    root: Path,
    builds: list[_DocumentBuild],
) -> None:
    expected: dict[str, str] = {}
    for build in builds:
        artifacts = build.manifest_record["artifacts"]
        if not isinstance(artifacts, dict):
            raise TypeError("generated sidecar artifacts must be an object")
        for name in ("retrieval", "grounding_evidence"):
            binding = artifacts[name]
            if not isinstance(binding, dict):
                raise TypeError("generated sidecar binding must be an object")
            path = binding.get("path")
            sha256 = binding.get("sha256")
            if not isinstance(path, str) or not isinstance(sha256, str):
                raise TypeError("generated sidecar binding is invalid")
            if path in expected:
                raise ValueError("generated sidecar contains duplicate artifact paths")
            expected[path] = sha256

    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if any(path.is_symlink() for path in entries):
        raise ValueError("grounding sidecar bundle must not contain symlinks")
    actual = {path.relative_to(root).as_posix(): path for path in entries if path.is_file()}
    expected_paths = {"manifest.json", *expected}
    if set(actual) != expected_paths:
        missing = len(expected_paths - set(actual))
        unexpected = len(set(actual) - expected_paths)
        raise ValueError(
            f"grounding sidecar inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    for relative_path, expected_sha256 in expected.items():
        if _sha256_file(actual[relative_path]) != expected_sha256:
            raise ValueError(f"grounding sidecar SHA-256 mismatch: {relative_path}")


def _assert_existing_sidecar_inventory(
    root: Path,
    manifest: _SidecarManifest,
) -> None:
    expected: dict[str, tuple[str, int, str]] = {}
    for document in manifest.documents:
        for label, binding, maximum_bytes in (
            (
                "retrieval artifact",
                document.artifacts.retrieval,
                _MAX_RETRIEVAL_ARTIFACT_BYTES,
            ),
            (
                "grounding evidence",
                document.artifacts.grounding_evidence,
                _MAX_GROUNDING_EVIDENCE_BYTES,
            ),
        ):
            expected[binding.path] = (binding.sha256, maximum_bytes, label)
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if any(path.is_symlink() for path in entries):
        raise ModelBGroundingSidecarBuildError("grounding sidecar bundle must not contain symlinks")
    actual = {path.relative_to(root).as_posix(): path for path in entries if path.is_file()}
    expected_paths = {"manifest.json", *expected}
    if set(actual) != expected_paths:
        missing = len(expected_paths - set(actual))
        unexpected = len(set(actual) - expected_paths)
        raise ModelBGroundingSidecarBuildError(
            f"grounding sidecar inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    for relative_path, (expected_sha256, maximum_bytes, label) in expected.items():
        payload = _read_bounded_file(
            actual[relative_path],
            maximum_bytes,
            label,
        )
        if _sha256_bytes(payload) != expected_sha256:
            raise ModelBGroundingSidecarBuildError(
                f"grounding sidecar SHA-256 mismatch: {relative_path}"
            )


def _parse_candidate_manifest(payload: bytes) -> _CandidateManifest:
    _assert_strict_json(payload, "candidate manifest")
    try:
        return _CandidateManifest.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ModelBGroundingSidecarBuildError(
            "candidate manifest is not valid strict JSON"
        ) from exc


def _parse_capability_profile(payload: bytes) -> _CapabilityProfile:
    _assert_strict_json(payload, "capability profile")
    try:
        return _CapabilityProfile.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ModelBGroundingSidecarBuildError(
            "capability profile is not valid strict JSON"
        ) from exc


def _assert_candidate_reference_bindings(
    candidate_manifest: _CandidateManifest,
    capability: _CapabilityProfile,
    corpus_sha256: str,
    capability_sha256: str,
) -> None:
    if candidate_manifest.knowledge_corpus_sha256 != corpus_sha256:
        raise ModelBGroundingSidecarBuildError(
            "candidate manifest knowledge corpus SHA-256 does not match input"
        )
    if candidate_manifest.capability_profile.sha256 != capability_sha256:
        raise ModelBGroundingSidecarBuildError(
            "candidate manifest capability profile SHA-256 does not match input"
        )
    if candidate_manifest.capability_profile.id != capability.profile_id:
        raise ModelBGroundingSidecarBuildError(
            "candidate manifest capability profile id does not match input"
        )


def _assert_inputs_unchanged(snapshots: list[_InputSnapshot]) -> None:
    for snapshot in snapshots:
        current = _read_bounded_file(
            _required_regular_file(snapshot.path, snapshot.label),
            snapshot.maximum_bytes,
            snapshot.label,
        )
        if _sha256_bytes(current) != snapshot.sha256:
            raise RuntimeError(f"{snapshot.label} changed during sidecar build")


def _assert_output_is_separate(
    destination: Path,
    input_roots: tuple[Path, ...],
) -> None:
    for input_root in input_roots:
        resolved_input = input_root.resolve()
        if destination == resolved_input or destination.is_relative_to(resolved_input):
            raise ValueError("output must not overlap source inputs")


def _resolve_registered_file(root: Path, reference: str, label: str) -> Path:
    pure = PurePosixPath(reference)
    if (
        pure.is_absolute()
        or "\\" in reference
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ModelBGroundingSidecarBuildError(f"{label} path is unsafe")
    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ModelBGroundingSidecarBuildError(f"{label} must not use symlinks")
    resolved = _required_regular_file(candidate, label)
    if not resolved.is_relative_to(root):
        raise ModelBGroundingSidecarBuildError(f"{label} is outside candidate bundle")
    return resolved


def _required_directory(path: Path, label: str) -> Path:
    try:
        path_metadata = path.lstat()
        _require_directory_non_reparse(path_metadata, label)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    _require_directory_non_reparse(resolved.lstat(), label)
    return resolved


def _required_regular_file(path: Path, label: str) -> Path:
    try:
        path_metadata = path.lstat()
        _require_regular_non_reparse_file(path_metadata, label)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    _require_regular_non_reparse_file(resolved.lstat(), label)
    return resolved


def _read_bounded_file(path: Path, maximum_bytes: int, label: str) -> bytes:
    descriptor: int | None = None
    try:
        path_metadata = path.lstat()
        _require_regular_non_reparse_file(path_metadata, label)
        if path_metadata.st_size > maximum_bytes:
            raise ModelBGroundingSidecarBuildError(
                f"{label} exceeds the {maximum_bytes}-byte limit"
            )
        flags = os.O_RDONLY
        for flag_name in (
            "O_BINARY",
            "O_CLOEXEC",
            "O_NOINHERIT",
            "O_NOFOLLOW",
            "O_NONBLOCK",
        ):
            flags |= getattr(os, flag_name, 0)
        descriptor = os.open(path, flags)
        opened_metadata = os.fstat(descriptor)
        _require_regular_non_reparse_file(opened_metadata, label)
        if opened_metadata.st_size > maximum_bytes:
            raise ModelBGroundingSidecarBuildError(
                f"{label} exceeds the {maximum_bytes}-byte limit"
            )
        _require_same_file_snapshot(path_metadata, opened_metadata, label)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = _read_stream_bounded(
                stream,
                label,
                maximum_bytes=maximum_bytes,
            )
            read_metadata = os.fstat(stream.fileno())
        _require_same_file_snapshot(opened_metadata, read_metadata, label)
        final_path_metadata = path.lstat()
        _require_regular_non_reparse_file(final_path_metadata, label)
        _require_same_file_snapshot(read_metadata, final_path_metadata, label)
    except ModelBGroundingSidecarBuildError:
        raise
    except OSError as exc:
        raise ModelBGroundingSidecarBuildError(f"{label} is missing or unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return payload


def _read_stream_bounded(
    stream: BinaryIO,
    label: str,
    *,
    maximum_bytes: int,
) -> bytes:
    payload = bytearray()
    while True:
        remaining = maximum_bytes + 1 - len(payload)
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > maximum_bytes:
            raise ModelBGroundingSidecarBuildError(
                f"{label} exceeds the {maximum_bytes}-byte limit"
            )
    return bytes(payload)


def _require_regular_non_reparse_file(
    metadata: os.stat_result,
    label: str,
) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(file_attributes & reparse_attribute)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ModelBGroundingSidecarBuildError(f"{label} must be a regular non-symlink file")


def _require_directory_non_reparse(
    metadata: os.stat_result,
    label: str,
) -> None:
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(file_attributes & reparse_attribute)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ModelBGroundingSidecarBuildError(f"{label} must be a regular non-symlink directory")


def _require_same_file_snapshot(
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    unchanged = (
        os.path.samestat(before, after)
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )
    if not unchanged:
        raise ModelBGroundingSidecarBuildError(f"{label} changed while it was read")


def _publish_directory_create_only(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    if os.name == "nt":
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
            raise RuntimeError("atomic create-only directory publish is unavailable")
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
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise RuntimeError("atomic create-only directory publish is unavailable")


def _assert_strict_json(payload: bytes, label: str) -> None:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite number: {value}")

    try:
        json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ModelBGroundingSidecarBuildError(f"{label} must be valid strict UTF-8 JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
