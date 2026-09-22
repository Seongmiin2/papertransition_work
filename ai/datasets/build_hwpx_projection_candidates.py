from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import pymupdf

from scan2hwpx.blueprint.model_a import classify_role
from scan2hwpx.contracts import (
    BBox,
    ColumnBreakPlanItem,
    ContentIR,
    ContentPlanItem,
    ContentRole,
    EvidenceIR,
    EvidenceObservation,
    EvidencePage,
    EvidenceSource,
    EvidenceSourceKind,
    FormulaContentNode,
    FormulaFormat,
    HwpDocumentPlan,
    ImageContentNode,
    LayoutIntent,
    ObservationKind,
    OcrCandidate,
    PageBreakPlanItem,
    PageLayoutIntent,
    StyleIntent,
    TableCell,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.hwpx.validate import validate_hwpx
from scan2hwpx.knowledge.hancom import (
    HancomChunk,
    load_chunks_bytes,
    retrieve_hwpx_plan_chunks,
)
from scan2hwpx.reference.hwpx import (
    HwpxProjectionSource,
    ProjectedCell,
    ProjectedNode,
    ProjectedPageLayout,
    ProjectedStyle,
)

PAIR_SCHEMA = "hwp-hwpx-projection-pairs/1.0"
CANDIDATE_SCHEMA = "hwp-human-review-candidates/1.0"
PROJECTION_SCHEMA = "hwpx-projection-source/1.0"
SPLITS = frozenset({"train", "validation", "test"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ISSUE_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*")
UNCALIBRATED_CONFIDENCE = 0.0
TEXT_ALIGNMENT_THRESHOLD = 0.45
HWPUNIT_TO_MM = 25.4 / 7200
IMAGE_ASPECT_RATIO_TOLERANCE = math.log(1.25)
_PARAGRAPH_LOCATOR_PART = re.compile(r"^p\d+$")
_SAFE_LATEX_IDENTITY = re.compile(r"^[A-Za-z0-9\s{}_^+*/=().,|<>\[\]-]+$")


class ProjectionModel(Protocol):
    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


class ProjectionReader(Protocol):
    def __call__(
        self,
        path: str | Path,
        *,
        source_document_sha256: str,
        expected_hwpx_sha256: str,
        expected_page_count: int,
    ) -> ProjectionModel: ...


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PairDocument:
    source_hwp_sha256: str
    split: str
    page_count: int
    hwpx: ArtifactBinding
    pdf: ArtifactBinding


@dataclass(frozen=True, slots=True)
class PairBundle:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    source_dataset_report_sha256: str
    source_inventory_sha256: str
    producer: dict[str, str]
    rights_manifest: dict[str, str]
    documents: tuple[PairDocument, ...]
    file_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class KnowledgeIndex:
    sha256: str
    chunks: tuple[HancomChunk, ...]
    ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class PdfObservation:
    id: str
    page_no: int
    kind: ObservationKind
    text: str | None
    bbox: tuple[float, float, float, float]
    source_ref: str
    asset_source_ref: str


@dataclass(frozen=True, slots=True)
class PdfEvidence:
    contract: EvidenceIR
    observations: tuple[PdfObservation, ...]
    region_by_page: dict[int, PdfObservation]
    page_image_bindings: tuple[dict[str, str], ...]
    image_crop_bindings: tuple[dict[str, str], ...]
    issue_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    refs: tuple[str, ...]
    score: float
    used_page_fallback: bool
    page_no: int


@dataclass(frozen=True, slots=True)
class ProjectionGroup:
    id: str
    kind: str
    source_nodes: tuple[ProjectedNode, ...]
    text: str
    role_hint: str | None
    style_id: str | None
    page_break: bool
    column_break: bool

    @property
    def source(self) -> ProjectedNode:
        return self.source_nodes[0]


@dataclass(frozen=True, slots=True)
class AlignmentBundle:
    results: dict[str, AlignmentResult]
    target_character_count: int
    matched_character_count: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build human-review-only IR candidates from paired HWPX and PDF artifacts"
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--capability-profile",
        type=Path,
        default=Path("ai/knowledge/hancom/capability_profile.json"),
    )
    parser.add_argument(
        "--knowledge-corpus",
        type=Path,
        default=Path("output/datasets/hancom-knowledge/processed/chunks.jsonl"),
    )
    args = parser.parse_args()

    manifest = build_hwpx_projection_candidates(
        args.pairs,
        args.capability_profile,
        args.knowledge_corpus,
        args.out,
        max_workers=args.max_workers,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "document_count": manifest["document_count"],
                "page_count": manifest["page_count"],
                "splits": manifest["splits"],
                "research_only": manifest["research_only"],
                "human_review_required": manifest["human_review_required"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_hwpx_projection_candidates(
    pairs_root: str | Path,
    capability_profile_path: str | Path,
    knowledge_corpus_path: str | Path,
    output_dir: str | Path,
    *,
    max_workers: int,
    dpi: int = 300,
    projection_reader: ProjectionReader | None = None,
) -> dict[str, Any]:
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise ValueError("dpi must be an integer between 72 and 600")

    bundle = _load_pair_bundle(Path(pairs_root))
    capability_path = _required_regular_file(Path(capability_profile_path), "capability profile")
    corpus_path = _required_regular_file(Path(knowledge_corpus_path), "knowledge corpus")
    capability_bytes = capability_path.read_bytes()
    corpus_bytes = corpus_path.read_bytes()
    capability = _load_capability_profile(capability_bytes)
    knowledge = _load_knowledge_index(corpus_path, corpus_bytes)
    destination = Path(output_dir).resolve()
    for input_root in (bundle.root, capability_path.parent, corpus_path.parent):
        if destination == input_root or destination.is_relative_to(input_root):
            raise ValueError("output must not overlap source inputs")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")

    if projection_reader is None:
        from scan2hwpx.reference.hwpx import read_hwpx_projection

        projection_reader = cast(ProjectionReader, read_hwpx_projection)

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
        results: dict[str, dict[str, Any]] = {}
        failures: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sha = {
                executor.submit(
                    _build_document_candidate,
                    document,
                    bundle,
                    staging,
                    capability,
                    knowledge,
                    dpi,
                    projection_reader,
                ): document.source_hwp_sha256
                for document in bundle.documents
            }
            for future in as_completed(future_to_sha):
                source_sha256 = future_to_sha[future]
                try:
                    results[source_sha256] = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures[source_sha256] = exc
        if failures:
            failed = ", ".join(sorted(failures))
            first = failures[min(failures)]
            raise RuntimeError(f"projection candidate build failed for source hashes: {failed}") from first

        ordered = [results[document.source_hwp_sha256] for document in bundle.documents]
        split_counts: dict[str, dict[str, int]] = {}
        for split in sorted(SPLITS):
            selected = [item for item in ordered if item["split"] == split]
            split_counts[split] = {
                "documents": len(selected),
                "pages": sum(int(item["page_count"]) for item in selected),
            }
        issue_counts = Counter(
            issue
            for item in ordered
            for issue in cast(list[str], item["issue_codes"])
        )
        manifest: dict[str, Any] = {
            "schema_version": CANDIDATE_SCHEMA,
            "artifact_role": "human_review_candidate",
            "research_only": True,
            "human_review_required": True,
            "golden_eligible": False,
            "training_eligible": False,
            "release_eligible": False,
            "rights_manifest": dict(bundle.rights_manifest),
            "source_bundle": {
                "schema_version": PAIR_SCHEMA,
                "manifest_sha256": bundle.manifest_sha256,
                "source_dataset_report_sha256": bundle.source_dataset_report_sha256,
                "source_inventory_sha256": bundle.source_inventory_sha256,
                "producer": bundle.producer,
            },
            "producer": {
                "name": "scan2hwpx-hwpx-projection-candidate-builder",
                "version": "1.3",
                "pymupdf_version": package_version("pymupdf"),
                "dpi": dpi,
                "pdf_text_order": "native",
            },
            "capability_profile": {
                "id": capability["profile_id"],
                "sha256": hashlib.sha256(capability_bytes).hexdigest(),
            },
            "knowledge_corpus_sha256": knowledge.sha256,
            "contract_versions": {
                "projection_source": PROJECTION_SCHEMA,
                "evidence_ir": "evidence-ir/1.0",
                "content_ir": "content-ir/1.0",
                "hwp_document_plan": "hwp-document-plan/1.0",
            },
            "document_count": len(ordered),
            "page_count": sum(int(item["page_count"]) for item in ordered),
            "splits": split_counts,
            "issue_counts": dict(sorted(issue_counts.items())),
            "documents": ordered,
        }
        _write_bytes(staging / "manifest.json", _json_bytes(manifest))
        _assert_candidate_inventory(staging, manifest)
        _assert_inputs_unchanged(
            bundle,
            capability_path,
            capability_bytes,
            corpus_path,
            corpus_bytes,
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output appeared during build: {destination}")
        staging.rename(destination)
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _load_pair_bundle(root_path: Path) -> PairBundle:
    root = _required_directory(root_path, "pairs")
    manifest_path = _required_regular_file(root / "manifest.json", "pairs manifest")
    manifest_bytes = manifest_path.read_bytes()
    try:
        value = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pairs manifest must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("pairs manifest must be an object")
    expected_keys = {
        "schema_version",
        "artifact_role",
        "research_only",
        "rights_manifest",
        "source_dataset_report_sha256",
        "source_inventory_sha256",
        "document_count",
        "splits",
        "producer",
        "documents",
    }
    if set(value) != expected_keys:
        raise ValueError("pairs manifest has an unexpected field set")
    if value["schema_version"] != PAIR_SCHEMA:
        raise ValueError(f"pairs manifest schema_version must be {PAIR_SCHEMA}")
    if value["artifact_role"] != "paired_source_for_projection":
        raise ValueError("pairs manifest artifact_role is not a projection source")
    if value["research_only"] is not True:
        raise ValueError("pairs manifest must remain research_only")
    if value["rights_manifest"] != {"status": "missing"}:
        raise ValueError("pairs manifest rights status must remain explicitly missing")
    for field in ("source_dataset_report_sha256", "source_inventory_sha256"):
        _required_sha256(value[field], f"pairs manifest {field}")
    producer = _strict_string_map(value["producer"], {"name", "version"}, "producer")
    raw_documents = value["documents"]
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("pairs manifest documents must be a non-empty list")
    if isinstance(value["document_count"], bool) or value["document_count"] != len(raw_documents):
        raise ValueError("pairs manifest document_count mismatch")

    documents: list[PairDocument] = []
    seen_sources: set[str] = set()
    seen_source_refs: set[str] = set()
    seen_refs: set[str] = set()
    artifact_split_by_hash: dict[str, str] = {}
    for index, item in enumerate(raw_documents):
        if not isinstance(item, dict) or set(item) != {
            "source_sha256",
            "source_ref",
            "split",
            "page_count",
            "hwpx",
            "pdf",
        }:
            raise ValueError(f"pairs manifest document {index} has an unexpected field set")
        source_sha256 = _required_sha256(item["source_sha256"], f"document {index} source")
        if source_sha256 in seen_sources:
            raise ValueError(f"pairs manifest contains duplicate source sha256: {source_sha256}")
        seen_sources.add(source_sha256)
        split = item["split"]
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"pairs manifest document {index} has invalid split")
        page_count = item["page_count"]
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
            raise ValueError(f"pairs manifest document {index} has invalid page_count")
        source_ref = item["source_ref"]
        if not isinstance(source_ref, str) or not source_ref or _has_control_character(source_ref):
            raise ValueError(f"pairs manifest document {index} has invalid source_ref")
        normalized_source_ref = source_ref.replace("\\", "/").casefold()
        if normalized_source_ref in seen_source_refs:
            raise ValueError("pairs manifest contains duplicate source refs")
        seen_source_refs.add(normalized_source_ref)
        hwpx = _load_artifact_binding(item["hwpx"], source_sha256, "hwpx", index)
        pdf = _load_artifact_binding(item["pdf"], source_sha256, "pdf", index)
        for binding in (hwpx, pdf):
            normalized_ref = binding.path.casefold()
            if normalized_ref in seen_refs:
                raise ValueError("pairs manifest contains duplicate artifact refs")
            seen_refs.add(normalized_ref)
            previous_split = artifact_split_by_hash.get(binding.sha256)
            if previous_split is not None and previous_split != split:
                raise ValueError("pairs manifest contains artifact hash leakage across splits")
            artifact_split_by_hash[binding.sha256] = split
        documents.append(
            PairDocument(
                source_hwp_sha256=source_sha256,
                split=split,
                page_count=page_count,
                hwpx=hwpx,
                pdf=pdf,
            )
        )
    documents.sort(key=lambda document: document.source_hwp_sha256)
    _validate_pair_split_counts(value["splits"], documents)
    file_hashes = _verify_pair_inventory(root, manifest_path, documents)
    return PairBundle(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        source_dataset_report_sha256=cast(str, value["source_dataset_report_sha256"]),
        source_inventory_sha256=cast(str, value["source_inventory_sha256"]),
        producer=producer,
        rights_manifest={"status": "missing"},
        documents=tuple(documents),
        file_hashes=file_hashes,
    )
def _load_artifact_binding(
    value: object,
    source_sha256: str,
    kind: str,
    document_index: int,
) -> ArtifactBinding:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(
            f"pairs manifest document {document_index} has invalid {kind} binding"
        )
    expected_path = f"{source_sha256}/source.{kind}"
    path = value["path"]
    if not isinstance(path, str) or path != expected_path:
        raise ValueError(
            f"pairs manifest document {document_index} has non-canonical {kind} path"
        )
    sha256 = _required_sha256(
        value["sha256"], f"document {document_index} {kind} sha256"
    )
    return ArtifactBinding(path=path, sha256=sha256)


def _validate_pair_split_counts(
    value: object,
    documents: Sequence[PairDocument],
) -> None:
    if not isinstance(value, dict) or set(value) != SPLITS:
        raise ValueError("pairs manifest splits must contain train, validation, and test")
    expected = Counter(document.split for document in documents)
    for split in SPLITS:
        count = value[split]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"pairs manifest split {split} has invalid document count")
        if count != expected[split]:
            raise ValueError(f"pairs manifest split {split} document count mismatch")


def _verify_pair_inventory(
    root: Path,
    manifest_path: Path,
    documents: Sequence[PairDocument],
) -> dict[str, str]:
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if any(path.is_symlink() for path in entries):
        raise ValueError("pairs bundle must not contain symlinks")
    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in entries
        if path.is_file()
    }
    expected_files = {"manifest.json"}
    expected_files.update(
        binding.path
        for document in documents
        for binding in (document.hwpx, document.pdf)
    )
    if set(actual_files) != expected_files:
        missing = len(expected_files - set(actual_files))
        unexpected = len(set(actual_files) - expected_files)
        raise ValueError(
            f"pairs bundle artifact inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    if actual_files["manifest.json"].resolve(strict=True) != manifest_path:
        raise ValueError("pairs bundle manifest path changed during validation")

    hashes: dict[str, str] = {"manifest.json": _sha256(manifest_path)}
    for document in documents:
        hwpx_path = actual_files[document.hwpx.path]
        pdf_path = actual_files[document.pdf.path]
        for binding, path in ((document.hwpx, hwpx_path), (document.pdf, pdf_path)):
            actual_sha256 = _sha256(path)
            if actual_sha256 != binding.sha256:
                raise ValueError(f"pairs bundle artifact SHA-256 mismatch: {binding.path}")
            hashes[binding.path] = actual_sha256
        if not validate_hwpx(hwpx_path).valid:
            raise ValueError(f"pairs bundle HWPX package is invalid: {document.source_hwp_sha256}")
        if _pdf_page_count(pdf_path) != document.page_count:
            raise ValueError(
                f"pairs bundle PDF page_count mismatch: {document.source_hwp_sha256}"
            )
    return hashes


def _load_capability_profile(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("capability profile must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("capability profile must be an object")
    if not isinstance(value.get("profile_id"), str) or not value["profile_id"].strip():
        raise ValueError("capability profile must contain profile_id")
    if value.get("target_format") != "hwpx":
        raise ValueError("capability profile target_format must be hwpx")
    if value.get("content_policy") != "immutable":
        raise ValueError("capability profile content_policy must be immutable")
    if value.get("planner_must_reference_content") is not True:
        raise ValueError("capability profile must require content references")
    forbidden = value.get("planner_forbidden")
    if not isinstance(forbidden, list) or not {
        "raw_xml",
        "local_file_path",
        "external_url",
    }.issubset(forbidden):
        raise ValueError("capability profile does not contain required planner prohibitions")
    return value


def _load_knowledge_index(path: Path, payload: bytes) -> KnowledgeIndex:
    chunks = load_chunks_bytes(payload)
    if path.read_bytes() != payload:
        raise ValueError("knowledge corpus changed while it was being loaded")
    return KnowledgeIndex(
        sha256=hashlib.sha256(payload).hexdigest(),
        chunks=chunks,
        ids=frozenset(chunk.id for chunk in chunks),
    )


def _build_pdf_evidence(
    document: PairDocument,
    bundle: PairBundle,
    document_dir: Path,
    dpi: int,
) -> PdfEvidence:
    pdf_path = bundle.root.joinpath(*document.pdf.path.split("/"))
    pages_dir = document_dir / "pages"
    assets_dir = document_dir / "assets"
    pages_dir.mkdir()
    sources: list[EvidenceSource] = []
    pages: list[EvidencePage] = []
    observations: list[PdfObservation] = []
    region_by_page: dict[int, PdfObservation] = {}
    page_bindings: list[dict[str, str]] = []
    crop_bindings: list[dict[str, str]] = []
    issue_codes: set[str] = {"pdf_observation_confidence_uncalibrated"}
    pdf_source_id = "input-pdf"
    sources.append(
        EvidenceSource(
            id=pdf_source_id,
            kind=EvidenceSourceKind.ORIGINAL_DOCUMENT,
            artifact_ref=f"artifact://model-a-input/{document.pdf.sha256}/source.pdf",
            producer="paired-pdf",
            sha256=document.pdf.sha256,
        )
    )
    scale = dpi / 72.0
    with pymupdf.open(pdf_path) as pdf:  # type: ignore[no-untyped-call]
        if pdf.needs_pass:
            raise ValueError("paired PDF must not require a password")
        if pdf.page_count != document.page_count:
            raise ValueError("paired PDF page_count changed during candidate build")
        for page_index, page in enumerate(pdf):
            page_no = page_index + 1
            page_token = f"page-{page_no:04d}"
            page_path = pages_dir / f"{page_token}.png"
            matrix = pymupdf.Matrix(scale, scale)  # type: ignore[no-untyped-call]
            pixmap = page.get_pixmap(
                matrix=matrix,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            pixmap.save(str(page_path))
            page_sha256 = _sha256(page_path)
            page_source_id = f"{page_token}-image"
            sources.append(
                EvidenceSource(
                    id=page_source_id,
                    kind=EvidenceSourceKind.PAGE_IMAGE,
                    artifact_ref=(
                        f"artifact://model-a-input/{document.pdf.sha256}/pages/{page_token}.png"
                    ),
                    producer=f"pymupdf-render-{dpi}dpi",
                    sha256=page_sha256,
                )
            )
            page_bindings.append(
                {
                    "path": f"{document.source_hwp_sha256}/pages/{page_token}.png",
                    "sha256": page_sha256,
                }
            )
            page_observations: list[EvidenceObservation] = []
            region = PdfObservation(
                id=f"{page_token}-region",
                page_no=page_no,
                kind=ObservationKind.REGION,
                text=None,
                bbox=(0.0, 0.0, float(pixmap.width), float(pixmap.height)),
                source_ref=page_source_id,
                asset_source_ref=page_source_id,
            )
            observations.append(region)
            region_by_page[page_no] = region
            page_observations.append(
                EvidenceObservation(
                    id=region.id,
                    kind=ObservationKind.REGION,
                    bbox=_bbox_contract(region.bbox, pixmap.width, pixmap.height),
                    confidence=UNCALIBRATED_CONFIDENCE,
                    source_refs=(page_source_id,),
                )
            )

            payload = page.get_text("dict", sort=False)
            if not isinstance(payload, dict):
                raise TypeError("PyMuPDF returned an invalid page text dictionary")
            line_index = 0
            image_index = 0
            blocks = payload.get("blocks", []) if int(page.rotation) == 0 else []
            if int(page.rotation) != 0:
                issue_codes.add("pdf_rotated_page_uses_region_fallback")
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        if not isinstance(line, dict):
                            continue
                        spans = line.get("spans", [])
                        if not isinstance(spans, list):
                            continue
                        text = _normalize_display_text(
                            "".join(
                                str(span.get("text", ""))
                                for span in spans
                                if isinstance(span, dict)
                            )
                        )
                        if not text:
                            continue
                        pixel_bbox = _scaled_pdf_bbox(
                            line.get("bbox"), scale, pixmap.width, pixmap.height
                        )
                        if pixel_bbox is None:
                            issue_codes.add("pdf_text_line_bbox_invalid")
                            continue
                        line_index += 1
                        observation_id = f"{page_token}-text-{line_index:05d}"
                        pdf_observation = PdfObservation(
                            id=observation_id,
                            page_no=page_no,
                            kind=ObservationKind.TEXT_LINE,
                            text=text,
                            bbox=pixel_bbox,
                            source_ref=pdf_source_id,
                            asset_source_ref=page_source_id,
                        )
                        observations.append(pdf_observation)
                        page_observations.append(
                            EvidenceObservation(
                                id=observation_id,
                                kind=ObservationKind.TEXT_LINE,
                                bbox=_bbox_contract(pixel_bbox, pixmap.width, pixmap.height),
                                confidence=UNCALIBRATED_CONFIDENCE,
                                source_refs=(pdf_source_id, page_source_id),
                                ocr_candidates=(
                                    OcrCandidate(
                                        text=text,
                                        provider="pdf-text-layer-uncalibrated",
                                        confidence=UNCALIBRATED_CONFIDENCE,
                                        quality=None,
                                        source_ref=pdf_source_id,
                                        selected=True,
                                    ),
                                ),
                            )
                        )
                elif block.get("type") == 1:
                    pixel_bbox = _scaled_pdf_bbox(
                        block.get("bbox"), scale, pixmap.width, pixmap.height
                    )
                    if pixel_bbox is None:
                        issue_codes.add("pdf_image_bbox_invalid")
                        continue
                    image_index += 1
                    observation_id = f"{page_token}-image-{image_index:04d}"
                    crop_path = assets_dir / f"{observation_id}.png"
                    clip = pymupdf.Rect(  # type: ignore[no-untyped-call]
                        pixel_bbox[0] / scale,
                        pixel_bbox[1] / scale,
                        pixel_bbox[2] / scale,
                        pixel_bbox[3] / scale,
                    )
                    crop = page.get_pixmap(
                        matrix=matrix,
                        clip=clip,
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    if crop.width < 1 or crop.height < 1:
                        issue_codes.add("pdf_image_crop_empty")
                        continue
                    crop_path.parent.mkdir(exist_ok=True)
                    crop.save(str(crop_path))
                    crop_sha256 = _sha256(crop_path)
                    crop_source_id = f"{observation_id}-crop"
                    sources.append(
                        EvidenceSource(
                            id=crop_source_id,
                            kind=EvidenceSourceKind.CROP,
                            artifact_ref=(
                                f"artifact://model-a-input/{document.pdf.sha256}/assets/"
                                f"{observation_id}.png"
                            ),
                            producer=f"pymupdf-render-{dpi}dpi",
                            sha256=crop_sha256,
                        )
                    )
                    crop_bindings.append(
                        {
                            "path": (
                                f"{document.source_hwp_sha256}/assets/{observation_id}.png"
                            ),
                            "sha256": crop_sha256,
                        }
                    )
                    pdf_observation = PdfObservation(
                        id=observation_id,
                        page_no=page_no,
                        kind=ObservationKind.IMAGE,
                        text=None,
                        bbox=pixel_bbox,
                        source_ref=crop_source_id,
                        asset_source_ref=crop_source_id,
                    )
                    observations.append(pdf_observation)
                    page_observations.append(
                        EvidenceObservation(
                            id=observation_id,
                            kind=ObservationKind.IMAGE,
                            bbox=_bbox_contract(pixel_bbox, pixmap.width, pixmap.height),
                            confidence=UNCALIBRATED_CONFIDENCE,
                            source_refs=(crop_source_id, page_source_id),
                        )
                    )
            if int(page.rotation) != 0:
                issue_codes.add("pdf_page_rotation_present")
            pages.append(
                EvidencePage(
                    id=page_token,
                    page_no=page_no,
                    width=float(pixmap.width),
                    height=float(pixmap.height),
                    rotation=int(page.rotation),
                    image_source_ref=page_source_id,
                    observations=tuple(page_observations),
                )
            )

    evidence = EvidenceIR(
        id=f"evidence-{document.pdf.sha256[:24]}",
        source_document_sha256=document.pdf.sha256,
        sources=tuple(sources),
        pages=tuple(pages),
    )
    return PdfEvidence(
        contract=evidence,
        observations=tuple(observations),
        region_by_page=region_by_page,
        page_image_bindings=tuple(page_bindings),
        image_crop_bindings=tuple(crop_bindings),
        issue_codes=tuple(sorted(issue_codes)),
    )


def _build_document_candidate(
    document: PairDocument,
    bundle: PairBundle,
    staging: Path,
    capability: dict[str, Any],
    knowledge: KnowledgeIndex,
    dpi: int,
    projection_reader: ProjectionReader,
) -> dict[str, Any]:
    document_dir = staging / document.source_hwp_sha256
    document_dir.mkdir()
    hwpx_path = bundle.root.joinpath(*document.hwpx.path.split("/"))
    raw_projection = projection_reader(
        hwpx_path,
        source_document_sha256=document.source_hwp_sha256,
        expected_hwpx_sha256=document.hwpx.sha256,
        expected_page_count=document.page_count,
    )
    projection = HwpxProjectionSource.model_validate(raw_projection.model_dump(mode="json"))
    if projection.schema_version != PROJECTION_SCHEMA:
        raise ValueError("projection reader returned an unsupported schema")
    if projection.source_document_sha256 != document.source_hwp_sha256:
        raise ValueError("projection reader returned the wrong source document digest")
    if projection.hwpx_sha256 != document.hwpx.sha256:
        raise ValueError("projection reader returned the wrong HWPX digest")
    if projection.page_count != document.page_count:
        raise ValueError("projection reader returned the wrong page count")
    if not projection.nodes:
        raise ValueError("projection contains no reviewable nodes")

    groups, grouping_issues = _group_projection_nodes(projection)
    pdf_evidence = _build_pdf_evidence(document, bundle, document_dir, dpi)
    alignment = _align_projection_groups(groups, pdf_evidence)
    content, node_pages, content_issues = _build_content_candidate(
        document,
        groups,
        alignment,
        pdf_evidence,
    )
    content.assert_evidence_integrity(pdf_evidence.contract)
    plan, plan_issues = _build_plan_candidate(
        document,
        projection,
        groups,
        content,
        node_pages,
        capability,
        knowledge,
    )
    plan.assert_content_integrity(content)
    _assert_evidence_is_pdf_only(pdf_evidence.contract, document)

    projection_binding = _write_json_artifact(
        document_dir / "projection_source.json",
        projection.model_dump(mode="json"),
        staging,
    )
    evidence_binding = _write_contract_artifact(
        document_dir / "evidence_ir.json", pdf_evidence.contract, staging
    )
    content_binding = _write_contract_artifact(
        document_dir / "content_ir.candidate.json", content, staging
    )
    plan_binding = _write_contract_artifact(
        document_dir / "hwp_document_plan.candidate.json", plan, staging
    )

    issue_codes = {
        issue.code for issue in projection.issues
    } | set(grouping_issues) | set(pdf_evidence.issue_codes) | set(content_issues) | set(
        plan_issues
    )
    issue_codes.update(
        "text_alignment_below_threshold"
        for group in groups
        if group.text
        and alignment.results[group.id].score < TEXT_ALIGNMENT_THRESHOLD
    )
    issue_codes.add("human_review_required")
    issue_codes.add("rights_manifest_missing")
    issue_codes.add("page_assignment_from_pdf_alignment")
    invalid_issue_codes = sorted(
        code for code in issue_codes if ISSUE_CODE_PATTERN.fullmatch(code) is None
    )
    if invalid_issue_codes:
        raise ValueError("candidate produced invalid issue codes: " + ", ".join(invalid_issue_codes))
    report = {
        "schema_version": "hwp-projection-candidate-report/1.0",
        "document_id": f"candidate-{document.source_hwp_sha256[:24]}",
        "lineage_id": document.source_hwp_sha256,
        "status": "needs_human_review",
        "research_only": True,
        "human_review_required": True,
        "training_eligible": False,
        "release_eligible": False,
        "rights_status": "missing",
        "counts": {
            "pages": document.page_count,
            "projection_nodes": len(projection.nodes),
            "content_nodes": len(content.nodes),
            "paragraph_groups": sum(group.kind == "text" for group in groups),
            "tables": sum(group.kind == "table" for group in groups),
            "images": sum(group.kind == "image" for group in groups),
            "formulas": sum(group.kind == "formula" for group in groups),
            "pdf_observations": len(pdf_evidence.observations),
            "pdf_image_crops": len(pdf_evidence.image_crop_bindings),
        },
        "alignment": {
            "target_character_count": alignment.target_character_count,
            "matched_character_count": alignment.matched_character_count,
            "coverage": (
                round(
                    alignment.matched_character_count / alignment.target_character_count,
                    6,
                )
                if alignment.target_character_count
                else 0.0
            ),
            "below_threshold_groups": sum(
                bool(group.text)
                and alignment.results[group.id].score < TEXT_ALIGNMENT_THRESHOLD
                for group in groups
            ),
            "page_fallback_groups": sum(
                alignment.results[group.id].used_page_fallback for group in groups
            ),
        },
        "contract_sha256": {
            "evidence_ir": contract_sha256(pdf_evidence.contract),
            "content_ir": contract_sha256(content),
            "hwp_document_plan": contract_sha256(plan),
        },
        "issue_counts": dict(sorted(Counter(issue.code for issue in projection.issues).items())),
        "issue_codes": sorted(issue_codes),
        "hard_flags": [
            "not_golden",
            "not_training_eligible",
            "not_release_eligible",
            "rights_manifest_missing",
        ],
    }
    report_binding = _write_json_artifact(
        document_dir / "report.json", report, staging
    )
    return {
        "document_id": report["document_id"],
        "lineage_id": document.source_hwp_sha256,
        "split": document.split,
        "page_count": document.page_count,
        "status": "needs_human_review",
        "source_artifacts": {
            "hwp": {"sha256": document.source_hwp_sha256},
            "hwpx": {"sha256": document.hwpx.sha256},
            "pdf": {"sha256": document.pdf.sha256},
        },
        "artifacts": {
            "projection_source": projection_binding,
            "evidence_ir": evidence_binding,
            "content_ir_candidate": content_binding,
            "hwp_document_plan_candidate": plan_binding,
            "report": report_binding,
            "page_images": list(pdf_evidence.page_image_bindings),
            "image_crops": list(pdf_evidence.image_crop_bindings),
        },
        "issue_codes": sorted(issue_codes),
    }


def _group_projection_nodes(
    projection: HwpxProjectionSource,
) -> tuple[tuple[ProjectionGroup, ...], tuple[str, ...]]:
    groups: list[ProjectionGroup] = []
    issues: set[str] = set()
    pending: list[ProjectedNode] = []
    pending_key: str | None = None

    def flush() -> None:
        nonlocal pending, pending_key
        if not pending:
            return
        style_ids = tuple(dict.fromkeys(node.style_id for node in pending if node.style_id))
        role_hints = tuple(dict.fromkeys(node.role_hint for node in pending if node.role_hint))
        if len(style_ids) > 1:
            issues.add("inline_style_runs_flattened")
        if len(role_hints) > 1:
            issues.add("paragraph_role_hints_conflict")
        groups.append(
            _projection_group(
                tuple(pending),
                text="".join(node.text for node in pending),
                role_hint=role_hints[0] if role_hints else None,
                style_id=style_ids[0] if style_ids else None,
            )
        )
        pending = []
        pending_key = None

    for node in projection.nodes:
        if node.kind == "text":
            key = _paragraph_locator(node.locator)
            if pending and key != pending_key:
                flush()
            pending.append(node)
            pending_key = key
            continue
        flush()
        groups.append(
            _projection_group(
                (node,),
                text=node.text,
                role_hint=node.role_hint,
                style_id=node.style_id,
            )
        )
    flush()
    paragraph_kinds: dict[str, set[str]] = {}
    for group in groups:
        for node in group.source_nodes:
            paragraph_kinds.setdefault(_paragraph_locator(node.locator), set()).add(node.kind)
    if any(len(kinds) > 1 for kinds in paragraph_kinds.values()):
        issues.add("mixed_content_paragraph_layout_flattened")
    return tuple(groups), tuple(sorted(issues))


def _projection_group(
    nodes: tuple[ProjectedNode, ...],
    *,
    text: str,
    role_hint: str | None,
    style_id: str | None,
) -> ProjectionGroup:
    signature = hashlib.sha256("|".join(node.id for node in nodes).encode("utf-8")).hexdigest()
    return ProjectionGroup(
        id=f"content-{signature[:24]}",
        kind=nodes[0].kind,
        source_nodes=nodes,
        text=unicodedata.normalize("NFC", text),
        role_hint=role_hint,
        style_id=style_id,
        page_break=any(node.page_break for node in nodes),
        column_break=any(node.column_break for node in nodes),
    )


def _paragraph_locator(locator: str) -> str:
    parts = locator.split("/")
    positions = [
        index for index, part in enumerate(parts) if _PARAGRAPH_LOCATOR_PART.fullmatch(part)
    ]
    if not positions:
        return locator.rpartition("/")[0]
    return "/".join(parts[: positions[-1] + 1])


def _align_projection_groups(
    groups: Sequence[ProjectionGroup],
    evidence: PdfEvidence,
) -> AlignmentBundle:
    units: list[tuple[str, str]] = []
    table_cells: dict[str, tuple[str, ...]] = {}
    for group in groups:
        if group.kind == "table":
            units.append((group.id, ""))
            cell_keys: list[str] = []
            for cell in group.source.cells:
                key = _cell_alignment_key(group.id, cell)
                units.append((key, cell.text))
                cell_keys.append(key)
            table_cells[group.id] = tuple(cell_keys)
        elif group.kind == "image":
            units.append((group.id, ""))
        else:
            units.append((group.id, group.text))

    target_parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    target_offset = 0
    for key, value in units:
        normalized = _normalize_alignment_text(value)
        start = target_offset
        target_parts.append(normalized)
        target_offset += len(normalized)
        spans[key] = (start, target_offset)
    target_text = "".join(target_parts)

    pdf_parts: list[str] = []
    pdf_owners: list[str] = []
    observation_page = {observation.id: observation.page_no for observation in evidence.observations}
    for observation in evidence.observations:
        if observation.kind != ObservationKind.TEXT_LINE or not observation.text:
            continue
        normalized = _normalize_alignment_text(observation.text)
        if not normalized:
            continue
        pdf_parts.append(normalized)
        pdf_owners.extend([observation.id] * len(normalized))
    pdf_text = "".join(pdf_parts)

    target_owners: list[str | None] = [None] * len(target_text)
    matched_character_count = 0
    if target_text and pdf_text:
        matcher = SequenceMatcher(None, target_text, pdf_text, autojunk=False)
        for target_start, pdf_start, size in matcher.get_matching_blocks():
            if size == 0:
                continue
            matched_character_count += size
            for offset in range(size):
                target_owners[target_start + offset] = pdf_owners[pdf_start + offset]

    preliminary: dict[str, tuple[tuple[str, ...], float, int | None]] = {}
    ordered_keys = [key for key, _ in units]
    for key in ordered_keys:
        start, end = spans[key]
        owners = [owner for owner in target_owners[start:end] if owner is not None]
        refs = tuple(dict.fromkeys(owners))
        length = end - start
        score = round(len(owners) / length, 6) if length else 0.0
        pages = Counter(observation_page[owner] for owner in owners)
        page_no = min(
            (page for page, count in pages.items() if count == max(pages.values())),
            default=None,
        )
        preliminary[key] = (refs, score, page_no)

    inferred_pages: list[int | None] = [preliminary[key][2] for key in ordered_keys]
    last_page: int | None = None
    for index, page_no in enumerate(inferred_pages):
        if page_no is not None:
            last_page = page_no
        elif last_page is not None:
            inferred_pages[index] = last_page
    next_page: int | None = None
    for index in range(len(inferred_pages) - 1, -1, -1):
        page_no = inferred_pages[index]
        if page_no is not None:
            next_page = page_no
        elif next_page is not None:
            inferred_pages[index] = next_page

    results: dict[str, AlignmentResult] = {}
    for index, key in enumerate(ordered_keys):
        refs, score, aligned_page = preliminary[key]
        page_no = inferred_pages[index] or 1
        if not 1 <= page_no <= len(evidence.contract.pages):
            raise ValueError("alignment inferred a page outside the EvidenceIR")
        used_fallback = not refs
        if used_fallback:
            refs = (evidence.region_by_page[page_no].id,)
        results[key] = AlignmentResult(
            refs=refs,
            score=score,
            used_page_fallback=used_fallback or aligned_page is None,
            page_no=page_no,
        )

    for group_id, table_cell_keys in table_cells.items():
        cell_results = [results[key] for key in table_cell_keys]
        refs = tuple(dict.fromkeys(ref for item in cell_results for ref in item.refs))
        page_counts = Counter(item.page_no for item in cell_results)
        highest = max(page_counts.values(), default=0)
        page_no = min(
            (page for page, count in page_counts.items() if count == highest),
            default=results[group_id].page_no,
        )
        lengths = [
            len(_normalize_alignment_text(cell.text)) for cell in _table_cells_for_group(groups, group_id)
        ]
        weighted_total = sum(lengths)
        weighted_score = (
            sum(result.score * length for result, length in zip(cell_results, lengths, strict=True))
            / weighted_total
            if weighted_total
            else 0.0
        )
        results[group_id] = AlignmentResult(
            refs=refs or (evidence.region_by_page[page_no].id,),
            score=round(weighted_score, 6),
            used_page_fallback=all(item.used_page_fallback for item in cell_results),
            page_no=page_no,
        )
    return AlignmentBundle(
        results=results,
        target_character_count=len(target_text),
        matched_character_count=matched_character_count,
    )


def _table_cells_for_group(
    groups: Sequence[ProjectionGroup], group_id: str
) -> tuple[ProjectedCell, ...]:
    for group in groups:
        if group.id == group_id:
            return group.source.cells
    raise KeyError(group_id)


def _cell_alignment_key(group_id: str, cell: ProjectedCell) -> str:
    return f"{group_id}:cell-r{cell.row:04d}-c{cell.column:04d}"


def _normalize_alignment_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return "".join(
        character
        for character in normalized
        if not character.isspace() and unicodedata.category(character) != "Cf"
    )


def _build_content_candidate(
    document: PairDocument,
    groups: Sequence[ProjectionGroup],
    alignment: AlignmentBundle,
    pdf_evidence: PdfEvidence,
) -> tuple[ContentIR, dict[str, tuple[int, bool]], tuple[str, ...]]:
    nodes: list[
        TextContentNode | TableContentNode | ImageContentNode | FormulaContentNode
    ] = []
    node_pages: dict[str, tuple[int, bool]] = {}
    issue_codes: set[str] = set()
    images_by_page: dict[int, list[PdfObservation]] = {}
    used_image_observation_ids: set[str] = set()
    for observation in pdf_evidence.observations:
        if observation.kind == ObservationKind.IMAGE:
            images_by_page.setdefault(observation.page_no, []).append(observation)

    question_started = False
    for group in groups:
        aligned = alignment.results[group.id]
        reliable_page = (
            not aligned.used_page_fallback and aligned.score >= TEXT_ALIGNMENT_THRESHOLD
        )
        node_pages[group.id] = (aligned.page_no, reliable_page)
        if group.kind == "text":
            role = _content_role(
                group.text,
                group.role_hint,
                question_started=question_started,
            )
            if role == ContentRole.QUESTION:
                question_started = True
            if role in {ContentRole.HEADER, ContentRole.FOOTER}:
                issue_codes.add("header_footer_special_placement_unimplemented")
            nodes.append(
                TextContentNode(
                    id=group.id,
                    role=role,
                    text=group.text,
                    evidence_refs=aligned.refs,
                    confidence=aligned.score,
                    needs_review=True,
                )
            )
            continue

        if group.kind == "table":
            source = group.source
            if source.rows is None or source.columns is None:
                raise ValueError("projected table lost its declared dimensions")
            cells = tuple(
                TableCell(
                    row=cell.row,
                    column=cell.column,
                    row_span=cell.row_span,
                    column_span=cell.column_span,
                    text=cell.text,
                    evidence_refs=alignment.results[
                        _cell_alignment_key(group.id, cell)
                    ].refs,
                )
                for cell in source.cells
            )
            nodes.append(
                TableContentNode(
                    id=group.id,
                    rows=source.rows,
                    columns=source.columns,
                    cells=cells,
                    evidence_refs=aligned.refs,
                    confidence=aligned.score,
                    needs_review=True,
                )
            )
            continue

        if group.kind == "image":
            issue_codes.add("image_pairing_unverified")
            page_images = images_by_page.get(aligned.page_no, [])
            image_observation = _unique_pdf_image_candidate(
                group,
                page_images,
                used_image_observation_ids,
            )
            if image_observation is None:
                image_observation = pdf_evidence.region_by_page[aligned.page_no]
                issue_codes.add("image_uses_page_region_fallback")
            else:
                used_image_observation_ids.add(image_observation.id)
                issue_codes.add("image_pairing_aspect_ratio_candidate")
            nodes.append(
                ImageContentNode(
                    id=group.id,
                    asset_ref=image_observation.asset_source_ref,
                    evidence_refs=(image_observation.id,),
                    confidence=UNCALIBRATED_CONFIDENCE,
                    needs_review=True,
                )
            )
            continue

        if group.kind != "formula":
            raise ValueError(f"unsupported projected node kind: {group.kind}")
        expression = (group.source.formula_script or group.text).strip()
        if _is_safe_latex_identity(expression):
            nodes.append(
                FormulaContentNode(
                    id=group.id,
                    expression=expression,
                    format=FormulaFormat.LATEX,
                    evidence_refs=aligned.refs,
                    confidence=aligned.score,
                    needs_review=True,
                )
            )
            issue_codes.add("hwp_equation_identity_subset_requires_review")
        else:
            nodes.append(
                TextContentNode(
                    id=group.id,
                    role=ContentRole.FORMULA,
                    text=expression,
                    evidence_refs=aligned.refs,
                    confidence=aligned.score,
                    needs_review=True,
                )
            )
            issue_codes.add("hwp_equation_dialect_not_converted")

    content = ContentIR(
        id=f"content-candidate-{document.source_hwp_sha256[:24]}",
        evidence_ir_id=pdf_evidence.contract.id,
        evidence_ir_sha256=contract_sha256(pdf_evidence.contract),
        revision=1,
        nodes=tuple(nodes),
        reading_order=tuple(node.id for node in nodes),
    )
    return content, node_pages, tuple(sorted(issue_codes))


def _content_role(
    text: str,
    role_hint: str | None,
    *,
    question_started: bool,
) -> ContentRole:
    if role_hint == "header":
        return ContentRole.HEADER
    if role_hint == "footer":
        return ContentRole.FOOTER
    if role_hint == "other":
        return ContentRole.OTHER
    inferred = classify_role(text, question_started=question_started).value
    return {
        "title": ContentRole.TITLE,
        "instruction": ContentRole.INSTRUCTION,
        "passage": ContentRole.PASSAGE,
        "question": ContentRole.QUESTION,
        "choice": ContentRole.CHOICE,
        "caption": ContentRole.CAPTION,
        "header": ContentRole.HEADER,
        "footer": ContentRole.FOOTER,
        "page_number": ContentRole.FOOTER,
        "source": ContentRole.CAPTION,
    }.get(inferred, ContentRole.OTHER)


def _is_safe_latex_identity(expression: str) -> bool:
    if not expression or _SAFE_LATEX_IDENTITY.fullmatch(expression) is None:
        return False
    depth = 0
    for character in expression:
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _unique_pdf_image_candidate(
    group: ProjectionGroup,
    candidates: Sequence[PdfObservation],
    used_observation_ids: set[str],
) -> PdfObservation | None:
    width = group.source.width_hwp
    height = group.source.height_hwp
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    target_ratio = width / height
    compatible: list[PdfObservation] = []
    for candidate in candidates:
        if candidate.id in used_observation_ids:
            continue
        x0, y0, x1, y1 = candidate.bbox
        candidate_width = x1 - x0
        candidate_height = y1 - y0
        if candidate_width <= 0 or candidate_height <= 0:
            continue
        ratio_error = abs(math.log((candidate_width / candidate_height) / target_ratio))
        if ratio_error <= IMAGE_ASPECT_RATIO_TOLERANCE:
            compatible.append(candidate)
    return compatible[0] if len(compatible) == 1 else None


def _build_plan_candidate(
    document: PairDocument,
    projection: HwpxProjectionSource,
    groups: Sequence[ProjectionGroup],
    content: ContentIR,
    node_pages: dict[str, tuple[int, bool]],
    capability: dict[str, Any],
    knowledge: KnowledgeIndex,
) -> tuple[HwpDocumentPlan, tuple[str, ...]]:
    issue_codes: set[str] = set()
    section_layouts = tuple(section.layout for section in projection.sections)
    if len({layout.model_dump_json() for layout in section_layouts}) > 1:
        issue_codes.add("multiple_section_layouts_flattened")
    source_layout = section_layouts[0]
    columns = source_layout.columns
    if columns > 4:
        columns = 4
        issue_codes.add("column_count_exceeds_plan_contract")
    page_layout = PageLayoutIntent(
        width_mm=_hwpunit_to_mm(source_layout.width_hwp),
        height_mm=_hwpunit_to_mm(source_layout.height_hwp),
        margin_top_mm=_hwpunit_to_mm(source_layout.margin_top_hwp),
        margin_right_mm=_hwpunit_to_mm(source_layout.margin_right_hwp),
        margin_bottom_mm=_hwpunit_to_mm(source_layout.margin_bottom_hwp),
        margin_left_mm=_hwpunit_to_mm(source_layout.margin_left_hwp),
        columns=columns,
        column_gap_mm=_hwpunit_to_mm(source_layout.column_gap_hwp),
    )

    content_by_id = {node.id: node for node in content.nodes}
    projection_styles = {style.id: style for style in projection.styles}
    style_intents: dict[str, StyleIntent] = {}
    style_ref_by_group: dict[str, str] = {}
    for group in groups:
        content_node = content_by_id[group.id]
        if not isinstance(content_node, TextContentNode) or group.style_id is None:
            continue
        source_style = projection_styles.get(group.style_id)
        if source_style is None:
            issue_codes.add("projection_style_reference_missing")
            continue
        style_intent = _style_intent(source_style, content_node.role)
        style_intents.setdefault(style_intent.id, style_intent)
        style_ref_by_group[group.id] = style_intent.id

    flow: list[ContentPlanItem | PageBreakPlanItem | ColumnBreakPlanItem] = []
    previous_reliable_page: int | None = None
    explicit_breaks_since_reliable_page = 0
    for index, group in enumerate(groups):
        page_no, page_reliable = node_pages[group.id]
        explicit_break = group.page_break and bool(flow)
        explicit_break_count = int(explicit_break)
        if explicit_break:
            explicit_breaks_since_reliable_page += 1
        inferred_break_count = 0
        if page_reliable:
            if previous_reliable_page is not None:
                if page_no < previous_reliable_page:
                    issue_codes.add("pdf_page_alignment_regression")
                else:
                    page_delta = page_no - previous_reliable_page
                    inferred_break_count = max(
                        0,
                        page_delta - explicit_breaks_since_reliable_page,
                    )
                    previous_reliable_page = page_no
            else:
                previous_reliable_page = page_no
            explicit_breaks_since_reliable_page = 0
        break_count = explicit_break_count + inferred_break_count
        if inferred_break_count and not explicit_break:
            issue_codes.add("page_break_inferred_from_pdf_alignment")
        for break_index in range(break_count):
            flow.append(
                PageBreakPlanItem(id=f"flow-page-break-{index:06d}-{break_index:02d}")
            )
        if group.column_break and flow:
            flow.append(ColumnBreakPlanItem(id=f"flow-column-break-{index:06d}"))
        width_fraction, width_issue = _width_fraction(group, source_layout)
        if width_issue is not None:
            issue_codes.add(width_issue)
        content_node = content_by_id[group.id]
        flow.append(
            ContentPlanItem(
                id=f"flow-content-{index:06d}",
                render_as=_render_as(content_node),
                content_ref=content_node.id,
                style_ref=style_ref_by_group.get(group.id),
                layout=LayoutIntent(width_fraction=width_fraction),
            )
        )

    official_spec_refs = _official_spec_refs(knowledge)
    if any(reference not in knowledge.ids for reference in official_spec_refs):
        raise ValueError("plan contains an unknown official specification reference")
    plan = HwpDocumentPlan(
        id=f"hwp-plan-candidate-{document.source_hwp_sha256[:24]}",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=contract_sha256(content),
        capability_profile_id=cast(str, capability["profile_id"]),
        design_profile_id="source-projection-candidate-v1",
        official_spec_refs=official_spec_refs,
        page_layout=page_layout,
        styles=tuple(sorted(style_intents.values(), key=lambda style: style.id)),
        flow=tuple(flow),
    )
    return plan, tuple(sorted(issue_codes))


def _style_intent(source: ProjectedStyle, role: ContentRole) -> StyleIntent:
    font_family = (
        _normalize_display_text(source.font_family) if source.font_family is not None else None
    )
    facts = {
        "semantic_role": role.value,
        "font_family": font_family or None,
        "font_size_pt": source.font_size_pt,
        "bold": source.bold,
        "italic": source.italic,
        "alignment": source.alignment,
        "line_spacing": source.line_spacing,
    }
    signature = hashlib.sha256(
        json.dumps(facts, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return StyleIntent(id=f"plan-style-{signature[:24]}", **facts)


def _width_fraction(
    group: ProjectionGroup,
    page_layout: ProjectedPageLayout,
) -> tuple[float | None, str | None]:
    width_hwp = group.source.width_hwp
    if width_hwp is None or width_hwp <= 0:
        return None, None
    printable_width = (
        page_layout.width_hwp
        - page_layout.margin_left_hwp
        - page_layout.margin_right_hwp
    )
    if printable_width <= 0:
        return None, "page_printable_width_invalid"
    fraction = width_hwp / printable_width
    if not 0 < fraction <= 1:
        return None, "object_width_exceeds_plan_contract"
    return round(fraction, 6), None


def _render_as(
    node: TextContentNode | TableContentNode | ImageContentNode | FormulaContentNode,
) -> Literal["paragraph", "table", "image", "formula"]:
    if isinstance(node, TableContentNode):
        return "table"
    if isinstance(node, ImageContentNode):
        return "image"
    if isinstance(node, FormulaContentNode):
        return "formula"
    return "paragraph"


def _official_spec_refs(knowledge: KnowledgeIndex) -> tuple[str, ...]:
    return tuple(chunk.id for chunk in retrieve_hwpx_plan_chunks(chunks=knowledge.chunks))


def _hwpunit_to_mm(value: int) -> float:
    return round(value * HWPUNIT_TO_MM, 6)


def _write_contract_artifact(
    path: Path,
    contract: EvidenceIR | ContentIR | HwpDocumentPlan,
    root: Path,
) -> dict[str, str]:
    binding = _write_json_artifact(path, contract.model_dump(mode="json"), root)
    binding["contract_sha256"] = contract_sha256(contract)
    return binding


def _write_json_artifact(
    path: Path,
    value: dict[str, Any],
    root: Path,
) -> dict[str, str]:
    payload = _json_bytes(value)
    _write_bytes(path, payload)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"candidate artifact already exists: {path.name}")
    with path.open("xb") as stream:
        stream.write(payload)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _assert_evidence_is_pdf_only(evidence: EvidenceIR, document: PairDocument) -> None:
    if evidence.source_document_sha256 != document.pdf.sha256:
        raise ValueError("EvidenceIR must be bound to the paired PDF digest")
    source_by_id = {source.id: source for source in evidence.sources}
    original = source_by_id.get("input-pdf")
    if original is None or original.sha256 != document.pdf.sha256:
        raise ValueError("EvidenceIR is missing its paired-PDF source binding")
    serialized = json.dumps(
        evidence.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if document.hwpx.sha256 in serialized or document.source_hwp_sha256 in serialized:
        raise ValueError("HWP/HWPX target lineage leaked into EvidenceIR")
    for source in evidence.sources:
        if document.pdf.sha256 not in source.artifact_ref:
            raise ValueError("EvidenceIR contains a source outside the paired PDF lineage")


def _bbox_contract(
    pixel: tuple[float, float, float, float],
    width: int,
    height: int,
) -> BBox:
    x0, y0, x1, y1 = pixel
    if width < 1 or height < 1 or min(pixel) < 0 or x0 >= x1 or y0 >= y1:
        raise ValueError("invalid pixel bbox")
    if x1 > width or y1 > height or not all(math.isfinite(value) for value in pixel):
        raise ValueError("pixel bbox lies outside its rendered page")
    return BBox(
        pixel=(x0, y0, x1, y1),
        normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
    )


def _scaled_pdf_bbox(
    value: object,
    scale: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    raw = tuple(float(item) * scale for item in value)
    if not all(math.isfinite(item) for item in raw):
        return None
    x0 = min(max(raw[0], 0.0), float(width))
    y0 = min(max(raw[1], 0.0), float(height))
    x1 = min(max(raw[2], 0.0), float(width))
    y1 = min(max(raw[3], 0.0), float(height))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _normalize_display_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFC", value)
    visible = "".join(
        " " if character in "\r\n\t" else character
        for character in normalized
        if unicodedata.category(character) not in {"Cc", "Cf"}
        or character in "\r\n\t"
    )
    return " ".join(visible.split())


def _assert_candidate_inventory(root: Path, manifest: dict[str, Any]) -> None:
    expected: dict[str, str] = {}
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise TypeError("candidate manifest documents must be a list")
    for document in documents:
        if not isinstance(document, dict):
            raise TypeError("candidate manifest document must be an object")
        lineage_id = _required_sha256(document.get("lineage_id"), "candidate lineage")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TypeError("candidate manifest artifacts must be an object")
        for name in (
            "projection_source",
            "evidence_ir",
            "content_ir_candidate",
            "hwp_document_plan_candidate",
            "report",
        ):
            _collect_candidate_binding(expected, artifacts.get(name), lineage_id)
        for name in ("page_images", "image_crops"):
            bindings = artifacts.get(name)
            if not isinstance(bindings, list):
                raise TypeError(f"candidate artifact {name} must be a list")
            for binding in bindings:
                _collect_candidate_binding(expected, binding, lineage_id)

    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if any(path.is_symlink() for path in entries):
        raise ValueError("candidate bundle must not contain symlinks")
    actual_files = {
        path.relative_to(root).as_posix(): path for path in entries if path.is_file()
    }
    expected_files = {"manifest.json", *expected}
    if set(actual_files) != expected_files:
        missing = len(expected_files - set(actual_files))
        unexpected = len(set(actual_files) - expected_files)
        raise ValueError(
            f"candidate bundle artifact inventory mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for reference, expected_sha256 in expected.items():
        if _sha256(actual_files[reference]) != expected_sha256:
            raise ValueError(f"candidate artifact SHA-256 mismatch: {reference}")
    if _metadata_contains_forbidden_key(manifest, {"text", "source_ref", "locator"}):
        raise ValueError("candidate root manifest contains source text or private references")
    for document in documents:
        if not isinstance(document, dict):
            raise TypeError("candidate manifest document must be an object")
        artifacts = cast(dict[str, Any], document["artifacts"])
        report_binding = cast(dict[str, str], artifacts["report"])
        report_path = actual_files[report_binding["path"]]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if _metadata_contains_forbidden_key(report, {"text", "source_ref", "locator"}):
            raise ValueError("candidate report contains source text or private references")


def _collect_candidate_binding(
    expected: dict[str, str],
    value: object,
    lineage_id: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError("candidate artifact binding must be an object")
    path = value.get("path")
    sha256 = _required_sha256(value.get("sha256"), "candidate artifact")
    if not isinstance(path, str) or not path:
        raise ValueError("candidate artifact path must be non-empty")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise ValueError("candidate artifact path is unsafe")
    if not pure.parts or pure.parts[0] != lineage_id:
        raise ValueError("candidate artifact path is outside its lineage directory")
    normalized = path.casefold()
    if any(reference.casefold() == normalized for reference in expected):
        raise ValueError("candidate manifest contains duplicate artifact paths")
    expected[path] = sha256


def _metadata_contains_forbidden_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            key in forbidden
            or _metadata_contains_forbidden_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_metadata_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _assert_inputs_unchanged(
    bundle: PairBundle,
    capability_path: Path,
    capability_bytes: bytes,
    corpus_path: Path,
    corpus_bytes: bytes,
) -> None:
    if _sha256(bundle.manifest_path) != bundle.manifest_sha256:
        raise RuntimeError("pairs manifest changed during candidate build")
    if _verify_pair_inventory(bundle.root, bundle.manifest_path, bundle.documents) != bundle.file_hashes:
        raise RuntimeError("pairs bundle changed during candidate build")
    if _required_regular_file(capability_path, "capability profile").read_bytes() != capability_bytes:
        raise RuntimeError("capability profile changed during candidate build")
    if _required_regular_file(corpus_path, "knowledge corpus").read_bytes() != corpus_bytes:
        raise RuntimeError("knowledge corpus changed during candidate build")


def _required_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return resolved


def _required_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"{label} not found: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    return resolved


def _strict_string_map(value: object, keys: set[str], label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has an unexpected field set")
    result: dict[str, str] = {}
    for key in sorted(keys):
        item = value[key]
        if not isinstance(item, str) or not item.strip() or _has_control_character(item):
            raise ValueError(f"{label} field {key} must be a non-empty string")
        result[key] = item
    return result


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_count(path: Path) -> int:
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        if document.needs_pass:
            raise ValueError("paired PDF must not require a password")
        if document.page_count < 1:
            raise ValueError("paired PDF must contain at least one page")
        return cast(int, document.page_count)


if __name__ == "__main__":
    raise SystemExit(main())
