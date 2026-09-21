from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from scan2hwpx.contracts import (
    ContentIR,
    ContentPlanItem,
    EvidenceIR,
    EvidenceSourceKind,
    HwpDocumentPlan,
    ImageContentNode,
    LayoutIntent,
    ObservationKind,
    PageBreakPlanItem,
    PageLayoutIntent,
    TableContentNode,
    TextContentNode,
    contract_sha256,
)
from scan2hwpx.evaluation.safe_artifact_io import (
    SafeArtifactIOError,
    publish_file_create_only,
    read_bounded_regular_file,
    remove_staging_directory,
    sha256_bounded_regular_file,
)
from scan2hwpx.evaluation.strict_json import require_strict_json_bytes
from scan2hwpx.hwpx import (
    IMAGE_ASSET_BUNDLE_VERSION,
    IMAGE_DESIGN_PROFILE_ID,
    MAX_IMAGE_ASSET_BYTES,
    MAX_IMAGE_ASSET_TOTAL_BYTES,
    MAX_IMAGE_ASSETS,
    SUPPORTED_CAPABILITY_PROFILE_ID,
    ImageAssetBundle,
    PlanCompileResult,
    PngImageAsset,
    build_image_asset_bundle,
    compile_plan_hwpx,
    image_asset_bundle_sha256,
    validate_hwpx,
)
from scan2hwpx.model_a.inference import ModelAPageAnalysis
from scan2hwpx.model_a.table_fusion import (
    ModelATableDocumentAssemblyArtifact,
    ModelATablePageFusionArtifact,
)

_MAX_ARTIFACT_JSON_BYTES = 64 * 1024 * 1024
_MAX_PAGE_JSON_BYTES = 16 * 1024 * 1024
_MAX_CANDIDATE_PLAN_BYTES = 64 * 1024 * 1024
_MAX_SHA_FILE_BYTES = 256
_MAX_JSON_DEPTH = 96
_MAX_JSON_NODES = 2_000_000

_CANONICAL_PAGE_LAYOUT = PageLayoutIntent(
    width_mm=210.0,
    height_mm=297.0,
    margin_top_mm=20.0,
    margin_right_mm=30.0,
    margin_bottom_mm=15.0,
    margin_left_mm=30.0,
    columns=2,
    column_gap_mm=8.0,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class TableFusionHwpxSmokeError(ValueError):
    """The persisted fusion batch is not safe to compile as a smoke artifact."""


@dataclass(frozen=True, slots=True)
class _FusionBatch:
    root: Path
    assembly: ModelATableDocumentAssemblyArtifact
    assembly_artifact_sha256: str
    promoted_evidence_ir: EvidenceIR
    content_ir: ContentIR
    page_analyses: tuple[ModelAPageAnalysis, ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one persisted Model A table-fusion batch to a nonrelease HWPX smoke "
            "artifact."
        )
    )
    parser.add_argument("--fusion-batch-dir", type=Path, required=True)
    parser.add_argument("--candidate-document-dir", type=Path, required=True)
    parser.add_argument("--output-hwpx", type=Path, required=True)
    return parser.parse_args(argv)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
        _require_directory_metadata(metadata, label)
        resolved = path.resolve(strict=True)
        _require_directory_metadata(resolved.lstat(), label)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise TableFusionHwpxSmokeError(f"{label} is missing or unreadable") from exc
    return resolved


def _require_directory_metadata(metadata: object, label: str) -> None:
    mode = getattr(metadata, "st_mode", None)
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not isinstance(mode, int)
        or stat.S_ISLNK(mode)
        or not stat.S_ISDIR(mode)
        or bool(attributes & reparse_attribute)
    ):
        raise TableFusionHwpxSmokeError(f"{label} must be a non-reparse directory")


def _direct_child(root: Path, name: str, label: str) -> Path:
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise TableFusionHwpxSmokeError(f"{label} is not a safe direct-child name")
    child = root / name
    if child.parent != root:
        raise TableFusionHwpxSmokeError(f"{label} escapes its configured root")
    return child


def _read_bounded(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        return read_bounded_regular_file(path, max_bytes=max_bytes, label=label)
    except SafeArtifactIOError as exc:
        raise TableFusionHwpxSmokeError(str(exc)) from exc


def _validate_sha_sidecar(path: Path, payload: bytes, *, label: str) -> str:
    digest = _sha256(payload)
    sidecar_path = path.with_suffix(".sha256")
    sidecar = _read_bounded(sidecar_path, max_bytes=_MAX_SHA_FILE_BYTES, label=f"{label} SHA")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar != expected:
        raise TableFusionHwpxSmokeError(f"{label} SHA file mismatch")
    return digest


def _load_model(
    path: Path,
    model: type[_ModelT],
    *,
    max_bytes: int,
    label: str,
    canonical: bool,
    sha_sidecar: bool,
) -> tuple[_ModelT, bytes, str]:
    raw = _read_bounded(path, max_bytes=max_bytes, label=label)
    require_strict_json_bytes(
        raw,
        max_depth=_MAX_JSON_DEPTH,
        max_nodes=_MAX_JSON_NODES,
        label=label,
    )
    try:
        artifact = model.model_validate_json(raw, strict=True)
    except (TypeError, ValueError) as exc:
        raise TableFusionHwpxSmokeError(f"{label} does not match {model.__name__}") from exc
    if canonical and raw != _canonical_json_bytes(artifact.model_dump(mode="json")):
        raise TableFusionHwpxSmokeError(f"{label} bytes are not canonical")
    digest = (
        _validate_sha_sidecar(path, raw, label=label) if sha_sidecar else _sha256(raw)
    )
    return artifact, raw, digest


def _page_output_key(page_no: int, page_id: str) -> str:
    return f"page-{page_no:04d}-{_sha256(page_id.encode('utf-8'))[:12]}"


def _load_fusion_batch(root: Path) -> _FusionBatch:
    assembly, _assembly_raw, assembly_sha256 = _load_model(
        _direct_child(root, "assembly.json", "assembly artifact"),
        ModelATableDocumentAssemblyArtifact,
        max_bytes=_MAX_ARTIFACT_JSON_BYTES,
        label="table fusion assembly",
        canonical=True,
        sha_sidecar=True,
    )
    promoted, _promoted_raw, _promoted_raw_sha256 = _load_model(
        _direct_child(root, "promoted-evidence-ir.json", "promoted EvidenceIR"),
        EvidenceIR,
        max_bytes=_MAX_ARTIFACT_JSON_BYTES,
        label="promoted EvidenceIR",
        canonical=True,
        sha_sidecar=True,
    )
    content, _content_raw, _content_raw_sha256 = _load_model(
        _direct_child(root, "content-ir.json", "fused ContentIR"),
        ContentIR,
        max_bytes=_MAX_ARTIFACT_JSON_BYTES,
        label="fused ContentIR",
        canonical=True,
        sha_sidecar=True,
    )

    source = assembly.source
    promoted_sha256 = contract_sha256(promoted)
    if (
        source.promoted_evidence_ir_id != promoted.id
        or source.promoted_evidence_ir_sha256 != promoted_sha256
        or source.source_document_sha256 != promoted.source_document_sha256
    ):
        raise TableFusionHwpxSmokeError("assembly promoted EvidenceIR lineage mismatch")
    if (
        source.content_ir_id != content.id
        or source.content_ir_revision != content.revision
        or assembly.content_ir != content
        or assembly.content_ir_sha256 != contract_sha256(content)
    ):
        raise TableFusionHwpxSmokeError("assembly fused ContentIR lineage mismatch")
    try:
        content.assert_evidence_integrity(promoted)
    except ValueError as exc:
        raise TableFusionHwpxSmokeError("fused ContentIR evidence lineage mismatch") from exc

    page_analyses: list[ModelAPageAnalysis] = []
    for page in assembly.pages:
        page_key = _page_output_key(page.page_no, page.page_id)
        page_dir = _require_directory(
            _direct_child(root, page_key, "fusion page directory"),
            "fusion page directory",
        )
        fusion, _fusion_raw, fusion_raw_sha256 = _load_model(
            _direct_child(page_dir, "fusion.json", "page fusion artifact"),
            ModelATablePageFusionArtifact,
            max_bytes=_MAX_PAGE_JSON_BYTES,
            label=f"page fusion {page.page_id}",
            canonical=True,
            sha_sidecar=True,
        )
        analysis, _analysis_raw, _analysis_raw_sha256 = _load_model(
            _direct_child(page_dir, "fused-page-analysis.json", "fused page analysis"),
            ModelAPageAnalysis,
            max_bytes=_MAX_PAGE_JSON_BYTES,
            label=f"fused page analysis {page.page_id}",
            canonical=True,
            sha_sidecar=True,
        )
        analysis_sha256 = contract_sha256(analysis)
        if (
            page.fusion_artifact_sha256 != fusion_raw_sha256
            or page.fused_page_analysis_sha256 != analysis_sha256
            or fusion.fused_page_analysis != analysis
            or fusion.fused_page_analysis_sha256 != analysis_sha256
        ):
            raise TableFusionHwpxSmokeError(f"page fusion digest mismatch: {page.page_id}")
        fusion_source = fusion.source
        if (
            fusion_source.base_evidence_ir_id != promoted.id
            or fusion_source.promoted_evidence_ir_sha256 != promoted_sha256
            or fusion_source.promotion_artifact_sha256 != source.promotion_artifact_sha256
            or fusion_source.target_page_id != page.page_id
            or fusion_source.target_page_no != page.page_no
        ):
            raise TableFusionHwpxSmokeError(f"page fusion source lineage mismatch: {page.page_id}")
        if (
            analysis.evidence_ir_id != promoted.id
            or analysis.evidence_ir_sha256 != promoted_sha256
            or analysis.page_id != page.page_id
            or analysis.page_no != page.page_no
        ):
            raise TableFusionHwpxSmokeError(f"fused page analysis lineage mismatch: {page.page_id}")
        if not analysis.reading_order:
            raise TableFusionHwpxSmokeError(
                f"empty fused pages cannot be represented by the compiler: {page.page_id}"
            )
        page_analyses.append(analysis)

    flattened_nodes = tuple(node for page in page_analyses for node in page.nodes)
    flattened_order = tuple(ref for page in page_analyses for ref in page.reading_order)
    if flattened_nodes != content.nodes:
        raise TableFusionHwpxSmokeError("page nodes do not flatten to assembled ContentIR")
    if flattened_order != content.reading_order:
        raise TableFusionHwpxSmokeError("page reading order does not flatten to assembled ContentIR")

    return _FusionBatch(
        root=root,
        assembly=assembly,
        assembly_artifact_sha256=assembly_sha256,
        promoted_evidence_ir=promoted,
        content_ir=content,
        page_analyses=tuple(page_analyses),
    )


def _load_official_spec_refs(candidate_root: Path) -> tuple[str, ...]:
    plan, _raw, _raw_sha256 = _load_model(
        _direct_child(
            candidate_root,
            "hwp_document_plan.candidate.json",
            "candidate HwpDocumentPlan",
        ),
        HwpDocumentPlan,
        max_bytes=_MAX_CANDIDATE_PLAN_BYTES,
        label="candidate HwpDocumentPlan",
        canonical=False,
        sha_sidecar=False,
    )
    return plan.official_spec_refs


def _image_width_fraction(
    evidence: EvidenceIR,
    analysis: ModelAPageAnalysis,
    node: ImageContentNode,
) -> float:
    pages = tuple(
        page
        for page in evidence.pages
        if page.id == analysis.page_id and page.page_no == analysis.page_no
    )
    if len(pages) != 1:
        raise TableFusionHwpxSmokeError(
            f"image analysis page is missing or ambiguous: {node.id}"
        )
    if len(node.evidence_refs) != 1:
        raise TableFusionHwpxSmokeError(
            f"image node must reference exactly one IMAGE observation: {node.id}"
        )

    observation_ref = node.evidence_refs[0]
    observations = tuple(
        observation
        for observation in pages[0].observations
        if observation.id == observation_ref
    )
    if len(observations) != 1 or observations[0].kind != ObservationKind.IMAGE:
        raise TableFusionHwpxSmokeError(
            f"image node does not reference an IMAGE observation on its analysis page: {node.id}"
        )
    observation = observations[0]
    if node.asset_ref not in observation.source_refs:
        raise TableFusionHwpxSmokeError(
            f"image observation does not reference the node asset: {node.id}"
        )

    page_layout = _CANONICAL_PAGE_LAYOUT
    total_gap_mm = page_layout.column_gap_mm * (page_layout.columns - 1)
    usable_width_mm = (
        page_layout.width_mm
        - page_layout.margin_left_mm
        - page_layout.margin_right_mm
        - total_gap_mm
    )
    column_width_mm = usable_width_mm / page_layout.columns
    normalized_x0, _normalized_y0, normalized_x1, _normalized_y1 = (
        observation.bbox.normalized
    )
    image_width_mm = (normalized_x1 - normalized_x0) * page_layout.width_mm
    if column_width_mm <= 0 or image_width_mm <= 0:
        raise TableFusionHwpxSmokeError(
            f"image width cannot be projected onto the canonical page: {node.id}"
        )
    return round(min(1.0, image_width_mm / column_width_mm), 8)


def _build_smoke_plan(
    batch: _FusionBatch,
    official_spec_refs: tuple[str, ...],
) -> HwpDocumentPlan:
    content = batch.content_ir
    content_by_id = {node.id: node for node in content.nodes}
    flow: list[ContentPlanItem | PageBreakPlanItem] = []
    ordered_content_refs: list[str] = []
    ordinal = 0

    for page_index, analysis in enumerate(batch.page_analyses, start=1):
        if page_index > 1:
            flow.append(PageBreakPlanItem(id=f"flow-page-break-{page_index:04d}"))
        for content_ref in analysis.reading_order:
            ordinal += 1
            node = content_by_id[content_ref]
            if isinstance(node, TextContentNode):
                render_as = "paragraph"
                layout = LayoutIntent()
            elif isinstance(node, TableContentNode):
                render_as = "table"
                layout = LayoutIntent()
            elif isinstance(node, ImageContentNode):
                render_as = "image"
                layout = LayoutIntent(
                    width_fraction=_image_width_fraction(
                        batch.promoted_evidence_ir,
                        analysis,
                        node,
                    )
                )
            else:
                raise TableFusionHwpxSmokeError(
                    f"unsupported fused content kind for compiler smoke: {node.kind}"
                )
            flow.append(
                ContentPlanItem(
                    id=f"flow-content-{ordinal:06d}",
                    render_as=render_as,
                    content_ref=content_ref,
                    layout=layout,
                )
            )
            ordered_content_refs.append(content_ref)

    if tuple(ordered_content_refs) != content.reading_order:
        raise TableFusionHwpxSmokeError("plan content order differs from fused ContentIR")
    content_sha256 = contract_sha256(content)
    plan = HwpDocumentPlan(
        id=f"table-fusion-hwpx-smoke-{content_sha256[:24]}",
        content_ir_id=content.id,
        content_ir_revision=content.revision,
        content_ir_sha256=content_sha256,
        capability_profile_id=SUPPORTED_CAPABILITY_PROFILE_ID,
        design_profile_id=IMAGE_DESIGN_PROFILE_ID,
        official_spec_refs=official_spec_refs,
        page_layout=_CANONICAL_PAGE_LAYOUT,
        styles=(),
        flow=tuple(flow),
    )
    try:
        plan.assert_content_integrity(content)
    except ValueError as exc:
        raise TableFusionHwpxSmokeError("generated smoke plan failed content integrity") from exc
    return plan


def _asset_file_name(artifact_ref: str, asset_ref: str) -> str:
    name = artifact_ref.rsplit("/", maxsplit=1)[-1]
    if (
        not name
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or Path(name).suffix.casefold() != ".png"
    ):
        raise TableFusionHwpxSmokeError(
            f"image source does not name a safe direct-child PNG: {asset_ref}"
        )
    return name


def _build_image_assets(batch: _FusionBatch, candidate_root: Path) -> ImageAssetBundle:
    assets_root = _require_directory(
        _direct_child(candidate_root, "assets", "candidate assets directory"),
        "candidate assets directory",
    )
    image_nodes = tuple(
        node for node in batch.content_ir.nodes if isinstance(node, ImageContentNode)
    )
    if len(image_nodes) > MAX_IMAGE_ASSETS:
        raise TableFusionHwpxSmokeError("image asset count exceeds the bundle limit")

    sources = {source.id: source for source in batch.promoted_evidence_ir.sources}
    file_by_ref: dict[str, str] = {}
    for node in image_nodes:
        source = sources.get(node.asset_ref)
        if source is None:
            raise TableFusionHwpxSmokeError(f"missing image EvidenceSource: {node.asset_ref}")
        if source.kind not in (EvidenceSourceKind.PAGE_IMAGE, EvidenceSourceKind.CROP):
            raise TableFusionHwpxSmokeError(
                f"image EvidenceSource has an unsupported kind: {node.asset_ref}"
            )
        if source.sha256 is None:
            raise TableFusionHwpxSmokeError(
                f"image EvidenceSource has no SHA-256: {node.asset_ref}"
            )
        file_by_ref[node.asset_ref] = _asset_file_name(source.artifact_ref, node.asset_ref)

    expected_names = set(file_by_ref.values())
    if len(expected_names) != len(file_by_ref):
        raise TableFusionHwpxSmokeError("distinct image refs resolve to the same candidate asset")
    try:
        actual_names = {
            entry.name
            for entry in assets_root.iterdir()
            if entry.suffix.casefold() == ".png"
        }
    except OSError as exc:
        raise TableFusionHwpxSmokeError("candidate assets directory is unreadable") from exc
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise TableFusionHwpxSmokeError(
            "candidate PNG coverage mismatch" + (": " + "; ".join(details) if details else "")
        )

    total_bytes = 0
    assets: list[PngImageAsset] = []
    for asset_ref in sorted(file_by_ref):
        source = sources[asset_ref]
        file_name = file_by_ref[asset_ref]
        raw = _read_bounded(
            _direct_child(assets_root, file_name, f"candidate image {asset_ref}"),
            max_bytes=MAX_IMAGE_ASSET_BYTES,
            label=f"candidate image {asset_ref}",
        )
        total_bytes += len(raw)
        if total_bytes > MAX_IMAGE_ASSET_TOTAL_BYTES:
            raise TableFusionHwpxSmokeError("image assets exceed the bundle total byte limit")
        digest = _sha256(raw)
        if digest != source.sha256:
            raise TableFusionHwpxSmokeError(
                f"candidate image does not match EvidenceSource SHA-256: {asset_ref}"
            )
        assets.append(
            PngImageAsset(
                asset_ref=asset_ref,
                media_type="image/png",
                sha256=digest,
                payload=raw,
            )
        )

    try:
        return build_image_asset_bundle(
            batch.promoted_evidence_ir,
            batch.content_ir,
            tuple(assets),
        )
    except (TypeError, ValueError) as exc:
        raise TableFusionHwpxSmokeError("image asset bundle validation failed") from exc


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _resolve_new_output(
    output_path: Path,
    *,
    input_roots: tuple[Path, ...],
) -> tuple[Path, Path]:
    if output_path.name in {"", ".", ".."}:
        raise ValueError("smoke output must name a file")
    if output_path.suffix.casefold() != ".hwpx":
        raise ValueError("smoke output path must use the .hwpx suffix")
    preliminary = output_path.resolve(strict=False)
    if any(_is_within(preliminary, root) for root in input_roots):
        raise ValueError("smoke output must be outside input artifact directories")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parent = _require_directory(output_path.parent, "smoke output parent")
    destination = parent / output_path.name
    if any(_is_within(destination, root) for root in input_roots):
        raise ValueError("smoke output must be outside input artifact directories")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    return destination, parent


def _compile_create_only(
    batch: _FusionBatch,
    plan: HwpDocumentPlan,
    image_assets: ImageAssetBundle,
    destination: Path,
    parent: Path,
) -> PlanCompileResult:
    staging = Path(
        tempfile.mkdtemp(
            dir=parent,
            prefix=f".{destination.stem}.",
            suffix=".partial",
        )
    )
    staged_output = staging / destination.name
    try:
        result = compile_plan_hwpx(
            batch.content_ir,
            plan,
            staged_output,
            evidence_ir=batch.promoted_evidence_ir,
            image_assets=image_assets,
        )
        validation = validate_hwpx(staged_output)
        if not validation.valid:
            raise TableFusionHwpxSmokeError(
                "compiled smoke HWPX failed validation: " + "; ".join(validation.errors)
            )
        try:
            staged_size = staged_output.lstat().st_size
            staged_sha256 = sha256_bounded_regular_file(
                staged_output,
                max_bytes=staged_size,
                label="staged smoke HWPX",
            )
        except (OSError, SafeArtifactIOError) as exc:
            raise TableFusionHwpxSmokeError("compiled smoke HWPX cannot be verified") from exc
        if staged_sha256 != result.artifact_sha256:
            raise TableFusionHwpxSmokeError("compiled smoke HWPX digest mismatch")
        try:
            publish_file_create_only(staged_output, destination)
        except SafeArtifactIOError as exc:
            raise TableFusionHwpxSmokeError("smoke output changed during publication") from exc
        return result
    finally:
        remove_staging_directory(staging, parent=parent)


def _summary(
    batch: _FusionBatch,
    plan: HwpDocumentPlan,
    image_assets: ImageAssetBundle,
    result: PlanCompileResult,
) -> dict[str, object]:
    counts = Counter(node.kind for node in batch.content_ir.nodes)
    return {
        "schema_version": "model-a-table-fusion-hwpx-smoke/1.0",
        "artifact_role": "nonrelease_table_fusion_hwpx_smoke",
        "status": "compiled_unverified_smoke",
        "research_only": True,
        "human_review_required": True,
        "verification_status": "unverified",
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "execution_policy": "strict_local_create_only_no_network_no_retry",
        "assembly_artifact_sha256": batch.assembly_artifact_sha256,
        "promoted_evidence_ir_sha256": contract_sha256(batch.promoted_evidence_ir),
        "content_ir_sha256": contract_sha256(batch.content_ir),
        "hwp_document_plan_sha256": contract_sha256(plan),
        "image_asset_bundle_version": IMAGE_ASSET_BUNDLE_VERSION,
        "image_asset_bundle_sha256": image_asset_bundle_sha256(image_assets),
        "compiler_version": result.compiler_version,
        "output_hwpx_sha256": result.artifact_sha256,
        "page_count": len(batch.page_analyses),
        "content_node_count": len(batch.content_ir.nodes),
        "text_node_count": counts["text"],
        "table_node_count": counts["table"],
        "image_node_count": counts["image"],
        "flow_item_count": len(plan.flow),
        "style_count": len(plan.styles),
        "official_spec_ref_count": len(plan.official_spec_refs),
        "paragraph_count": result.paragraph_count,
        "page_break_count": result.page_break_count,
        "hwpx_package_valid": True,
        "limitations": [
            "The source table-fusion artifacts remain unverified and human-review required.",
            "This smoke does not perform visual comparison, DVC, or Hancom round-trip testing.",
            "The compiled artifact is not eligible for golden, training, or release use.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    fusion_root = _require_directory(args.fusion_batch_dir, "fusion batch directory")
    candidate_root = _require_directory(
        args.candidate_document_dir,
        "candidate document directory",
    )
    destination, parent = _resolve_new_output(
        args.output_hwpx,
        input_roots=(fusion_root, candidate_root),
    )

    batch = _load_fusion_batch(fusion_root)
    official_spec_refs = _load_official_spec_refs(candidate_root)
    plan = _build_smoke_plan(batch, official_spec_refs)
    image_assets = _build_image_assets(batch, candidate_root)
    result = _compile_create_only(
        batch,
        plan,
        image_assets,
        destination,
        parent,
    )
    print(json.dumps(_summary(batch, plan, image_assets, result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
