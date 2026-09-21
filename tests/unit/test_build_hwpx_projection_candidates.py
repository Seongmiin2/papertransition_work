from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Never

import pymupdf
import pytest
from lxml import etree  # type: ignore[import-untyped]

from ai.datasets.build_hwpx_projection_candidates import (
    ArtifactBinding,
    KnowledgeIndex,
    PairDocument,
    PdfObservation,
    ProjectionGroup,
    _build_plan_candidate,
    _unique_pdf_image_candidate,
    build_hwpx_projection_candidates,
)
from scan2hwpx.contracts import (
    ContentIR,
    ContentRole,
    EvidenceIR,
    HwpDocumentPlan,
    ObservationKind,
    TextContentNode,
)
from scan2hwpx.hwpx import ensure_minimal_template
from scan2hwpx.knowledge.hancom import HancomChunk
from scan2hwpx.reference.hwpx import (
    HwpxProjectionSource,
    ProjectedNode,
    ProjectedPageLayout,
    ProjectedSection,
)

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_hwpx(path: Path) -> None:
    ensure_minimal_template(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    section = etree.fromstring(entries["Contents/section0.xml"])
    paragraph = etree.Element(
        f"{{{HP}}}p",
        id="999",
        paraPrIDRef="20",
        styleIDRef="0",
        pageBreak="0",
        columnBreak="0",
        merged="0",
    )
    first = etree.SubElement(paragraph, f"{{{HP}}}run", charPrIDRef="3")
    etree.SubElement(first, f"{{{HP}}}t").text = "LeftTop LeftBottom "
    second = etree.SubElement(paragraph, f"{{{HP}}}run", charPrIDRef="4")
    etree.SubElement(second, f"{{{HP}}}t").text = "RightTop RightBottom"
    section.append(paragraph)
    entries["Contents/section0.xml"] = etree.tostring(
        section,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            entries["mimetype"],
            compress_type=zipfile.ZIP_STORED,
        )
        for name in sorted(entries):
            if name != "mimetype":
                archive.writestr(name, entries[name], compress_type=zipfile.ZIP_DEFLATED)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, str, str, str]:
    source_sha256 = hashlib.sha256(b"original-hwp-source").hexdigest()
    pair_root = tmp_path / "pairs"
    document_dir = pair_root / source_sha256
    document_dir.mkdir(parents=True)
    hwpx_path = document_dir / "source.hwpx"
    pdf_path = document_dir / "source.pdf"
    _write_hwpx(hwpx_path)
    with pymupdf.open() as pdf:  # type: ignore[no-untyped-call]
        page = pdf.new_page(width=300, height=400)
        page.insert_text((20, 40), "LeftTop")
        page.insert_text((20, 100), "LeftBottom")
        page.insert_text((170, 40), "RightTop")
        page.insert_text((170, 100), "RightBottom")
        pdf.save(pdf_path)
    hwpx_sha256 = _sha256(hwpx_path)
    pdf_sha256 = _sha256(pdf_path)
    manifest = {
        "schema_version": "hwp-hwpx-projection-pairs/1.0",
        "artifact_role": "paired_source_for_projection",
        "research_only": True,
        "rights_manifest": {"status": "missing"},
        "source_dataset_report_sha256": "1" * 64,
        "source_inventory_sha256": "2" * 64,
        "document_count": 1,
        "splits": {"train": 1, "validation": 0, "test": 0},
        "producer": {"name": "fixture", "version": "1"},
        "documents": [
            {
                "source_sha256": source_sha256,
                "source_ref": r"C:\private\student-name\exam.hwp",
                "split": "train",
                "page_count": 1,
                "hwpx": {
                    "path": f"{source_sha256}/source.hwpx",
                    "sha256": hwpx_sha256,
                },
                "pdf": {
                    "path": f"{source_sha256}/source.pdf",
                    "sha256": pdf_sha256,
                },
            }
        ],
    }
    (pair_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    capability_dir = tmp_path / "capability"
    capability_dir.mkdir()
    capability_path = capability_dir / "profile.json"
    capability_path.write_text(
        json.dumps(
            {
                "profile_id": "exam2hwpx-authoring-v1",
                "target_format": "hwpx",
                "content_policy": "immutable",
                "planner_must_reference_content": True,
                "planner_forbidden": ["raw_xml", "local_file_path", "external_url"],
            }
        ),
        encoding="utf-8",
    )
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    knowledge_path = knowledge_dir / "chunks.jsonl"
    knowledge_path.write_text(
        json.dumps(
            {
                "id": "official-hwpx-fixture",
                "source_id": "hancom-hwpx-format",
                "title": "HWPX format and validation",
                "section": "package paragraph table style validation",
                "text": "HWPX package manifest header section paragraph table style validation.",
                "tags": [
                    "hwpx",
                    "package",
                    "paragraph",
                    "table",
                    "style-reference",
                    "validation",
                ],
                "source_url": "https://tech.hancom.com/hwpxformat/",
                "source_sha256": "f" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        pair_root,
        capability_path,
        knowledge_path,
        source_sha256,
        hwpx_sha256,
        pdf_sha256,
    )


def test_builds_review_only_pdf_grounded_candidate_bundle(tmp_path: Path) -> None:
    pair_root, capability, knowledge, source_sha, hwpx_sha, pdf_sha = _write_inputs(
        tmp_path
    )
    output = tmp_path / "candidates"

    manifest = build_hwpx_projection_candidates(
        pair_root,
        capability,
        knowledge,
        output,
        max_workers=2,
        dpi=72,
    )

    assert manifest["research_only"] is True
    assert manifest["human_review_required"] is True
    assert manifest["training_eligible"] is False
    assert manifest["release_eligible"] is False
    assert manifest["document_count"] == 1
    document_dir = output / source_sha
    projection = HwpxProjectionSource.model_validate_json(
        (document_dir / "projection_source.json").read_bytes()
    )
    evidence_path = document_dir / "evidence_ir.json"
    evidence = EvidenceIR.model_validate_json(evidence_path.read_bytes())
    content = ContentIR.model_validate_json(
        (document_dir / "content_ir.candidate.json").read_bytes()
    )
    plan = HwpDocumentPlan.model_validate_json(
        (document_dir / "hwp_document_plan.candidate.json").read_bytes()
    )
    content.assert_evidence_integrity(evidence)
    plan.assert_content_integrity(content)
    assert projection.source_document_sha256 == source_sha
    assert evidence.source_document_sha256 == pdf_sha
    assert len(content.nodes) == 1
    assert content.nodes[0].kind == "text"
    assert content.nodes[0].text == "LeftTop LeftBottom RightTop RightBottom"
    assert all(node.needs_review for node in content.nodes)
    evidence_bytes = evidence_path.read_bytes()
    assert hwpx_sha.encode() not in evidence_bytes
    assert source_sha.encode() not in evidence_bytes

    root_metadata = (output / "manifest.json").read_text(encoding="utf-8")
    report_metadata = (document_dir / "report.json").read_text(encoding="utf-8")
    report = json.loads(report_metadata)
    assert report["alignment"]["coverage"] > 0.99
    for private_value in (
        "LeftTop LeftBottom RightTop RightBottom",
        "student-name",
        r"C:\private",
    ):
        assert private_value not in root_metadata
        assert private_value not in report_metadata
    for binding in manifest["documents"][0]["artifacts"]["page_images"]:
        artifact = output.joinpath(*binding["path"].split("/"))
        assert _sha256(artifact) == binding["sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        build_hwpx_projection_candidates(
            pair_root,
            capability,
            knowledge,
            output,
            max_workers=1,
            dpi=72,
        )


class _FailingProjectionReader:
    def __call__(
        self,
        path: str | Path,
        *,
        source_document_sha256: str,
        expected_hwpx_sha256: str,
        expected_page_count: int,
    ) -> Never:
        del path, source_document_sha256, expected_hwpx_sha256, expected_page_count
        raise RuntimeError("injected projection failure")


def test_worker_failure_does_not_publish_partial_bundle(tmp_path: Path) -> None:
    pair_root, capability, knowledge, source_sha, _, _ = _write_inputs(tmp_path)
    output = tmp_path / "failed-candidates"

    with pytest.raises(RuntimeError, match=source_sha):
        build_hwpx_projection_candidates(
            pair_root,
            capability,
            knowledge,
            output,
            max_workers=1,
            dpi=72,
            projection_reader=_FailingProjectionReader(),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".failed-candidates.*.partial"))


def test_plan_infers_page_break_across_unreliable_intermediate_node() -> None:
    source_sha = "1" * 64
    node_ids = tuple(f"content-{index}" for index in range(1, 5))
    projected_nodes = tuple(
        ProjectedNode(
            id=f"projection-{index}",
            locator=f"section0/p{index}",
            kind="text",
            text=f"node {index}",
        )
        for index in range(1, 5)
    )
    groups = tuple(
        ProjectionGroup(
            id=node_id,
            kind="text",
            source_nodes=(projected_node,),
            text=projected_node.text,
            role_hint=None,
            style_id=None,
            page_break=False,
            column_break=False,
        )
        for node_id, projected_node in zip(node_ids, projected_nodes, strict=True)
    )
    layout = ProjectedPageLayout(
        width_hwp=59528,
        height_hwp=84188,
        margin_top_hwp=4252,
        margin_right_hwp=4252,
        margin_bottom_hwp=4252,
        margin_left_hwp=4252,
        margin_header_hwp=0,
        margin_footer_hwp=0,
        margin_gutter_hwp=0,
    )
    projection = HwpxProjectionSource(
        source_document_sha256=source_sha,
        hwpx_sha256="2" * 64,
        page_count=3,
        sections=(
            ProjectedSection(
                index=0,
                layout=layout,
                node_refs=tuple(node.id for node in projected_nodes),
            ),
        ),
        nodes=projected_nodes,
        styles=(),
        assets=(),
        issues=(),
        stats={},
    )
    content = ContentIR(
        id="content-review-candidate",
        evidence_ir_id="evidence-pdf",
        evidence_ir_sha256="3" * 64,
        nodes=tuple(
            TextContentNode(
                id=node_id,
                role=ContentRole.OTHER,
                text=f"node {index}",
                evidence_refs=(f"observation-{index}",),
                confidence=1.0,
                needs_review=True,
            )
            for index, node_id in enumerate(node_ids, start=1)
        ),
        reading_order=node_ids,
    )
    document = PairDocument(
        source_hwp_sha256=source_sha,
        split="train",
        page_count=3,
        hwpx=ArtifactBinding(path=f"{source_sha}/source.hwpx", sha256="2" * 64),
        pdf=ArtifactBinding(path=f"{source_sha}/source.pdf", sha256="4" * 64),
    )
    node_pages = {
        node_ids[0]: (1, True),
        node_ids[1]: (2, False),
        node_ids[2]: (2, True),
        node_ids[3]: (3, True),
    }

    plan, issues = _build_plan_candidate(
        document,
        projection,
        groups,
        content,
        node_pages,
        {"profile_id": "exam2hwpx-authoring-v1"},
        KnowledgeIndex(
            sha256="5" * 64,
            chunks=(
                HancomChunk(
                    id="official-hwpx-fixture",
                    source_id="hancom-hwpx-format",
                    title="HWPX format and validation",
                    section="package paragraph table style validation",
                    text=(
                        "HWPX package manifest header section paragraph table "
                        "style reference validation conformance DVC."
                    ),
                    tags=(
                        "hwpx",
                        "package",
                        "paragraph",
                        "table",
                        "style-reference",
                        "validation",
                    ),
                    source_url="https://tech.hancom.com/hwpxformat/",
                    source_sha256="f" * 64,
                ),
            ),
            ids=frozenset({"official-hwpx-fixture"}),
        ),
    )

    assert sum(item.kind == "page_break" for item in plan.flow) == 2
    assert "page_break_inferred_from_pdf_alignment" in issues


def test_image_candidate_requires_a_unique_compatible_aspect_ratio() -> None:
    source = ProjectedNode(
        id="projection-image",
        locator="section0/p1/image1",
        kind="image",
        asset_id="target-image",
        width_hwp=200,
        height_hwp=100,
    )
    group = ProjectionGroup(
        id="content-image",
        kind="image",
        source_nodes=(source,),
        text="",
        role_hint=None,
        style_id=None,
        page_break=False,
        column_break=False,
    )
    wrong = PdfObservation(
        id="pdf-image-square",
        page_no=1,
        kind=ObservationKind.IMAGE,
        text=None,
        bbox=(0, 0, 100, 100),
        source_ref="page-image",
        asset_source_ref="square-crop",
    )
    matching = PdfObservation(
        id="pdf-image-wide",
        page_no=1,
        kind=ObservationKind.IMAGE,
        text=None,
        bbox=(0, 0, 200, 100),
        source_ref="page-image",
        asset_source_ref="wide-crop",
    )

    assert _unique_pdf_image_candidate(group, (wrong, matching), set()) == matching
    duplicate_match = PdfObservation(
        id="pdf-image-wide-2",
        page_no=1,
        kind=ObservationKind.IMAGE,
        text=None,
        bbox=(0, 0, 200, 100),
        source_ref="page-image",
        asset_source_ref="wide-crop-2",
    )
    assert (
        _unique_pdf_image_candidate(group, (matching, duplicate_match), set())
        is None
    )
