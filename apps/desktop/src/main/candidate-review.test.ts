import { createHash } from "node:crypto";
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { openCandidateReviewBundle } from "./candidate-review.js";

const roots: string[] = [];
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

interface Fixture {
  root: string;
  lineageId: string;
  contentPath: string;
  trackedPaths: string[];
  manifest: Record<string, any>;
  writeManifest(): void;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("candidate review bundle", () => {
  it("loads one document lazily and derives page overlays without changing the bundle", () => {
    const fixture = writeFixture();
    const before = snapshot(fixture.trackedPaths);

    const bundle = openCandidateReviewBundle(fixture.root);
    expect(bundle.summary).toMatchObject({
      schemaVersion: "hwp-human-review-candidates/1.0",
      documentCount: 1,
      pageCount: 2,
    });
    expect(bundle.documents).toEqual([
      expect.objectContaining({
        lineageId: fixture.lineageId,
        pageCount: 2,
        status: "needs_human_review",
      }),
    ]);

    const document = bundle.loadDocument(fixture.lineageId);
    expect(document.pages.map((page) => page.pageNo)).toEqual([1, 2]);
    expect(document.overlays).toHaveLength(2);
    expect(document.overlays[0]).toMatchObject({
      contentRef: "content-text",
      pageNumbers: [1, 2],
      primaryPageNo: 1,
      spansMultiplePages: true,
      usesPageRegionFallback: false,
    });
    expect(document.overlays[0].boxes.map((box) => box.observationId)).toEqual([
      "page-0001-text-00001",
      "page-0002-text-00001",
    ]);
    expect(document.overlays[1]).toMatchObject({
      contentRef: "content-image",
      pageNumbers: [2],
      spansMultiplePages: false,
      usesPageRegionFallback: true,
    });
    expect(document.imageGroundingCandidates).toEqual([
      {
        observationRef: "page-0002-image-00001",
        assetRef: "page-0002-image-00001-crop",
        pageNo: 2,
        sourceKind: "crop",
        sha256: sha256(cropBytes()),
      },
    ]);

    const bytes = bundle.readPageImage(fixture.lineageId, 1);
    expect(bytes.mimeType).toBe("image/png");
    expect(Buffer.from(bytes.bytes)).toEqual(pageBytes("one"));
    const dataUrl = bundle.readPageImageDataUrl(fixture.lineageId, 2);
    expect(dataUrl.dataUrl).toBe(
      `data:image/png;base64,${pageBytes("two").toString("base64")}`,
    );
    expect(
      JSON.stringify({
        summary: bundle.summary,
        documents: bundle.documents,
        document,
        dataUrl,
      }),
    ).not.toContain(fixture.root);
    expect(snapshot(fixture.trackedPaths)).toEqual(before);
  });

  it("does not touch registered artifacts until their document is loaded", () => {
    const fixture = writeFixture();
    rmSync(fixture.contentPath);

    const bundle = openCandidateReviewBundle(fixture.root);
    expect(bundle.summary.documentCount).toBe(1);
    expect(() => bundle.loadDocument(fixture.lineageId)).toThrow(/ContentIR is missing/);
  });

  it("rejects a bundle after its selected manifest snapshot changes", () => {
    const fixture = writeFixture();
    const bundle = openCandidateReviewBundle(fixture.root);
    const manifestPath = join(fixture.root, "manifest.json");
    const original = readFileSync(manifestPath, "utf8");

    writeFileSync(manifestPath, `${original.trimEnd()}\n\n`, "utf8");

    expect(() => bundle.assertCurrentSnapshot()).toThrow(
      /no longer matches the selected snapshot/,
    );
  });

  it("rejects unregistered page images even when a file exists", () => {
    const fixture = writeFixture();
    writeFileSync(
      join(fixture.root, fixture.lineageId, "pages", "page-0003.png"),
      pageBytes("not-registered"),
    );

    const bundle = openCandidateReviewBundle(fixture.root);
    expect(() => bundle.readPageImage(fixture.lineageId, 3)).toThrow(/not registered/);
  });

  it("rejects unsafe registered artifact paths while opening the manifest", () => {
    const fixture = writeFixture();
    fixture.manifest.documents[0].artifacts.content_ir_candidate.path =
      `${fixture.lineageId}/../outside.json`;
    fixture.writeManifest();

    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(/path is unsafe/);
  });

  it("rejects artifacts whose bytes no longer match their registered SHA-256", () => {
    const fixture = writeFixture();
    writeFileSync(fixture.contentPath, "{}", "utf8");

    const bundle = openCandidateReviewBundle(fixture.root);
    expect(() => bundle.loadDocument(fixture.lineageId)).toThrow(/ContentIR SHA-256 mismatch/);
  });

  it("rejects symbolic links in registered artifact paths", () => {
    const fixture = writeFixture();
    const original = readFileSync(fixture.contentPath);
    const target = join(fixture.root, "outside-content.json");
    writeFileSync(target, original);
    rmSync(fixture.contentPath);
    try {
      symlinkSync(target, fixture.contentPath, "file");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EPERM") return;
      throw error;
    }

    const bundle = openCandidateReviewBundle(fixture.root);
    expect(() => bundle.loadDocument(fixture.lineageId)).toThrow(/symbolic link/);
  });

  it("rejects manifests that relax review-only eligibility flags", () => {
    const fixture = writeFixture();
    fixture.manifest.training_eligible = true;
    fixture.writeManifest();

    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(
      /training_eligible must remain false/,
    );
  });

  it("rejects manifests that claim rights or falsify issue totals", () => {
    const fixture = writeFixture();
    fixture.manifest.rights_manifest.status = "verified";
    fixture.writeManifest();
    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(/rights status.*missing/);

    fixture.manifest.rights_manifest.status = "missing";
    fixture.manifest.issue_counts.human_review_required = 2;
    fixture.writeManifest();
    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(
      /issue_counts does not match/,
    );
  });

  it("rejects missing, unexpected, or inconsistent strict manifest metadata", () => {
    let fixture = writeFixture();
    delete fixture.manifest.source_bundle;
    fixture.writeManifest();
    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(
      /missing or unexpected fields/,
    );

    fixture = writeFixture();
    fixture.manifest.unexpected = true;
    fixture.writeManifest();
    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(
      /missing or unexpected fields/,
    );

    fixture = writeFixture();
    fixture.manifest.splits.train.pages = 1;
    fixture.writeManifest();
    expect(() => openCandidateReviewBundle(fixture.root)).toThrow(
      /split counts do not match/,
    );
  });
});

function writeFixture(): Fixture {
  const root = mkdtempSync(join(tmpdir(), "exam2hwpx-candidate-review-"));
  roots.push(root);
  const lineageId = "a".repeat(64);
  const hwpxSha256 = "b".repeat(64);
  const pdfSha256 = "c".repeat(64);
  const documentDir = join(root, lineageId);
  mkdirSync(join(documentDir, "pages"), { recursive: true });
  mkdirSync(join(documentDir, "assets"));

  const pageOneBytes = pageBytes("one");
  const pageTwoBytes = pageBytes("two");
  const imageCropBytes = cropBytes();

  const evidenceContractSha256 = sha256(Buffer.from("evidence-contract"));
  const contentContractSha256 = sha256(Buffer.from("content-contract"));
  const planContractSha256 = sha256(Buffer.from("plan-contract"));
  const evidence = {
    schema_version: "evidence-ir/1.0",
    id: "evidence-test",
    source_document_sha256: pdfSha256,
    sources: [
      {
        id: "input-pdf",
        kind: "original_document",
        artifact_ref: `artifact://fixture/${pdfSha256}/source.pdf`,
        producer: "fixture",
        sha256: pdfSha256,
      },
      {
        id: "page-0001-image",
        kind: "page_image",
        artifact_ref: `artifact://fixture/${pdfSha256}/pages/page-0001.png`,
        producer: "fixture",
        sha256: sha256(pageOneBytes),
      },
      {
        id: "page-0002-image",
        kind: "page_image",
        artifact_ref: `artifact://fixture/${pdfSha256}/pages/page-0002.png`,
        producer: "fixture",
        sha256: sha256(pageTwoBytes),
      },
      {
        id: "page-0002-image-00001-crop",
        kind: "crop",
        artifact_ref: `artifact://fixture/${pdfSha256}/assets/page-0002-image-00001.png`,
        producer: "fixture",
        sha256: sha256(imageCropBytes),
      },
    ],
    pages: [
      evidencePage(1, [
        observation(
          "page-0001-text-00001",
          "text_line",
          [10, 20, 60, 40],
          [0.1, 0.1, 0.6, 0.2],
        ),
      ]),
      evidencePage(2, [
        observation(
          "page-0002-region",
          "region",
          [0, 0, 100, 200],
          [0, 0, 1, 1],
          ["input-pdf", "page-0002-image"],
        ),
        observation(
          "page-0002-image-00001",
          "image",
          [25, 70, 75, 130],
          [0.25, 0.35, 0.75, 0.65],
          ["input-pdf", "page-0002-image-00001-crop"],
        ),
        observation(
          "page-0002-text-00001",
          "text_line",
          [20, 40, 80, 60],
          [0.2, 0.2, 0.8, 0.3],
        ),
      ]),
    ],
  };
  const content = {
    schema_version: "content-ir/1.0",
    id: "content-test",
    evidence_ir_id: evidence.id,
    evidence_ir_sha256: evidenceContractSha256,
    revision: 1,
    nodes: [
      {
        id: "content-text",
        kind: "text",
        role: "question",
        text: "question",
        evidence_refs: ["page-0001-text-00001", "page-0002-text-00001"],
        confidence: 0.9,
        needs_review: true,
      },
      {
        id: "content-image",
        kind: "image",
        role: "image",
        asset_ref: "page-0002-image",
        evidence_refs: ["page-0002-region", "page-0002-text-00001"],
        confidence: 0,
        needs_review: true,
      },
    ],
    reading_order: ["content-text", "content-image"],
  };
  const plan = {
    schema_version: "hwp-document-plan/1.0",
    id: "plan-test",
    content_ir_id: content.id,
    content_ir_revision: content.revision,
    content_ir_sha256: contentContractSha256,
    capability_profile_id: "test-profile",
    design_profile_id: "test-design",
    official_spec_refs: ["spec-test"],
    page_layout: {
      width_mm: 210,
      height_mm: 297,
      margin_top_mm: 10,
      margin_right_mm: 10,
      margin_bottom_mm: 10,
      margin_left_mm: 10,
      columns: 1,
      column_gap_mm: 0,
    },
    styles: [],
    flow: [
      {
        id: "flow-1",
        kind: "content",
        render_as: "paragraph",
        content_ref: "content-text",
        style_ref: null,
        layout: {},
      },
      {
        id: "flow-2",
        kind: "content",
        render_as: "image",
        content_ref: "content-image",
        style_ref: null,
        layout: {},
      },
    ],
  };
  const projection = {
    schema_version: "hwpx-projection-source/1.0",
    source_document_sha256: lineageId,
    hwpx_sha256: hwpxSha256,
    page_count: 2,
    sections: [],
    nodes: [],
    styles: [],
    assets: [],
    issues: [],
    stats: {},
  };
  const report = {
    schema_version: "hwp-projection-candidate-report/1.0",
    document_id: `candidate-${lineageId.slice(0, 24)}`,
    lineage_id: lineageId,
    status: "needs_human_review",
    research_only: true,
    human_review_required: true,
    training_eligible: false,
    release_eligible: false,
    rights_status: "missing",
    counts: { pages: 2 },
    alignment: {},
    contract_sha256: {
      evidence_ir: evidenceContractSha256,
      content_ir: contentContractSha256,
      hwp_document_plan: planContractSha256,
    },
    issue_counts: {},
    issue_codes: ["human_review_required", "rights_manifest_missing"],
    hard_flags: ["not_golden", "not_training_eligible", "not_release_eligible"],
  };

  const trackedPaths: string[] = [];
  const bindJson = (name: string, payload: unknown, contractSha256?: string) => {
    const path = join(documentDir, name);
    const bytes = Buffer.from(JSON.stringify(payload, null, 2), "utf8");
    writeFileSync(path, bytes);
    trackedPaths.push(path);
    return {
      path: `${lineageId}/${name.replaceAll("\\", "/")}`,
      sha256: sha256(bytes),
      ...(contractSha256 ? { contract_sha256: contractSha256 } : {}),
    };
  };
  const projectionBinding = bindJson("projection_source.json", projection);
  const evidenceBinding = bindJson("evidence_ir.json", evidence, evidenceContractSha256);
  const contentBinding = bindJson("content_ir.candidate.json", content, contentContractSha256);
  const planBinding = bindJson(
    "hwp_document_plan.candidate.json",
    plan,
    planContractSha256,
  );
  const reportBinding = bindJson("report.json", report);
  const pageBindings = [1, 2].map((pageNo) => {
    const path = join(documentDir, "pages", `page-${String(pageNo).padStart(4, "0")}.png`);
    const bytes = pageNo === 1 ? pageOneBytes : pageTwoBytes;
    writeFileSync(path, bytes);
    trackedPaths.push(path);
    return {
      path: `${lineageId}/pages/page-${String(pageNo).padStart(4, "0")}.png`,
      sha256: sha256(bytes),
    };
  });
  const cropPath = join(documentDir, "assets", "page-0002-image-00001.png");
  writeFileSync(cropPath, imageCropBytes);
  trackedPaths.push(cropPath);
  const cropBinding = {
    path: `${lineageId}/assets/page-0002-image-00001.png`,
    sha256: sha256(imageCropBytes),
  };
  const manifest: Record<string, any> = {
    schema_version: "hwp-human-review-candidates/1.0",
    artifact_role: "human_review_candidate",
    research_only: true,
    human_review_required: true,
    golden_eligible: false,
    training_eligible: false,
    release_eligible: false,
    rights_manifest: { status: "missing" },
    source_bundle: {
      schema_version: "hwp-hwpx-projection-pairs/1.0",
      manifest_sha256: "d".repeat(64),
      source_dataset_report_sha256: "e".repeat(64),
      source_inventory_sha256: "f".repeat(64),
      producer: { name: "fixture", version: "1.0" },
    },
    producer: {
      name: "scan2hwpx-hwpx-projection-candidate-builder",
      version: "1.3",
      pymupdf_version: "1.28.2",
      dpi: 300,
      pdf_text_order: "native",
    },
    capability_profile: {
      id: "exam2hwpx-authoring-v1",
      sha256: "1".repeat(64),
    },
    knowledge_corpus_sha256: "2".repeat(64),
    contract_versions: {
      projection_source: "hwpx-projection-source/1.0",
      evidence_ir: "evidence-ir/1.0",
      content_ir: "content-ir/1.0",
      hwp_document_plan: "hwp-document-plan/1.0",
    },
    document_count: 1,
    page_count: 2,
    splits: {
      train: { documents: 1, pages: 2 },
      validation: { documents: 0, pages: 0 },
      test: { documents: 0, pages: 0 },
    },
    issue_counts: { human_review_required: 1, rights_manifest_missing: 1 },
    documents: [
      {
        document_id: report.document_id,
        lineage_id: lineageId,
        split: "train",
        page_count: 2,
        status: "needs_human_review",
        source_artifacts: {
          hwp: { sha256: lineageId },
          hwpx: { sha256: hwpxSha256 },
          pdf: { sha256: pdfSha256 },
        },
        artifacts: {
          projection_source: projectionBinding,
          evidence_ir: evidenceBinding,
          content_ir_candidate: contentBinding,
          hwp_document_plan_candidate: planBinding,
          report: reportBinding,
          page_images: pageBindings,
          image_crops: [cropBinding],
        },
        issue_codes: report.issue_codes,
      },
    ],
  };
  const manifestPath = join(root, "manifest.json");
  const writeManifest = () => writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  writeManifest();
  trackedPaths.push(manifestPath);
  return {
    root,
    lineageId,
    contentPath: join(documentDir, "content_ir.candidate.json"),
    trackedPaths,
    manifest,
    writeManifest,
  };
}

function evidencePage(pageNo: number, observations: unknown[]) {
  return {
    id: `page-${String(pageNo).padStart(4, "0")}`,
    page_no: pageNo,
    width: 100,
    height: 200,
    rotation: 0,
    image_source_ref: `page-${String(pageNo).padStart(4, "0")}-image`,
    observations,
  };
}

function observation(
  id: string,
  kind: string,
  pixel: [number, number, number, number],
  normalized: [number, number, number, number],
  sourceRefs: string[] = ["input-pdf"],
) {
  return {
    id,
    kind,
    bbox: { pixel, normalized },
    confidence: 0,
    source_refs: sourceRefs,
    ocr_candidates: kind === "text_line" ? [{ text: "x" }] : [],
  };
}

function pageBytes(label: string): Buffer {
  return Buffer.concat([PNG_SIGNATURE, Buffer.from(label, "utf8")]);
}

function cropBytes(): Buffer {
  return Buffer.concat([PNG_SIGNATURE, Buffer.from("crop", "utf8")]);
}

function snapshot(paths: readonly string[]): Record<string, string> {
  return Object.fromEntries(paths.map((path) => [path, sha256(readFileSync(path))]));
}

function sha256(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
