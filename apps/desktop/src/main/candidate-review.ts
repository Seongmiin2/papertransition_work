import { createHash } from "node:crypto";
import {
  lstatSync,
  readFileSync,
  realpathSync,
  type Stats,
} from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

const CANDIDATE_SCHEMA = "hwp-human-review-candidates/1.0";
const SOURCE_BUNDLE_SCHEMA = "hwp-hwpx-projection-pairs/1.0";
const PROJECTION_SCHEMA = "hwpx-projection-source/1.0";
const EVIDENCE_SCHEMA = "evidence-ir/1.0";
const CONTENT_SCHEMA = "content-ir/1.0";
const PLAN_SCHEMA = "hwp-document-plan/1.0";
const REPORT_SCHEMA = "hwp-projection-candidate-report/1.0";
const SHA256 = /^[0-9a-f]{64}$/;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const ISSUE_CODE = /^[a-z][a-z0-9_.-]*$/;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_JSON_ARTIFACT_BYTES = 64 * 1024 * 1024;
const MAX_PAGE_IMAGE_BYTES = 128 * 1024 * 1024;
const MAX_IMAGE_GROUNDING_CANDIDATES = 10_000;
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  [key: string]: JsonValue;
}

interface ArtifactBinding {
  path: string;
  sha256: string;
  contractSha256: string | null;
}

interface SourceArtifacts {
  hwpSha256: string;
  hwpxSha256: string;
  pdfSha256: string;
}

interface CandidateDocument {
  documentId: string;
  lineageId: string;
  split: string;
  pageCount: number;
  status: "needs_human_review";
  issueCodes: readonly string[];
  sourceArtifacts: SourceArtifacts;
  projection: ArtifactBinding;
  evidence: ArtifactBinding;
  content: ArtifactBinding;
  plan: ArtifactBinding;
  report: ArtifactBinding;
  pageImages: ReadonlyMap<number, ArtifactBinding>;
  imageCrops: readonly ArtifactBinding[];
}

interface CandidateManifest {
  documents: readonly CandidateDocument[];
  documentCount: number;
  pageCount: number;
  issueCounts: Readonly<Record<string, number>>;
}

export interface CandidateBundleSummary {
  schemaVersion: typeof CANDIDATE_SCHEMA;
  manifestSha256: string;
  documentCount: number;
  pageCount: number;
  issueCounts: Readonly<Record<string, number>>;
}

export interface CandidateDocumentSummary {
  documentId: string;
  lineageId: string;
  split: string;
  pageCount: number;
  status: "needs_human_review";
  issueCodes: readonly string[];
}

export interface CandidatePageSummary {
  pageNo: number;
  width: number;
  height: number;
  rotation: number;
}

export interface EvidenceOverlayBox {
  observationId: string;
  kind: string;
  pageNo: number;
  pixel: readonly [number, number, number, number];
  normalized: readonly [number, number, number, number];
}

export interface ContentNodeOverlay {
  contentRef: string;
  contentKind: string;
  confidence: number;
  needsReview: boolean;
  evidenceRefs: readonly string[];
  pageNumbers: readonly number[];
  primaryPageNo: number;
  spansMultiplePages: boolean;
  usesPageRegionFallback: boolean;
  boxes: readonly EvidenceOverlayBox[];
}

export interface CandidateImageGroundingCandidate {
  observationRef: string;
  assetRef: string;
  pageNo: number;
  sourceKind: "page_image" | "crop";
  sha256: string;
}

export interface CandidateReviewDocument {
  document: CandidateDocumentSummary;
  artifactDigests: {
    evidenceIrSha256: string;
    contentIrSha256: string;
    hwpDocumentPlanSha256: string;
  };
  pages: readonly CandidatePageSummary[];
  overlays: readonly ContentNodeOverlay[];
  imageGroundingCandidates: readonly CandidateImageGroundingCandidate[];
  evidenceIr: JsonObject;
  contentIr: JsonObject;
  hwpDocumentPlan: JsonObject;
  report: JsonObject;
}

export interface CandidatePageImageBytes {
  pageNo: number;
  mimeType: "image/png";
  sha256: string;
  bytes: Uint8Array;
}

export interface CandidatePageImageDataUrl {
  pageNo: number;
  mimeType: "image/png";
  sha256: string;
  dataUrl: string;
}

interface ParsedEvidence {
  id: string;
  contractSha256: string;
  pages: readonly CandidatePageSummary[];
  observations: ReadonlyMap<string, EvidenceOverlayBox>;
  imageGroundingCandidates: readonly CandidateImageGroundingCandidate[];
  payload: JsonObject;
}

interface ParsedEvidenceSource {
  id: string;
  kind: string;
  sha256: string | null;
}

interface ParsedContent {
  id: string;
  revision: number;
  payload: JsonObject;
  overlays: readonly ContentNodeOverlay[];
  nodeIds: ReadonlySet<string>;
}

/**
 * Read-only access to one immutable HWPX projection candidate bundle.
 *
 * Opening validates only the root manifest and registered paths. Large document
 * JSON and page images are checked and read when the caller asks for them.
 */
export class CandidateReviewBundle {
  readonly summary: CandidateBundleSummary;
  readonly documents: readonly CandidateDocumentSummary[];

  private readonly root: string;
  private readonly manifest: CandidateManifest;
  private readonly documentsByLineage: ReadonlyMap<string, CandidateDocument>;

  private constructor(root: string, manifest: CandidateManifest, manifestSha256: string) {
    this.root = root;
    this.manifest = manifest;
    this.documentsByLineage = new Map(
      manifest.documents.map((document) => [document.lineageId, document]),
    );
    this.summary = {
      schemaVersion: CANDIDATE_SCHEMA,
      manifestSha256,
      documentCount: manifest.documentCount,
      pageCount: manifest.pageCount,
      issueCounts: { ...manifest.issueCounts },
    };
    this.documents = manifest.documents.map(documentSummary);
  }

  static open(rootPath: string): CandidateReviewBundle {
    if (!rootPath.trim()) throw new TypeError("candidate bundle path must not be empty");
    const selectedRoot = resolve(rootPath);
    const selectedStats = checkedLstat(selectedRoot, "candidate bundle");
    if (selectedStats.isSymbolicLink()) {
      throw new Error("candidate bundle root must not be a symbolic link");
    }
    if (!selectedStats.isDirectory()) {
      throw new Error("candidate bundle root must be a directory");
    }
    const root = realpathSync.native(selectedRoot);
    const manifestPath = join(root, "manifest.json");
    const manifestStats = checkedLstat(manifestPath, "candidate manifest");
    if (manifestStats.isSymbolicLink()) {
      throw new Error("candidate manifest must not be a symbolic link");
    }
    if (!manifestStats.isFile()) throw new Error("candidate manifest must be a file");
    if (manifestStats.size > MAX_MANIFEST_BYTES) {
      throw new Error("candidate manifest exceeds the size limit");
    }
    const manifestBytes = readFileSync(manifestPath);
    if (manifestBytes.byteLength > MAX_MANIFEST_BYTES) {
      throw new Error("candidate manifest exceeds the size limit");
    }
    const manifest = parseManifest(parseJsonObject(manifestBytes, "candidate manifest"));
    return new CandidateReviewBundle(root, manifest, sha256(manifestBytes));
  }

  get canonicalRootPath(): string {
    return this.root;
  }

  assertCurrentSnapshot(): void {
    let current: CandidateReviewBundle;
    try {
      current = CandidateReviewBundle.open(this.root);
    } catch {
      throw new Error("candidate bundle no longer matches the selected snapshot");
    }
    if (current.summary.manifestSha256 !== this.summary.manifestSha256) {
      throw new Error("candidate bundle no longer matches the selected snapshot");
    }
  }

  loadDocument(lineageId: string): CandidateReviewDocument {
    const document = this.requiredDocument(lineageId);
    const projection = this.readJsonArtifact(
      document.projection,
      "projection source",
    );
    validateProjection(projection, document);

    const evidencePayload = this.readJsonArtifact(document.evidence, "EvidenceIR");
    const evidence = parseEvidence(evidencePayload, document);
    const contentPayload = this.readJsonArtifact(document.content, "ContentIR");
    const content = parseContent(contentPayload, document, evidence);
    const plan = this.readJsonArtifact(document.plan, "HwpDocumentPlan");
    validatePlan(plan, document, content);
    const report = this.readJsonArtifact(document.report, "candidate report");
    validateReport(report, document);

    const reportContractDigests = requiredObject(report.contract_sha256, "report contract_sha256");
    assertEquals(
      requiredSha256(reportContractDigests.evidence_ir, "report EvidenceIR digest"),
      requiredContractSha(document.evidence, "EvidenceIR"),
      "candidate report EvidenceIR digest mismatch",
    );
    assertEquals(
      requiredSha256(reportContractDigests.content_ir, "report ContentIR digest"),
      requiredContractSha(document.content, "ContentIR"),
      "candidate report ContentIR digest mismatch",
    );
    assertEquals(
      requiredSha256(
        reportContractDigests.hwp_document_plan,
        "report HwpDocumentPlan digest",
      ),
      requiredContractSha(document.plan, "HwpDocumentPlan"),
      "candidate report HwpDocumentPlan digest mismatch",
    );

    return {
      document: documentSummary(document),
      artifactDigests: {
        evidenceIrSha256: document.evidence.sha256,
        contentIrSha256: document.content.sha256,
        hwpDocumentPlanSha256: document.plan.sha256,
      },
      pages: evidence.pages,
      overlays: content.overlays,
      imageGroundingCandidates: evidence.imageGroundingCandidates,
      evidenceIr: evidence.payload,
      contentIr: content.payload,
      hwpDocumentPlan: plan,
      report,
    };
  }

  readPageImage(lineageId: string, pageNo: number): CandidatePageImageBytes {
    const document = this.requiredDocument(lineageId);
    if (!Number.isSafeInteger(pageNo) || pageNo < 1) {
      throw new TypeError("page number must be a positive integer");
    }
    const binding = document.pageImages.get(pageNo);
    if (!binding) {
      throw new Error(`page ${pageNo} is not registered for candidate ${lineageId}`);
    }
    const bytes = this.readArtifact(binding, "candidate page image", MAX_PAGE_IMAGE_BYTES);
    if (bytes.byteLength < PNG_SIGNATURE.byteLength || !bytes.subarray(0, 8).equals(PNG_SIGNATURE)) {
      throw new Error(`registered page ${pageNo} is not a PNG image`);
    }
    return {
      pageNo,
      mimeType: "image/png",
      sha256: binding.sha256,
      bytes: new Uint8Array(bytes),
    };
  }

  readPageImageDataUrl(lineageId: string, pageNo: number): CandidatePageImageDataUrl {
    const image = this.readPageImage(lineageId, pageNo);
    return {
      pageNo: image.pageNo,
      mimeType: image.mimeType,
      sha256: image.sha256,
      dataUrl: `data:${image.mimeType};base64,${Buffer.from(image.bytes).toString("base64")}`,
    };
  }

  private requiredDocument(lineageId: string): CandidateDocument {
    const document = this.documentsByLineage.get(lineageId);
    if (!document) throw new Error(`unknown candidate lineage: ${lineageId}`);
    return document;
  }

  private readJsonArtifact(binding: ArtifactBinding, label: string): JsonObject {
    return parseJsonObject(
      this.readArtifact(binding, label, MAX_JSON_ARTIFACT_BYTES),
      label,
    );
  }

  private readArtifact(binding: ArtifactBinding, label: string, sizeLimit: number): Buffer {
    const path = resolveRegisteredPath(this.root, binding.path, label);
    const stats = checkedLstat(path, label);
    if (stats.isSymbolicLink()) throw new Error(`${label} must not be a symbolic link`);
    if (!stats.isFile()) throw new Error(`${label} must be a regular file`);
    if (stats.size > sizeLimit) throw new Error(`${label} exceeds the size limit`);
    assertNoSymlinkSegments(this.root, binding.path, label);
    const realPath = realpathSync.native(path);
    assertContained(this.root, realPath, label);
    const bytes = readFileSync(path);
    if (bytes.byteLength > sizeLimit) throw new Error(`${label} exceeds the size limit`);
    if (sha256(bytes) !== binding.sha256) throw new Error(`${label} SHA-256 mismatch`);
    if (lstatSync(path).isSymbolicLink() || realpathSync.native(path) !== realPath) {
      throw new Error(`${label} changed while it was being read`);
    }
    return bytes;
  }
}

export function openCandidateReviewBundle(rootPath: string): CandidateReviewBundle {
  return CandidateReviewBundle.open(rootPath);
}

function parseManifest(value: JsonObject): CandidateManifest {
  requireExactKeys(
    value,
    [
      "artifact_role",
      "capability_profile",
      "contract_versions",
      "document_count",
      "documents",
      "golden_eligible",
      "human_review_required",
      "issue_counts",
      "knowledge_corpus_sha256",
      "page_count",
      "producer",
      "release_eligible",
      "research_only",
      "rights_manifest",
      "schema_version",
      "source_bundle",
      "splits",
      "training_eligible",
    ],
    "candidate manifest",
  );
  assertEquals(
    requiredString(value.schema_version, "candidate schema_version"),
    CANDIDATE_SCHEMA,
    "unsupported candidate manifest schema",
  );
  assertEquals(
    requiredString(value.artifact_role, "candidate artifact_role"),
    "human_review_candidate",
    "candidate artifact role is invalid",
  );
  requireLiteral(value.research_only, true, "research_only");
  requireLiteral(value.human_review_required, true, "human_review_required");
  requireLiteral(value.golden_eligible, false, "golden_eligible");
  requireLiteral(value.training_eligible, false, "training_eligible");
  requireLiteral(value.release_eligible, false, "release_eligible");
  const rights = requiredObject(value.rights_manifest, "rights_manifest");
  requireExactKeys(rights, ["status"], "rights_manifest");
  assertEquals(rights.status, "missing", "candidate rights status must remain missing");

  const sourceBundle = requiredObject(value.source_bundle, "source_bundle");
  requireExactKeys(
    sourceBundle,
    [
      "manifest_sha256",
      "producer",
      "schema_version",
      "source_dataset_report_sha256",
      "source_inventory_sha256",
    ],
    "source_bundle",
  );
  assertEquals(
    sourceBundle.schema_version,
    SOURCE_BUNDLE_SCHEMA,
    "source bundle schema mismatch",
  );
  requiredSha256(sourceBundle.manifest_sha256, "source bundle manifest SHA-256");
  requiredSha256(
    sourceBundle.source_dataset_report_sha256,
    "source bundle dataset report SHA-256",
  );
  requiredSha256(
    sourceBundle.source_inventory_sha256,
    "source bundle inventory SHA-256",
  );
  requireStringRecord(sourceBundle.producer, "source bundle producer");

  const producer = requiredObject(value.producer, "candidate producer");
  requireExactKeys(
    producer,
    ["dpi", "name", "pdf_text_order", "pymupdf_version", "version"],
    "candidate producer",
  );
  requiredString(producer.name, "candidate producer name");
  requiredString(producer.version, "candidate producer version");
  requiredString(producer.pymupdf_version, "candidate PyMuPDF version");
  requiredPositiveInteger(producer.dpi, "candidate producer dpi");
  assertEquals(
    producer.pdf_text_order,
    "native",
    "candidate PDF text order is invalid",
  );

  const capability = requiredObject(value.capability_profile, "capability_profile");
  requireExactKeys(capability, ["id", "sha256"], "capability_profile");
  requiredString(capability.id, "capability profile id");
  requiredSha256(capability.sha256, "capability profile SHA-256");
  requiredSha256(value.knowledge_corpus_sha256, "knowledge corpus SHA-256");

  const versions = requiredObject(value.contract_versions, "contract_versions");
  requireExactKeys(
    versions,
    ["content_ir", "evidence_ir", "hwp_document_plan", "projection_source"],
    "contract_versions",
  );
  assertEquals(versions.projection_source, PROJECTION_SCHEMA, "projection schema mismatch");
  assertEquals(versions.evidence_ir, EVIDENCE_SCHEMA, "EvidenceIR schema mismatch");
  assertEquals(versions.content_ir, CONTENT_SCHEMA, "ContentIR schema mismatch");
  assertEquals(versions.hwp_document_plan, PLAN_SCHEMA, "HwpDocumentPlan schema mismatch");

  const documentCount = requiredPositiveInteger(value.document_count, "document_count");
  const pageCount = requiredPositiveInteger(value.page_count, "page_count");
  const rawDocuments = requiredArray(value.documents, "candidate documents");
  if (rawDocuments.length !== documentCount) {
    throw new Error("candidate document_count does not match documents");
  }
  const registeredPaths = new Set<string>();
  const documents = rawDocuments.map((item, index) =>
    parseManifestDocument(item, index, registeredPaths),
  );
  rejectDuplicates(
    documents.map((document) => document.documentId.toLocaleLowerCase("en-US")),
    "candidate document ids",
  );
  rejectDuplicates(documents.map((document) => document.lineageId), "candidate lineage ids");
  if (documents.reduce((total, document) => total + document.pageCount, 0) !== pageCount) {
    throw new Error("candidate page_count does not match documents");
  }

  const splits = requiredObject(value.splits, "candidate splits");
  requireExactKeys(splits, ["test", "train", "validation"], "candidate splits");
  for (const split of ["train", "validation", "test"]) {
    const counts = requiredObject(splits[split], `candidate ${split} split`);
    requireExactKeys(counts, ["documents", "pages"], `candidate ${split} split`);
    const expectedDocuments = documents.filter((document) => document.split === split);
    if (
      requiredInteger(counts.documents, `candidate ${split} documents`) !==
        expectedDocuments.length ||
      requiredInteger(counts.pages, `candidate ${split} pages`) !==
        expectedDocuments.reduce((total, document) => total + document.pageCount, 0)
    ) {
      throw new Error(`candidate split counts do not match for ${split}`);
    }
  }

  const issueCountsValue = requiredObject(value.issue_counts, "candidate issue_counts");
  const issueCounts: Record<string, number> = {};
  for (const [code, count] of Object.entries(issueCountsValue)) {
    if (!ISSUE_CODE.test(code) || !Number.isSafeInteger(count) || Number(count) < 0) {
      throw new Error("candidate issue_counts is invalid");
    }
    issueCounts[code] = Number(count);
  }
  const expectedIssueCounts: Record<string, number> = {};
  for (const document of documents) {
    for (const code of document.issueCodes) {
      expectedIssueCounts[code] = (expectedIssueCounts[code] ?? 0) + 1;
    }
  }
  if (
    Object.keys(issueCounts).length !== Object.keys(expectedIssueCounts).length ||
    Object.entries(expectedIssueCounts).some(([code, count]) => issueCounts[code] !== count)
  ) {
    throw new Error("candidate issue_counts does not match document issues");
  }
  return { documents, documentCount, pageCount, issueCounts };
}

function parseManifestDocument(
  value: JsonValue,
  index: number,
  registeredPaths: Set<string>,
): CandidateDocument {
  const item = requiredObject(value, `candidate document ${index}`);
  requireExactKeys(
    item,
    [
      "artifacts",
      "document_id",
      "issue_codes",
      "lineage_id",
      "page_count",
      "source_artifacts",
      "split",
      "status",
    ],
    `candidate document ${index}`,
  );
  const documentId = requiredString(item.document_id, "candidate document_id");
  if (!IDENTIFIER.test(documentId)) throw new Error("candidate document_id is invalid");
  const lineageId = requiredSha256(item.lineage_id, "candidate lineage_id");
  const split = requiredString(item.split, "candidate split");
  if (!["train", "validation", "test"].includes(split)) {
    throw new Error("candidate split is invalid");
  }
  const pageCount = requiredPositiveInteger(item.page_count, "candidate page_count");
  assertEquals(item.status, "needs_human_review", "candidate status is invalid");
  const issueCodes = requiredStringArray(item.issue_codes, "candidate issue_codes");
  if (issueCodes.some((code) => !ISSUE_CODE.test(code))) {
    throw new Error("candidate issue_codes is invalid");
  }
  rejectDuplicates(issueCodes, "candidate issue codes");

  const source = requiredObject(item.source_artifacts, "candidate source_artifacts");
  requireExactKeys(source, ["hwp", "hwpx", "pdf"], "candidate source_artifacts");
  const sourceArtifacts: SourceArtifacts = {
    hwpSha256: artifactDigest(source.hwp, "source HWP"),
    hwpxSha256: artifactDigest(source.hwpx, "source HWPX"),
    pdfSha256: artifactDigest(source.pdf, "source PDF"),
  };
  assertEquals(sourceArtifacts.hwpSha256, lineageId, "source HWP digest mismatch");

  const artifacts = requiredObject(item.artifacts, "candidate artifacts");
  requireExactKeys(
    artifacts,
    [
      "content_ir_candidate",
      "evidence_ir",
      "hwp_document_plan_candidate",
      "image_crops",
      "page_images",
      "projection_source",
      "report",
    ],
    "candidate artifacts",
  );
  const projection = parseBinding(
    artifacts.projection_source,
    lineageId,
    "projection source",
    registeredPaths,
    false,
  );
  const evidence = parseBinding(
    artifacts.evidence_ir,
    lineageId,
    "EvidenceIR",
    registeredPaths,
    true,
  );
  const content = parseBinding(
    artifacts.content_ir_candidate,
    lineageId,
    "ContentIR",
    registeredPaths,
    true,
  );
  const plan = parseBinding(
    artifacts.hwp_document_plan_candidate,
    lineageId,
    "HwpDocumentPlan",
    registeredPaths,
    true,
  );
  const report = parseBinding(
    artifacts.report,
    lineageId,
    "candidate report",
    registeredPaths,
    false,
  );

  const pageImages = new Map<number, ArtifactBinding>();
  for (const rawBinding of requiredArray(artifacts.page_images, "candidate page_images")) {
    const binding = parseBinding(
      rawBinding,
      lineageId,
      "candidate page image",
      registeredPaths,
      false,
    );
    const segments = binding.path.split("/");
    const match = /^page-(\d{4,})\.png$/.exec(segments.at(-1) ?? "");
    if (segments.length !== 3 || segments[1] !== "pages" || !match) {
      throw new Error("candidate page image path is invalid");
    }
    const pageNo = Number(match[1]);
    if (!Number.isSafeInteger(pageNo) || pageNo < 1 || pageNo > pageCount) {
      throw new Error("candidate page image number is outside the document");
    }
    if (pageImages.has(pageNo)) throw new Error(`duplicate candidate page image ${pageNo}`);
    pageImages.set(pageNo, binding);
  }
  if (pageImages.size !== pageCount) {
    throw new Error("candidate page image count does not match page_count");
  }
  for (let pageNo = 1; pageNo <= pageCount; pageNo += 1) {
    if (!pageImages.has(pageNo)) throw new Error(`candidate page image ${pageNo} is missing`);
  }

  const imageCrops = requiredArray(artifacts.image_crops, "candidate image_crops").map(
    (binding) =>
      parseBinding(
        binding,
        lineageId,
        "candidate image crop",
        registeredPaths,
        false,
      ),
  );
  return {
    documentId,
    lineageId,
    split,
    pageCount,
    status: "needs_human_review",
    issueCodes,
    sourceArtifacts,
    projection,
    evidence,
    content,
    plan,
    report,
    pageImages,
    imageCrops,
  };
}

function parseBinding(
  value: JsonValue | undefined,
  lineageId: string,
  label: string,
  registeredPaths: Set<string>,
  requireContractSha: boolean,
): ArtifactBinding {
  const item = requiredObject(value, `${label} binding`);
  requireExactKeys(
    item,
    requireContractSha ? ["contract_sha256", "path", "sha256"] : ["path", "sha256"],
    `${label} binding`,
  );
  const path = requiredString(item.path, `${label} path`);
  validateLogicalPath(path, lineageId, label);
  const normalizedPath = path.toLocaleLowerCase("en-US");
  if (registeredPaths.has(normalizedPath)) {
    throw new Error(`duplicate candidate artifact path: ${path}`);
  }
  registeredPaths.add(normalizedPath);
  const contractSha256 =
    item.contract_sha256 === undefined
      ? null
      : requiredSha256(item.contract_sha256, `${label} contract_sha256`);
  if (requireContractSha && contractSha256 === null) {
    throw new Error(`${label} binding requires contract_sha256`);
  }
  return {
    path,
    sha256: requiredSha256(item.sha256, `${label} sha256`),
    contractSha256,
  };
}

function validateProjection(value: JsonObject, document: CandidateDocument): void {
  assertEquals(value.schema_version, PROJECTION_SCHEMA, "projection schema mismatch");
  assertEquals(
    value.source_document_sha256,
    document.lineageId,
    "projection source document mismatch",
  );
  assertEquals(value.hwpx_sha256, document.sourceArtifacts.hwpxSha256, "projection HWPX mismatch");
  assertEquals(value.page_count, document.pageCount, "projection page count mismatch");
}

function parseEvidence(value: JsonObject, document: CandidateDocument): ParsedEvidence {
  assertEquals(value.schema_version, EVIDENCE_SCHEMA, "EvidenceIR schema mismatch");
  const id = requiredString(value.id, "EvidenceIR id");
  assertEquals(
    value.source_document_sha256,
    document.sourceArtifacts.pdfSha256,
    "EvidenceIR source PDF mismatch",
  );
  const sources = new Map<string, ParsedEvidenceSource>();
  for (const rawSource of requiredArray(value.sources, "EvidenceIR sources")) {
    const source = requiredObject(rawSource, "EvidenceIR source");
    const sourceId = requiredString(source.id, "EvidenceIR source id");
    if (sources.has(sourceId)) {
      throw new Error(`duplicate EvidenceIR source id: ${sourceId}`);
    }
    sources.set(sourceId, {
      id: sourceId,
      kind: requiredString(source.kind, "EvidenceIR source kind"),
      sha256:
        source.sha256 === undefined || source.sha256 === null
          ? null
          : requiredSha256(source.sha256, "EvidenceIR source SHA-256"),
    });
  }
  const registeredPageSha256 = new Set(
    [...document.pageImages.values()].map((binding) => binding.sha256),
  );
  const registeredCropSha256 = new Set(
    document.imageCrops.map((binding) => binding.sha256),
  );
  const rawPages = requiredArray(value.pages, "EvidenceIR pages");
  if (rawPages.length !== document.pageCount) {
    throw new Error("EvidenceIR page count does not match candidate document");
  }
  const pages: CandidatePageSummary[] = [];
  const observations = new Map<string, EvidenceOverlayBox>();
  const imageGroundingCandidates: CandidateImageGroundingCandidate[] = [];
  const imageGroundingPairs = new Set<string>();
  const pageNumbers = new Set<number>();
  for (const rawPage of rawPages) {
    const page = requiredObject(rawPage, "EvidenceIR page");
    const pageNo = requiredPositiveInteger(page.page_no, "EvidenceIR page_no");
    if (pageNo > document.pageCount || pageNumbers.has(pageNo)) {
      throw new Error("EvidenceIR contains an invalid or duplicate page number");
    }
    pageNumbers.add(pageNo);
    const width = requiredPositiveNumber(page.width, "EvidenceIR page width");
    const height = requiredPositiveNumber(page.height, "EvidenceIR page height");
    const rotation = requiredInteger(page.rotation, "EvidenceIR page rotation");
    const pageImageSourceRef = requiredString(
      page.image_source_ref,
      "EvidenceIR page image_source_ref",
    );
    const pageImageSource = sources.get(pageImageSourceRef);
    const registeredPage = document.pageImages.get(pageNo);
    if (
      !pageImageSource ||
      pageImageSource.kind !== "page_image" ||
      pageImageSource.sha256 === null ||
      pageImageSource.sha256 !== registeredPage?.sha256
    ) {
      throw new Error("EvidenceIR page image source is not registered for its page");
    }
    pages.push({ pageNo, width, height, rotation });
    for (const rawObservation of requiredArray(page.observations, "EvidenceIR observations")) {
      const observation = requiredObject(rawObservation, "EvidenceIR observation");
      const observationId = requiredString(observation.id, "EvidenceIR observation id");
      if (observations.has(observationId)) {
        throw new Error(`duplicate EvidenceIR observation id: ${observationId}`);
      }
      const bbox = requiredObject(observation.bbox, "EvidenceIR observation bbox");
      const pixel = requiredBbox(bbox.pixel, "EvidenceIR pixel bbox", false);
      const normalized = requiredBbox(bbox.normalized, "EvidenceIR normalized bbox", true);
      const kind = requiredString(observation.kind, "EvidenceIR observation kind");
      const sourceRefs = requiredStringArray(
        observation.source_refs,
        "EvidenceIR observation source_refs",
      );
      rejectDuplicates(sourceRefs, "EvidenceIR observation source refs");
      if (sourceRefs.some((sourceRef) => !sources.has(sourceRef))) {
        throw new Error(`EvidenceIR observation has an unknown source ref: ${observationId}`);
      }
      observations.set(observationId, {
        observationId,
        kind,
        pageNo,
        pixel,
        normalized,
      });
      if (kind === "image") {
        for (const sourceRef of sourceRefs) {
          const source = sources.get(sourceRef);
          if (!source || source.sha256 === null) continue;
          const registered =
            (source.kind === "page_image" && registeredPageSha256.has(source.sha256)) ||
            (source.kind === "crop" && registeredCropSha256.has(source.sha256));
          if (!registered || (source.kind !== "page_image" && source.kind !== "crop")) {
            continue;
          }
          const pairKey = `${observationId}\u0000${source.id}`;
          if (imageGroundingPairs.has(pairKey)) {
            throw new Error("duplicate image grounding candidate");
          }
          imageGroundingPairs.add(pairKey);
          imageGroundingCandidates.push({
            observationRef: observationId,
            assetRef: source.id,
            pageNo,
            sourceKind: source.kind,
            sha256: source.sha256,
          });
          if (imageGroundingCandidates.length > MAX_IMAGE_GROUNDING_CANDIDATES) {
            throw new Error("image grounding candidate count exceeds the limit");
          }
        }
      }
    }
  }
  pages.sort((left, right) => left.pageNo - right.pageNo);
  imageGroundingCandidates.sort(
    (left, right) =>
      left.pageNo - right.pageNo ||
      compareStrings(left.observationRef, right.observationRef) ||
      compareStrings(left.assetRef, right.assetRef),
  );
  const contractSha256 = requiredContractSha(document.evidence, "EvidenceIR");
  return {
    id,
    contractSha256,
    pages,
    observations,
    imageGroundingCandidates,
    payload: value,
  };
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function parseContent(
  value: JsonObject,
  document: CandidateDocument,
  evidence: ParsedEvidence,
): ParsedContent {
  assertEquals(value.schema_version, CONTENT_SCHEMA, "ContentIR schema mismatch");
  const id = requiredString(value.id, "ContentIR id");
  assertEquals(value.evidence_ir_id, evidence.id, "ContentIR EvidenceIR id mismatch");
  assertEquals(
    value.evidence_ir_sha256,
    evidence.contractSha256,
    "ContentIR EvidenceIR digest mismatch",
  );
  const revision = requiredPositiveInteger(value.revision, "ContentIR revision");
  const rawNodes = requiredArray(value.nodes, "ContentIR nodes");
  if (!rawNodes.length) throw new Error("ContentIR must contain at least one node");
  const nodeIds = new Set<string>();
  const overlays: ContentNodeOverlay[] = [];
  for (const rawNode of rawNodes) {
    const node = requiredObject(rawNode, "ContentIR node");
    const contentRef = requiredString(node.id, "ContentIR node id");
    if (nodeIds.has(contentRef)) throw new Error(`duplicate ContentIR node id: ${contentRef}`);
    nodeIds.add(contentRef);
    const evidenceRefs = collectEvidenceRefs(node);
    const boxes = evidenceRefs.map((reference) => {
      const observation = evidence.observations.get(reference);
      if (!observation) throw new Error(`unknown ContentIR evidence ref: ${reference}`);
      return observation;
    });
    const pageNumbers = [...new Set(boxes.map((box) => box.pageNo))].sort(
      (left, right) => left - right,
    );
    if (!pageNumbers.length) throw new Error(`ContentIR node ${contentRef} has no page evidence`);
    overlays.push({
      contentRef,
      contentKind: requiredString(node.kind, "ContentIR node kind"),
      confidence: requiredConfidence(node.confidence, "ContentIR node confidence"),
      needsReview: requiredBoolean(node.needs_review, "ContentIR node needs_review"),
      evidenceRefs,
      pageNumbers,
      primaryPageNo: pageNumbers[0],
      spansMultiplePages: pageNumbers.length > 1,
      usesPageRegionFallback: boxes.some((box) => box.kind === "region"),
      boxes,
    });
  }
  const readingOrder = requiredStringArray(value.reading_order, "ContentIR reading_order");
  rejectDuplicates(readingOrder, "ContentIR reading_order");
  if (
    readingOrder.length !== nodeIds.size ||
    readingOrder.some((reference) => !nodeIds.has(reference))
  ) {
    throw new Error("ContentIR reading_order does not cover every node exactly once");
  }
  if (!document.content.contractSha256) throw new Error("ContentIR contract digest is missing");
  return { id, revision, payload: value, overlays, nodeIds };
}

function validatePlan(
  value: JsonObject,
  _document: CandidateDocument,
  content: ParsedContent,
): void {
  assertEquals(value.schema_version, PLAN_SCHEMA, "HwpDocumentPlan schema mismatch");
  assertEquals(value.content_ir_id, content.id, "HwpDocumentPlan ContentIR id mismatch");
  assertEquals(
    value.content_ir_revision,
    content.revision,
    "HwpDocumentPlan ContentIR revision mismatch",
  );
  assertEquals(
    value.content_ir_sha256,
    requiredContractSha(_document.content, "ContentIR"),
    "HwpDocumentPlan ContentIR digest mismatch",
  );
  const planned = requiredArray(value.flow, "HwpDocumentPlan flow")
    .map((rawItem) => requiredObject(rawItem, "HwpDocumentPlan flow item"))
    .filter((item) => item.kind === "content")
    .map((item) => requiredString(item.content_ref, "HwpDocumentPlan content_ref"));
  rejectDuplicates(planned, "HwpDocumentPlan content refs");
  if (planned.length !== content.nodeIds.size || planned.some((ref) => !content.nodeIds.has(ref))) {
    throw new Error("HwpDocumentPlan does not cover every ContentIR node exactly once");
  }
}

function validateReport(value: JsonObject, document: CandidateDocument): void {
  assertEquals(value.schema_version, REPORT_SCHEMA, "candidate report schema mismatch");
  assertEquals(value.document_id, document.documentId, "candidate report document mismatch");
  assertEquals(value.lineage_id, document.lineageId, "candidate report lineage mismatch");
  assertEquals(value.status, "needs_human_review", "candidate report status is invalid");
  requireLiteral(value.research_only, true, "candidate report research_only");
  requireLiteral(
    value.human_review_required,
    true,
    "candidate report human_review_required",
  );
  requireLiteral(value.training_eligible, false, "candidate report training_eligible");
  requireLiteral(value.release_eligible, false, "candidate report release_eligible");
  const reportIssues = requiredStringArray(value.issue_codes, "candidate report issue_codes");
  if (!sameStringSet(reportIssues, document.issueCodes)) {
    throw new Error("candidate report issue codes do not match the manifest");
  }
}

function collectEvidenceRefs(node: JsonObject): string[] {
  const refs = requiredStringArray(node.evidence_refs, "ContentIR node evidence_refs");
  if (!refs.length) throw new Error("ContentIR node evidence_refs must not be empty");
  if (node.kind === "table") {
    for (const rawCell of requiredArray(node.cells, "ContentIR table cells")) {
      const cell = requiredObject(rawCell, "ContentIR table cell");
      refs.push(...requiredStringArray(cell.evidence_refs, "ContentIR table cell evidence_refs"));
    }
  }
  return [...new Set(refs)];
}

function documentSummary(document: CandidateDocument): CandidateDocumentSummary {
  return {
    documentId: document.documentId,
    lineageId: document.lineageId,
    split: document.split,
    pageCount: document.pageCount,
    status: document.status,
    issueCodes: [...document.issueCodes],
  };
}

function validateLogicalPath(path: string, lineageId: string, label: string): void {
  if (path.includes("\\") || path.includes("\0") || path.startsWith("/")) {
    throw new Error(`${label} path is unsafe`);
  }
  const segments = path.split("/");
  if (
    segments.length < 2 ||
    segments[0] !== lineageId ||
    segments.some((segment) => !segment || segment === "." || segment === ".." || segment.includes(":"))
  ) {
    throw new Error(`${label} path is unsafe`);
  }
}

function resolveRegisteredPath(root: string, logicalPath: string, label: string): string {
  const path = resolve(root, ...logicalPath.split("/"));
  assertContained(root, path, label);
  return path;
}

function assertNoSymlinkSegments(root: string, logicalPath: string, label: string): void {
  let current = root;
  for (const segment of logicalPath.split("/")) {
    current = join(current, segment);
    if (checkedLstat(current, label).isSymbolicLink()) {
      throw new Error(`${label} path must not contain symbolic links`);
    }
  }
}

function assertContained(root: string, path: string, label: string): void {
  const child = relative(root, path);
  if (!child || isAbsolute(child) || child === ".." || child.startsWith(`..${sep}`)) {
    throw new Error(`${label} path escapes the candidate bundle`);
  }
}

function checkedLstat(path: string, label: string): Stats {
  try {
    return lstatSync(path);
  } catch {
    throw new Error(`${label} is missing or unreadable`);
  }
}

function parseJsonObject(bytes: Uint8Array, label: string): JsonObject {
  let decoded: string;
  try {
    decoded = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${label} must be valid UTF-8`, { cause: error });
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(decoded);
  } catch (error) {
    throw new Error(`${label} must be valid JSON`, { cause: error });
  }
  return requiredObject(parsed as JsonValue, label);
}

function requireExactKeys(
  value: JsonObject,
  expected: readonly string[],
  label: string,
): void {
  const actual = Object.keys(value).sort();
  const required = [...expected].sort();
  if (
    actual.length !== required.length ||
    actual.some((key, index) => key !== required[index])
  ) {
    throw new Error(`${label} contains missing or unexpected fields`);
  }
}


function requiredObject(value: JsonValue | undefined, label: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value;
}

function requireStringRecord(
  value: JsonValue | undefined,
  label: string,
): JsonObject {
  const record = requiredObject(value, label);
  if (Object.values(record).some((item) => typeof item !== "string")) {
    throw new TypeError(`${label} values must be strings`);
  }
  return record;
}

function requiredArray(value: JsonValue | undefined, label: string): JsonValue[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
  return value;
}

function requiredString(value: JsonValue | undefined, label: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new TypeError(`${label} must be a non-empty string`);
  }
  return value;
}

function requiredStringArray(value: JsonValue | undefined, label: string): string[] {
  return requiredArray(value, label).map((item) => requiredString(item, label));
}

function requiredSha256(value: JsonValue | undefined, label: string): string {
  const digest = requiredString(value, label);
  if (!SHA256.test(digest)) throw new TypeError(`${label} must be a lowercase SHA-256`);
  return digest;
}

function requiredContractSha(binding: ArtifactBinding, label: string): string {
  if (!binding.contractSha256) throw new Error(`${label} contract digest is missing`);
  return binding.contractSha256;
}

function artifactDigest(value: JsonValue | undefined, label: string): string {
  const binding = requiredObject(value, label);
  requireExactKeys(binding, ["sha256"], label);
  return requiredSha256(binding.sha256, `${label} sha256`);
}

function requiredPositiveInteger(value: JsonValue | undefined, label: string): number {
  const result = requiredInteger(value, label);
  if (result < 1) throw new TypeError(`${label} must be positive`);
  return result;
}

function requiredInteger(value: JsonValue | undefined, label: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new TypeError(`${label} must be an integer`);
  }
  return value;
}

function requiredPositiveNumber(value: JsonValue | undefined, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive number`);
  }
  return value;
}

function requiredConfidence(value: JsonValue | undefined, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError(`${label} must be between zero and one`);
  }
  return value;
}

function requiredBoolean(value: JsonValue | undefined, label: string): boolean {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean`);
  return value;
}

function requiredBbox(
  value: JsonValue | undefined,
  label: string,
  normalized: boolean,
): [number, number, number, number] {
  const values = requiredArray(value, label);
  if (
    values.length !== 4 ||
    values.some((item) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    throw new TypeError(`${label} must contain four finite numbers`);
  }
  const bbox = values as [number, number, number, number];
  if (bbox[0] < 0 || bbox[1] < 0 || bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) {
    throw new Error(`${label} is unordered or negative`);
  }
  if (normalized && bbox.some((item) => item < 0 || item > 1)) {
    throw new Error(`${label} must stay between zero and one`);
  }
  return bbox;
}

function requireLiteral(value: JsonValue | undefined, expected: boolean, label: string): void {
  if (value !== expected) throw new Error(`${label} must remain ${String(expected)}`);
}

function assertEquals(actual: unknown, expected: unknown, message: string): void {
  if (actual !== expected) throw new Error(message);
}

function rejectDuplicates(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) throw new Error(`${label} contain duplicates`);
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((item) => right.includes(item));
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
