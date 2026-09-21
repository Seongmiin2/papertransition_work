import { createHash } from "node:crypto";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { startOrReopenCandidateReviewDraft } from "../dist-electron/main/candidate-review-draft.js";
import { openCandidateReviewBundle } from "../dist-electron/main/candidate-review.js";
import {
  applyCandidateReviewDraftPatch,
  completeCandidateReviewDraft,
  loadCandidateReviewDraftView,
} from "../dist-electron/main/candidate-review-draft-patch.js";

const candidateArgument = process.argv[2];
if (!candidateArgument) {
  console.error("usage: node scripts/smoke-candidate-review.mjs <candidate-root>");
  process.exit(2);
}

const projectRoot = realpathSync.native(resolve(import.meta.dirname, ".."));
const requestedCandidateRoot = resolve(candidateArgument);
if (!isAbsolute(requestedCandidateRoot)) {
  throw new Error("candidate root must resolve to an absolute path");
}
const candidateRoot = realpathSync.native(requestedCandidateRoot);
const manifestPath = join(candidateRoot, "manifest.json");
const manifestBytes = readFileSync(manifestPath);
const manifest = JSON.parse(manifestBytes.toString("utf8"));
const manifestSha256 = sha256(manifestBytes);
const document = manifest.documents?.[0];
if (
  typeof document?.document_id !== "string" ||
  typeof document?.lineage_id !== "string"
) {
  throw new Error("candidate manifest has no smoke-test document");
}

const temporaryRoot = realpathSync.native(tmpdir());
const userDataRoot = mkdtempSync(join(temporaryRoot, "pw-review-"));
if (!isStrictChild(temporaryRoot, userDataRoot)) {
  throw new Error("unsafe smoke-test directory");
}

try {
  const options = {
    projectRoot,
    userDataRoot,
    candidateRoot,
    manifestSha256,
    lineageId: document.lineage_id,
    documentId: document.document_id,
  };
  const created = await startOrReopenCandidateReviewDraft({
    ...options,
    reviewerLabel: "smoke_reviewer",
  });
  const initial = await loadCandidateReviewDraftView(options);
  if (!initial.nodes.length) throw new Error("candidate draft view has no nodes");
  const reviewDocument = openCandidateReviewBundle(candidateRoot).loadDocument(
    document.lineage_id,
  );
  const overlaysByContentRef = new Map(
    reviewDocument.overlays.map((overlay) => [overlay.contentRef, overlay]),
  );
  const droppedFallbackImageIds = new Set(
    initial.nodes
      .filter(
        (node) =>
          node.kind === "image" &&
          overlaysByContentRef.get(node.id)?.usesPageRegionFallback === true,
      )
      .map((node) => node.id),
  );
  const smokeOnlyOperations = [
    ...[...droppedFallbackImageIds].map((nodeId) => ({
      op: "drop_image_node",
      nodeId,
    })),
    ...initial.nodes
      .filter((node) => node.needsReview && !droppedFallbackImageIds.has(node.id))
      .map((node) => ({
        op: "set_needs_review",
        nodeId: node.id,
        needsReview: false,
      })),
    ...initial.issueDispositions
      .filter((issue) => issue.disposition === "pending")
      .map((issue) => ({
        op: "set_issue_disposition",
        issueCode: issue.issueCode,
        disposition: "accepted_limitation",
        note: "ephemeral smoke test only; not a human approval",
      })),
  ];
  let saved = initial;
  for (let offset = 0; offset < smokeOnlyOperations.length; offset += 1_000) {
    saved = await applyCandidateReviewDraftPatch({
      ...options,
      request: {
        expectedDraftRevision: saved.draftRevision,
        operations: smokeOnlyOperations.slice(offset, offset + 1_000),
      },
    });
  }
  const completed = await completeCandidateReviewDraft({
    ...options,
    request: { expectedDraftRevision: saved.draftRevision },
  });
  const reopened = await startOrReopenCandidateReviewDraft({
    ...options,
    reviewerLabel: "smoke_reviewer",
  });
  const verified = await loadCandidateReviewDraftView(options);
  const draftDirectory = join(
    userDataRoot,
    "candidate-reviews",
    manifestSha256,
    document.lineage_id,
  );
  const immutableRevisions = Array.from(
    { length: completed.draftRevision },
    (_, index) =>
      index === 0
        ? join(draftDirectory, "draft.json")
        : join(draftDirectory, `draft.r${index + 1}.json`),
  ).every(existsSync);
  const summary = {
    operation: created.operation,
    initialRevision: initial.draftRevision,
    savedRevision: saved.draftRevision,
    completedRevision: completed.draftRevision,
    completedStatus: completed.status,
    reopenedRevision: reopened.draftRevision,
    verifiedRevision: verified.draftRevision,
    immutableRevisions,
    candidateManifestUnchanged: sha256(readFileSync(manifestPath)) === manifestSha256,
    rightsStatus: verified.rightsStatus,
    trainingEligible: verified.trainingEligible,
    goldenEligible: verified.goldenEligible,
    releaseEligible: verified.releaseEligible,
    unresolvedNodes: verified.nodes.filter((node) => node.needsReview).length,
    pendingIssues: verified.issueDispositions.filter(
      (issue) => issue.disposition === "pending",
    ).length,
    nodeCount: verified.nodes.length,
    issueCount: verified.issueDispositions.length,
    droppedFallbackImages: droppedFallbackImageIds.size,
    availableExactImageGroundings: reviewDocument.imageGroundingCandidates.length,
    testOnlyAutoResolution: true,
  };
  if (
    summary.operation !== "created" ||
    summary.initialRevision !== 1 ||
    summary.completedRevision !== summary.savedRevision + 1 ||
    summary.completedStatus !== "complete" ||
    summary.reopenedRevision !== summary.completedRevision ||
    summary.verifiedRevision !== summary.completedRevision ||
    !summary.immutableRevisions ||
    !summary.candidateManifestUnchanged ||
    summary.rightsStatus !== "unverified" ||
    summary.trainingEligible !== false ||
    summary.goldenEligible !== false ||
    summary.releaseEligible !== false ||
    summary.unresolvedNodes !== 0 ||
    summary.pendingIssues !== 0
  ) {
    throw new Error("candidate review smoke invariants failed");
  }
  console.log(JSON.stringify(summary));
} finally {
  if (!isStrictChild(temporaryRoot, userDataRoot)) {
    throw new Error("refusing unsafe smoke-test cleanup");
  }
  rmSync(userDataRoot, { recursive: true, force: true });
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function isStrictChild(parent, child) {
  const offset = relative(parent, resolve(child));
  return (
    offset !== "" &&
    offset !== ".." &&
    !offset.startsWith(`..${sep}`) &&
    !isAbsolute(offset)
  );
}
