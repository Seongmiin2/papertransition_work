import { useState } from "react";
import {
  candidateReviewTextRoles,
  type CandidateImageGroundingCandidate,
  type CandidateReviewContentRole,
  type CandidateReviewDraftIssueDisposition,
  type CandidateReviewDraftImageNode,
  type CandidateReviewDraftNode,
  type CandidateReviewDraftPatchOperation,
  type CandidateReviewDraftTableCell,
  type CandidateReviewDraftTableNode,
  type CandidateReviewDraftTextNode,
  type CandidateReviewIssueDisposition,
  type CandidateReviewTextRole,
} from "../types/contracts";

const MAX_TEXT_LENGTH = 1_000_000;
const TABLE_PAGE_SIZE = 40;

const ROLE_LABELS: Readonly<Record<CandidateReviewContentRole, string>> = {
  title: "제목",
  instruction: "지시문",
  passage: "지문",
  question: "문항",
  choice: "선택지",
  caption: "캡션",
  header: "머리말",
  footer: "꼬리말",
  other: "기타",
  table: "표",
  image: "이미지",
  formula: "수식",
};

export interface DraftNodeEditorProps {
  node: CandidateReviewDraftNode | null;
  imageGroundingCandidates?: readonly CandidateImageGroundingCandidate[];
  primaryPageNo?: number | null;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: (editorKey: string, dirty: boolean) => void;
  onSave: (operations: readonly CandidateReviewDraftPatchOperation[]) => void;
}

export function DraftNodeEditor({
  node,
  imageGroundingCandidates = [],
  primaryPageNo = null,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: DraftNodeEditorProps) {
  if (!node) {
    return <p className="draftEditorEmpty">현재 페이지에서 편집할 노드를 선택하세요.</p>;
  }
  if (node.kind === "text") {
    return (
      <TextNodeEditor
        node={node}
        saving={saving}
        readOnly={readOnly}
        activeDirtyEditorKey={activeDirtyEditorKey}
        onDirtyChange={onDirtyChange}
        onSave={onSave}
      />
    );
  }
  if (node.kind === "table") {
    return (
      <TableNodeEditor
        node={node}
        saving={saving}
        readOnly={readOnly}
        activeDirtyEditorKey={activeDirtyEditorKey}
        onDirtyChange={onDirtyChange}
        onSave={onSave}
      />
    );
  }
  if (node.kind === "image") {
    return (
      <ImageNodeEditor
        node={node}
        imageGroundingCandidates={imageGroundingCandidates}
        primaryPageNo={primaryPageNo}
        saving={saving}
        readOnly={readOnly}
        activeDirtyEditorKey={activeDirtyEditorKey}
        onDirtyChange={onDirtyChange}
        onSave={onSave}
      />
    );
  }
  return (
    <ReviewFlagEditor
      node={node}
      saving={saving}
      readOnly={readOnly}
      activeDirtyEditorKey={activeDirtyEditorKey}
      onDirtyChange={onDirtyChange}
      onSave={onSave}
    />
  );
}

function ImageNodeEditor({
  node,
  imageGroundingCandidates,
  primaryPageNo,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: {
  node: CandidateReviewDraftImageNode;
  imageGroundingCandidates: readonly CandidateImageGroundingCandidate[];
  primaryPageNo: number | null;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: DraftNodeEditorProps["onDirtyChange"];
  onSave: DraftNodeEditorProps["onSave"];
}) {
  const editorKey = `node-review:${node.id}`;
  const mutationLocked = activeDirtyEditorKey !== null;
  const orderedCandidates = orderImageGroundingCandidates(
    imageGroundingCandidates,
    primaryPageNo,
  );
  const [selectedIndex, setSelectedIndex] = useState(
    orderedCandidates.length > 0 ? 0 : -1,
  );
  const selectedCandidate = orderedCandidates[selectedIndex] ?? null;

  return (
    <div className="draftNodeEditor">
      <ReviewFlagEditor
        node={node}
        saving={saving}
        readOnly={readOnly}
        activeDirtyEditorKey={activeDirtyEditorKey}
        onDirtyChange={onDirtyChange}
        onSave={onSave}
      />
      <p className="draftEditorHint">
        원본과 대응하지 않는 페이지 전체 대체 이미지는 제거할 수 있습니다. 제거는 새 리비전으로
        기록됩니다.
      </p>
      <label>
        <span>정확한 이미지 근거</span>
        <select
          value={selectedCandidate ? String(selectedIndex) : ""}
          disabled={saving || readOnly || mutationLocked || orderedCandidates.length === 0}
          onChange={(event) => setSelectedIndex(Number(event.target.value))}
        >
          {orderedCandidates.length === 0 && (
            <option value="">연결 가능한 IMAGE 근거 없음</option>
          )}
          {orderedCandidates.map((candidate, index) => (
            <option
              key={`${candidate.observationRef}:${candidate.assetRef}`}
              value={index}
            >
              {candidate.pageNo}쪽 · {candidate.sourceKind === "crop" ? "잘라낸 이미지" : "페이지 이미지"} · {candidate.assetRef} · {candidate.sha256.slice(0, 12)}…
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        className="secondary"
        disabled={saving || readOnly || mutationLocked || selectedCandidate === null}
        onClick={() =>
          selectedCandidate &&
          onSave([buildSetImageGroundingPatchOperation(node, selectedCandidate)])
        }
      >
        {readOnly ? "읽기 전용" : "선택 근거로 연결"}
      </button>
      <button
        type="button"
        className="secondary"
        disabled={saving || readOnly || mutationLocked}
        onClick={() => onSave([buildDropImageNodePatchOperation(node)])}
      >
        {readOnly ? "읽기 전용" : "이 이미지 노드 제거"}
      </button>
    </div>
  );
}

function TextNodeEditor({
  node,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: {
  node: CandidateReviewDraftTextNode;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: DraftNodeEditorProps["onDirtyChange"];
  onSave: DraftNodeEditorProps["onSave"];
}) {
  const [text, setText] = useState(node.text);
  const [role, setRole] = useState<CandidateReviewContentRole>(node.role);
  const [needsReview, setNeedsReview] = useState(node.needsReview);
  const editorKey = `node-text:${node.id}`;
  const locked = isEditorLocked(activeDirtyEditorKey, editorKey);
  const operations = buildTextNodePatchOperations(node, {
    text,
    role,
    needsReview,
  });
  const invalid = text.length === 0 || text.length > MAX_TEXT_LENGTH;

  return (
    <div className="draftNodeEditor">
      <label>
        <span>문서 역할</span>
        <select
          value={role}
          disabled={saving || readOnly || locked}
          onChange={(event) => {
            const nextRole = event.target.value as CandidateReviewContentRole;
            setRole(nextRole);
            onDirtyChange(
              editorKey,
              text !== node.text ||
                nextRole !== node.role ||
                needsReview !== node.needsReview,
            );
          }}
        >
          {!isCandidateReviewTextRole(node.role) && (
            <option value={node.role}>{ROLE_LABELS[node.role]} (현재 값)</option>
          )}
          {candidateReviewTextRoles.map((value) => (
            <option key={value} value={value}>{ROLE_LABELS[value]}</option>
          ))}
        </select>
      </label>
      <label>
        <span>검수 텍스트</span>
        <textarea
          value={text}
          maxLength={MAX_TEXT_LENGTH}
          disabled={saving || readOnly || locked}
          aria-invalid={invalid}
          onChange={(event) => {
            const nextText = event.target.value;
            setText(nextText);
            onDirtyChange(
              editorKey,
              nextText !== node.text ||
                role !== node.role ||
                needsReview !== node.needsReview,
            );
          }}
        />
      </label>
      <label className="draftCheck">
        <input
          type="checkbox"
          checked={needsReview}
          disabled={saving || readOnly || locked}
          onChange={(event) => {
            const nextNeedsReview = event.target.checked;
            setNeedsReview(nextNeedsReview);
            onDirtyChange(
              editorKey,
              text !== node.text ||
                role !== node.role ||
                nextNeedsReview !== node.needsReview,
            );
          }}
        />
        추가 검수 필요
      </label>
      <button
        type="button"
        className="primary draftSave"
        disabled={saving || readOnly || locked || invalid || operations.length === 0}
        onClick={() => onSave(operations)}
      >
        {readOnly
          ? "읽기 전용 완료본"
          : saving
            ? "새 리비전 저장 중"
            : "변경을 새 리비전으로 저장"}
      </button>
    </div>
  );
}

function TableNodeEditor({
  node,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: {
  node: CandidateReviewDraftTableNode;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: DraftNodeEditorProps["onDirtyChange"];
  onSave: DraftNodeEditorProps["onSave"];
}) {
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(node.cells.length / TABLE_PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visibleCells = node.cells.slice(
    safePage * TABLE_PAGE_SIZE,
    (safePage + 1) * TABLE_PAGE_SIZE,
  );

  return (
    <div className="draftNodeEditor">
      <ReviewFlagEditor
        node={node}
        saving={saving}
        readOnly={readOnly}
        activeDirtyEditorKey={activeDirtyEditorKey}
        onDirtyChange={onDirtyChange}
        onSave={onSave}
      />
      <div className="draftTableToolbar">
        <strong>표 셀 {node.cells.length}개</strong>
        <span>
          <button
            type="button"
            disabled={activeDirtyEditorKey !== null || safePage === 0}
            onClick={() => setPage(safePage - 1)}
          >이전</button>
          {safePage + 1} / {pageCount}
          <button
            type="button"
            disabled={
              activeDirtyEditorKey !== null || safePage + 1 >= pageCount
            }
            onClick={() => setPage(safePage + 1)}
          >다음</button>
        </span>
      </div>
      <div className="draftCellList">
        {visibleCells.map((cell) => (
          <TableCellEditor
            key={tableCellKey(cell)}
            nodeId={node.id}
            cell={cell}
            saving={saving}
            readOnly={readOnly}
            activeDirtyEditorKey={activeDirtyEditorKey}
            onDirtyChange={onDirtyChange}
            onSave={onSave}
          />
        ))}
      </div>
    </div>
  );
}

function TableCellEditor({
  nodeId,
  cell,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: {
  nodeId: string;
  cell: CandidateReviewDraftTableCell;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: DraftNodeEditorProps["onDirtyChange"];
  onSave: DraftNodeEditorProps["onSave"];
}) {
  const [text, setText] = useState(cell.text);
  const editorKey = `table-cell:${nodeId}:${tableCellKey(cell)}`;
  const locked = isEditorLocked(activeDirtyEditorKey, editorKey);
  const operation = buildTableCellPatchOperation(nodeId, cell, text);

  return (
    <label className="draftCell">
      <span>
        행 {cell.row + 1}, 열 {cell.column + 1}
        {(cell.rowSpan > 1 || cell.columnSpan > 1) &&
          ` · 병합 ${cell.rowSpan}×${cell.columnSpan}`}
      </span>
      <div>
        <textarea
          value={text}
          maxLength={MAX_TEXT_LENGTH}
          disabled={saving || readOnly || locked}
          onChange={(event) => {
            const nextText = event.target.value;
            setText(nextText);
            onDirtyChange(editorKey, nextText !== cell.text);
          }}
        />
        <button
          type="button"
          className="secondary"
          disabled={saving || readOnly || locked || operation === null}
          onClick={() => operation && onSave([operation])}
        >
          {readOnly ? "읽기 전용" : "셀 저장"}
        </button>
      </div>
    </label>
  );
}

function ReviewFlagEditor({
  node,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: {
  node: CandidateReviewDraftNode;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: DraftNodeEditorProps["onDirtyChange"];
  onSave: DraftNodeEditorProps["onSave"];
}) {
  const [needsReview, setNeedsReview] = useState(node.needsReview);
  const editorKey = `node-review:${node.id}`;
  const locked = isEditorLocked(activeDirtyEditorKey, editorKey);
  const operation = buildNeedsReviewPatchOperation(node, needsReview);

  return (
    <div className="draftReviewFlag">
      <label className="draftCheck">
        <input
          type="checkbox"
          checked={needsReview}
          disabled={saving || readOnly || locked}
          onChange={(event) => {
            const nextNeedsReview = event.target.checked;
            setNeedsReview(nextNeedsReview);
            onDirtyChange(editorKey, nextNeedsReview !== node.needsReview);
          }}
        />
        추가 검수 필요
      </label>
      <button
        type="button"
        className="secondary"
        disabled={saving || readOnly || locked || operation === null}
        onClick={() => operation && onSave([operation])}
      >
        {readOnly ? "읽기 전용" : "상태 저장"}
      </button>
    </div>
  );
}

export function DraftIssueEditor({
  issue,
  description,
  saving,
  readOnly,
  activeDirtyEditorKey,
  onDirtyChange,
  onSave,
}: {
  issue: CandidateReviewDraftIssueDisposition;
  description: string;
  saving: boolean;
  readOnly: boolean;
  activeDirtyEditorKey: string | null;
  onDirtyChange: DraftNodeEditorProps["onDirtyChange"];
  onSave: DraftNodeEditorProps["onSave"];
}) {
  const [disposition, setDisposition] =
    useState<CandidateReviewIssueDisposition>(issue.disposition);
  const [note, setNote] = useState(issue.note);
  const editorKey = `issue:${issue.issueCode}`;
  const locked = isEditorLocked(activeDirtyEditorKey, editorKey);
  const operation = buildIssueDispositionPatchOperation(issue, disposition, note);
  const missingReason =
    disposition === "accepted_limitation" && note.trim().length === 0;

  return (
    <article className="draftIssueEditor">
      <code>{issue.issueCode}</code>
      <p>{description}</p>
      <select
        value={disposition}
        disabled={saving || readOnly || locked}
        onChange={(event) => {
          const nextDisposition = event.target.value as CandidateReviewIssueDisposition;
          setDisposition(nextDisposition);
          onDirtyChange(
            editorKey,
            nextDisposition !== issue.disposition || note !== issue.note,
          );
        }}
      >
        <option value="pending">검수 대기</option>
        <option value="resolved">해결됨</option>
        <option value="accepted_limitation">한계 수용</option>
      </select>
      <textarea
        value={note}
        maxLength={MAX_TEXT_LENGTH}
        disabled={saving || readOnly || locked}
        placeholder={disposition === "accepted_limitation" ? "한계를 수용한 이유(필수)" : "검수 메모"}
        aria-invalid={missingReason}
        onChange={(event) => {
          const nextNote = event.target.value;
          setNote(nextNote);
          onDirtyChange(
            editorKey,
            disposition !== issue.disposition || nextNote !== issue.note,
          );
        }}
      />
      <button
        type="button"
        className="secondary"
        disabled={saving || readOnly || locked || missingReason || operation === null}
        onClick={() => operation && onSave([operation])}
      >
        {readOnly ? "읽기 전용 완료본" : "이슈 저장"}
      </button>
    </article>
  );
}

export function buildTextNodePatchOperations(
  node: CandidateReviewDraftTextNode,
  next: {
    text: string;
    role: CandidateReviewContentRole;
    needsReview: boolean;
  },
): CandidateReviewDraftPatchOperation[] {
  if (
    next.text.length === 0 ||
    next.text.length > MAX_TEXT_LENGTH ||
    (next.role !== node.role && !isCandidateReviewTextRole(next.role))
  ) {
    return [];
  }
  const operations: CandidateReviewDraftPatchOperation[] = [];
  if (next.text !== node.text) {
    operations.push({ op: "set_text", nodeId: node.id, text: next.text });
  }
  if (next.role !== node.role) {
    operations.push({
      op: "set_role",
      nodeId: node.id,
      role: next.role as CandidateReviewTextRole,
    });
  }
  if (next.needsReview !== node.needsReview) {
    operations.push({
      op: "set_needs_review",
      nodeId: node.id,
      needsReview: next.needsReview,
    });
  }
  return operations;
}

export function buildTableCellPatchOperation(
  nodeId: string,
  cell: CandidateReviewDraftTableCell,
  text: string,
): CandidateReviewDraftPatchOperation | null {
  if (text === cell.text || text.length > MAX_TEXT_LENGTH) return null;
  return {
    op: "set_table_cell_text",
    nodeId,
    row: cell.row,
    column: cell.column,
    rowSpan: cell.rowSpan,
    columnSpan: cell.columnSpan,
    text,
  };
}

export function buildNeedsReviewPatchOperation(
  node: CandidateReviewDraftNode,
  needsReview: boolean,
): CandidateReviewDraftPatchOperation | null {
  return needsReview === node.needsReview
    ? null
    : { op: "set_needs_review", nodeId: node.id, needsReview };
}

export function buildDropImageNodePatchOperation(
  node: CandidateReviewDraftImageNode,
): CandidateReviewDraftPatchOperation {
  return { op: "drop_image_node", nodeId: node.id };
}

export function buildSetImageGroundingPatchOperation(
  node: CandidateReviewDraftImageNode,
  candidate: CandidateImageGroundingCandidate,
): CandidateReviewDraftPatchOperation {
  return {
    op: "set_image_grounding",
    nodeId: node.id,
    assetRef: candidate.assetRef,
    observationRef: candidate.observationRef,
  };
}

export function buildIssueDispositionPatchOperation(
  issue: CandidateReviewDraftIssueDisposition,
  disposition: CandidateReviewIssueDisposition,
  note: string,
): CandidateReviewDraftPatchOperation | null {
  if (
    note.length > MAX_TEXT_LENGTH ||
    (disposition === "accepted_limitation" && note.trim().length === 0)
  ) {
    return null;
  }
  if (disposition === issue.disposition && note === issue.note) return null;
  return {
    op: "set_issue_disposition",
    issueCode: issue.issueCode,
    disposition,
    note,
  };
}

function tableCellKey(cell: CandidateReviewDraftTableCell): string {
  return [cell.row, cell.column, cell.rowSpan, cell.columnSpan].join(":");
}

function orderImageGroundingCandidates(
  candidates: readonly CandidateImageGroundingCandidate[],
  primaryPageNo: number | null,
): CandidateImageGroundingCandidate[] {
  return [...candidates].sort((left, right) => {
    const leftPrimary = left.pageNo === primaryPageNo ? 0 : 1;
    const rightPrimary = right.pageNo === primaryPageNo ? 0 : 1;
    return (
      leftPrimary - rightPrimary ||
      left.pageNo - right.pageNo ||
      left.observationRef.localeCompare(right.observationRef) ||
      left.assetRef.localeCompare(right.assetRef)
    );
  });
}

export function isEditorLocked(
  activeDirtyEditorKey: string | null,
  editorKey: string,
): boolean {
  return activeDirtyEditorKey !== null && activeDirtyEditorKey !== editorKey;
}

function isCandidateReviewTextRole(
  value: CandidateReviewContentRole,
): value is CandidateReviewTextRole {
  return (candidateReviewTextRoles as readonly string[]).includes(value);
}
