from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scan2hwpx.contracts import (
    ContentIR,
    EvidenceIR,
    EvidencePage,
    EvidenceSourceKind,
    ObservationKind,
    TableContentNode,
    contract_sha256,
)
from scan2hwpx.evaluation.strict_json import require_strict_json_bytes
from scan2hwpx.model_a.inference import ModelAPageImage
from scan2hwpx.model_a.table_topology import (
    TABLE_TOPOLOGY_POLICY_SHA256,
    BuiltModelATableTopology,
    ModelATableTopologyArtifact,
    ModelATableTopologyTable,
    build_model_a_table_topology,
)
from scan2hwpx.reference.hwpx import HwpxProjectionSource, ProjectedNode

CellKey = tuple[int, int, int, int]
NormalizedBBox = tuple[float, float, float, float]
AssignmentPair = tuple[CellKey, str]


class _BuiltTopologyLike(Protocol):
    artifact: ModelATableTopologyArtifact


@dataclass(frozen=True, slots=True)
class _ProxyCell:
    key: CellKey
    text: str
    ordered_observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProxyTable:
    source_node_id: str
    content_node_id: str
    page_id: str
    page_no: int
    rows: int
    columns: int
    bbox_normalized: NormalizedBBox | None
    cells: tuple[_ProxyCell, ...]

    @property
    def ordered_observation_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                observation_id
                for cell in self.cells
                for observation_id in cell.ordered_observation_ids
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic Model A table-topology detection and compare it with an "
            "unverified exact-HWPX projection reference."
        )
    )
    parser.add_argument("--evidence-ir", type=Path, required=True)
    parser.add_argument("--projection-source", type=Path, required=True)
    parser.add_argument("--proxy-content-ir", type=Path, required=True)
    parser.add_argument("--pages-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-id", action="append", dest="page_ids")
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _f1(true_positive: int, predicted: int, reference: int) -> float | None:
    precision = _rate(true_positive, predicted)
    recall = _rate(true_positive, reference)
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return "".join(
        character
        for character in normalized
        if not character.isspace() and unicodedata.category(character) != "Cf"
    )


def _cell_key(cell: Any) -> CellKey:
    if isinstance(cell, _ProxyCell):
        return cell.key
    return (cell.row, cell.column, cell.row_span, cell.column_span)


def _projection_content_id(node: ProjectedNode) -> str:
    signature = hashlib.sha256(node.id.encode("utf-8")).hexdigest()
    return f"content-{signature[:24]}"


def _union_bbox(boxes: Sequence[NormalizedBBox]) -> NormalizedBBox | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_iou(first: NormalizedBBox | None, second: NormalizedBBox) -> float | None:
    if first is None:
        return None
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union == 0 else intersection / union


def _build_proxy_tables(
    projection: HwpxProjectionSource,
    content: ContentIR,
    evidence: EvidenceIR,
) -> tuple[_ProxyTable, ...]:
    if projection.page_count != len(evidence.pages):
        raise ValueError("projection and EvidenceIR page counts differ")
    content_tables = {node.id: node for node in content.nodes if isinstance(node, TableContentNode)}
    source_tables = tuple(node for node in projection.nodes if node.kind == "table")
    expected_ids = tuple(_projection_content_id(node) for node in source_tables)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("projected table ids collide after candidate id derivation")
    if set(content_tables) != set(expected_ids):
        raise ValueError("proxy ContentIR tables do not exactly cover projected HWPX tables")

    observation_page: dict[str, tuple[str, int]] = {}
    observation_kind: dict[str, ObservationKind] = {}
    observation_bbox: dict[str, NormalizedBBox] = {}
    for page in evidence.pages:
        for observation in page.observations:
            observation_page[observation.id] = (page.id, page.page_no)
            observation_kind[observation.id] = observation.kind
            observation_bbox[observation.id] = observation.bbox.normalized

    proxies: list[_ProxyTable] = []
    for source in source_tables:
        if source.id != f"{projection.source_document_sha256}:{source.locator}":
            raise ValueError("projected table id is not bound to its source digest and locator")
        if source.rows is None or source.columns is None:
            raise ValueError("projected table lost its declared dimensions")
        content_id = _projection_content_id(source)
        candidate = content_tables[content_id]
        if candidate.rows != source.rows or candidate.columns != source.columns:
            raise ValueError("proxy table dimensions differ from projected HWPX table")

        source_cells = {_cell_key(cell): cell for cell in source.cells}
        candidate_cells = {_cell_key(cell): cell for cell in candidate.cells}
        if len(source_cells) != len(source.cells) or len(candidate_cells) != len(candidate.cells):
            raise ValueError("table contains duplicate cell topology keys")
        if set(source_cells) != set(candidate_cells):
            raise ValueError("proxy table grid/span differs from projected HWPX table")
        for key, source_cell in source_cells.items():
            if candidate_cells[key].text != source_cell.text:
                raise ValueError("proxy table cell text differs from projected HWPX table")

        nested_refs = tuple(
            dict.fromkeys(ref for cell in candidate.cells for ref in cell.evidence_refs)
        )
        if candidate.evidence_refs != nested_refs:
            raise ValueError("proxy table parent refs do not equal its ordered cell-ref union")
        pages = {observation_page[ref] for ref in candidate.evidence_refs}
        if len(pages) != 1:
            raise ValueError("proxy table evidence refs do not bind exactly one page")
        page_id, page_no = next(iter(pages))

        proxy_cells = tuple(
            _ProxyCell(
                key=_cell_key(cell),
                text=cell.text,
                ordered_observation_ids=tuple(
                    ref
                    for ref in cell.evidence_refs
                    if observation_kind[ref] == ObservationKind.TEXT_LINE
                ),
            )
            for cell in candidate.cells
        )
        text_ids = tuple(
            dict.fromkeys(
                observation_id
                for cell in proxy_cells
                for observation_id in cell.ordered_observation_ids
            )
        )
        proxies.append(
            _ProxyTable(
                source_node_id=source.id,
                content_node_id=content_id,
                page_id=page_id,
                page_no=page_no,
                rows=source.rows,
                columns=source.columns,
                bbox_normalized=_union_bbox(
                    tuple(observation_bbox[observation_id] for observation_id in text_ids)
                ),
                cells=proxy_cells,
            )
        )
    return tuple(proxies)


def _maximum_overlap_matching(
    references: Sequence[_ProxyTable],
    predictions: Sequence[ModelATableTopologyTable],
) -> tuple[tuple[int, int], ...]:
    if not references or not predictions:
        return ()
    reference_ids = [set(table.ordered_observation_ids) for table in references]
    prediction_ids = [set(table.ordered_observation_ids) for table in predictions]
    weights = [
        [len(reference & prediction) for prediction in prediction_ids]
        for reference in reference_ids
    ]
    size = max(len(references), len(predictions))
    maximum = max((weight for row in weights for weight in row), default=0)
    costs = [
        [
            maximum
            - (weights[row][column] if row < len(references) and column < len(predictions) else 0)
            for column in range(size)
        ]
        for row in range(size)
    ]

    # Hungarian assignment on the padded square cost matrix. Stable traversal
    # makes equal-overlap ties deterministic.
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    column_owner = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row in range(1, size + 1):
        column_owner[0] = row
        minimum = [10**18] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            active_row = column_owner[column]
            delta = 10**18
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                current = (
                    costs[active_row - 1][candidate_column - 1]
                    - u[active_row]
                    - v[candidate_column]
                )
                if current < minimum[candidate_column]:
                    minimum[candidate_column] = current
                    previous_column[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    u[column_owner[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if column_owner[column] == 0:
                break
        while True:
            previous = previous_column[column]
            column_owner[column] = column_owner[previous]
            column = previous
            if column == 0:
                break

    return tuple(
        sorted(
            (row - 1, column - 1)
            for column, row in enumerate(column_owner[1:], start=1)
            if row > 0
            and row <= len(references)
            and column <= len(predictions)
            and weights[row - 1][column - 1] > 0
        )
    )


def _assignment_counter(
    table: _ProxyTable | ModelATableTopologyTable,
) -> Counter[AssignmentPair]:
    return Counter(
        (_cell_key(cell), observation_id)
        for cell in table.cells
        for observation_id in cell.ordered_observation_ids
    )


def _ambiguous_reference_ids(table: _ProxyTable) -> frozenset[str]:
    counts = Counter(
        observation_id for cell in table.cells for observation_id in cell.ordered_observation_ids
    )
    return frozenset(observation_id for observation_id, count in counts.items() if count > 1)


def _table_comparison(
    reference: _ProxyTable,
    prediction: ModelATableTopologyTable | None,
) -> dict[str, object]:
    reference_cells = {cell.key: cell for cell in reference.cells}
    predicted_cells = (
        {} if prediction is None else {_cell_key(cell): cell for cell in prediction.cells}
    )
    reference_signature = tuple(sorted(reference_cells))
    predicted_signature = tuple(sorted(predicted_cells))
    ambiguous_ids = _ambiguous_reference_ids(reference)
    ambiguous_cell_keys = {
        cell.key for cell in reference.cells if set(cell.ordered_observation_ids) & ambiguous_ids
    }
    unambiguous_keys = set(reference_cells) - ambiguous_cell_keys
    assignment_exact = {
        key: (
            key in predicted_cells
            and reference_cells[key].ordered_observation_ids
            == predicted_cells[key].ordered_observation_ids
        )
        for key in reference_cells
    }
    text_exact = {
        key: (
            key in predicted_cells
            and _normalize_text(reference_cells[key].text)
            == _normalize_text(predicted_cells[key].text)
        )
        for key in reference_cells
    }
    reference_pairs = _assignment_counter(reference)
    predicted_pairs: Counter[AssignmentPair] = (
        Counter() if prediction is None else _assignment_counter(prediction)
    )
    common_pairs = reference_pairs & predicted_pairs
    missing_pairs = reference_pairs - predicted_pairs
    extra_pairs = predicted_pairs - reference_pairs
    unambiguous_reference_pairs = Counter(
        {pair: count for pair, count in reference_pairs.items() if pair[1] not in ambiguous_ids}
    )
    unambiguous_predicted_pairs = Counter(
        {pair: count for pair, count in predicted_pairs.items() if pair[1] not in ambiguous_ids}
    )
    unambiguous_common = unambiguous_reference_pairs & unambiguous_predicted_pairs
    reference_ids = set(reference.ordered_observation_ids)
    predicted_ids = set() if prediction is None else set(prediction.ordered_observation_ids)
    outside_ids = sorted(predicted_ids - reference_ids)
    reference_empty = tuple(sorted(cell.key for cell in reference.cells if cell.text == ""))
    predicted_empty = tuple(
        sorted(
            _cell_key(cell)
            for cell in (() if prediction is None else prediction.cells)
            if cell.text == "" and not cell.ordered_observation_ids
        )
    )
    strict_exact_count = sum(assignment_exact.values())
    unambiguous_exact_count = sum(assignment_exact[key] for key in unambiguous_keys)
    normalized_text_exact_count = sum(text_exact.values())
    unambiguous_text_exact_count = sum(text_exact[key] for key in unambiguous_keys)
    predicted_nested_counts = Counter(
        observation_id
        for cell in (() if prediction is None else prediction.cells)
        for observation_id in cell.ordered_observation_ids
    )
    predicted_shared_ids = frozenset(
        observation_id for observation_id, count in predicted_nested_counts.items() if count > 1
    )
    return {
        "reference_content_node_id": reference.content_node_id,
        "reference_source_node_id": reference.source_node_id,
        "predicted_table_grid_observation_id": (
            None if prediction is None else prediction.table_grid_observation_id
        ),
        "shared_text_observation_count": len(reference_ids & predicted_ids),
        "bbox_iou_against_reference_text_union": (
            None
            if prediction is None
            else _bbox_iou(reference.bbox_normalized, prediction.bbox_normalized)
        ),
        "reference_rows": reference.rows,
        "reference_columns": reference.columns,
        "predicted_rows": None if prediction is None else prediction.rows,
        "predicted_columns": None if prediction is None else prediction.columns,
        "rows_columns_exact": (
            prediction is not None
            and (reference.rows, reference.columns) == (prediction.rows, prediction.columns)
        ),
        "grid_span_exact": prediction is not None and reference_signature == predicted_signature,
        "reference_physical_cell_count": len(reference.cells),
        "predicted_physical_cell_count": 0 if prediction is None else len(prediction.cells),
        "reference_grid_slot_count": reference.rows * reference.columns,
        "reference_spanning_cell_count": sum(key[2:] != (1, 1) for key in reference_signature),
        "reference_empty_cell_count": len(reference_empty),
        "predicted_empty_cell_count": len(predicted_empty),
        "empty_cell_signature_exact": (
            prediction is not None and reference_empty == predicted_empty
        ),
        "reference_ambiguous_text_observation_ids": sorted(ambiguous_ids),
        "reference_ambiguous_text_observation_count": len(ambiguous_ids),
        "reference_ambiguous_cell_count": len(ambiguous_cell_keys),
        "strict_cell_assignment_exact_count": strict_exact_count,
        "strict_cell_assignment_sample_count": len(reference_cells),
        "strict_cell_assignment_exact": (
            prediction is not None
            and reference_signature == predicted_signature
            and strict_exact_count == len(reference_cells)
        ),
        "unambiguous_cell_assignment_exact_count": unambiguous_exact_count,
        "unambiguous_cell_assignment_sample_count": len(unambiguous_keys),
        "normalized_cell_text_exact_count": normalized_text_exact_count,
        "normalized_cell_text_sample_count": len(reference_cells),
        "unambiguous_normalized_cell_text_exact_count": unambiguous_text_exact_count,
        "unambiguous_normalized_cell_text_sample_count": len(unambiguous_keys),
        "predicted_shared_text_observation_ids": sorted(predicted_shared_ids),
        "shared_text_observation_signature_exact": (
            prediction is not None and ambiguous_ids == predicted_shared_ids
        ),
        "reference_assignment_pair_count": sum(reference_pairs.values()),
        "predicted_assignment_pair_count": sum(predicted_pairs.values()),
        "assignment_pair_true_positive_count": sum(common_pairs.values()),
        "missing_assignment_pair_count": sum(missing_pairs.values()),
        "extra_assignment_pair_count": sum(extra_pairs.values()),
        "unambiguous_reference_assignment_pair_count": sum(unambiguous_reference_pairs.values()),
        "unambiguous_predicted_assignment_pair_count": sum(unambiguous_predicted_pairs.values()),
        "unambiguous_assignment_pair_true_positive_count": sum(unambiguous_common.values()),
        "missing_reference_text_observation_ids": sorted(reference_ids - predicted_ids),
        "outside_absorbed_text_observation_ids": outside_ids,
        "outside_absorbed_text_observation_count": len(outside_ids),
    }


def _page_comparison(
    page_id: str,
    page_no: int,
    references: Sequence[_ProxyTable],
    prediction: ModelATableTopologyArtifact,
) -> dict[str, object]:
    predicted_tables = prediction.tables
    matches = _maximum_overlap_matching(references, predicted_tables)
    prediction_by_reference = {reference: predicted for reference, predicted in matches}
    matched_predictions = {predicted for _reference, predicted in matches}
    details = [
        _table_comparison(reference, predicted_tables[prediction_by_reference[index]])
        if index in prediction_by_reference
        else _table_comparison(reference, None)
        for index, reference in enumerate(references)
    ]
    extra_predictions = [
        table for index, table in enumerate(predicted_tables) if index not in matched_predictions
    ]
    reference_ids = {
        observation_id for table in references for observation_id in table.ordered_observation_ids
    }
    predicted_ids = {
        observation_id
        for table in predicted_tables
        for observation_id in table.ordered_observation_ids
    }
    extra_pair_count = sum(
        len(cell.ordered_observation_ids) for table in extra_predictions for cell in table.cells
    )
    return {
        "page_id": page_id,
        "page_no": page_no,
        "reference_table_count": len(references),
        "predicted_table_count": len(predicted_tables),
        "table_count_exact": len(references) == len(predicted_tables),
        "matched_table_count": len(matches),
        "matching_basis": "maximum_total_shared_unique_text_observation_count_one_to_one",
        "reference_text_observation_count": len(reference_ids),
        "predicted_table_text_observation_count": len(predicted_ids),
        "document_non_table_absorbed_observation_ids": sorted(predicted_ids - reference_ids),
        "document_non_table_absorbed_observation_count": len(predicted_ids - reference_ids),
        "missed_reference_table_observation_ids": sorted(reference_ids - predicted_ids),
        "missed_reference_table_observation_count": len(reference_ids - predicted_ids),
        "extra_predicted_assignment_pair_count": extra_pair_count,
        "unmatched_predicted_table_grid_observation_ids": [
            table.table_grid_observation_id for table in extra_predictions
        ],
        "tables": details,
    }


def _sum_details(pages: Sequence[dict[str, Any]], field: str) -> int:
    return sum(int(table[field]) for page in pages for table in page["tables"])


def _aggregate_metrics(pages: Sequence[dict[str, Any]]) -> dict[str, object]:
    reference_tables = sum(int(page["reference_table_count"]) for page in pages)
    predicted_tables = sum(int(page["predicted_table_count"]) for page in pages)
    matched_tables = sum(int(page["matched_table_count"]) for page in pages)
    rows_columns_exact = sum(
        int(bool(table["rows_columns_exact"])) for page in pages for table in page["tables"]
    )
    grid_span_exact = sum(
        int(bool(table["grid_span_exact"])) for page in pages for table in page["tables"]
    )
    empty_exact = sum(
        int(bool(table["empty_cell_signature_exact"])) for page in pages for table in page["tables"]
    )
    pair_tp = _sum_details(pages, "assignment_pair_true_positive_count")
    reference_pairs = _sum_details(pages, "reference_assignment_pair_count")
    predicted_pairs = _sum_details(pages, "predicted_assignment_pair_count") + sum(
        int(page["extra_predicted_assignment_pair_count"]) for page in pages
    )
    unambiguous_tp = _sum_details(pages, "unambiguous_assignment_pair_true_positive_count")
    unambiguous_reference_pairs = _sum_details(pages, "unambiguous_reference_assignment_pair_count")
    unambiguous_predicted_pairs = _sum_details(
        pages, "unambiguous_predicted_assignment_pair_count"
    ) + sum(int(page["extra_predicted_assignment_pair_count"]) for page in pages)
    strict_cells = _sum_details(pages, "strict_cell_assignment_sample_count")
    strict_cells_exact = _sum_details(pages, "strict_cell_assignment_exact_count")
    unambiguous_cells = _sum_details(pages, "unambiguous_cell_assignment_sample_count")
    unambiguous_cells_exact = _sum_details(pages, "unambiguous_cell_assignment_exact_count")
    normalized_text_cells = _sum_details(pages, "normalized_cell_text_sample_count")
    normalized_text_exact = _sum_details(pages, "normalized_cell_text_exact_count")
    unambiguous_text_cells = _sum_details(pages, "unambiguous_normalized_cell_text_sample_count")
    unambiguous_text_exact = _sum_details(pages, "unambiguous_normalized_cell_text_exact_count")
    absorbed = sum(int(page["document_non_table_absorbed_observation_count"]) for page in pages)
    predicted_unique_ids = sum(
        int(page["predicted_table_text_observation_count"]) for page in pages
    )
    return {
        "table_count_exact": all(bool(page["table_count_exact"]) for page in pages),
        "reference_table_count": reference_tables,
        "predicted_table_count": predicted_tables,
        "matched_table_count": matched_tables,
        "table_detection_precision": _rate(matched_tables, predicted_tables),
        "table_detection_recall": _rate(matched_tables, reference_tables),
        "rows_columns_exact_count": rows_columns_exact,
        "rows_columns_exact_rate": _rate(rows_columns_exact, reference_tables),
        "grid_span_exact_count": grid_span_exact,
        "grid_span_exact_rate": _rate(grid_span_exact, reference_tables),
        "empty_cell_signature_exact_count": empty_exact,
        "empty_cell_signature_exact_rate": _rate(empty_exact, reference_tables),
        "reference_physical_cell_count": _sum_details(pages, "reference_physical_cell_count"),
        "reference_grid_slot_count": _sum_details(pages, "reference_grid_slot_count"),
        "reference_spanning_cell_count": _sum_details(pages, "reference_spanning_cell_count"),
        "reference_empty_cell_count": _sum_details(pages, "reference_empty_cell_count"),
        "reference_ambiguous_text_observation_count": _sum_details(
            pages, "reference_ambiguous_text_observation_count"
        ),
        "reference_ambiguous_cell_count": _sum_details(pages, "reference_ambiguous_cell_count"),
        "strict_cell_assignment_exact_count": strict_cells_exact,
        "strict_cell_assignment_sample_count": strict_cells,
        "strict_cell_assignment_exact_rate": _rate(strict_cells_exact, strict_cells),
        "unambiguous_cell_assignment_exact_count": unambiguous_cells_exact,
        "unambiguous_cell_assignment_sample_count": unambiguous_cells,
        "unambiguous_cell_assignment_exact_rate": _rate(unambiguous_cells_exact, unambiguous_cells),
        "normalized_cell_text_exact_count": normalized_text_exact,
        "normalized_cell_text_sample_count": normalized_text_cells,
        "normalized_cell_text_exact_rate": _rate(normalized_text_exact, normalized_text_cells),
        "unambiguous_normalized_cell_text_exact_count": unambiguous_text_exact,
        "unambiguous_normalized_cell_text_sample_count": unambiguous_text_cells,
        "unambiguous_normalized_cell_text_exact_rate": _rate(
            unambiguous_text_exact, unambiguous_text_cells
        ),
        "reference_assignment_pair_count": reference_pairs,
        "predicted_assignment_pair_count": predicted_pairs,
        "assignment_pair_true_positive_count": pair_tp,
        "assignment_pair_precision": _rate(pair_tp, predicted_pairs),
        "assignment_pair_recall": _rate(pair_tp, reference_pairs),
        "assignment_pair_f1": _f1(pair_tp, predicted_pairs, reference_pairs),
        "unambiguous_reference_assignment_pair_count": unambiguous_reference_pairs,
        "unambiguous_predicted_assignment_pair_count": unambiguous_predicted_pairs,
        "unambiguous_assignment_pair_true_positive_count": unambiguous_tp,
        "unambiguous_assignment_pair_precision": _rate(unambiguous_tp, unambiguous_predicted_pairs),
        "unambiguous_assignment_pair_recall": _rate(unambiguous_tp, unambiguous_reference_pairs),
        "unambiguous_assignment_pair_f1": _f1(
            unambiguous_tp,
            unambiguous_predicted_pairs,
            unambiguous_reference_pairs,
        ),
        "document_non_table_absorbed_observation_count": absorbed,
        "document_non_table_absorption_rate": _rate(absorbed, predicted_unique_ids),
        "missed_reference_table_observation_count": sum(
            int(page["missed_reference_table_observation_count"]) for page in pages
        ),
    }


def _selected_pages(
    evidence: EvidenceIR,
    page_ids: list[str] | None,
) -> tuple[EvidencePage, ...]:
    ordered = tuple(sorted(evidence.pages, key=lambda page: (page.page_no, page.id)))
    if page_ids is None:
        return ordered
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("--page-id values must be unique")
    page_by_id = {page.id: page for page in ordered}
    unknown = sorted(set(page_ids) - set(page_by_id))
    if unknown:
        raise ValueError("unknown --page-id values: " + ", ".join(unknown))
    requested = set(page_ids)
    return tuple(page for page in ordered if page.id in requested)


def _page_image_path(pages_root: Path, page_id: str) -> Path:
    matches: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = (pages_root / f"{page_id}{suffix}").resolve()
        if candidate.parent != pages_root:
            raise ValueError("page id cannot escape the configured pages root")
        if candidate.is_file():
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"page image must resolve exactly once for {page_id}: {matches}")
    return matches[0]


def _media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }[path.suffix.casefold()]


def _validate_built_binding(
    built: BuiltModelATableTopology,
    evidence: EvidenceIR,
    page: EvidencePage,
) -> None:
    source = built.artifact.source
    if (
        source.evidence_ir_id != evidence.id
        or source.evidence_ir_contract_sha256 != contract_sha256(evidence)
        or source.source_document_sha256 != evidence.source_document_sha256
        or source.target_page_id != page.id
        or source.target_page_no != page.page_no
        or source.page_image_source_ref != page.image_source_ref
    ):
        raise ValueError("table topology artifact does not bind the selected EvidenceIR page")
    source_by_id = {item.id: item for item in evidence.sources}
    image_source = source_by_id[page.image_source_ref]
    if image_source.kind != EvidenceSourceKind.PAGE_IMAGE or (
        image_source.sha256 is not None and image_source.sha256 != source.page_image_sha256
    ):
        raise ValueError("table topology artifact page image provenance differs")
    observation_by_id = {observation.id: observation for observation in page.observations}
    for table in built.artifact.tables:
        for observation_id in table.ordered_observation_ids:
            observation = observation_by_id.get(observation_id)
            if observation is None or observation.kind != ObservationKind.TEXT_LINE:
                raise ValueError("detected table references a non-page or non-text observation")


def _evidence_promotion_block(
    evidence: EvidenceIR,
    pages: Sequence[EvidencePage],
    results: Sequence[_BuiltTopologyLike],
) -> dict[str, object]:
    existing = {observation.id: observation for page in pages for observation in page.observations}
    source_by_id = {source.id: source for source in evidence.sources}
    expected_ids: list[str] = []
    actual_ids: list[str] = []
    promotion_records: list[dict[str, object]] = []
    shared_nested_ids: set[str] = set()
    excess_nested_assignments = 0
    planned_cell_grid_ref_count = 0
    exact_materialized_ids: list[str] = []
    materialization_mismatch_ids: list[str] = []
    for page, result in zip(pages, results, strict=True):
        for index, table in enumerate(result.artifact.tables):
            expected = f"{page.id}:table-grid:{TABLE_TOPOLOGY_POLICY_SHA256[:12]}:{index:04d}"
            expected_ids.append(expected)
            actual_ids.append(table.table_grid_observation_id)
            planned_cell_grid_ref_count += len(table.cells)
            nested_counts = Counter(
                observation_id
                for cell in table.cells
                for observation_id in cell.ordered_observation_ids
            )
            duplicates = {
                observation_id: count
                for observation_id, count in nested_counts.items()
                if count > 1
            }
            shared_nested_ids.update(duplicates)
            excess_nested_assignments += sum(count - 1 for count in duplicates.values())
            promotion_records.append(
                {
                    "page_id": page.id,
                    "page_no": page.page_no,
                    "table_grid_observation_id": table.table_grid_observation_id,
                    "bbox_normalized": table.bbox_normalized,
                    "cell_count": len(table.cells),
                }
            )
            existing_observation = existing.get(table.table_grid_observation_id)
            if existing_observation is None:
                continue
            layout_sources = tuple(
                source_by_id[source_ref]
                for source_ref in existing_observation.source_refs
                if source_by_id[source_ref].kind == EvidenceSourceKind.LAYOUT_PROVIDER
            )
            if (
                existing_observation.kind == ObservationKind.TABLE_GRID
                and existing_observation.bbox.normalized == table.bbox_normalized
                and page.image_source_ref in existing_observation.source_refs
                and len(layout_sources) == 1
            ):
                exact_materialized_ids.append(table.table_grid_observation_id)
            else:
                materialization_mismatch_ids.append(table.table_grid_observation_id)
    collisions = sorted(
        observation_id
        for observation_id in actual_ids
        if observation_id in existing
        and existing[observation_id].kind != ObservationKind.TABLE_GRID
    )
    deterministic_ids_exact = expected_ids == actual_ids
    return {
        "original_table_grid_observation_count": sum(
            observation.kind == ObservationKind.TABLE_GRID for observation in existing.values()
        ),
        "predicted_table_grid_observation_count": len(actual_ids),
        "planned_cell_grid_ref_count": planned_cell_grid_ref_count,
        "cell_grid_ref_injection_required": bool(actual_ids),
        "deterministic_grid_ids_exact": deterministic_ids_exact,
        "grid_id_collision_count": len(collisions),
        "grid_id_collisions": collisions,
        "promotion_contract_ready": deterministic_ids_exact and not collisions,
        "promotion_materialized_in_input_evidence_count": len(exact_materialized_ids),
        "promotion_materialization_mismatch_ids": materialization_mismatch_ids,
        "promotion_materialized_in_input_evidence": (
            len(exact_materialized_ids) == len(actual_ids)
        ),
        "current_evidence_ir_model_a_table_compatible": (
            len(exact_materialized_ids) == len(actual_ids)
        ),
        "nested_shared_text_observation_count": len(shared_nested_ids),
        "nested_excess_text_assignment_count": excess_nested_assignments,
        "cell_text_ownership_review_free": excess_nested_assignments == 0,
        "promotion_records": promotion_records,
    }


def main() -> int:
    args = _parse_args()
    evidence_path = args.evidence_ir.resolve()
    projection_path = args.projection_source.resolve()
    content_path = args.proxy_content_ir.resolve()
    evidence_raw = evidence_path.read_bytes()
    projection_raw = projection_path.read_bytes()
    content_raw = content_path.read_bytes()
    require_strict_json_bytes(evidence_raw, max_depth=64, label="EvidenceIR")
    require_strict_json_bytes(projection_raw, max_depth=64, label="HWPX projection source")
    require_strict_json_bytes(content_raw, max_depth=64, label="proxy ContentIR")
    evidence = EvidenceIR.model_validate_json(evidence_raw, strict=True)
    projection = HwpxProjectionSource.model_validate_json(projection_raw, strict=True)
    content = ContentIR.model_validate_json(content_raw, strict=True)
    content.assert_evidence_integrity(evidence)
    proxy_tables = _build_proxy_tables(projection, content, evidence)

    selected_pages = _selected_pages(evidence, args.page_ids)
    if not selected_pages:
        raise ValueError("at least one page must be selected")
    selected_page_ids = {page.id for page in selected_pages}
    selected_proxies = tuple(table for table in proxy_tables if table.page_id in selected_page_ids)
    pages_root = (
        args.pages_root.resolve()
        if args.pages_root is not None
        else (evidence_path.parent / "pages").resolve()
    )

    built_results: list[BuiltModelATableTopology] = []
    for page in selected_pages:
        image_path = _page_image_path(pages_root, page.id)
        image_bytes = image_path.read_bytes()
        page_image = ModelAPageImage(
            page_id=page.id,
            page_no=page.page_no,
            image_source_ref=page.image_source_ref,
            media_type=_media_type(image_path),
            raw_bytes=image_bytes,
            sha256=_sha256(image_bytes),
        )
        built = build_model_a_table_topology(evidence, page_image=page_image)
        _validate_built_binding(built, evidence, page)
        built_results.append(built)

    page_reports = [
        _page_comparison(
            page.id,
            page.page_no,
            tuple(table for table in selected_proxies if table.page_id == page.id),
            built.artifact,
        )
        for page, built in zip(selected_pages, built_results, strict=True)
    ]
    aggregate = _aggregate_metrics(page_reports)
    promotion = _evidence_promotion_block(evidence, selected_pages, built_results)
    detector_keys = {
        (
            built.artifact.detector.detector_id,
            built.artifact.detector.detector_version,
            built.artifact.detector.detector_policy_sha256,
        )
        for built in built_results
    }
    if len(detector_keys) != 1:
        raise ValueError("selected pages do not use exactly one detector policy")
    detector = built_results[0].artifact.detector

    report = {
        "schema_version": "model-a-table-topology-proxy-comparison/1.0",
        "metric_status": "diagnostic_only_not_accuracy_not_release_evidence",
        "reference_basis": "exact_original_hwpx_projection_unverified_not_human_gold",
        "research_only": True,
        "human_review_required": True,
        "golden_eligible": False,
        "training_eligible": False,
        "release_eligible": False,
        "source_lineage": {
            "projection_source_document_sha256": projection.source_document_sha256,
            "projection_hwpx_sha256": projection.hwpx_sha256,
            "evidence_source_document_sha256": evidence.source_document_sha256,
            "evidence_ir_id": evidence.id,
            "evidence_ir_raw_sha256": _sha256(evidence_raw),
            "evidence_ir_contract_sha256": contract_sha256(evidence),
            "projection_source_raw_sha256": _sha256(projection_raw),
            "projection_source_contract_sha256": contract_sha256(projection),
            "proxy_content_ir_raw_sha256": _sha256(content_raw),
            "proxy_content_ir_contract_sha256": contract_sha256(content),
        },
        "selected_page_ids": [page.id for page in selected_pages],
        "detector_provenance": detector.model_dump(mode="json"),
        "page_topology_artifact_sha256s": {
            page.id: built.artifact_sha256
            for page, built in zip(selected_pages, built_results, strict=True)
        },
        "metrics": aggregate,
        "evidence_promotion": promotion,
        "pages": page_reports,
        "limitations": [
            "The HWPX projection reference is machine-derived and not human-verified Gold.",
            "Agreement is diagnostic only and must not be reported as accuracy.",
            "Reference table bboxes are text-observation unions, not source border coordinates.",
            "Tables are matched one-to-one only through shared EvidenceIR text observations.",
            (
                "The projection reference has shared line observations when one PDF line spans "
                "cells; unambiguous metrics exclude every cell touched by those observations."
            ),
            (
                "A deterministic grid id is only a promotion plan; it is not a TABLE_GRID "
                "observation until augmented EvidenceIR materializes it and rebuilds bindings."
            ),
        ],
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(report, pretty=True))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
