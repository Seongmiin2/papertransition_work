from __future__ import annotations

import json
import math
from typing import Any

DEFAULT_MAX_JSON_NODES = 500_000
_MIN_SIGNED_64 = -(2**63)
_MAX_SIGNED_64 = 2**63 - 1


class StrictJSONError(ValueError):
    """Raised when bytes are not a bounded-shape, interoperable JSON value."""


def require_strict_json_bytes(
    payload: bytes,
    *,
    max_depth: int,
    label: str,
    max_nodes: int = DEFAULT_MAX_JSON_NODES,
) -> None:
    """Reject ambiguous JSON and bound numeric, nesting, and node complexity.

    Pydantic's JSON decoder remains responsible for the typed schema. This pass is
    deliberately separate because duplicate object members otherwise lose evidence
    before typed validation sees them.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJSONError(f"{label} must be UTF-8 JSON") from exc
    _require_nesting_within_limit(text, max_depth=max_depth, label=label)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise StrictJSONError(f"{label} contains a duplicate object key")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise StrictJSONError(f"{label} contains a non-finite number: {value}")

    def parse_bounded_int(value: str) -> int:
        parsed = int(value)
        if not _MIN_SIGNED_64 <= parsed <= _MAX_SIGNED_64:
            raise StrictJSONError(f"{label} contains an integer outside signed 64-bit range")
        return parsed

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise StrictJSONError(f"{label} contains a non-finite number")
        return parsed

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
            parse_int=parse_bounded_int,
            parse_float=parse_finite_float,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise StrictJSONError(f"{label} is not valid strict JSON") from exc
    _require_node_count_within_limit(decoded, max_nodes=max_nodes, label=label)


def _require_nesting_within_limit(text: str, *, max_depth: int, label: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise StrictJSONError(f"{label} exceeds the JSON nesting limit")
        elif character in "]}":
            depth -= 1


def _require_node_count_within_limit(
    value: Any,
    *,
    max_nodes: int,
    label: str,
) -> None:
    node_count = 0
    pending = [value]
    while pending:
        current = pending.pop()
        node_count += 1
        if node_count > max_nodes:
            raise StrictJSONError(f"{label} exceeds the JSON node limit")
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
