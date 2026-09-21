from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai.modeling import compare_model_a_role_proxy as comparison
from scan2hwpx.contracts import ContentIR, EvidenceIR, ObservationKind, contract_sha256
from scan2hwpx.model_a import (
    BuiltModelARoleVectorRequest,
    execute_model_a_role_vector_chunked_page,
)
from tests.unit.test_model_a_role_vector import _evidence, _image, _json_bytes, _model


class _Provider:
    def __init__(self, roles: tuple[str, ...]) -> None:
        self._roles = roles

    def generate(self, request: BuiltModelARoleVectorRequest, /) -> bytes:
        assert len(request.artifact.ordered_text_observation_ids) == len(self._roles)
        return _json_bytes({"roles": self._roles})


def _write_comparison_inputs(
    root: Path,
    evidence: EvidenceIR,
    *,
    raw_roles: tuple[str, ...],
) -> tuple[Path, Path, Path]:
    result = execute_model_a_role_vector_chunked_page(
        evidence,
        page_image=_image(),
        model=_model(),
        provider=_Provider(raw_roles),
    )
    analysis = result.artifact.compiled_page_analysis
    assert analysis is not None
    assert result.compiled_page_analysis_bytes is not None
    proxy = ContentIR(
        id="proxy-content",
        evidence_ir_id=evidence.id,
        evidence_ir_sha256=contract_sha256(evidence),
        nodes=analysis.nodes,
        reading_order=analysis.reading_order,
    )

    evidence_path = root / "evidence.json"
    proxy_path = root / "proxy.json"
    batch_dir = root / "batch"
    page_dir = batch_dir / analysis.page_id
    page_dir.mkdir(parents=True)
    evidence_path.write_bytes(_json_bytes(evidence.model_dump(mode="json")))
    proxy_path.write_bytes(_json_bytes(proxy.model_dump(mode="json")))
    (page_dir / "result.json").write_bytes(result.result_bytes)
    (page_dir / "compiled-page-analysis.json").write_bytes(result.compiled_page_analysis_bytes)
    return evidence_path, proxy_path, batch_dir


def _run_comparison(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    evidence: EvidenceIR,
    *,
    raw_roles: tuple[str, ...],
) -> dict[str, object]:
    evidence_path, proxy_path, batch_dir = _write_comparison_inputs(
        root,
        evidence,
        raw_roles=raw_roles,
    )
    output = root / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_model_a_role_proxy.py",
            "--evidence-ir",
            str(evidence_path),
            "--proxy-content-ir",
            str(proxy_path),
            "--batch-dir",
            str(batch_dir),
            "--output",
            str(output),
        ],
    )

    assert comparison.main() == 0
    return json.loads(output.read_bytes())


def test_report_separates_raw_model_from_resolved_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_comparison(
        monkeypatch,
        tmp_path,
        _evidence(),
        raw_roles=("passage", "passage", "passage"),
    )

    assert report["schema_version"] == "model-a-unverified-role-proxy-comparison/1.2"
    comparisons = report["comparisons"]
    assert comparisons["raw_model"]["exact_agreement_rate"] == pytest.approx(2 / 3)
    assert comparisons["resolved_pipeline"]["exact_agreement_rate"] == 1
    assert report["recorded_anchor_provenance"]["resolution_count"] == 1
    assert report["recorded_anchor_provenance"]["override_count"] == 1
    assert report["compiler_provenance"]["compiler_sha256"]
    assert "unverified_proxy_exact_agreement_rate" not in report


def test_report_handles_a_page_without_text_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _evidence()
    page = base.pages[0].model_copy(
        update={
            "observations": tuple(
                observation
                for observation in base.pages[0].observations
                if observation.kind == ObservationKind.IMAGE
            )
        }
    )
    evidence = base.model_copy(update={"pages": (page,)})

    report = _run_comparison(
        monkeypatch,
        tmp_path,
        evidence,
        raw_roles=(),
    )

    assert report["text_observation_count"] == 0
    for basis in ("raw_model", "resolved_pipeline"):
        block = report["comparisons"][basis]
        assert block["aligned_count"] == 0
        assert block["exact_agreement_rate"] is None
        assert block["role_macro_f1"] is None
