from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[
    str,
    Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@+-]*$"),
]
LogicalArtifactRef = Annotated[
    str,
    Field(pattern=r"^(?:artifact|repo)://[A-Za-z0-9][A-Za-z0-9._/@+-]*$"),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ComponentStatus = Literal["placeholder", "candidate", "deployed"]
BundleStatus = Literal["candidate", "deployed", "retired"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ModelComponent(_StrictModel):
    status: ComponentStatus
    model_id: Identifier
    revision: Identifier
    artifact_ref: LogicalArtifactRef | None
    artifact_sha256: Sha256 | None

    @field_validator("artifact_ref")
    @classmethod
    def validate_artifact_ref(cls, value: str | None) -> str | None:
        return _validate_logical_ref(value) if value is not None else None

    @model_validator(mode="after")
    def validate_artifact_presence(self) -> Self:
        present = self.artifact_ref is not None and self.artifact_sha256 is not None
        absent = self.artifact_ref is None and self.artifact_sha256 is None
        if self.status == "placeholder" and not absent:
            raise ValueError("placeholder component must not claim an artifact")
        if self.status != "placeholder" and not present:
            raise ValueError("candidate or deployed component requires artifact ref and sha256")
        return self


class ContractVersions(_StrictModel):
    evidence_ir: Literal["evidence-ir/1.0"]
    content_ir: Literal["content-ir/1.0"]
    hwp_document_plan: Literal["hwp-document-plan/1.0"]
    quality_report: Literal["quality-report/1.0"]


class ModelBundleManifest(_StrictModel):
    schema_version: Literal["1.0"]
    bundle_id: Identifier
    status: BundleStatus
    created_at: datetime
    model_a: ModelComponent
    model_b: ModelComponent
    dataset_manifest_ref: LogicalArtifactRef
    dataset_manifest_sha256: Sha256
    release_gate_set_id: Identifier
    capability_profile_ref: LogicalArtifactRef
    capability_profile_sha256: Sha256
    hancom_knowledge_manifest_ref: LogicalArtifactRef
    hancom_knowledge_manifest_sha256: Sha256
    hancom_knowledge_corpus_ref: LogicalArtifactRef
    hancom_knowledge_corpus_sha256: Sha256
    compiler_version: Identifier
    compiler_sha256: Sha256
    prompt_version: Identifier
    contract_versions: ContractVersions

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @field_validator(
        "dataset_manifest_ref",
        "capability_profile_ref",
        "hancom_knowledge_manifest_ref",
        "hancom_knowledge_corpus_ref",
    )
    @classmethod
    def validate_artifact_refs(cls, value: str) -> str:
        return _validate_logical_ref(value)

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.model_a.model_id.casefold() == self.model_b.model_id.casefold():
            raise ValueError("model_a and model_b must not have duplicate model_id values")

        refs = [
            self.dataset_manifest_ref,
            self.capability_profile_ref,
            self.hancom_knowledge_manifest_ref,
            self.hancom_knowledge_corpus_ref,
        ]
        refs.extend(
            component.artifact_ref
            for component in (self.model_a, self.model_b)
            if component.artifact_ref is not None
        )
        duplicate_refs = _duplicates(refs)
        if duplicate_refs:
            raise ValueError("duplicate artifact refs: " + ", ".join(duplicate_refs))
        if self.status == "deployed" and not self.components_complete:
            raise ValueError("deployed bundle cannot contain placeholder components")
        return self

    @property
    def components_complete(self) -> bool:
        return all(component.status != "placeholder" for component in (self.model_a, self.model_b))


def load_model_bundle_manifest(path: Path | str) -> ModelBundleManifest:
    """Load a strict, self-contained model release provenance manifest."""
    return ModelBundleManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _validate_logical_ref(value: str) -> str:
    path = value.split("://", maxsplit=1)[1]
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("artifact ref must not contain empty or traversal path segments")
    return value


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if normalized in seen:
            duplicates.add(value)
        seen.add(normalized)
    return sorted(duplicates)
