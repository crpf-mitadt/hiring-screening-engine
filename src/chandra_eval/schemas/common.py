"""Common JSON-serializable schema types."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model that keeps schema drift visible during tests."""

    model_config = ConfigDict(extra="forbid")


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    CONTRADICTED = "contradicted"


class DatePrecision(str, Enum):
    EXACT = "exact"
    MONTH = "month"
    YEAR = "year"
    UNKNOWN = "unknown"


class EvidenceRef(StrictBaseModel):
    source_type: str
    source_document_id: str
    page_number: int | None = None
    block_index: int | None = None
    bbox: list[float] | None = None
    exact_text: str
    normalized_text: str | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.SUPPORTED
    notes: str | None = None


class TextEvidence(StrictBaseModel):
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
