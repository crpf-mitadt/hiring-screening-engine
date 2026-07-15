"""Evaluation result schemas."""

from enum import Enum
from typing import Any

from pydantic import Field

from chandra_eval.schemas.common import EvidenceRef, StrictBaseModel


class CriterionStatus(str, Enum):
    MET = "met"
    PARTIAL = "partial"
    NOT_MET = "not_met"
    AMBIGUOUS = "ambiguous"


class EligibilityStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class CriterionEvaluation(StrictBaseModel):
    criterion_id: str
    match_fraction: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=0)
    status: CriterionStatus
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    ambiguity: list[str] = Field(default_factory=list)
    model_assessment: dict[str, Any] | None = None


class EligibilityResult(StrictBaseModel):
    status: EligibilityStatus
    checked_requirements: list[str] = Field(default_factory=list)
    failed_requirements: list[str] = Field(default_factory=list)
    ambiguous_requirements: list[str] = Field(default_factory=list)
    explanation: str


class CandidateEvaluationReport(StrictBaseModel):
    evaluation_id: str
    candidate_id: str
    job_id: str
    eligibility: EligibilityResult
    criterion_evaluations: list[CriterionEvaluation]
    score_before_gates: float
    final_score: float
    maximum_score: float = 100.0
    evidence_completeness: float = Field(ge=0, le=1)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommendation_label: str
    human_review_required: bool = True
    human_review_disclaimer: str = (
        "Decision support only. A qualified human reviewer must review the "
        "evidence before any employment decision."
    )
