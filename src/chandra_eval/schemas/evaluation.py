"""Evaluation result schemas."""

from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from chandra_eval.schemas.common import EvidenceRef, StrictBaseModel


class CriterionStatus(str, Enum):
    MET = "met"
    PARTIALLY_MET = "partially_met"
    PARTIAL = "partial"
    NOT_MET = "not_met"
    NOT_FOUND = "not_found"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"
    AMBIGUOUS = "ambiguous"


class EligibilityStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class CriterionEvaluation(StrictBaseModel):
    criterion_id: str
    criterion_name: str = ""
    category: str = ""
    weight: float = Field(default=0, ge=0)
    requirement_ids: list[str] = Field(default_factory=list)
    match_fraction: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=0)
    status: CriterionStatus
    explanation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    ambiguity: list[str] = Field(default_factory=list)
    verification_needed: list[str] = Field(default_factory=list)
    candidate_requirement: str = ""
    evidence_found: list[str] = Field(default_factory=list)
    component_assessment: dict[str, Any] = Field(default_factory=dict)
    model_assessment: dict[str, Any] | None = Field(default=None, exclude=True)

    @field_validator("match_fraction")
    @classmethod
    def _approved_fraction(cls, value: float) -> float:
        approved = {0.0, 0.25, 0.5, 0.75, 1.0}
        rounded = round(float(value), 3)
        if rounded not in approved:
            raise ValueError("match_fraction must be one of 0.00, 0.25, 0.50, 0.75, 1.00")
        return rounded

    @model_validator(mode="after")
    def _positive_score_requires_evidence(self) -> "CriterionEvaluation":
        if self.model_assessment and not self.component_assessment:
            self.component_assessment = self.model_assessment
        if self.component_assessment and self.model_assessment is None:
            self.model_assessment = self.component_assessment
        if self.category in {"skill", "leadership"} and not self.component_assessment:
            raise ValueError("skill and leadership criteria require populated component_assessment")
        if self.weighted_score > 0 and not self.evidence:
            raise ValueError("weighted_score > 0 requires at least one valid evidence item")
        if self.weighted_score > 0 and self.component_assessment:
            scopes = self.component_assessment.get("evidence_scopes", [])
            if scopes and not any(scope == "in_scope" for scope in scopes):
                raise ValueError("weighted_score > 0 requires in-scope criterion evidence")
        return self


class EligibilityResult(StrictBaseModel):
    status: EligibilityStatus
    checked_requirements: list[str] = Field(default_factory=list)
    failed_requirements: list[str] = Field(default_factory=list)
    ambiguous_requirements: list[str] = Field(default_factory=list)
    requirement_statuses: dict[str, str] = Field(default_factory=dict)
    requirement_details: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str


class CandidateEvaluationReport(StrictBaseModel):
    evaluation_id: str
    candidate_id: str
    job_id: str
    scoring_version: str = "scoring-v2-evidence-discrete"
    warnings: list[str] = Field(default_factory=list)
    mock_evaluation: bool = False
    eligibility: EligibilityResult
    criterion_evaluations: list[CriterionEvaluation]
    score_before_gates: float
    final_score: float
    maximum_score: float = 100.0
    evidence_completeness: float = Field(ge=0, le=1)
    evidence_completeness_diagnostics: dict[str, Any] = Field(default_factory=dict)
    experience_summary: dict[str, Any] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommendation_label: str
    human_review_required: bool = True
    human_review_disclaimer: str = (
        "Decision support only. A qualified human reviewer must review the "
        "evidence before any employment decision."
    )
