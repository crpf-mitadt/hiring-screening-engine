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
        if self.component_assessment:
            self._validate_component_assessment()
        if self.weighted_score > 0 and not self.evidence:
            raise ValueError("weighted_score > 0 requires at least one valid evidence item")
        if self.weighted_score > 0 and self.component_assessment:
            scopes = self.component_assessment.get("evidence_scopes", [])
            if scopes and not any(scope == "in_scope" for scope in scopes):
                raise ValueError("weighted_score > 0 requires in-scope criterion evidence")
        return self

    def _validate_component_assessment(self) -> None:
        assessment = self.component_assessment
        required_core = set(assessment.get("required_core_components", []))
        preferred = set(assessment.get("preferred_components", []))
        dimensions = set(assessment.get("evaluation_dimension_components", []))
        controlled_scope = set(assessment.get("controlled_equivalent_components", []))
        adjacent_scope = set(assessment.get("adjacent_components_considered", []))
        supported_core = set(assessment.get("supported_core_components", []))
        supported_preferred = set(assessment.get("supported_preferred_components", []))
        supported_dimensions = set(assessment.get("supported_evaluation_dimensions", []))
        supported_controlled = set(assessment.get("supported_controlled_equivalents", []))
        supported_adjacent = set(assessment.get("supported_adjacent_components", []))
        missing_core = set(assessment.get("missing_core_components", []))
        missing_preferred = set(assessment.get("missing_preferred_components", []))
        missing_dimensions = set(assessment.get("missing_evaluation_dimensions", []))
        if required_core & preferred:
            raise ValueError("preferred components cannot appear in required/core arrays")
        if required_core & dimensions or preferred & dimensions:
            raise ValueError("evaluation dimensions cannot appear in required/preferred component arrays")
        if not supported_core.issubset(required_core):
            raise ValueError("supported_core_components must be a subset of required_core_components")
        if not supported_preferred.issubset(preferred):
            raise ValueError("supported_preferred_components must be a subset of preferred_components")
        if not supported_dimensions.issubset(dimensions):
            raise ValueError("supported_evaluation_dimensions must be a subset of evaluation_dimension_components")
        if supported_controlled and not controlled_scope:
            raise ValueError("supported_controlled_equivalents require controlled_equivalent_components")
        if not supported_adjacent.issubset(adjacent_scope):
            raise ValueError("supported_adjacent_components must be a subset of adjacent_components_considered")
        if missing_core != required_core - supported_core:
            raise ValueError("missing_core_components must equal unmatched required_core_components")
        if missing_preferred != preferred - supported_preferred:
            raise ValueError("missing_preferred_components must equal unmatched preferred_components")
        if missing_dimensions != dimensions - supported_dimensions:
            raise ValueError("missing_evaluation_dimensions must equal unmatched evaluation_dimension_components")
        matched_evidence_keys: set[tuple] = set()
        for component in assessment.get("components", []):
            evidence_items = component.get("evidence", []) or []
            component_class = component.get("classification") or component.get("class")
            source = component.get("source")
            if component.get("matched") and not evidence_items:
                raise ValueError("every matched component must have evidence")
            if component.get("matched"):
                for evidence in evidence_items:
                    if not (
                        str(evidence.get("source_type") or "").strip()
                        and str(evidence.get("source_document_id") or "").strip()
                        and str(evidence.get("exact_text") or evidence.get("normalized_text") or "").strip()
                    ):
                        raise ValueError("matched component evidence must include source provenance")
            if source == "preferred_skill_names" and component_class != "preferred":
                raise ValueError("source=preferred_skill_names implies class=preferred")
            if str(source or "").startswith("evaluation_dimensions") and component_class != "evaluation_dimension":
                raise ValueError("source=evaluation_dimensions implies class=evaluation_dimension")
            if str(source or "").startswith("adjacent-skill rule") and component_class != "adjacent/transferable":
                raise ValueError("source=adjacent-skill rule implies class=adjacent/transferable")
            if not str(source or "").startswith("evaluation_dimensions") and source not in {
                "required_skill_names",
                "preferred_skill_names",
                "evaluation_dimensions",
                "adjacent-skill rule",
                "adjacent_skill_names",
                "criterion description",
                "criterion description fallback",
                "required_skills",
                "preferred_skills",
            } and "linked requirement" not in str(component.get("source", "")):
                if not str(source or "").startswith("adjacent-skill rule"):
                    raise ValueError("component source must identify a structured field or linked requirement")
            if component.get("matched"):
                for evidence in evidence_items:
                    matched_evidence_keys.add(_evidence_key(evidence))
        criterion_keys = {_evidence_key(item.model_dump(mode="json")) for item in self.evidence}
        if criterion_keys != matched_evidence_keys:
            raise ValueError("criterion evidence must be the deduplicated union of matched component evidence")


def _evidence_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("source_type"),
        item.get("source_document_id"),
        item.get("page_number"),
        item.get("block_index"),
        item.get("normalized_text") or item.get("exact_text"),
    )


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
