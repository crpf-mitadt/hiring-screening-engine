"""Job-description schemas."""

from enum import Enum

from pydantic import Field, model_validator

from chandra_eval.schemas.common import EvidenceRef, ExtractionMethod, StrictBaseModel


class RequirementCategory(str, Enum):
    EDUCATION = "education"
    EXPERIENCE = "experience"
    SKILL = "skill"
    CERTIFICATION = "certification"
    DOMAIN = "domain"
    RESPONSIBILITY = "responsibility"
    RESEARCH = "research"
    TEACHING = "teaching"
    LEADERSHIP = "leadership"
    PROJECT = "project"
    LANGUAGE = "language"
    OTHER = "other"


class RequirementType(str, Enum):
    MINIMUM = "minimum"
    PREFERRED = "preferred"
    EQUIVALENT_ACCEPTED = "equivalent_accepted"
    INFORMATIONAL = "informational"


class JDRequirement(StrictBaseModel):
    requirement_id: str
    category: RequirementCategory
    description: str
    requirement_type: RequirementType = RequirementType.INFORMATIONAL
    mandatory: bool = False
    preferred: bool = False
    minimum_value: float | None = None
    unit: str | None = None
    accepted_equivalents: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScoringCriterion(StrictBaseModel):
    criterion_id: str
    name: str
    category: RequirementCategory
    description: str
    weight: float = Field(ge=0)
    mandatory: bool = False
    accepted_evidence: list[str] = Field(default_factory=list)
    scoring_guidance: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    required_skill_names: list[str] = Field(default_factory=list)
    preferred_skill_names: list[str] = Field(default_factory=list)
    evaluation_dimensions: list[str] = Field(default_factory=list)


class StructuredJD(StrictBaseModel):
    job_id: str
    source_document_id: str
    extraction_method: ExtractionMethod = ExtractionMethod.DETERMINISTIC_FALLBACK
    job_title: str
    department: str = ""
    role_summary: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    mandatory_requirements: list[JDRequirement] = Field(default_factory=list)
    preferred_requirements: list[JDRequirement] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience_months: int | None = None
    domain: str = ""
    seniority: str = ""
    scoring_rubric: list[ScoringCriterion] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_rubric_weight(self) -> "StructuredJD":
        if self.scoring_rubric:
            ids = [criterion.criterion_id for criterion in self.scoring_rubric]
            duplicate_ids = sorted({criterion_id for criterion_id in ids if ids.count(criterion_id) > 1})
            if duplicate_ids:
                raise ValueError(f"scoring_rubric criterion_ids must be unique: {', '.join(duplicate_ids)}")
            total = sum(criterion.weight for criterion in self.scoring_rubric)
            if round(total, 3) != 100:
                raise ValueError(f"scoring_rubric weights must total 100, got {total}")
        return self
