"""Candidate profile schemas."""

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from chandra_eval.schemas.common import (
    DatePrecision,
    EvidenceRef,
    StrictBaseModel,
)


class DegreeLevel(str, Enum):
    SECONDARY = "secondary"
    DIPLOMA = "diploma"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    POSTDOCTORATE = "postdoctorate"
    PROFESSIONAL = "professional"
    CERTIFICATION = "certification"
    UNKNOWN = "unknown"


class CompletionStatus(str, Enum):
    COMPLETED = "completed"
    PURSUING = "pursuing"
    INCOMPLETE = "incomplete"
    UNCLEAR = "unclear"
    NOT_FOUND = "not_found"


class EducationEntry(StrictBaseModel):
    degree_level: DegreeLevel = DegreeLevel.UNKNOWN
    degree_name: str = ""
    field_of_study: str = ""
    institution: str = ""
    start_date: str | None = None
    end_date: str | None = None
    graduation_year: int | None = None
    completion_status: CompletionStatus = CompletionStatus.UNCLEAR
    grade: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExperienceEntry(StrictBaseModel):
    title: str = ""
    organization: str = ""
    industry_or_domain: str = ""
    employment_type: str = ""
    start_date: str | None = None
    end_date: str | None = None
    start_date_precision: DatePrecision = DatePrecision.UNKNOWN
    end_date_precision: DatePrecision = DatePrecision.UNKNOWN
    currently_active: bool = False
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScoringSafeProfile(StrictBaseModel):
    candidate_id: str
    summary: str = ""
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    teaching_experience: list[str] = Field(default_factory=list)
    research_experience: list[str] = Field(default_factory=list)
    leadership_experience: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)


class CandidateProfile(StrictBaseModel):
    candidate_id: str
    source_document_id: str
    summary: str = ""
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    teaching_experience: list[str] = Field(default_factory=list)
    research_experience: list[str] = Field(default_factory=list)
    leadership_experience: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    personal_information: dict[str, Any] | None = None
    scoring_safe_profile: ScoringSafeProfile | None = None

    @model_validator(mode="after")
    def ensure_safe_profile(self) -> "CandidateProfile":
        if self.scoring_safe_profile is None:
            self.scoring_safe_profile = ScoringSafeProfile(
                candidate_id=self.candidate_id,
                summary=self.summary,
                education=self.education,
                experience=self.experience,
                skills=self.skills,
                certifications=self.certifications,
                projects=self.projects,
                publications=self.publications,
                teaching_experience=self.teaching_experience,
                research_experience=self.research_experience,
                leadership_experience=self.leadership_experience,
                achievements=self.achievements,
                languages=self.languages,
                missing_information=self.missing_information,
                extraction_warnings=self.extraction_warnings,
                source_pages=self.source_pages,
            )
        return self
