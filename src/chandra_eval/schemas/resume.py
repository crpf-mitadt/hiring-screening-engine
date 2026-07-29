"""Candidate profile schemas."""

from enum import Enum
import re
from typing import Any

from pydantic import Field, field_validator, model_validator

from chandra_eval.schemas.common import (
    DatePrecision,
    EvidenceRef,
    ExtractionMethod,
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


def _none_to_list(value):
    return [] if value is None else value


_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _normalize_resume_date(value: Any) -> tuple[str | None, DatePrecision, bool]:
    if value is None:
        return None, DatePrecision.UNKNOWN, False
    text = str(value).strip()
    if not text:
        return None, DatePrecision.UNKNOWN, False
    lowered = text.lower().strip(".")
    if lowered in {"present", "current", "now"}:
        return None, DatePrecision.UNKNOWN, True
    match = re.fullmatch(r"(\d{4})(?:[-/](\d{1,2}))?", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2) or "1")
        if not 1 <= month <= 12:
            return text, DatePrecision.UNKNOWN, False
        if match.group(2):
            return f"{year:04d}-{month:02d}", DatePrecision.MONTH, False
        return f"{year:04d}", DatePrecision.YEAR, False
    match = re.fullmatch(r"(\d{1,2})[-/](\d{4})", text)
    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}", DatePrecision.MONTH, False
    match = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", text)
    if match:
        month = _MONTH_NAMES.get(match.group(1).lower())
        if month is not None:
            return f"{int(match.group(2)):04d}-{month:02d}", DatePrecision.MONTH, False
    return text, DatePrecision.UNKNOWN, False


class EducationEntry(StrictBaseModel):
    degree_level: DegreeLevel = DegreeLevel.UNKNOWN
    degree_name: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    graduation_year: int | None = None
    completion_status: CompletionStatus = CompletionStatus.UNCLEAR
    grade: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("evidence", "warnings", mode="before")
    @classmethod
    def _collections_default_empty(cls, value):
        return _none_to_list(value)


class ExperienceEntry(StrictBaseModel):
    title: str | None = None
    organization: str | None = None
    industry_or_domain: str | None = None
    employment_type: str | None = None
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

    @field_validator(
        "responsibilities",
        "achievements",
        "skills_used",
        "evidence",
        "warnings",
        mode="before",
    )
    @classmethod
    def _collections_default_empty(cls, value):
        return _none_to_list(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize_dates(cls, values):
        if not isinstance(values, dict):
            return values
        output = dict(values)
        start, start_precision, _ = _normalize_resume_date(output.get("start_date"))
        end, end_precision, active = _normalize_resume_date(output.get("end_date"))
        output["start_date"] = start
        output["end_date"] = end
        if output.get("start_date_precision") in {None, "", DatePrecision.UNKNOWN, "unknown"}:
            output["start_date_precision"] = start_precision
        if active:
            output["currently_active"] = True
            output["end_date_precision"] = DatePrecision.UNKNOWN
        elif output.get("end_date_precision") in {None, "", DatePrecision.UNKNOWN, "unknown"}:
            output["end_date_precision"] = end_precision
        return output


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

    @field_validator(
        "education",
        "experience",
        "skills",
        "certifications",
        "projects",
        "publications",
        "teaching_experience",
        "research_experience",
        "leadership_experience",
        "achievements",
        "languages",
        "missing_information",
        "extraction_warnings",
        "source_pages",
        mode="before",
    )
    @classmethod
    def _collections_default_empty(cls, value):
        return _none_to_list(value)


class CandidateProfile(StrictBaseModel):
    candidate_id: str
    source_document_id: str
    extraction_method: ExtractionMethod = ExtractionMethod.DETERMINISTIC_FALLBACK
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

    @field_validator(
        "education",
        "experience",
        "skills",
        "certifications",
        "projects",
        "publications",
        "teaching_experience",
        "research_experience",
        "leadership_experience",
        "achievements",
        "languages",
        "missing_information",
        "extraction_warnings",
        "source_pages",
        mode="before",
    )
    @classmethod
    def _collections_default_empty(cls, value):
        return _none_to_list(value)

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
