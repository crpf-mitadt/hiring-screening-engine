"""Build scoring-safe candidate payloads."""

from copy import deepcopy
from typing import Any

from chandra_eval.safety.prohibited_attributes import (
    PROHIBITED_PERSONAL_INFORMATION_KEYS,
)
from chandra_eval.schemas.resume import CandidateProfile, ScoringSafeProfile


def sanitize_mapping(value: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key.lower() in PROHIBITED_PERSONAL_INFORMATION_KEYS:
            continue
        if isinstance(item, dict):
            cleaned[key] = sanitize_mapping(item)
        else:
            cleaned[key] = deepcopy(item)
    return cleaned


def build_scoring_safe_profile(candidate: CandidateProfile) -> ScoringSafeProfile:
    """Return a profile containing only job-related fields."""

    return ScoringSafeProfile(
        candidate_id=candidate.candidate_id,
        summary=candidate.summary,
        education=deepcopy(candidate.education),
        experience=deepcopy(candidate.experience),
        skills=list(candidate.skills),
        certifications=list(candidate.certifications),
        projects=list(candidate.projects),
        publications=list(candidate.publications),
        teaching_experience=list(candidate.teaching_experience),
        research_experience=list(candidate.research_experience),
        leadership_experience=list(candidate.leadership_experience),
        achievements=list(candidate.achievements),
        languages=list(candidate.languages),
        missing_information=list(candidate.missing_information),
        extraction_warnings=list(candidate.extraction_warnings),
        source_pages=list(candidate.source_pages),
    )
