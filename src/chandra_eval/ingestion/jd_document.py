"""Job-description loading and simple synthetic extraction."""

import json
import re
from pathlib import Path

from chandra_eval.schemas.common import EvidenceRef, EvidenceStatus
from chandra_eval.schemas.jd import (
    JDRequirement,
    RequirementCategory,
    RequirementType,
    ScoringCriterion,
    StructuredJD,
)


def load_structured_jd(path: Path) -> StructuredJD:
    return StructuredJD.model_validate_json(path.read_text(encoding="utf-8"))


def write_structured_jd(jd: StructuredJD, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jd.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def extract_synthetic_jd(text: str, job_id: str, source_document_id: str) -> StructuredJD:
    """Heuristic JD extraction for tests and demos; live pilots should use Gemma."""

    lowered = text.lower()
    skills = _extract_list_after(text, "required skills")
    preferred_skills = _extract_list_after(text, "preferred skills")
    months = _extract_experience_months(lowered)
    requirements: list[JDRequirement] = []
    for index, skill in enumerate(skills, start=1):
        requirements.append(
            JDRequirement(
                requirement_id=f"req-skill-{index}",
                category=RequirementCategory.SKILL,
                description=skill,
                requirement_type=RequirementType.MINIMUM,
                mandatory=True,
                evidence=[
                    EvidenceRef(
                        source_type="jd",
                        source_document_id=source_document_id,
                        exact_text=skill,
                        evidence_status=EvidenceStatus.SUPPORTED,
                    )
                ],
            )
        )
    if months is not None:
        requirements.append(
            JDRequirement(
                requirement_id="req-experience-months",
                category=RequirementCategory.EXPERIENCE,
                description=f"At least {months} months of relevant experience",
                requirement_type=RequirementType.MINIMUM,
                mandatory=True,
                minimum_value=months,
                unit="months",
            )
        )
    return StructuredJD(
        job_id=job_id,
        source_document_id=source_document_id,
        job_title=_extract_title(text),
        role_summary=text.splitlines()[0].strip() if text.splitlines() else "",
        mandatory_requirements=requirements,
        required_skills=skills,
        preferred_skills=preferred_skills,
        minimum_experience_months=months,
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="required-skills",
                name="Required skills",
                category=RequirementCategory.SKILL,
                description="Match required JD skills",
                weight=45,
                mandatory=True,
            ),
            ScoringCriterion(
                criterion_id="minimum-experience",
                name="Minimum experience",
                category=RequirementCategory.EXPERIENCE,
                description="Meet minimum experience duration",
                weight=35,
                mandatory=True,
            ),
            ScoringCriterion(
                criterion_id="preferred-skills",
                name="Preferred skills",
                category=RequirementCategory.SKILL,
                description="Match preferred JD skills",
                weight=20,
                mandatory=False,
            ),
        ],
        extraction_warnings=["Synthetic heuristic extraction; verify before live use."],
    )


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip().removeprefix("Title:").strip()
    return "Untitled role"


def _extract_list_after(text: str, label: str) -> list[str]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*(.+)", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return [item.strip() for item in re.split(r",|;", match.group(1)) if item.strip()]
    return []


def _extract_experience_months(text: str) -> int | None:
    match = re.search(r"(\d+)\+?\s*(years|year|yrs|yr)", text)
    if match:
        return int(match.group(1)) * 12
    match = re.search(r"(\d+)\+?\s*(months|month|mos|mo)", text)
    if match:
        return int(match.group(1))
    return None
