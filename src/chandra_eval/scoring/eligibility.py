"""Mandatory requirement gates."""

from datetime import date

from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.experience_basis import ExperienceQualification
from chandra_eval.scoring.education import evaluate_education_requirement
from chandra_eval.scoring.experience_basis import qualify_quantitative_experience
from chandra_eval.scoring.matching import build_candidate_evidence_index, match_qualitative_experience, match_requirement
from chandra_eval.schemas.evaluation import EligibilityResult, EligibilityStatus
from chandra_eval.schemas.jd import RequirementCategory, StructuredJD
from chandra_eval.schemas.resume import ScoringSafeProfile


def evaluate_eligibility(
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    *,
    experience_qualifications: dict[str, ExperienceQualification] | None = None,
    evaluation_date: date | None = None,
) -> EligibilityResult:
    checked: list[str] = []
    failed: list[str] = []
    ambiguous: list[str] = []
    requirement_statuses: dict[str, str] = {}
    requirement_details: list[dict] = []
    index = build_candidate_evidence_index(candidate)
    duration = calculate_experience_duration(candidate.experience, evaluation_date=evaluation_date)
    experience_qualifications = experience_qualifications or {}

    for requirement in jd.mandatory_requirements:
        checked.append(requirement.requirement_id)
        detail = {
            "requirement_id": requirement.requirement_id,
            "description": requirement.description,
            "category": requirement.category.value,
            "status": "unclear",
            "evidence": [],
            "missing_evidence": "",
            "ambiguity": [],
            "explanation": "",
        }
        if requirement.category == RequirementCategory.SKILL:
            match = match_requirement(requirement.description, index, requirement.accepted_equivalents)
            if match.fraction > 0:
                requirement_statuses[requirement.requirement_id] = "met"
                detail["status"] = "met"
                detail["evidence"] = [item.model_dump(mode="json") for item in match.evidence]
                if not match.evidence:
                    detail["missing_evidence"] = "Matched scoring-safe profile text without source block evidence; verify manually."
            else:
                requirement_statuses[requirement.requirement_id] = "not_found"
                detail["status"] = "not_found"
                detail["missing_evidence"] = "No direct resume evidence found for this mandatory skill."
                ambiguous.append(requirement.requirement_id)
        elif requirement.category == RequirementCategory.EXPERIENCE:
            if requirement.minimum_value is None:
                match = match_qualitative_experience(requirement.description, index)
                if match.evidence:
                    requirement_statuses[requirement.requirement_id] = "met"
                    detail["status"] = "met"
                    detail["evidence"] = [item.model_dump(mode="json") for item in match.evidence]
                    detail["ambiguity"].append("Qualitative experience requirement assessed without a numeric duration.")
                else:
                    requirement_statuses[requirement.requirement_id] = "not_found"
                    detail["status"] = "not_found"
                    detail["missing_evidence"] = "No direct qualitative experience evidence found for this mandatory requirement."
                    detail["ambiguity"].extend(match.ambiguity)
                    ambiguous.append(requirement.requirement_id)
            else:
                qualification = experience_qualifications.get(requirement.requirement_id) or qualify_quantitative_experience(
                    candidate,
                    required_months=int(requirement.minimum_value),
                    requirement_text=requirement.description,
                    evaluation_date=evaluation_date,
                )
                detail["confirmed_months"] = qualification.confirmed_months
                detail["required_months"] = qualification.required_months
                detail["fraction"] = qualification.fraction
                detail["ambiguity"].extend(qualification.ambiguity)
                if qualification.confirmed_months >= int(requirement.minimum_value) and qualification.evidence and qualification.fraction >= 1:
                    requirement_statuses[requirement.requirement_id] = "met"
                    detail["status"] = "met"
                    detail["evidence"] = [item.model_dump(mode="json") for item in qualification.evidence]
                    detail["explanation"] = qualification.explanation
                    detail["ambiguity"] = []
                elif qualification.ambiguity and qualification.confirmed_months + duration.possible_additional_months >= int(requirement.minimum_value):
                    requirement_statuses[requirement.requirement_id] = "unclear"
                    detail["status"] = "unclear"
                    detail["missing_evidence"] = qualification.explanation
                    ambiguous.append(requirement.requirement_id)
                elif qualification.confirmed_months == 0 and duration.uncertain_ranges:
                    requirement_statuses[requirement.requirement_id] = "unclear"
                    detail["status"] = "unclear"
                    detail["missing_evidence"] = qualification.explanation
                    detail["ambiguity"].extend(duration.date_warnings)
                    ambiguous.append(requirement.requirement_id)
                elif qualification.confirmed_months == 0:
                    requirement_statuses[requirement.requirement_id] = "not_found"
                    detail["status"] = "not_found"
                    detail["missing_evidence"] = qualification.explanation
                    ambiguous.append(requirement.requirement_id)
                else:
                    requirement_statuses[requirement.requirement_id] = "not_met"
                    detail["status"] = "not_met"
                    detail["missing_evidence"] = qualification.explanation
                    failed.append(requirement.requirement_id)
        elif requirement.category == RequirementCategory.EDUCATION:
            education_match = evaluate_education_requirement(requirement.description, candidate.education)
            if education_match.matched:
                requirement_statuses[requirement.requirement_id] = "met"
                detail["status"] = "met"
                detail["evidence"] = [item.model_dump(mode="json") for item in education_match.evidence]
                detail["ambiguity"].extend(education_match.notes)
            else:
                requirement_statuses[requirement.requirement_id] = "not_found"
                detail["status"] = "not_found"
                detail["missing_evidence"] = "; ".join(education_match.missing)
                detail["ambiguity"].extend(education_match.ambiguity or education_match.notes)
                ambiguous.append(requirement.requirement_id)
        elif requirement.category in {RequirementCategory.LANGUAGE, RequirementCategory.DOMAIN}:
            allowed_sources = {"education"} if requirement.category == RequirementCategory.EDUCATION else None
            match = match_requirement(
                requirement.description,
                index,
                requirement.accepted_equivalents,
                allowed_sources=allowed_sources,
            )
            if match.fraction > 0:
                requirement_statuses[requirement.requirement_id] = (
                    "partially_met" if requirement.category == RequirementCategory.EDUCATION else "met"
                )
                detail["status"] = requirement_statuses[requirement.requirement_id]
                detail["evidence"] = [item.model_dump(mode="json") for item in match.evidence]
                if not match.evidence:
                    detail["missing_evidence"] = "Matched scoring-safe profile text without source block evidence; verify manually."
                ambiguous.append(requirement.requirement_id) if requirement.category == RequirementCategory.EDUCATION else None
            else:
                requirement_statuses[requirement.requirement_id] = "not_found"
                detail["status"] = "not_found"
                detail["missing_evidence"] = "No direct resume evidence found for this mandatory requirement."
                ambiguous.append(requirement.requirement_id)
        else:
            requirement_statuses[requirement.requirement_id] = "unclear"
            detail["status"] = "unclear"
            detail["ambiguity"].append("No deterministic mandatory checker exists for this category.")
            ambiguous.append(requirement.requirement_id)
        requirement_details.append(detail)

    if failed:
        status = EligibilityStatus.FAIL
        explanation = "One or more mandatory requirements were definitively below a required threshold."
    elif ambiguous:
        status = EligibilityStatus.REVIEW
        explanation = "Some mandatory requirements are unclear, partial, or not found and require human review."
    else:
        status = EligibilityStatus.PASS
        explanation = "All deterministic mandatory checks passed."

    return EligibilityResult(
        status=status,
        checked_requirements=checked,
        failed_requirements=failed,
        ambiguous_requirements=ambiguous,
        requirement_statuses=requirement_statuses,
        requirement_details=requirement_details,
        explanation=explanation,
    )
