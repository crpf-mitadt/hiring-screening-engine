"""Mandatory requirement gates."""

from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.matching import build_candidate_evidence_index, match_requirement
from chandra_eval.schemas.evaluation import EligibilityResult, EligibilityStatus
from chandra_eval.schemas.jd import RequirementCategory, StructuredJD
from chandra_eval.schemas.resume import ScoringSafeProfile


def evaluate_eligibility(
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
) -> EligibilityResult:
    checked: list[str] = []
    failed: list[str] = []
    ambiguous: list[str] = []
    requirement_statuses: dict[str, str] = {}
    index = build_candidate_evidence_index(candidate)
    duration = calculate_experience_duration(candidate.experience)

    for requirement in jd.mandatory_requirements:
        checked.append(requirement.requirement_id)
        if requirement.category == RequirementCategory.SKILL:
            match = match_requirement(requirement.description, index, requirement.accepted_equivalents)
            if match.fraction > 0:
                requirement_statuses[requirement.requirement_id] = "met"
            else:
                requirement_statuses[requirement.requirement_id] = "not_found"
                ambiguous.append(requirement.requirement_id)
        elif requirement.category == RequirementCategory.EXPERIENCE:
            if requirement.minimum_value is None:
                requirement_statuses[requirement.requirement_id] = "unclear"
                ambiguous.append(requirement.requirement_id)
            elif duration.confirmed_months >= int(requirement.minimum_value):
                requirement_statuses[requirement.requirement_id] = "met"
            elif duration.uncertain_ranges:
                requirement_statuses[requirement.requirement_id] = "unclear"
                ambiguous.append(requirement.requirement_id)
            elif duration.confirmed_months == 0:
                requirement_statuses[requirement.requirement_id] = "not_found"
                ambiguous.append(requirement.requirement_id)
            else:
                requirement_statuses[requirement.requirement_id] = "not_met"
                failed.append(requirement.requirement_id)
        elif requirement.category in {RequirementCategory.EDUCATION, RequirementCategory.LANGUAGE, RequirementCategory.DOMAIN}:
            match = match_requirement(requirement.description, index, requirement.accepted_equivalents)
            if match.fraction > 0:
                requirement_statuses[requirement.requirement_id] = (
                    "partially_met" if requirement.category == RequirementCategory.EDUCATION else "met"
                )
                ambiguous.append(requirement.requirement_id) if requirement.category == RequirementCategory.EDUCATION else None
            else:
                requirement_statuses[requirement.requirement_id] = "not_found"
                ambiguous.append(requirement.requirement_id)
        else:
            requirement_statuses[requirement.requirement_id] = "unclear"
            ambiguous.append(requirement.requirement_id)

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
        explanation=explanation,
    )
