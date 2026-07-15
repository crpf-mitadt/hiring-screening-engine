"""Mandatory requirement gates."""

from chandra_eval.scoring.experience import total_experience_months
from chandra_eval.scoring.normalization import contains_equivalent
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

    for requirement in jd.mandatory_requirements:
        checked.append(requirement.requirement_id)
        if requirement.category == RequirementCategory.SKILL:
            if not contains_equivalent(
                candidate.skills,
                requirement.description,
                requirement.accepted_equivalents,
            ):
                failed.append(requirement.requirement_id)
        elif requirement.category == RequirementCategory.EXPERIENCE:
            if requirement.minimum_value is None:
                ambiguous.append(requirement.requirement_id)
            elif total_experience_months(candidate.experience) < int(requirement.minimum_value):
                failed.append(requirement.requirement_id)
        else:
            ambiguous.append(requirement.requirement_id)

    if failed:
        status = EligibilityStatus.FAIL
        explanation = "One or more mandatory requirements were not supported."
    elif ambiguous:
        status = EligibilityStatus.REVIEW
        explanation = "Some mandatory requirements require human review."
    else:
        status = EligibilityStatus.PASS
        explanation = "All deterministic mandatory checks passed."

    return EligibilityResult(
        status=status,
        checked_requirements=checked,
        failed_requirements=failed,
        ambiguous_requirements=ambiguous,
        explanation=explanation,
    )
