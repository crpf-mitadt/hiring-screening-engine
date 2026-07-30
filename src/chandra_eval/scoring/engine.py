"""Deterministic candidate-to-JD scoring engine."""

from uuid import uuid4

from chandra_eval.scoring.eligibility import evaluate_eligibility
from chandra_eval.scoring.education import evaluate_education_requirement
from chandra_eval.scoring.evidence_completeness import evidence_completeness
from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.experience_basis import qualify_quantitative_experience
from chandra_eval.scoring.matching import (
    build_candidate_evidence_index,
    criterion_terms,
    match_qualitative_experience,
    match_terms,
)
from chandra_eval.schemas.evaluation import (
    CandidateEvaluationReport,
    CriterionEvaluation,
    CriterionStatus,
    EligibilityStatus,
)
from chandra_eval.schemas.jd import RequirementCategory, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import ScoringSafeProfile


APPROVED_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def discrete_match_fraction(value: float) -> float:
    if value <= 0:
        return 0.0
    if value < 0.375:
        return 0.25
    if value < 0.625:
        return 0.5
    if value < 0.875:
        return 0.75
    return 1.0


def _status(fraction: float, *, has_terms: bool, ambiguous: bool = False) -> CriterionStatus:
    if not has_terms:
        return CriterionStatus.NOT_APPLICABLE
    if ambiguous and fraction > 0:
        return CriterionStatus.UNCLEAR
    if fraction >= 0.999:
        return CriterionStatus.MET
    if fraction > 0:
        return CriterionStatus.PARTIALLY_MET
    return CriterionStatus.NOT_FOUND


def _score_criterion(
    criterion: ScoringCriterion,
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    *,
    experience_summary: dict,
) -> CriterionEvaluation:
    missing: list[str] = []
    ambiguity: list[str] = []
    fraction = 0.0
    explanation = ""
    index = build_candidate_evidence_index(candidate)
    verification_needed = list(criterion.accepted_evidence)

    if criterion.category == RequirementCategory.EXPERIENCE:
        required_months = jd.minimum_experience_months or _minimum_from_requirements(jd)
        if required_months:
            qualification = qualify_quantitative_experience(
                candidate,
                required_months=required_months,
                requirement_text=criterion.description,
            )
            fraction = qualification.fraction
            evidence = qualification.evidence
            missing = qualification.missing
            ambiguity = qualification.ambiguity
            if qualification.confirmed_months < required_months and experience_summary.get("uncertain_ranges"):
                ambiguity.append("One or more experience date ranges are ambiguous and may affect the minimum.")
            fraction = discrete_match_fraction(fraction)
            explanation = qualification.explanation
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                match_fraction=fraction,
                weighted_score=round(criterion.weight * fraction, 6),
                status=_status(fraction, has_terms=True, ambiguous=bool(ambiguity)),
                explanation=explanation,
                evidence=evidence,
                missing_information=missing,
                ambiguity=ambiguity,
                verification_needed=verification_needed,
                candidate_requirement=criterion.description,
                evidence_found=_evidence_found(evidence),
            )
        else:
            match = match_qualitative_experience(criterion.description, index)
            terms = match.terms
            fraction = discrete_match_fraction(match.fraction)
            missing = match.missing_terms
            evidence = match.evidence
            ambiguity = list(match.ambiguity)
            explanation = _match_explanation(match.matches, missing, ambiguity)
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                match_fraction=fraction,
                weighted_score=round(criterion.weight * fraction, 6),
                status=_status(fraction, has_terms=bool(terms), ambiguous=bool(ambiguity)),
                explanation=explanation,
                evidence=evidence,
                missing_information=missing,
                ambiguity=ambiguity,
                verification_needed=verification_needed,
                candidate_requirement=criterion.description,
                evidence_found=_evidence_found(evidence),
            )
    if criterion.category == RequirementCategory.EDUCATION:
        education_match = evaluate_education_requirement(criterion.description, candidate.education)
        fraction = 1.0 if education_match.matched else 0.0
        explanation = (
            "Matched education requirement components. "
            + "; ".join(education_match.notes)
            if education_match.matched
            else "Education requirement components were not fully matched. "
            + "; ".join([*education_match.missing, *education_match.ambiguity])
        )
        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            match_fraction=fraction,
            weighted_score=round(criterion.weight * fraction, 6),
            status=_status(fraction, has_terms=True, ambiguous=bool(education_match.ambiguity)),
            explanation=explanation,
            evidence=education_match.evidence,
            missing_information=education_match.missing,
            ambiguity=education_match.ambiguity,
            verification_needed=verification_needed,
            candidate_requirement=criterion.description,
            evidence_found=_evidence_found(education_match.evidence),
        )

    terms = criterion_terms(criterion, jd)
    allowed_sources = {"education"} if criterion.category == RequirementCategory.EDUCATION else None
    match = match_terms(terms, index, allowed_sources=allowed_sources)
    fraction = discrete_match_fraction(match.fraction)
    missing = match.missing_terms
    ambiguity = list(match.ambiguity)
    if criterion.category == RequirementCategory.EDUCATION and fraction > 0 and missing:
        ambiguity.append("Education evidence is partial or specialization is unclear.")
    explanation = _match_explanation(match.matches, missing, ambiguity)

    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        match_fraction=fraction,
        weighted_score=round(criterion.weight * fraction, 6),
        status=_status(fraction, has_terms=bool(terms), ambiguous=bool(ambiguity)),
        explanation=explanation,
        evidence=match.evidence,
        missing_information=missing,
        ambiguity=ambiguity,
        verification_needed=verification_needed,
        candidate_requirement=criterion.description,
        evidence_found=_evidence_found(match.evidence),
    )


def evaluate_candidate(
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    evaluation_id: str | None = None,
) -> CandidateEvaluationReport:
    criteria = jd.scoring_rubric or [
        ScoringCriterion(
            criterion_id="skills",
            name="Required skills",
            category=RequirementCategory.SKILL,
            description="Required skills from the job description",
            weight=45,
            mandatory=True,
        ),
        ScoringCriterion(
            criterion_id="experience",
            name="Relevant experience",
            category=RequirementCategory.EXPERIENCE,
            description="Minimum relevant experience",
            weight=35,
            mandatory=True,
        ),
        ScoringCriterion(
            criterion_id="preferred-skills",
            name="Preferred skills",
            category=RequirementCategory.SKILL,
            description="Preferred skills from the job description",
            weight=20,
            mandatory=False,
        ),
    ]
    total_weight = sum(criterion.weight for criterion in criteria) or 1.0
    duration = calculate_experience_duration(candidate.experience)
    experience_summary = duration.as_dict()
    evaluations = [
        _score_criterion(criterion, candidate, jd, experience_summary=experience_summary)
        for criterion in criteria
    ]
    raw_score = sum(item.weighted_score for item in evaluations) * (100 / total_weight)
    eligibility = evaluate_eligibility(candidate, jd)
    final_score = min(100.0, round(raw_score, 3))

    gaps = sorted({gap for item in evaluations for gap in item.missing_information})
    criteria_by_id = {criterion.criterion_id: criterion for criterion in criteria}
    strengths = [
        criteria_by_id[item.criterion_id].name or criteria_by_id[item.criterion_id].description or item.criterion_id
        for item in evaluations
        if item.status == CriterionStatus.MET
    ]
    if eligibility.status == EligibilityStatus.FAIL:
        recommendation = "mandatory_requirement_failed_human_review_required"
    elif final_score >= 75 and eligibility.status == EligibilityStatus.PASS:
        recommendation = "strong_match_for_human_review"
    elif final_score >= 50:
        recommendation = "possible_match_for_human_review"
    else:
        recommendation = "weak_match_for_human_review"

    return CandidateEvaluationReport(
        evaluation_id=evaluation_id or f"eval-{uuid4().hex[:12]}",
        candidate_id=candidate.candidate_id,
        job_id=jd.job_id,
        eligibility=eligibility,
        criterion_evaluations=evaluations,
        score_before_gates=round(raw_score, 3),
        final_score=final_score,
        evidence_completeness=evidence_completeness(
            evaluations,
            date_warnings=experience_summary.get("date_warnings", []),
        ),
        experience_summary=experience_summary,
        strengths=strengths,
        gaps=gaps,
        missing_information=sorted(set(candidate.missing_information + gaps)),
        recommendation_label=recommendation,
        human_review_required=True,
    )


def _minimum_from_requirements(jd: StructuredJD) -> int:
    values = [
        int(requirement.minimum_value)
        for requirement in jd.mandatory_requirements
        if requirement.category == RequirementCategory.EXPERIENCE and requirement.minimum_value is not None
    ]
    return max(values) if values else 0


def _match_explanation(matches, missing: list[str], ambiguity: list[str]) -> str:
    supported = [item for item in matches if item.matched]
    supported_terms = [item.term for item in supported[:6]]
    parts: list[str] = []
    if supported_terms:
        parts.append(f"Matched {', '.join(supported_terms)} from resume evidence.")
    else:
        parts.append("No direct resume evidence was found for the criterion terms.")
    if missing:
        parts.append(f"Missing or unverified: {', '.join(missing[:6])}.")
    if ambiguity:
        parts.append(f"Ambiguity: {', '.join(ambiguity[:4])}.")
    return " ".join(parts)


def _evidence_found(evidence) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        text = (item.normalized_text or item.exact_text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text[:160])
    return output
