"""Deterministic candidate-to-JD scoring engine."""

from uuid import uuid4

from chandra_eval.scoring.eligibility import evaluate_eligibility
from chandra_eval.scoring.evidence_completeness import evidence_completeness
from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.matching import (
    build_candidate_evidence_index,
    criterion_terms,
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

    if criterion.category == RequirementCategory.EXPERIENCE:
        required_months = jd.minimum_experience_months or _minimum_from_requirements(jd)
        actual_months = int(experience_summary["confirmed_months"])
        if required_months:
            fraction = min(actual_months / required_months, 1.0)
            if actual_months < required_months:
                missing.append(f"{required_months - actual_months} additional months of experience")
                if experience_summary.get("uncertain_ranges"):
                    ambiguity.append(
                        "One or more experience date ranges are ambiguous and may affect the minimum."
                    )
        else:
            terms = criterion_terms(criterion, jd)
            match = match_terms(terms, index)
            fraction = match.fraction
            missing = match.missing_terms
            evidence = match.evidence
            explanation = _match_explanation(match.matches, missing, ambiguity)
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                match_fraction=round(fraction, 3),
                weighted_score=round(criterion.weight * fraction, 3),
                status=_status(fraction, has_terms=bool(terms), ambiguous=bool(ambiguity)),
                explanation=explanation,
                evidence=evidence,
                missing_information=missing,
                ambiguity=ambiguity,
            )
        terms = criterion_terms(criterion, jd)
        semantic = match_terms(terms, index)
        evidence = semantic.evidence
        if semantic.fraction > 0 and fraction < 1.0:
            fraction = max(fraction, min(semantic.fraction, 0.75))
        missing.extend(term for term in semantic.missing_terms if term not in missing)
        explanation = (
            f"Confirmed {actual_months} months against {required_months or 0} required months. "
            f"{_match_explanation(semantic.matches, semantic.missing_terms, ambiguity)}"
        )
        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            match_fraction=round(fraction, 3),
            weighted_score=round(criterion.weight * fraction, 3),
            status=_status(fraction, has_terms=True, ambiguous=bool(ambiguity)),
            explanation=explanation,
            evidence=evidence,
            missing_information=missing,
            ambiguity=ambiguity,
        )

    terms = criterion_terms(criterion, jd)
    match = match_terms(terms, index)
    fraction = match.fraction
    missing = match.missing_terms
    ambiguity = list(match.ambiguity)
    if criterion.category == RequirementCategory.EDUCATION and fraction > 0 and missing:
        ambiguity.append("Education evidence is partial or specialization is unclear.")
    explanation = _match_explanation(match.matches, missing, ambiguity)

    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        match_fraction=round(fraction, 3),
        weighted_score=round(criterion.weight * fraction, 3),
        status=_status(fraction, has_terms=bool(terms), ambiguous=bool(ambiguity)),
        explanation=explanation,
        evidence=match.evidence,
        missing_information=missing,
        ambiguity=ambiguity,
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
    strengths = [
        item.criterion_id
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
        evidence_completeness=evidence_completeness(evaluations),
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
