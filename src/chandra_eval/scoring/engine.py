"""Deterministic candidate-to-JD scoring engine."""

from uuid import uuid4

from chandra_eval.scoring.eligibility import evaluate_eligibility
from chandra_eval.scoring.evidence_completeness import evidence_completeness
from chandra_eval.scoring.experience import total_experience_months
from chandra_eval.scoring.skills import missing_skills, skill_match_fraction
from chandra_eval.schemas.evaluation import (
    CandidateEvaluationReport,
    CriterionEvaluation,
    CriterionStatus,
    EligibilityStatus,
)
from chandra_eval.schemas.jd import RequirementCategory, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import ScoringSafeProfile


def _status(fraction: float) -> CriterionStatus:
    if fraction >= 0.999:
        return CriterionStatus.MET
    if fraction > 0:
        return CriterionStatus.PARTIAL
    return CriterionStatus.NOT_MET


def _score_criterion(
    criterion: ScoringCriterion,
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
) -> CriterionEvaluation:
    missing: list[str] = []
    fraction = 0.0
    explanation = ""

    if criterion.category == RequirementCategory.SKILL:
        required = jd.required_skills if criterion.mandatory else jd.preferred_skills
        fraction = skill_match_fraction(required, candidate.skills)
        missing = missing_skills(required, candidate.skills)
        explanation = f"Matched {len(required) - len(missing)} of {len(required)} skills."
    elif criterion.category == RequirementCategory.EXPERIENCE:
        required_months = jd.minimum_experience_months or 0
        actual_months = total_experience_months(candidate.experience)
        fraction = 1.0 if required_months == 0 else min(actual_months / required_months, 1.0)
        if actual_months < required_months:
            missing = [f"{required_months - actual_months} additional months of experience"]
        explanation = f"Found {actual_months} months against {required_months} required months."
    elif criterion.category == RequirementCategory.CERTIFICATION:
        preferred = [
            requirement.description
            for requirement in jd.preferred_requirements + jd.mandatory_requirements
            if requirement.category == RequirementCategory.CERTIFICATION
        ]
        fraction = skill_match_fraction(preferred, candidate.certifications)
        missing = missing_skills(preferred, candidate.certifications)
        explanation = f"Matched {len(preferred) - len(missing)} of {len(preferred)} certifications."
    else:
        text = " ".join(
            [
                candidate.summary,
                *candidate.projects,
                *candidate.research_experience,
                *candidate.teaching_experience,
                *candidate.leadership_experience,
                *candidate.achievements,
            ]
        ).lower()
        tokens = [token for token in criterion.description.lower().split() if len(token) > 3]
        hits = sum(1 for token in tokens if token in text)
        fraction = 1.0 if not tokens else min(hits / len(tokens), 1.0)
        explanation = "Matched criterion terms against job-related profile text."

    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        match_fraction=round(fraction, 3),
        weighted_score=round(criterion.weight * fraction, 3),
        status=_status(fraction),
        explanation=explanation,
        evidence=[],
        missing_information=missing,
        ambiguity=[],
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
    evaluations = [_score_criterion(criterion, candidate, jd) for criterion in criteria]
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
        recommendation = "human_review_mandatory_requirement_not_supported"
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
        strengths=strengths,
        gaps=gaps,
        missing_information=sorted(set(candidate.missing_information + gaps)),
        recommendation_label=recommendation,
        human_review_required=True,
    )
