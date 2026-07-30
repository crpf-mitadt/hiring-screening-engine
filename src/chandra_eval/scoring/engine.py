"""Deterministic candidate-to-JD scoring engine."""

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import re
from uuid import uuid4

from chandra_eval.scoring.eligibility import evaluate_eligibility
from chandra_eval.scoring.education import evaluate_education_requirement
from chandra_eval.scoring.evidence_completeness import evidence_completeness, evidence_completeness_diagnostics
from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.experience_basis import ExperienceQualification, qualify_quantitative_experience
from chandra_eval.scoring.matching import (
    EvidenceText,
    build_candidate_evidence_index,
    criterion_terms,
    match_requirement,
    match_qualitative_experience,
    match_terms,
    _contains_phrase,
    _dedupe_evidence,
    _dedupe_terms,
)
from chandra_eval.scoring.normalization import normalize_token
from chandra_eval.schemas.common import EvidenceRef, EvidenceStatus
from chandra_eval.schemas.evaluation import (
    CandidateEvaluationReport,
    CriterionEvaluation,
    CriterionStatus,
    EligibilityStatus,
)
from chandra_eval.schemas.jd import RequirementCategory, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import ScoringSafeProfile


APPROVED_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
CLASS_REQUIRED = "required/core"
CLASS_PREFERRED = "preferred"
CLASS_ADJACENT = "adjacent/transferable"
CLASS_CONTROLLED = "controlled equivalent"
CLASS_EVALUATION_DIMENSION = "evaluation_dimension"

_ADJACENT_FULL_STACK_TERMS = (
    "JavaScript",
    "React.js",
    "Node.js",
    "Express.js",
    "Vue.js",
    "REST APIs",
    "frontend development",
    "backend development",
    "full-stack development",
)

_SOFT_SKILL_PATTERNS: dict[str, tuple[str, ...]] = {
    "communication": (
        "presented",
        "presented findings",
        "communicated with clients",
        "communicated with stakeholders",
        "client communication",
        "stakeholder communication",
        "authored documentation",
        "facilitated meetings",
        "trained",
        "mentored",
        "coordinated across",
        "coordinated with stakeholders",
    ),
    "ownership": (
        "owned",
        "ownership",
        "led delivery",
        "accountable",
        "end to end",
        "independently managed",
        "drove",
        "took responsibility",
        "production support",
    ),
    "proactivity": (
        "proactively",
        "proactive",
        "initiative",
        "initiated",
        "introduced",
        "automated",
        "identified",
        "anticipated",
        "proposed and implemented",
        "prevented",
    ),
    "problem-solving": (
        "resolved",
        "root cause",
        "debugged",
        "diagnosed",
        "reduced",
        "improved by",
        "fixed",
    ),
}


@dataclass
class ComponentEvaluation:
    term: str
    source: str
    component_class: str
    matched: bool = False
    evidence: list[EvidenceRef] = field(default_factory=list)
    note: str = ""


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
    if ambiguous:
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
    eligibility: "EligibilityResult",
    experience_qualifications: dict[str, ExperienceQualification],
    evaluation_date: date,
) -> CriterionEvaluation:
    missing: list[str] = []
    ambiguity: list[str] = []
    fraction = 0.0
    explanation = ""
    index = build_candidate_evidence_index(candidate)
    verification_needed = list(criterion.accepted_evidence)

    if criterion.category == RequirementCategory.EXPERIENCE and (
        criterion.required_skill_names
        or criterion.preferred_skill_names
        or criterion.adjacent_skill_names
        or criterion.evaluation_dimensions
    ):
        return _score_mixed_experience_skill_criterion(
            criterion,
            candidate,
            jd,
            eligibility=eligibility,
            index=index,
            experience_summary=experience_summary,
            verification_needed=verification_needed,
            experience_qualifications=experience_qualifications,
            evaluation_date=evaluation_date,
        )

    if criterion.category == RequirementCategory.EXPERIENCE:
        required_months = jd.minimum_experience_months or _minimum_from_requirements(jd)
        if required_months:
            qualification = _qualification_for_criterion(
                criterion,
                jd,
                candidate,
                experience_qualifications,
                required_months=required_months,
                requirement_text=criterion.description,
                evaluation_date=evaluation_date,
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
                criterion_name=criterion.name,
                category=criterion.category.value,
                weight=criterion.weight,
                requirement_ids=criterion.requirement_ids,
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
                component_assessment=_single_component_assessment(
                    criterion,
                    term=criterion.description,
                    source="linked requirement" if criterion.requirement_ids else "criterion description",
                    matched=fraction > 0,
                    evidence=evidence,
                ),
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
                criterion_name=criterion.name,
                category=criterion.category.value,
                weight=criterion.weight,
                requirement_ids=criterion.requirement_ids,
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
                component_assessment=_single_component_assessment(
                    criterion,
                    term=criterion.description,
                    source="criterion description",
                    matched=fraction > 0,
                    evidence=evidence,
                ),
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
            criterion_name=criterion.name,
            category=criterion.category.value,
            weight=criterion.weight,
            requirement_ids=criterion.requirement_ids,
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
            component_assessment=_single_component_assessment(
                criterion,
                term=criterion.description,
                source="criterion description",
                matched=fraction > 0,
                evidence=education_match.evidence,
            ),
        )

    if (
        criterion.category
        in {
            RequirementCategory.SKILL,
            RequirementCategory.LEADERSHIP,
            RequirementCategory.RESPONSIBILITY,
            RequirementCategory.OTHER,
        }
        or criterion.required_skill_names
        or criterion.preferred_skill_names
        or criterion.adjacent_skill_names
        or criterion.evaluation_dimensions
        or criterion.requirement_ids
    ):
        return _score_skill_criterion(
            criterion,
            candidate,
            jd,
            eligibility=eligibility,
            index=index,
            verification_needed=verification_needed,
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
        criterion_name=criterion.name,
        category=criterion.category.value,
        weight=criterion.weight,
        requirement_ids=criterion.requirement_ids,
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
        component_assessment=_single_component_assessment(
            criterion,
            term=criterion.description,
            source="criterion description",
            matched=fraction > 0,
            evidence=match.evidence,
        ),
    )


def evaluate_candidate(
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    evaluation_id: str | None = None,
) -> CandidateEvaluationReport:
    evaluation_date = date.today()
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
    duration = calculate_experience_duration(candidate.experience, evaluation_date=evaluation_date)
    experience_summary = duration.as_dict()
    experience_summary["evaluation_date"] = evaluation_date.isoformat()
    experience_qualifications = _experience_qualifications(candidate, jd, evaluation_date)
    eligibility = evaluate_eligibility(
        candidate,
        jd,
        experience_qualifications=experience_qualifications,
        evaluation_date=evaluation_date,
    )
    evaluations = [
        _score_criterion(
            criterion,
            candidate,
            jd,
            experience_summary=experience_summary,
            eligibility=eligibility,
            experience_qualifications=experience_qualifications,
            evaluation_date=evaluation_date,
        )
        for criterion in criteria
    ]
    _validate_criterion_results(criteria, evaluations, eligibility=eligibility)
    raw_score = sum(item.weighted_score for item in evaluations) * (100 / total_weight)
    final_score = min(100.0, round(raw_score, 3))

    gaps = _report_gaps(evaluations)
    criteria_by_id = {criterion.criterion_id: criterion for criterion in criteria}
    strengths = _report_strengths(evaluations, criteria_by_id)
    if eligibility.status == EligibilityStatus.FAIL:
        recommendation = "mandatory_requirement_failed_human_review_required"
    elif final_score >= 75 and eligibility.status == EligibilityStatus.PASS:
        recommendation = "strong_match_for_human_review"
    elif final_score >= 50:
        recommendation = "possible_match_for_human_review"
    else:
        recommendation = "weak_match_for_human_review"

    completeness_diagnostics = evidence_completeness_diagnostics(
        evaluations,
        date_warnings=experience_summary.get("date_warnings", []),
        warnings=candidate.extraction_warnings,
    )
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
            warnings=candidate.extraction_warnings,
        ),
        evidence_completeness_diagnostics=completeness_diagnostics,
        experience_summary=experience_summary,
        strengths=strengths,
        gaps=gaps,
        missing_information=sorted(set(candidate.missing_information + gaps)),
        recommendation_label=recommendation,
        human_review_required=True,
    )


def _score_skill_criterion(
    criterion: ScoringCriterion,
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    *,
    eligibility: "EligibilityResult",
    index: list[EvidenceText],
    verification_needed: list[str],
) -> CriterionEvaluation:
    components = _criterion_components(criterion, jd, eligibility, index)
    return _criterion_from_components(
        criterion,
        components,
        verification_needed=verification_needed,
        candidate_requirement=criterion.description,
    )


def _criterion_from_components(
    criterion: ScoringCriterion,
    components: list[ComponentEvaluation],
    *,
    verification_needed: list[str],
    candidate_requirement: str,
    ambiguity: list[str] | None = None,
) -> CriterionEvaluation:
    required_components = [item for item in components if item.component_class == CLASS_REQUIRED]
    preferred_components = [item for item in components if item.component_class == CLASS_PREFERRED]
    dimension_components = [item for item in components if item.component_class == CLASS_EVALUATION_DIMENSION]
    adjacent_components = [item for item in components if item.component_class == CLASS_ADJACENT]
    matched_components = [item for item in components if item.matched]
    missing_required = [item.term for item in [*required_components, *dimension_components] if not item.matched]
    evidence = _dedupe_evidence([evidence for item in matched_components for evidence in item.evidence])
    raw_fraction = _weighted_component_fraction(components)
    fraction = discrete_match_fraction(raw_fraction)
    explanation = _component_explanation(components)
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        criterion_name=criterion.name,
        category=criterion.category.value,
        weight=criterion.weight,
        requirement_ids=criterion.requirement_ids,
        match_fraction=fraction,
        weighted_score=round(criterion.weight * fraction, 6),
        status=_status(fraction, has_terms=bool(components), ambiguous=bool(ambiguity)),
        explanation=explanation,
        evidence=evidence,
        missing_information=missing_required,
        ambiguity=ambiguity or [],
        verification_needed=verification_needed,
        candidate_requirement=candidate_requirement,
        evidence_found=_evidence_found(evidence),
        component_assessment=_component_assessment(
            criterion,
            components=components,
            evidence=evidence,
        ),
    )


def _score_mixed_experience_skill_criterion(
    criterion: ScoringCriterion,
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    *,
    eligibility: "EligibilityResult",
    index: list[EvidenceText],
    experience_summary: dict,
    verification_needed: list[str],
    experience_qualifications: dict[str, ExperienceQualification],
    evaluation_date: date,
) -> CriterionEvaluation:
    components: list[ComponentEvaluation] = []
    ambiguity: list[str] = []
    required_months = jd.minimum_experience_months or _minimum_from_requirements(jd)
    if required_months:
        qualification = _qualification_for_criterion(
            criterion,
            jd,
            candidate,
            experience_qualifications,
            required_months=required_months,
            requirement_text=criterion.description,
            evaluation_date=evaluation_date,
        )
        ambiguity = list(qualification.ambiguity)
        if qualification.confirmed_months < required_months and experience_summary.get("uncertain_ranges"):
            ambiguity.append("One or more experience date ranges are ambiguous and may affect the minimum.")
        components.append(
            ComponentEvaluation(
                term="Relevant experience duration/relevance",
                source=_experience_component_source(criterion),
                component_class=CLASS_REQUIRED,
                matched=qualification.fraction >= 1 and bool(qualification.evidence),
                evidence=qualification.evidence if qualification.evidence else [],
                note=qualification.explanation,
            )
        )
    components.extend(_criterion_components(criterion, jd, eligibility, index, include_linked_experience=False))
    return _criterion_from_components(
        criterion,
        components,
        verification_needed=verification_needed,
        candidate_requirement=criterion.description,
        ambiguity=ambiguity,
    )


def _experience_component_source(criterion: ScoringCriterion) -> str:
    for requirement_id in criterion.requirement_ids:
        if "exp" in requirement_id.lower() or "experience" in requirement_id.lower():
            return f"linked requirement {requirement_id}"
    return "criterion description fallback"


def _criterion_components(
    criterion: ScoringCriterion,
    jd: StructuredJD,
    eligibility: "EligibilityResult",
    index: list[EvidenceText],
    *,
    include_linked_experience: bool = True,
) -> list[ComponentEvaluation]:
    components = _owned_components(criterion, jd, include_linked_experience=include_linked_experience)
    components = [
        _match_component(component.term, index, source=component.source, component_class=component.component_class)
        for component in components
    ]
    components = _merge_linked_requirement_components(
        components,
        criterion,
        jd,
        eligibility,
        include_linked_experience=include_linked_experience,
    )
    components.extend(_adjacent_components(criterion, components, index))
    components.extend(_dimension_components(criterion, index))
    return _dedupe_components(components)


def _owned_components(
    criterion: ScoringCriterion,
    jd: StructuredJD,
    *,
    include_linked_experience: bool = True,
) -> list[ComponentEvaluation]:
    components: list[ComponentEvaluation] = []
    for term in criterion.required_skill_names:
        components.append(ComponentEvaluation(term=term, source="required_skill_names", component_class=CLASS_REQUIRED))
    for term in criterion.preferred_skill_names:
        components.append(ComponentEvaluation(term=term, source="preferred_skill_names", component_class=CLASS_PREFERRED))
    linked_ids = set(criterion.requirement_ids)
    for requirement in [*jd.mandatory_requirements, *jd.preferred_requirements]:
        if not include_linked_experience and requirement.category == RequirementCategory.EXPERIENCE:
            continue
        if requirement.requirement_id in linked_ids and not _requirement_represented(
            requirement.description, [item.term for item in components]
        ):
            component_class = CLASS_REQUIRED if requirement.mandatory else CLASS_PREFERRED
            for term in _linked_requirement_component_terms(requirement.description):
                components.append(
                    ComponentEvaluation(
                        term=term,
                        source=f"linked requirement {requirement.requirement_id}",
                        component_class=component_class,
                    )
                )
    if not components and criterion.mandatory and _default_required_skill_criterion(criterion):
        components.extend(
            ComponentEvaluation(term=term, source="required_skills", component_class=CLASS_REQUIRED)
            for term in jd.required_skills
        )
    if not components and not criterion.mandatory and _default_preferred_skill_criterion(criterion):
        components.extend(
            ComponentEvaluation(term=term, source="preferred_skills", component_class=CLASS_PREFERRED)
            for term in jd.preferred_skills
        )
    if not components and not criterion.adjacent_skill_names and not criterion.evaluation_dimensions:
        components.append(
            ComponentEvaluation(
                term=criterion.description or criterion.name,
                source="criterion description fallback",
                component_class=CLASS_REQUIRED if criterion.mandatory else CLASS_PREFERRED,
            )
        )
    return _dedupe_components(components)


def _linked_requirement_component_terms(description: str) -> list[str]:
    normalized = normalize_token(description)
    if "database" in normalized or "mysql" in normalized or "postgres" in normalized or "sql" in normalized:
        return ["SQL/database management"]
    return _simple_requirement_skill_terms(description) or [description]


def _simple_requirement_skill_terms(description: str) -> list[str]:
    patterns = (
        r"^\s*required\s*:\s*(.+?)\s*[.;]?\s*$",
        r"^\s*(.+?)\s+is\s+(?:required|mandatory|essential)\s*[.;]?\s*$",
        r"^\s*(?:must\s+(?:know|have|use|support)|requires?)\s+(.+?)\s*[.;]?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            return _dedupe_terms([match.group(1).strip()])
    return []


def _requirement_represented(description: str, terms: list[str]) -> bool:
    requirement_norm = normalize_token(description)
    if any(_is_database_component(term) for term in terms) and (
        "database" in requirement_norm or "mysql" in requirement_norm or "postgres" in requirement_norm or "sql" in requirement_norm
    ):
        return True
    return any(_contains_phrase(requirement_norm, normalize_token(term)) for term in terms)


def _match_component(term: str, index: list[EvidenceText], *, source: str, component_class: str) -> ComponentEvaluation:
    match = match_requirement(term, index) if _is_database_component(term) else match_terms([term], index)
    return ComponentEvaluation(
        term=term,
        source=source,
        component_class=component_class,
        matched=match.fraction > 0 and bool(match.evidence),
        evidence=match.evidence,
        note="matched direct or controlled-equivalent evidence" if match.evidence else "",
    )


def _merge_linked_requirement_components(
    components: list[ComponentEvaluation],
    criterion: ScoringCriterion,
    jd: StructuredJD,
    eligibility: "EligibilityResult",
    *,
    include_linked_experience: bool = True,
) -> list[ComponentEvaluation]:
    linked_ids = set(criterion.requirement_ids)
    if not linked_ids:
        return components
    details = {detail.get("requirement_id"): detail for detail in eligibility.requirement_details}
    requirements = {item.requirement_id: item for item in [*jd.mandatory_requirements, *jd.preferred_requirements]}
    output = list(components)
    for requirement_id in linked_ids:
        detail = details.get(requirement_id)
        requirement = requirements.get(requirement_id)
        if requirement and not include_linked_experience and requirement.category == RequirementCategory.EXPERIENCE:
            continue
        if not detail or not requirement or detail.get("status") != "met":
            continue
        evidence = [EvidenceRef.model_validate(item) for item in detail.get("evidence", [])]
        if not evidence:
            continue
        merged = False
        for index, component in enumerate(output):
            if _component_related_to_requirement(component.term, requirement.description):
                output[index] = ComponentEvaluation(
                    term=component.term,
                    source=f"{component.source}; linked requirement {requirement_id}",
                    component_class=component.component_class,
                    matched=True,
                    evidence=_dedupe_evidence([*component.evidence, *evidence]),
                    note=f"linked requirement {requirement_id} supplied validated evidence",
                )
                merged = True
                break
        if not merged:
            output.append(
                ComponentEvaluation(
                    term=requirement.description,
                    source=f"linked requirement {requirement_id}",
                    component_class=CLASS_REQUIRED if requirement.mandatory else CLASS_PREFERRED,
                    matched=True,
                    evidence=evidence,
                    note=f"linked requirement {requirement_id} supplied validated evidence",
                )
            )
    return output


def _adjacent_components(
    criterion: ScoringCriterion,
    core_components: list[ComponentEvaluation],
    index: list[EvidenceText],
) -> list[ComponentEvaluation]:
    core_terms = [component.term for component in core_components]
    output: list[ComponentEvaluation] = []
    source_declared = _dedupe_terms(criterion.adjacent_skill_names)
    for term in source_declared:
        match = match_terms([term], index)
        output.append(
            ComponentEvaluation(
                term=term,
                source="adjacent_skill_names",
                component_class=CLASS_ADJACENT,
                matched=bool(match.evidence),
                evidence=match.evidence,
                note=(
                    f"{term} recognized as source-declared adjacent evidence, not a mandatory substitute"
                    if match.evidence
                    else ""
                ),
            )
        )
    if _allows_full_stack_adjacent(criterion):
        inferred = _dedupe_terms(
            [
                term
                for term in _ADJACENT_FULL_STACK_TERMS
                if not any(_same_or_related_term(term, core_term) for core_term in core_terms)
                and not any(_same_or_related_term(term, adjacent_term) for adjacent_term in source_declared)
            ]
        )
        for term in inferred:
            match = match_terms([term], index)
            if match.evidence:
                output.append(
                    ComponentEvaluation(
                        term=term,
                        source="adjacent-skill rule v1: full-stack-transferable",
                        component_class=CLASS_ADJACENT,
                        matched=True,
                        evidence=match.evidence,
                        note=f"{term} recognized as inferred adjacent full-stack evidence, not a mandatory substitute",
                    )
                )
    output.extend(_git_related_adjacent_components(criterion, index))
    return _dedupe_components(output)


def _dimension_components(criterion: ScoringCriterion, index: list[EvidenceText]) -> list[ComponentEvaluation]:
    return [_match_soft_skill_dimension(dimension, index) for dimension in criterion.evaluation_dimensions]


def _git_related_adjacent_components(criterion: ScoringCriterion, index: list[EvidenceText]) -> list[ComponentEvaluation]:
    preferred_hosts = {normalize_token(term) for term in criterion.preferred_skill_names if normalize_token(term) in {"github", "gitlab"}}
    if not preferred_hosts:
        return []
    evidence = _plain_git_evidence(index)
    if not evidence:
        return []
    return [
        ComponentEvaluation(
            term="Git version control",
            source="adjacent-skill rule v1: git-hosting-related",
            component_class=CLASS_ADJACENT,
            matched=True,
            evidence=evidence,
            note="Plain Git is related evidence for Git hosting tools, not an exact GitHub or GitLab match.",
        )
    ]


def _plain_git_evidence(index: list[EvidenceText]) -> list[EvidenceRef]:
    evidence: list[EvidenceRef] = []
    for item in index:
        if _contains_phrase(item.normalized, "git") and not (
            _contains_phrase(item.normalized, "github") or _contains_phrase(item.normalized, "gitlab")
        ):
            evidence.extend(
                evidence_item.model_copy(
                    update={
                        "evidence_status": EvidenceStatus.PARTIALLY_SUPPORTED,
                        "notes": (
                            f"{evidence_item.notes}; Plain Git is adjacent/related evidence, not exact GitHub/GitLab"
                            if evidence_item.notes
                            else "Plain Git is adjacent/related evidence, not exact GitHub/GitLab"
                        ),
                    }
                )
                for evidence_item in item.evidence
            )
    return _dedupe_evidence(evidence)


def _match_soft_skill_dimension(dimension: str, index: list[EvidenceText]) -> ComponentEvaluation:
    normalized = normalize_token(dimension)
    if "problem" in normalized and "solv" in normalized:
        key = "problem-solving"
    else:
        key = next((name for name in _SOFT_SKILL_PATTERNS if _contains_phrase(normalized, name)), normalized)
    patterns = _SOFT_SKILL_PATTERNS.get(key, ())
    for item in index:
        if item.source not in {"experience", "leadership", "achievement", "project", "summary"}:
            continue
        if _looks_purely_technical(item.normalized):
            continue
        for pattern in patterns:
            if _contains_phrase(item.normalized, normalize_token(pattern)):
                evidence = _soft_skill_evidence(item, pattern, dimension)
                return ComponentEvaluation(
                    term=dimension,
                    source="evaluation_dimensions",
                    component_class=CLASS_EVALUATION_DIMENSION,
                    matched=True,
                    evidence=evidence,
                    note=f"matched direct {dimension} evidence via trigger phrase '{pattern}'",
                )
    return ComponentEvaluation(term=dimension, source="evaluation_dimensions", component_class=CLASS_EVALUATION_DIMENSION)


def _soft_skill_evidence(item: EvidenceText, pattern: str, dimension: str) -> list[EvidenceRef]:
    pattern_norm = normalize_token(pattern)
    exact_norm = normalize_token(item.text)
    supporting = [
        evidence
        for evidence in item.evidence
        if normalize_token(evidence.normalized_text or evidence.exact_text) == exact_norm
        or _contains_phrase(normalize_token(evidence.normalized_text or evidence.exact_text), pattern_norm)
    ]
    return [
        evidence.model_copy(
            update={
                "evidence_status": EvidenceStatus.SUPPORTED,
                "notes": f"{evidence.notes}; Matched soft-skill dimension '{dimension}'" if evidence.notes else f"Matched soft-skill dimension '{dimension}'",
            }
        )
        for evidence in supporting
    ]


def _component_explanation(components: list[ComponentEvaluation]) -> str:
    parts: list[str] = []
    supported_required = [item.term for item in components if item.component_class == CLASS_REQUIRED and item.matched]
    supported_preferred = [item.term for item in components if item.component_class == CLASS_PREFERRED and item.matched]
    supported_dimensions = [item.term for item in components if item.component_class == CLASS_EVALUATION_DIMENSION and item.matched]
    supported_adjacent = [item for item in components if item.component_class == CLASS_ADJACENT and item.matched]
    missing_required = [item.term for item in components if item.component_class == CLASS_REQUIRED and not item.matched]
    missing_preferred = [item.term for item in components if item.component_class == CLASS_PREFERRED and not item.matched]
    missing_dimensions = [item.term for item in components if item.component_class == CLASS_EVALUATION_DIMENSION and not item.matched]
    if supported_required:
        parts.append(f"Supported required/core components: {', '.join(supported_required[:6])}.")
    if supported_preferred:
        parts.append(f"Supported preferred components: {', '.join(supported_preferred[:6])}.")
    if supported_dimensions:
        parts.append(f"Supported evaluation dimensions: {', '.join(supported_dimensions[:6])}.")
    if supported_adjacent:
        parts.append(
            "Adjacent or transferable components: "
            + ", ".join(item.term for item in supported_adjacent[:6])
            + "."
        )
    if missing_required:
        parts.append(f"Missing required/core components: {', '.join(missing_required[:8])}.")
    if missing_preferred:
        parts.append(f"Missing preferred components: {', '.join(missing_preferred[:8])}.")
    if missing_dimensions:
        parts.append(f"Unverified evaluation dimensions: {', '.join(missing_dimensions[:8])}.")
    if not parts:
        parts.append("No in-scope resume evidence was found for the criterion components.")
    return " ".join(parts)


def _component_assessment(
    criterion: ScoringCriterion,
    *,
    components: list[ComponentEvaluation],
    evidence: list[EvidenceRef],
) -> dict:
    required_components = [item for item in components if item.component_class == CLASS_REQUIRED]
    preferred_components = [item for item in components if item.component_class == CLASS_PREFERRED]
    adjacent_components = [item for item in components if item.component_class == CLASS_ADJACENT]
    dimension_components = [item for item in components if item.component_class == CLASS_EVALUATION_DIMENSION]
    controlled_components = [item for item in required_components if _is_database_component(item.term)]
    controlled_evidence = [
        evidence_item
        for component in controlled_components
        if component.matched
        for evidence_item in component.evidence
    ]
    return {
        "criterion_id": criterion.criterion_id,
        "criterion_name": criterion.name,
        "category": criterion.category.value,
        "weight": criterion.weight,
        "requirement_ids": list(criterion.requirement_ids),
        "represented_requirement_ids": list(criterion.requirement_ids),
        "required_skill_names": list(criterion.required_skill_names),
        "preferred_skill_names": list(criterion.preferred_skill_names),
        "adjacent_skill_names": list(criterion.adjacent_skill_names),
        "evaluation_dimensions": list(criterion.evaluation_dimensions),
        "source_criterion_hash": _source_criterion_hash(criterion),
        "required_core_components": [item.term for item in required_components],
        "preferred_components": [item.term for item in preferred_components],
        "controlled_equivalent_components": [item.term for item in controlled_components],
        "adjacent_components_considered": [item.term for item in adjacent_components],
        "source_declared_adjacent_components": [
            item.term for item in adjacent_components if item.source == "adjacent_skill_names"
        ],
        "inferred_adjacent_components": [
            item.term for item in adjacent_components if item.source.startswith("adjacent-skill rule")
        ],
        "adjacent_inference_rules": sorted(
            {item.source for item in adjacent_components if item.source.startswith("adjacent-skill rule")}
        ),
        "evaluation_dimension_components": [item.term for item in dimension_components],
        "components": [
            {
                "term": item.term,
                "source": item.source,
                "class": item.component_class,
                "classification": item.component_class,
                "matched": item.matched,
                "missing": not item.matched,
                "evidence_count": len(item.evidence),
                "evidence": [evidence_item.model_dump(mode="json") for evidence_item in item.evidence],
                "explanation": item.note,
                "note": item.note,
            }
            for item in components
        ],
        "supported_core_components": [item.term for item in required_components if item.matched],
        "supported_required_components": [item.term for item in required_components if item.matched],
        "missing_core_components": [item.term for item in required_components if not item.matched],
        "supported_preferred_components": [item.term for item in preferred_components if item.matched],
        "missing_preferred_components": [item.term for item in preferred_components if not item.matched],
        "supported_adjacent_components": [item.term for item in adjacent_components if item.matched],
        "supported_evaluation_dimensions": [item.term for item in dimension_components if item.matched],
        "missing_evaluation_dimensions": [item.term for item in dimension_components if not item.matched],
        "supported_controlled_equivalents": sorted(_controlled_equivalent_notes(controlled_evidence)),
        "controlled_equivalents_used": sorted(_controlled_equivalent_notes(controlled_evidence)),
        "evidence_scopes": ["in_scope" for _ in evidence],
    }


def _single_component_assessment(
    criterion: ScoringCriterion,
    *,
    term: str,
    source: str,
    matched: bool,
    evidence: list[EvidenceRef],
) -> dict:
    return _component_assessment(
        criterion,
        components=[
            ComponentEvaluation(
                term=term,
                source=source,
                component_class=CLASS_REQUIRED if criterion.mandatory or source != "preferred_skill_names" else CLASS_PREFERRED,
                matched=matched,
                evidence=evidence if matched else [],
                note="criterion evaluated from source rubric definition",
            )
        ],
        evidence=evidence if matched else [],
    )


def _weighted_component_fraction(components: list[ComponentEvaluation]) -> float:
    required_like = [item for item in components if item.component_class in {CLASS_REQUIRED, CLASS_EVALUATION_DIMENSION}]
    preferred = [item for item in components if item.component_class == CLASS_PREFERRED]
    adjacent = [item for item in components if item.component_class == CLASS_ADJACENT]
    if required_like:
        required_fraction = len([item for item in required_like if item.matched]) / len(required_like)
        preferred_fraction = len([item for item in preferred if item.matched]) / len(preferred) if preferred else 0.0
        adjacent_bonus = 0.25 if any(item.matched for item in adjacent) else 0.0
        if preferred:
            return min(1.0, (required_fraction * 0.75) + (preferred_fraction * 0.25) + adjacent_bonus)
        return min(1.0, required_fraction + adjacent_bonus)
    denominator = sum(_component_weight(item) for item in components)
    if denominator <= 0:
        return 0.0
    numerator = sum(_component_weight(item) for item in components if item.matched)
    return numerator / denominator


def _component_weight(component: ComponentEvaluation) -> float:
    if component.component_class in {CLASS_REQUIRED, CLASS_EVALUATION_DIMENSION}:
        return 1.0
    if component.component_class == CLASS_PREFERRED:
        return 0.5
    if component.component_class == CLASS_ADJACENT:
        return 0.25
    return 0.0


def _dedupe_components(components: list[ComponentEvaluation]) -> list[ComponentEvaluation]:
    output: list[ComponentEvaluation] = []
    seen: set[tuple[str, str]] = set()
    for component in components:
        key = (normalize_token(component.term), component.component_class)
        if key in seen:
            continue
        seen.add(key)
        output.append(component)
    return output


def _experience_qualifications(
    candidate: ScoringSafeProfile,
    jd: StructuredJD,
    evaluation_date: date,
) -> dict[str, ExperienceQualification]:
    qualifications: dict[str, ExperienceQualification] = {}
    for requirement in jd.mandatory_requirements:
        if requirement.category != RequirementCategory.EXPERIENCE or requirement.minimum_value is None:
            continue
        qualifications[requirement.requirement_id] = qualify_quantitative_experience(
            candidate,
            required_months=int(requirement.minimum_value),
            requirement_text=requirement.description,
            evaluation_date=evaluation_date,
        )
    return qualifications


def _qualification_for_criterion(
    criterion: ScoringCriterion,
    jd: StructuredJD,
    candidate: ScoringSafeProfile,
    experience_qualifications: dict[str, ExperienceQualification],
    *,
    required_months: int,
    requirement_text: str,
    evaluation_date: date,
) -> ExperienceQualification:
    requirements = {item.requirement_id: item for item in jd.mandatory_requirements}
    for requirement_id in criterion.requirement_ids:
        requirement = requirements.get(requirement_id)
        if (
            requirement
            and requirement.category == RequirementCategory.EXPERIENCE
            and int(requirement.minimum_value or 0) == int(required_months)
        ):
            qualification = experience_qualifications.get(requirement_id)
            if qualification is not None:
                return qualification
    return qualify_quantitative_experience(
        candidate,
        required_months=required_months,
        requirement_text=requirement_text,
        evaluation_date=evaluation_date,
    )


def _report_strengths(
    evaluations: list[CriterionEvaluation],
    criteria_by_id: dict[str, ScoringCriterion],
) -> list[str]:
    strengths: list[str] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        criterion = criteria_by_id[evaluation.criterion_id]
        if evaluation.status == CriterionStatus.MET:
            _append_unique(strengths, seen, criterion.name or criterion.description or evaluation.criterion_id)
        for component in evaluation.component_assessment.get("components", []):
            if not component.get("matched") or not component.get("evidence"):
                continue
            classification = component.get("classification") or component.get("class")
            if classification == CLASS_EVALUATION_DIMENSION:
                continue
            if classification == CLASS_REQUIRED:
                prefix = "required/core"
            elif classification == CLASS_PREFERRED:
                prefix = "preferred"
            elif classification == CLASS_ADJACENT:
                prefix = "adjacent"
            else:
                continue
            _append_unique(strengths, seen, f"{prefix}: {criterion.name}: {component.get('term')}")
    return strengths


def _report_gaps(evaluations: list[CriterionEvaluation]) -> list[str]:
    gaps: list[str] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        assessment = evaluation.component_assessment
        for term in evaluation.missing_information:
            _append_unique(gaps, seen, term)
        for term in assessment.get("missing_core_components", []) or []:
            _append_unique(gaps, seen, f"mandatory/core gap: {evaluation.criterion_name}: {term}")
        for term in assessment.get("missing_preferred_components", []) or []:
            _append_unique(gaps, seen, f"preferred gap: {evaluation.criterion_name}: {term}")
        for term in assessment.get("missing_evaluation_dimensions", []) or []:
            _append_unique(gaps, seen, f"unverified soft-skill dimension: {evaluation.criterion_name}: {term}")
    return gaps


def _append_unique(output: list[str], seen: set[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in seen:
        seen.add(cleaned)
        output.append(cleaned)


def _source_criterion_hash(criterion: ScoringCriterion) -> str:
    payload = {
        "criterion_id": criterion.criterion_id,
        "name": criterion.name,
        "category": criterion.category.value,
        "description": criterion.description,
        "weight": criterion.weight,
        "mandatory": criterion.mandatory,
        "accepted_evidence": criterion.accepted_evidence,
        "scoring_guidance": criterion.scoring_guidance,
        "requirement_ids": criterion.requirement_ids,
        "required_skill_names": criterion.required_skill_names,
        "preferred_skill_names": criterion.preferred_skill_names,
        "adjacent_skill_names": criterion.adjacent_skill_names,
        "evaluation_dimensions": criterion.evaluation_dimensions,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _default_required_skill_criterion(criterion: ScoringCriterion) -> bool:
    text = normalize_token(" ".join([criterion.criterion_id, criterion.name, criterion.description]))
    return "required skill" in text or "mandatory skill" in text


def _default_preferred_skill_criterion(criterion: ScoringCriterion) -> bool:
    text = normalize_token(" ".join([criterion.criterion_id, criterion.name, criterion.description]))
    return "preferred skill" in text or "additional technical" in text


def _is_database_component(term: str) -> bool:
    normalized = normalize_token(term)
    return any(
        _contains_phrase(normalized, value)
        for value in ("database", "sql", "mysql", "postgresql", "postgres")
    )


def _component_related_to_requirement(term: str, requirement: str) -> bool:
    term_norm = normalize_token(term)
    requirement_norm = normalize_token(requirement)
    if _is_database_component(term) and _is_database_component(requirement):
        return True
    return _contains_phrase(requirement_norm, term_norm) or _contains_phrase(term_norm, requirement_norm)


def _same_or_related_term(left: str, right: str) -> bool:
    left_norm = normalize_token(left)
    right_norm = normalize_token(right)
    return left_norm == right_norm or _contains_phrase(left_norm, right_norm) or _contains_phrase(right_norm, left_norm)


def _allows_full_stack_adjacent(criterion: ScoringCriterion) -> bool:
    text = normalize_token(
        " ".join(
            [
                criterion.criterion_id,
                criterion.name,
                criterion.description,
                criterion.scoring_guidance,
                *criterion.required_skill_names,
            ]
        )
    )
    if any(_contains_phrase(text, term) for term in ("communication", "ownership", "proactivity", "soft skill")):
        return False
    return any(
        _contains_phrase(text, term)
        for term in ("technical stack", "full stack", "fullstack", "web development", "php", "laravel", "wordpress")
    )


def _looks_purely_technical(text: str) -> bool:
    technical_terms = (
        "react",
        "javascript",
        "node",
        "express",
        "vue",
        "php",
        "laravel",
        "wordpress",
        "mysql",
        "postgres",
        "html",
        "css",
        "api",
        "database",
    )
    soft_terms = tuple(term for patterns in _SOFT_SKILL_PATTERNS.values() for term in patterns)
    return any(_contains_phrase(text, term) for term in technical_terms) and not any(
        _contains_phrase(text, normalize_token(term)) for term in soft_terms
    )


def _controlled_equivalent_notes(evidence: list[EvidenceRef]) -> set[str]:
    used: set[str] = set()
    for item in evidence:
        text = normalize_token(" ".join([item.exact_text, item.normalized_text or "", item.notes or ""]))
        if _contains_phrase(text, "mysql"):
            used.add("MySQL")
        if _contains_phrase(text, "postgresql") or _contains_phrase(text, "postgres"):
            used.add("PostgreSQL")
    return used


def _validate_criterion_results(
    criteria: list[ScoringCriterion],
    evaluations: list[CriterionEvaluation],
    *,
    eligibility: "EligibilityResult",
) -> None:
    criteria_by_id = {criterion.criterion_id: criterion for criterion in criteria}
    result_ids = [item.criterion_id for item in evaluations]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Duplicate criterion evaluation IDs are invalid.")
    if set(result_ids) != set(criteria_by_id):
        raise ValueError("Criterion evaluations must contain exactly one result per structured-JD criterion.")
    for item in evaluations:
        criterion = criteria_by_id[item.criterion_id]
        if item.criterion_name != criterion.name:
            raise ValueError(f"Criterion name mismatch for {item.criterion_id}.")
        if item.category != criterion.category.value:
            raise ValueError(f"Criterion category mismatch for {item.criterion_id}.")
        if round(item.weight, 6) != round(criterion.weight, 6):
            raise ValueError(f"Criterion weight mismatch for {item.criterion_id}.")
        if item.requirement_ids != criterion.requirement_ids:
            raise ValueError(f"Criterion linked requirement mismatch for {item.criterion_id}.")
        _validate_component_runtime_invariants(item)
        _validate_linked_experience_consistency(criterion, item, eligibility)


def _validate_component_runtime_invariants(evaluation: CriterionEvaluation) -> None:
    assessment = evaluation.component_assessment
    components = assessment.get("components", [])
    _require_terms_represented(
        assessment.get("required_skill_names", []),
        components,
        source="required_skill_names",
        criterion_id=evaluation.criterion_id,
    )
    _require_terms_represented(
        assessment.get("preferred_skill_names", []),
        components,
        source="preferred_skill_names",
        criterion_id=evaluation.criterion_id,
    )
    _require_terms_represented(
        assessment.get("adjacent_skill_names", []),
        components,
        source="adjacent_skill_names",
        criterion_id=evaluation.criterion_id,
    )
    _require_terms_represented(
        assessment.get("evaluation_dimensions", []),
        components,
        source="evaluation_dimensions",
        criterion_id=evaluation.criterion_id,
    )
    for requirement_id in assessment.get("requirement_ids", []):
        if requirement_id not in assessment.get("represented_requirement_ids", []):
            raise ValueError(f"linked requirement {requirement_id} disappeared from {evaluation.criterion_id}")
    for component in evaluation.component_assessment.get("components", []):
        source = str(component.get("source", ""))
        classification = component.get("classification") or component.get("class")
        if source == "preferred_skill_names" and classification != CLASS_PREFERRED:
            raise ValueError("source=preferred_skill_names implies class=preferred")
        if source == "evaluation_dimensions" and classification != CLASS_EVALUATION_DIMENSION:
            raise ValueError("source=evaluation_dimensions implies class=evaluation_dimension")
        if source.startswith("adjacent-skill rule") and classification != CLASS_ADJACENT:
            raise ValueError("source=adjacent-skill rule implies class=adjacent/transferable")
        if component.get("matched") and not component.get("evidence"):
            raise ValueError("supported components require evidence")
        if normalize_token(str(component.get("term", ""))) == "github":
            for evidence in component.get("evidence", []) or []:
                notes = normalize_token(str(evidence.get("notes", "")))
                if _contains_phrase(notes, "exact token phrase git"):
                    raise ValueError("Git evidence cannot be labelled an exact GitHub match")


def _require_terms_represented(
    terms: list[str],
    components: list[dict],
    *,
    source: str,
    criterion_id: str,
) -> None:
    represented = {
        normalize_token(str(component.get("term", "")))
        for component in components
        if source in str(component.get("source", ""))
    }
    missing = [term for term in terms if normalize_token(term) not in represented]
    if missing:
        raise ValueError(f"{source} terms disappeared from {criterion_id}: {', '.join(missing)}")


def _validate_linked_experience_consistency(
    criterion: ScoringCriterion,
    evaluation: CriterionEvaluation,
    eligibility: "EligibilityResult",
) -> None:
    details = {detail.get("requirement_id"): detail for detail in eligibility.requirement_details}
    for requirement_id in criterion.requirement_ids:
        detail = details.get(requirement_id)
        if not detail or "exp" not in requirement_id.lower():
            continue
        required_months = detail.get("required_months")
        if required_months is None:
            continue
        linked_components = [
            component
            for component in evaluation.component_assessment.get("components", [])
            if requirement_id in str(component.get("source", ""))
        ]
        if not linked_components:
            continue
        if detail.get("status") == "met" and any(not component.get("matched") for component in linked_components):
            raise ValueError(
                f"Linked experience criterion {evaluation.criterion_id} contradicts met requirement {requirement_id}."
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
