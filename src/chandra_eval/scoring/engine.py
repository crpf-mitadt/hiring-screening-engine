"""Deterministic candidate-to-JD scoring engine."""

from dataclasses import dataclass, field
import hashlib
import json
from uuid import uuid4

from chandra_eval.scoring.eligibility import evaluate_eligibility
from chandra_eval.scoring.education import evaluate_education_requirement
from chandra_eval.scoring.evidence_completeness import evidence_completeness, evidence_completeness_diagnostics
from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.experience_basis import qualify_quantitative_experience
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
        "communicated",
        "communication",
        "presented",
        "stakeholder",
        "client communication",
        "cross functional",
        "documentation",
    ),
    "ownership": (
        "owned",
        "ownership",
        "led",
        "lead",
        "accountable",
        "end to end",
        "independently delivered",
    ),
    "proactivity": (
        "proactively",
        "initiative",
        "initiated",
        "improved",
        "optimized",
        "automated",
        "identified",
        "anticipated",
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
    eligibility: "EligibilityResult",
) -> CriterionEvaluation:
    missing: list[str] = []
    ambiguity: list[str] = []
    fraction = 0.0
    explanation = ""
    index = build_candidate_evidence_index(candidate)
    verification_needed = list(criterion.accepted_evidence)

    if criterion.evaluation_dimensions and criterion.category in {
        RequirementCategory.SKILL,
        RequirementCategory.LEADERSHIP,
        RequirementCategory.RESPONSIBILITY,
        RequirementCategory.OTHER,
    }:
        return _score_soft_skill_criterion(criterion, index, verification_needed)

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

    if criterion.category == RequirementCategory.SKILL:
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
    eligibility = evaluate_eligibility(candidate, jd)
    evaluations = [
        _score_criterion(criterion, candidate, jd, experience_summary=experience_summary, eligibility=eligibility)
        for criterion in criteria
    ]
    _validate_criterion_results(criteria, evaluations)
    raw_score = sum(item.weighted_score for item in evaluations) * (100 / total_weight)
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
    if criterion.evaluation_dimensions:
        return _score_soft_skill_criterion(criterion, index, verification_needed)

    components = _criterion_components(criterion, jd, eligibility, index)
    core_components = [item for item in components if item.component_class in {"required/core", "preferred/core"}]
    adjacent_components = [item for item in components if item.component_class == "adjacent/transferable"]
    supported_core = [item for item in core_components if item.matched]
    supported_adjacent = [item for item in adjacent_components if item.matched]
    missing_core = [item.term for item in core_components if not item.matched]
    evidence = _dedupe_evidence([evidence for item in [*supported_core, *supported_adjacent] for evidence in item.evidence])
    if core_components:
        core_fraction = len(supported_core) / len(core_components)
    else:
        core_fraction = 0.0
    adjacent_bonus = 0.25 if supported_adjacent else 0.0
    if core_fraction >= 1:
        raw_fraction = 1.0
    elif supported_core:
        raw_fraction = min(0.75, core_fraction + adjacent_bonus)
    elif supported_adjacent:
        raw_fraction = 0.25
    else:
        raw_fraction = 0.0
    fraction = discrete_match_fraction(raw_fraction)
    explanation = _component_explanation(supported_core, supported_adjacent, missing_core)
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        criterion_name=criterion.name,
        category=criterion.category.value,
        weight=criterion.weight,
        requirement_ids=criterion.requirement_ids,
        match_fraction=fraction,
        weighted_score=round(criterion.weight * fraction, 6),
        status=_status(fraction, has_terms=bool(core_components or adjacent_components)),
        explanation=explanation,
        evidence=evidence,
        missing_information=missing_core,
        ambiguity=[],
        verification_needed=verification_needed,
        candidate_requirement=criterion.description,
        evidence_found=_evidence_found(evidence),
        component_assessment=_component_assessment(
            criterion,
            core_components=core_components,
            adjacent_components=adjacent_components,
            evidence=evidence,
        ),
    )


def _score_soft_skill_criterion(
    criterion: ScoringCriterion,
    index: list[EvidenceText],
    verification_needed: list[str],
) -> CriterionEvaluation:
    components = [_match_soft_skill_dimension(dimension, index) for dimension in criterion.evaluation_dimensions]
    supported = [item for item in components if item.matched]
    missing = [item.term for item in components if not item.matched]
    evidence = _dedupe_evidence([evidence for item in supported for evidence in item.evidence])
    fraction = discrete_match_fraction(len(supported) / len(components)) if components else 0.0
    explanation = _component_explanation(supported, [], missing)
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id,
        criterion_name=criterion.name,
        category=criterion.category.value,
        weight=criterion.weight,
        requirement_ids=criterion.requirement_ids,
        match_fraction=fraction,
        weighted_score=round(criterion.weight * fraction, 6),
        status=_status(fraction, has_terms=bool(components)),
        explanation=explanation,
        evidence=evidence,
        missing_information=missing,
        ambiguity=[],
        verification_needed=verification_needed,
        candidate_requirement=criterion.description,
        evidence_found=_evidence_found(evidence),
        component_assessment=_component_assessment(
            criterion,
            core_components=components,
            adjacent_components=[],
            evidence=evidence,
        ),
    )


def _criterion_components(
    criterion: ScoringCriterion,
    jd: StructuredJD,
    eligibility: "EligibilityResult",
    index: list[EvidenceText],
) -> list[ComponentEvaluation]:
    terms = _owned_core_terms(criterion, jd)
    components = [_match_component(term, index, source=_component_source(term, criterion), component_class=_core_class(criterion)) for term in terms]
    components = _merge_linked_requirement_components(components, criterion, jd, eligibility)
    components.extend(_adjacent_components(criterion, components, index))
    return components


def _owned_core_terms(criterion: ScoringCriterion, jd: StructuredJD) -> list[str]:
    terms: list[str] = []
    terms.extend(criterion.required_skill_names)
    terms.extend(criterion.preferred_skill_names)
    linked_ids = set(criterion.requirement_ids)
    for requirement in [*jd.mandatory_requirements, *jd.preferred_requirements]:
        if requirement.requirement_id in linked_ids and not _requirement_represented(requirement.description, terms):
            terms.extend(_linked_requirement_component_terms(requirement.description))
    if not terms and criterion.mandatory and _default_required_skill_criterion(criterion):
        terms.extend(jd.required_skills)
    if not terms and not criterion.mandatory and _default_preferred_skill_criterion(criterion):
        terms.extend(jd.preferred_skills)
    if not terms:
        terms.append(criterion.description or criterion.name)
    return _dedupe_terms(terms)


def _linked_requirement_component_terms(description: str) -> list[str]:
    normalized = normalize_token(description)
    if "database" in normalized or "mysql" in normalized or "postgres" in normalized or "sql" in normalized:
        return ["SQL/database management"]
    return [description]


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
                    component_class=_core_class(criterion),
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
    if not _allows_full_stack_adjacent(criterion):
        return []
    core_terms = [component.term for component in core_components]
    adjacent_terms = [
        term
        for term in _ADJACENT_FULL_STACK_TERMS
        if not any(_same_or_related_term(term, core_term) for core_term in core_terms)
    ]
    output: list[ComponentEvaluation] = []
    for term in adjacent_terms:
        match = match_terms([term], index)
        if match.evidence:
            output.append(
                ComponentEvaluation(
                    term=term,
                    source="adjacent-skill rule",
                    component_class="adjacent/transferable",
                    matched=True,
                    evidence=match.evidence,
                    note=f"{term} recognized as adjacent full-stack evidence, not a mandatory substitute",
                )
            )
    return output


def _match_soft_skill_dimension(dimension: str, index: list[EvidenceText]) -> ComponentEvaluation:
    normalized = normalize_token(dimension)
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
                    component_class="required/core",
                    matched=True,
                    evidence=evidence,
                    note=f"matched direct {dimension} evidence",
                )
    return ComponentEvaluation(term=dimension, source="evaluation_dimensions", component_class="required/core")


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


def _component_explanation(
    supported_core: list[ComponentEvaluation],
    supported_adjacent: list[ComponentEvaluation],
    missing_core: list[str],
) -> str:
    parts: list[str] = []
    direct = [item.term for item in supported_core if "linked requirement" not in item.source]
    linked = [item.term for item in supported_core if "linked requirement" in item.source]
    if direct:
        parts.append(f"Supported required/core components: {', '.join(direct[:6])}.")
    if linked:
        parts.append(f"Controlled or linked requirement evidence supported: {', '.join(linked[:6])}.")
    if supported_adjacent:
        parts.append(
            "Adjacent or transferable components: "
            + ", ".join(item.term for item in supported_adjacent[:6])
            + "."
        )
    if missing_core:
        parts.append(f"Missing required/core components: {', '.join(missing_core[:8])}.")
    if not parts:
        parts.append("No in-scope resume evidence was found for the criterion components.")
    return " ".join(parts)


def _component_assessment(
    criterion: ScoringCriterion,
    *,
    core_components: list[ComponentEvaluation],
    adjacent_components: list[ComponentEvaluation],
    evidence: list[EvidenceRef],
) -> dict:
    return {
        "criterion_id": criterion.criterion_id,
        "criterion_name": criterion.name,
        "category": criterion.category.value,
        "weight": criterion.weight,
        "requirement_ids": list(criterion.requirement_ids),
        "required_skill_names": list(criterion.required_skill_names),
        "preferred_skill_names": list(criterion.preferred_skill_names),
        "evaluation_dimensions": list(criterion.evaluation_dimensions),
        "source_criterion_hash": _source_criterion_hash(criterion),
        "required_core_components": [item.term for item in core_components if item.component_class == "required/core"],
        "controlled_equivalent_components": [item.term for item in core_components if _is_database_component(item.term)],
        "adjacent_components_considered": [item.term for item in adjacent_components],
        "components": [
            {
                "term": item.term,
                "source": item.source,
                "class": item.component_class,
                "matched": item.matched,
                "evidence_count": len(item.evidence),
                "evidence": [evidence_item.model_dump(mode="json") for evidence_item in item.evidence],
                "note": item.note,
            }
            for item in [*core_components, *adjacent_components]
        ],
        "supported_core_components": [item.term for item in core_components if item.matched],
        "supported_required_components": [item.term for item in core_components if item.matched],
        "missing_core_components": [item.term for item in core_components if not item.matched],
        "supported_adjacent_components": [item.term for item in adjacent_components if item.matched],
        "supported_controlled_equivalents": sorted(_controlled_equivalent_notes(evidence)),
        "controlled_equivalents_used": sorted(_controlled_equivalent_notes(evidence)),
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
        core_components=[
            ComponentEvaluation(
                term=term,
                source=source,
                component_class=_core_class(criterion),
                matched=matched,
                evidence=evidence if matched else [],
                note="criterion evaluated from source rubric definition",
            )
        ],
        adjacent_components=[],
        evidence=evidence if matched else [],
    )


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
        "evaluation_dimensions": criterion.evaluation_dimensions,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _core_class(criterion: ScoringCriterion) -> str:
    return "preferred/core" if criterion.preferred_skill_names and not criterion.required_skill_names else "required/core"


def _component_source(term: str, criterion: ScoringCriterion) -> str:
    if term in criterion.required_skill_names:
        return "required_skill_names"
    if term in criterion.preferred_skill_names:
        return "preferred_skill_names"
    return "linked requirement"


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


def _validate_criterion_results(criteria: list[ScoringCriterion], evaluations: list[CriterionEvaluation]) -> None:
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
