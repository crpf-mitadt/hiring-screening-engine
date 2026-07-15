"""Evidence-aware deterministic matching for candidate/JD comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from chandra_eval.scoring.normalization import normalize_token
from chandra_eval.schemas.common import EvidenceRef, EvidenceStatus
from chandra_eval.schemas.jd import RequirementCategory, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import EducationEntry, ExperienceEntry, ScoringSafeProfile


_STOPWORDS = {
    "and",
    "or",
    "the",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "role",
    "candidate",
    "requirement",
    "required",
    "preferred",
    "minimum",
    "expertise",
    "experience",
    "skill",
    "skills",
    "knowledge",
    "management",
    "support",
    "proficiency",
}

_ALIASES: dict[str, tuple[str, ...]] = {
    "ms office": ("excel", "word", "powerpoint", "presentation", "report"),
    "microsoft office": ("excel", "word", "powerpoint", "presentation", "report"),
    "ms excel": ("excel", "spreadsheet"),
    "ms powerpoint": ("powerpoint", "presentation", "client ready presentation"),
    "customer relationship management": ("crm", "lead database", "prospect database"),
    "crm management": ("crm", "lead database", "crm record", "pipeline management"),
    "crm": ("crm", "lead database", "prospect database", "sales opportunit", "pipeline"),
    "lead generation": ("lead generation", "lead qualification", "client acquisition", "demand generation"),
    "lead generation and qualification": ("lead generation", "lead qualification", "client acquisition"),
    "marketing campaign management": ("campaign management", "campaign analysis", "campaign execution"),
    "campaign management": ("campaign analysis", "campaign execution", "marketing campaign"),
    "social media management": ("social media marketing", "social media"),
    "social media marketing": ("social media management", "social media"),
    "event planning": ("event management", "event planning", "fair", "activation"),
    "pipeline tracking": ("sales pipeline", "tracked sales opportunit", "pipeline management"),
    "sales pipeline management": ("sales pipeline", "pipeline tracking", "tracked sales opportunit"),
    "content creation": ("proposal development", "content", "presentation"),
    "business development": ("client acquisition", "demand generation", "proposal development"),
    "multilingual communication": ("english", "hindi", "marathi", "communication"),
}


@dataclass(frozen=True)
class EvidenceText:
    text: str
    normalized: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    source: str = ""


@dataclass(frozen=True)
class TermMatch:
    term: str
    matched: bool
    evidence: list[EvidenceRef] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class MatchResult:
    terms: list[str]
    matches: list[TermMatch]
    missing_terms: list[str]
    evidence: list[EvidenceRef]
    ambiguity: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        if not self.terms:
            return 0.0
        return len([item for item in self.matches if item.matched]) / len(self.terms)


def build_candidate_evidence_index(candidate: ScoringSafeProfile) -> list[EvidenceText]:
    items: list[EvidenceText] = []
    _append(items, candidate.summary, [], "summary")
    for education in candidate.education:
        evidence = education.evidence
        _append_education(items, education, evidence)
    for experience in candidate.experience:
        evidence = experience.evidence
        _append_experience(items, experience, evidence)
    for source, values in (
        ("skill", candidate.skills),
        ("certification", candidate.certifications),
        ("project", candidate.projects),
        ("publication", candidate.publications),
        ("teaching", candidate.teaching_experience),
        ("research", candidate.research_experience),
        ("leadership", candidate.leadership_experience),
        ("achievement", candidate.achievements),
        ("language", candidate.languages),
    ):
        for value in values:
            _append(items, value, _synthetic_evidence(candidate, value, source), source)
    return items


def criterion_terms(criterion: ScoringCriterion, jd: StructuredJD) -> list[str]:
    seed_text = " ".join(
        [
            criterion.name,
            criterion.description,
            criterion.category.value,
            criterion.scoring_guidance,
            *criterion.accepted_evidence,
        ]
    )
    terms = list(criterion.accepted_evidence)
    relevant_required = [
        skill for skill in jd.required_skills if _phrase_relevant(skill, seed_text)
    ]
    relevant_preferred = [
        skill for skill in jd.preferred_skills if _phrase_relevant(skill, seed_text)
    ]
    relevant_requirements = [
        requirement.description
        for requirement in [*jd.mandatory_requirements, *jd.preferred_requirements]
        if requirement.category == criterion.category or _phrase_relevant(requirement.description, seed_text)
    ]
    terms.extend(relevant_requirements)
    terms.extend(relevant_required)
    if "preferred" in normalize_token(seed_text):
        terms.extend(relevant_preferred)
    else:
        terms.extend(skill for skill in relevant_preferred if _phrase_relevant(seed_text, skill))
    if criterion.category == RequirementCategory.SKILL and criterion.mandatory and not relevant_required:
        terms.extend(jd.required_skills)
    if criterion.category == RequirementCategory.SKILL and not criterion.mandatory and "preferred" in normalize_token(seed_text):
        terms.extend(jd.preferred_skills)
    if not terms:
        terms.extend(_important_phrases(seed_text))
    return _dedupe_terms(terms)


def match_terms(terms: list[str], index: list[EvidenceText]) -> MatchResult:
    matches: list[TermMatch] = []
    all_evidence: list[EvidenceRef] = []
    missing: list[str] = []
    for term in _dedupe_terms(terms):
        term_match = _match_term(term, index)
        if term_match.matched:
            all_evidence.extend(term_match.evidence)
        else:
            missing.append(term)
        matches.append(term_match)
    return MatchResult(
        terms=_dedupe_terms(terms),
        matches=matches,
        missing_terms=missing,
        evidence=_dedupe_evidence(all_evidence),
    )


def match_requirement(description: str, index: list[EvidenceText], equivalents: list[str] | None = None) -> MatchResult:
    terms = _dedupe_terms([description, *(equivalents or []), *_important_phrases(description)])
    return match_terms(terms[: max(1, len(terms))], index)


def _append_education(items: list[EvidenceText], education: EducationEntry, evidence: list[EvidenceRef]) -> None:
    values = [
        education.degree_level.value,
        education.degree_name,
        education.field_of_study,
        education.institution,
        education.completion_status.value,
        education.grade,
    ]
    for value in values:
        _append(items, value, evidence, "education")


def _append_experience(items: list[EvidenceText], experience: ExperienceEntry, evidence: list[EvidenceRef]) -> None:
    values = [
        experience.title,
        experience.organization,
        experience.industry_or_domain,
        experience.employment_type,
        *experience.responsibilities,
        *experience.achievements,
        *experience.skills_used,
    ]
    for value in values:
        _append(items, value, evidence, "experience")


def _append(items: list[EvidenceText], value: str | None, evidence: list[EvidenceRef], source: str) -> None:
    if not value:
        return
    text = str(value).strip()
    if not text:
        return
    items.append(EvidenceText(text=text, normalized=normalize_token(text), evidence=evidence, source=source))


def _match_term(term: str, index: list[EvidenceText]) -> TermMatch:
    normalized = normalize_token(term)
    options = _term_options(normalized)
    for item in index:
        for option in options:
            if _contains_phrase(item.normalized, option):
                return TermMatch(
                    term=term,
                    matched=True,
                    evidence=_mark_evidence(item.evidence, f"Matched '{term}' via '{option}'"),
                    reason=f"matched '{option}' in {item.source}: {item.text}",
                )
    semantic = _semantic_match(normalized, index)
    if semantic is not None:
        return TermMatch(term=term, matched=True, evidence=semantic[0], reason=semantic[1])
    return TermMatch(term=term, matched=False)


def _semantic_match(normalized: str, index: list[EvidenceText]) -> tuple[list[EvidenceRef], str] | None:
    patterns: list[tuple[set[str], str]] = []
    if any(token in normalized for token in ("crm", "pipeline", "database")):
        patterns.append(({"lead", "database"}, "responsibility supports CRM through lead database work"))
        patterns.append(({"sales", "opportunit"}, "responsibility supports CRM through sales opportunity tracking"))
        patterns.append(({"crm", "record"}, "responsibility directly supports CRM records"))
    if "office" in normalized or "excel" in normalized or "powerpoint" in normalized:
        patterns.append(({"excel"}, "direct MS Office evidence"))
        patterns.append(({"powerpoint"}, "direct presentation software evidence"))
        patterns.append(({"presentation"}, "presentation/reporting evidence supports office productivity"))
    if "lead generation" in normalized or "outreach" in normalized or "campaign" in normalized:
        patterns.append(({"client", "acquisition"}, "client acquisition supports lead generation/outreach"))
        patterns.append(({"demand", "generation"}, "demand generation supports lead generation/outreach"))
        patterns.append(({"campaign", "analysis"}, "campaign analysis supports marketing campaign work"))
        patterns.append(({"social", "media"}, "social media marketing supports outreach"))
        patterns.append(({"stakeholder", "engagement"}, "stakeholder engagement supports outreach"))
    if "graduate" in normalized or "degree" in normalized or "education" in normalized:
        patterns.append(({"mba"}, "MBA evidence supports postgraduate education but specialization remains reviewable"))
        patterns.append(({"master"}, "master degree evidence supports postgraduate education"))
        patterns.append(({"bachelor"}, "bachelor degree evidence supports graduate education"))
    for item in index:
        words = set(item.normalized.split())
        for required, reason in patterns:
            if required.issubset(words):
                return _mark_evidence(item.evidence, reason), f"{reason}: {item.text}"
    return None


def _term_options(normalized: str) -> list[str]:
    options = [normalized]
    options.extend(_ALIASES.get(normalized, ()))
    for alias, equivalents in _ALIASES.items():
        if normalized == alias or normalized in equivalents:
            options.append(alias)
            options.extend(equivalents)
    return _dedupe_terms(options)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return phrase in text


def _phrase_relevant(phrase: str, seed_text: str) -> bool:
    phrase_tokens = _meaningful_tokens(phrase)
    seed_tokens = _meaningful_tokens(seed_text)
    if phrase_tokens & seed_tokens:
        return True
    phrase_norm = normalize_token(phrase)
    seed_norm = normalize_token(seed_text)
    return any(option in seed_norm for option in _term_options(phrase_norm))


def _important_phrases(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"[,;/]|\band\b|\bor\b", text, flags=re.IGNORECASE)]
    return [chunk for chunk in chunks if len(_meaningful_tokens(chunk)) >= 1][:8]


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalize_token(value).split()
        if len(token) > 2 and token not in _STOPWORDS
    }


def _dedupe_terms(terms: list[str] | tuple[str, ...]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = str(term).strip()
        normalized = normalize_token(cleaned)
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        output.append(cleaned)
    return output


def _dedupe_evidence(items: list[EvidenceRef]) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    seen: set[tuple] = set()
    for item in items:
        key = (
            item.source_document_id,
            item.page_number,
            item.block_index,
            item.exact_text,
            item.notes,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _mark_evidence(items: list[EvidenceRef], reason: str) -> list[EvidenceRef]:
    if not items:
        return []
    return [
        item.model_copy(
            update={
                "evidence_status": EvidenceStatus.SUPPORTED,
                "notes": f"{item.notes}; {reason}" if item.notes else reason,
            }
        )
        for item in items
    ]


def _synthetic_evidence(candidate: ScoringSafeProfile, text: str, source: str) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            source_type="structured_resume",
            source_document_id=candidate.candidate_id,
            exact_text=text,
            normalized_text=text,
            evidence_status=EvidenceStatus.SUPPORTED,
            notes=f"Derived from scoring-safe {source}",
        )
    ]
