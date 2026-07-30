"""Source-backed experience qualification helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.scoring.matching import _contains_phrase, _dedupe_evidence, _mark_evidence
from chandra_eval.scoring.normalization import normalize_token
from chandra_eval.schemas.common import EvidenceRef
from chandra_eval.schemas.resume import ExperienceEntry, ScoringSafeProfile


_DATE_EVIDENCE_RE = re.compile(
    r"(\d{1,2}\s*[-/]\s*\d{4}|\d{4}\s*-\s*\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}|"
    r"present|current|now)",
    flags=re.IGNORECASE,
)
_ROLE_TOKENS = ("developer", "engineer", "software", "full stack", "fullstack")
_TECH_RELEVANCE_TOKENS = (
    "full stack",
    "fullstack",
    "software",
    "developer",
    "development",
    "node",
    "node.js",
    "express",
    "express.js",
    "react",
    "react.js",
    "vue",
    "vue.js",
    "rest api",
    "api",
    "frontend",
    "backend",
    "mysql",
    "postgresql",
    "postgres",
    "database",
)


@dataclass(frozen=True)
class ExperienceQualification:
    required_months: int
    confirmed_months: int
    fraction: float
    relevant_entries: list[ExperienceEntry] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    ambiguity: list[str] = field(default_factory=list)
    explanation: str = ""


def qualify_quantitative_experience(
    candidate: ScoringSafeProfile,
    *,
    required_months: int,
    requirement_text: str,
) -> ExperienceQualification:
    relevant_entries: list[ExperienceEntry] = []
    relevance_evidence: list[EvidenceRef] = []
    for entry in candidate.experience:
        entry_relevance_evidence = _entry_relevance_evidence(entry, requirement_text)
        if entry_relevance_evidence:
            relevant_entries.append(entry)
            relevance_evidence.extend(entry_relevance_evidence)

    duration = calculate_experience_duration(relevant_entries)
    evidence: list[EvidenceRef] = []
    for entry in relevant_entries:
        evidence.extend(_entry_basis_evidence(entry, requirement_text))
    evidence = _dedupe_evidence(evidence)

    missing: list[str] = []
    ambiguity = list(duration.date_warnings)
    if not relevant_entries:
        ambiguity.append("No source-backed relevant role evidence was found.")
    if duration.confirmed_months < required_months:
        missing.append(f"{required_months - duration.confirmed_months} additional months of relevant experience")
    if duration.confirmed_months >= required_months and not evidence:
        ambiguity.append("Duration met the numeric threshold but lacks traceable role/date/relevance evidence.")

    fraction = min(duration.confirmed_months / required_months, 1.0) if required_months else 0.0
    if not evidence:
        fraction = 0.0
    elif duration.confirmed_months >= required_months and not _has_relevance_evidence(evidence):
        fraction = min(fraction, 0.5)

    explanation = _experience_explanation(
        required_months=required_months,
        confirmed_months=duration.confirmed_months,
        entries=relevant_entries,
        evidence=evidence,
        missing=missing,
        ambiguity=ambiguity,
    )
    return ExperienceQualification(
        required_months=required_months,
        confirmed_months=duration.confirmed_months,
        fraction=fraction,
        relevant_entries=relevant_entries,
        evidence=evidence,
        missing=missing,
        ambiguity=ambiguity,
        explanation=explanation,
    )


def _entry_relevance_evidence(entry: ExperienceEntry, requirement_text: str) -> list[EvidenceRef]:
    requirement = normalize_token(requirement_text)
    required_terms = _relevance_terms(requirement)
    entry_text = normalize_token(
        " ".join(
            item
            for item in (
                entry.title,
                entry.organization,
                entry.industry_or_domain,
                *entry.responsibilities,
                *entry.achievements,
                *entry.skills_used,
            )
            if item
        )
    )
    has_role_signal = any(_contains_phrase(entry_text, token) for token in _ROLE_TOKENS)
    has_required_signal = any(_contains_phrase(entry_text, token) for token in required_terms)
    if not (has_role_signal and has_required_signal):
        return []
    return _mark_evidence(
        [
            item
            for item in entry.evidence
            if any(_contains_phrase(normalize_token(item.normalized_text or item.exact_text), token) for token in required_terms)
        ],
        "Relevant experience evidence",
    )


def _entry_basis_evidence(entry: ExperienceEntry, requirement_text: str) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    output.extend(_field_evidence(entry.evidence, entry.title, "Role title evidence"))
    output.extend(_field_evidence(entry.evidence, entry.organization, "Organization evidence"))
    output.extend(_date_evidence(entry.evidence))
    output.extend(_entry_relevance_evidence(entry, requirement_text))
    return _dedupe_evidence(output)


def _field_evidence(evidence: list[EvidenceRef], value: str | None, note: str) -> list[EvidenceRef]:
    if not value:
        return []
    normalized = normalize_token(value)
    return _mark_evidence(
        [
            item
            for item in evidence
            if _contains_phrase(normalize_token(item.normalized_text or item.exact_text), normalized)
        ],
        note,
    )


def _date_evidence(evidence: list[EvidenceRef]) -> list[EvidenceRef]:
    return _mark_evidence(
        [
            item
            for item in evidence
            if _DATE_EVIDENCE_RE.search(item.normalized_text or item.exact_text)
        ],
        "Role date evidence",
    )


def _relevance_terms(requirement_text: str) -> list[str]:
    terms = [term for term in _TECH_RELEVANCE_TOKENS if _contains_phrase(requirement_text, term)]
    if "full stack" in requirement_text or "fullstack" in requirement_text:
        terms.extend(["full stack", "fullstack", "frontend", "backend", "api"])
        return sorted(set(terms))
    if "development" in requirement_text or "developer" in requirement_text:
        terms.extend(["software", "developer", "development"])
    return sorted(set(terms)) or ["software", "developer", "development"]


def _has_relevance_evidence(evidence: list[EvidenceRef]) -> bool:
    return any(item.notes and "Relevant experience evidence" in item.notes for item in evidence)


def _experience_explanation(
    *,
    required_months: int,
    confirmed_months: int,
    entries: list[ExperienceEntry],
    evidence: list[EvidenceRef],
    missing: list[str],
    ambiguity: list[str],
) -> str:
    if entries and evidence:
        role = entries[0]
        title = role.title or "role"
        start = role.start_date or "unknown start"
        end = "the evaluation date" if role.currently_active else (role.end_date or "unknown end")
        return (
            f"{confirmed_months} confirmed months from {title}, {start} to {end}, "
            f"against {required_months} required months, supported by role/date/relevance evidence."
        )
    if missing:
        return f"Relevant confirmed experience is below the {required_months}-month requirement."
    if ambiguity:
        return "Relevant experience could not be confirmed from traceable source evidence."
    return "No source-backed relevant experience evidence was found."
