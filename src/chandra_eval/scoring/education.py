"""Deterministic education requirement checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from chandra_eval.scoring.matching import _mark_evidence
from chandra_eval.scoring.normalization import normalize_token
from chandra_eval.schemas.common import EvidenceRef
from chandra_eval.schemas.resume import CompletionStatus, DegreeLevel, EducationEntry


_TECHNICAL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "computer science": ("computer science", "cs"),
    "artificial intelligence": ("artificial intelligence", "ai"),
    "data science": ("data science",),
    "machine learning": ("machine learning",),
    "computer engineering": ("computer engineering",),
    "software engineering": ("software engineering",),
    "information technology": ("information technology", "it"),
    "statistics": ("statistics",),
    "mathematics": ("mathematics", "math"),
}

_EXECUTIVE_TERMS = ("executive", "certificate", "certification", "program", "leadership")


@dataclass(frozen=True)
class EducationRequirementMatch:
    matched: bool
    evidence: list[EvidenceRef] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    ambiguity: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def evaluate_education_requirement(
    description: str,
    education_entries: list[EducationEntry],
) -> EducationRequirementMatch:
    accepted_levels = _accepted_degree_levels(description)
    required_fields = _required_technical_fields(description)
    completion_required = _completion_required(description)
    best_notes: list[str] = []
    best_missing: list[str] = []
    best_ambiguity: list[str] = []

    for entry in education_entries:
        entry_text = _entry_text(entry)
        if _is_executive_program(entry):
            best_ambiguity.append(f"Executive/certificate program is not a degree equivalent: {entry_text}")
            continue
        level_ok = not accepted_levels or entry.degree_level in accepted_levels
        field_result = _field_match(entry.field_of_study, required_fields)
        completion_ok = not completion_required or entry.completion_status in {
            CompletionStatus.COMPLETED,
            CompletionStatus.PURSUING,
        }
        if level_ok and field_result[0] and completion_ok:
            notes = [
                f"Matched degree level {entry.degree_level.value}",
                f"matched field {entry.field_of_study or 'unspecified'} via {field_result[1]}",
            ]
            if completion_required:
                notes.append(f"completion status {entry.completion_status.value}")
            return EducationRequirementMatch(
                matched=True,
                evidence=_mark_evidence(entry.evidence, "; ".join(notes)),
                notes=notes,
            )
        missing: list[str] = []
        ambiguity: list[str] = []
        if not level_ok:
            missing.append(f"required degree level {', '.join(sorted(level.value for level in accepted_levels))}")
        if not field_result[0]:
            if entry.field_of_study:
                ambiguity.append(
                    f"Field mismatch: {entry.field_of_study} is not in the controlled technical field mapping."
                )
            else:
                missing.append("required technical field of study")
        if not completion_ok:
            missing.append("completed or pursuing status")
        if len(missing) + len(ambiguity) > len(best_missing) + len(best_ambiguity):
            best_notes = [f"Education entry checked: {entry_text}"]
            best_missing = missing
            best_ambiguity = ambiguity

    if not education_entries:
        best_missing = ["education evidence"]
    return EducationRequirementMatch(
        matched=False,
        missing=best_missing or ["education requirement components"],
        ambiguity=sorted(set(best_ambiguity)),
        notes=best_notes,
    )


def _accepted_degree_levels(description: str) -> set[DegreeLevel]:
    normalized = normalize_token(description)
    levels: set[DegreeLevel] = set()
    if "master" in normalized or "postgraduate" in normalized or "graduate degree" in normalized:
        levels.add(DegreeLevel.MASTER)
    if "bachelor" in normalized or "undergraduate" in normalized:
        levels.add(DegreeLevel.BACHELOR)
    if "doctor" in normalized or "phd" in normalized:
        levels.add(DegreeLevel.DOCTORATE)
    return levels


def _required_technical_fields(description: str) -> set[str]:
    normalized = normalize_token(description)
    fields: set[str] = set()
    for field_name, aliases in _TECHNICAL_FIELD_ALIASES.items():
        if any(_contains_token_phrase(normalized, normalize_token(alias)) for alias in aliases):
            fields.add(field_name)
    if "closely related" in normalized or "technical field" in normalized:
        fields.update(_TECHNICAL_FIELD_ALIASES)
    return fields


def _completion_required(description: str) -> bool:
    normalized = normalize_token(description)
    return any(token in normalized for token in ("holding", "pursuing", "completed", "degree"))


def _field_match(field: str | None, required_fields: set[str]) -> tuple[bool, str]:
    if not required_fields:
        return True, "no field restriction"
    normalized = normalize_token(field or "")
    if not normalized:
        return False, ""
    for field_name in sorted(required_fields):
        if any(
            _contains_token_phrase(normalized, normalize_token(alias))
            for alias in _TECHNICAL_FIELD_ALIASES[field_name]
        ):
            return True, field_name
    return False, ""


def _is_executive_program(entry: EducationEntry) -> bool:
    text = normalize_token(_entry_text(entry))
    if entry.degree_level not in {DegreeLevel.CERTIFICATION, DegreeLevel.UNKNOWN, DegreeLevel.PROFESSIONAL}:
        return False
    return any(_contains_token_phrase(text, term) for term in _EXECUTIVE_TERMS)


def _entry_text(entry: EducationEntry) -> str:
    return " ".join(
        item
        for item in (
            entry.degree_level.value,
            entry.degree_name,
            entry.field_of_study,
            entry.institution,
            entry.completion_status.value,
        )
        if item
    )


def _contains_token_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
