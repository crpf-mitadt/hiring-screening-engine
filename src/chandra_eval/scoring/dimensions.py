"""Deterministic evaluation-dimension classification and sharing rules."""

from __future__ import annotations

import re

from chandra_eval.scoring.matching import _contains_phrase
from chandra_eval.scoring.normalization import normalize_token


TECH_STACK_BREADTH_THRESHOLD = 2
TECH_STACK_BREADTH_TERMS = ("React.js", "Node.js", "Express")

_SOFT_SKILL_SIGNALS = (
    "soft",
    "soft skill",
    "communication",
    "communication clarity",
    "initiative",
    "ownership",
    "proactivity",
    "leadership",
    "collaboration",
    "stakeholder communication",
    "problem solving",
    "problem-solving",
)

_TECHNICAL_DIMENSION_TERMS = (
    "code quality",
    "database architecture",
    "cms customization",
)


def canonical_technical_breadth_term(term: str) -> str | None:
    raw = term.lower()
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    normalized = normalize_token(term)
    if "react" in raw or "reactjs" in compact or _contains_phrase(normalized, "react"):
        return "React.js"
    if "node" in raw or "nodejs" in compact or _contains_phrase(normalized, "node"):
        return "Node.js"
    if "express" in raw or "expressjs" in compact or _contains_phrase(normalized, "expres"):
        return "Express"
    return None


def dimension_domain(
    *,
    category: str,
    criterion_name: str = "",
    requirement_ids: list[str] | tuple[str, ...] = (),
    dimension: str = "",
    explicit_domain: str | None = None,
) -> str:
    if explicit_domain in {"technical", "experience", "soft_skill", "generic"}:
        return explicit_domain

    linked_text = normalize_token(" ".join(requirement_ids))
    if _has_soft_signal(linked_text):
        return "soft_skill"
    if _contains_phrase(linked_text, "experience") or _contains_phrase(linked_text, "exp"):
        return "experience"

    dimension_text = normalize_token(dimension)
    if is_experience_dimension(dimension_text):
        return "experience"
    if _is_known_technical_dimension(dimension_text):
        return "technical"

    criterion_text = normalize_token(criterion_name)
    if _has_soft_signal(criterion_text) or _has_soft_signal(dimension_text):
        return "soft_skill"

    if category == "experience":
        return "experience"
    if category == "leadership":
        return "soft_skill"
    if category == "skill":
        return "technical"
    return "generic"


def dimension_gap_label(domain: str) -> str:
    if domain == "technical":
        return "unverified technical dimension"
    if domain == "experience":
        return "unverified experience dimension"
    if domain == "soft_skill":
        return "unverified soft-skill dimension"
    return "unverified evaluation dimension"


def is_experience_dimension(normalized_or_text: str) -> bool:
    normalized = normalize_token(normalized_or_text)
    return (
        (_contains_phrase(normalized, "year") or _contains_phrase(normalized, "duration"))
        and _contains_phrase(normalized, "experience")
    )


def is_tech_stack_versatility_dimension(normalized_or_text: str) -> bool:
    normalized = normalize_token(normalized_or_text)
    return (
        _contains_phrase(normalized, "tech stack versatility")
        or _contains_phrase(normalized, "technical versatility")
        or (_contains_phrase(normalized, "versatility") and _contains_phrase(normalized, "stack"))
        or _contains_phrase(normalized, "technical breadth")
    )


def _has_soft_signal(normalized: str) -> bool:
    return any(_contains_phrase(normalized, normalize_token(term)) for term in _SOFT_SKILL_SIGNALS)


def _is_known_technical_dimension(normalized: str) -> bool:
    return any(_contains_phrase(normalized, normalize_token(term)) for term in _TECHNICAL_DIMENSION_TERMS)
