"""Canonical source-provenance representation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chandra_eval.scoring.dimensions import canonical_technical_breadth_term
from chandra_eval.scoring.normalization import normalize_token


SOURCE_FIELDS = (
    "required_skill_names",
    "preferred_skill_names",
    "adjacent_skill_names",
    "evaluation_dimensions",
    "required_skills",
    "preferred_skills",
)


def canonical_source_term(term: str) -> str:
    technical = canonical_technical_breadth_term(term)
    if technical is not None:
        return f"tech:{normalize_token(technical)}"
    return normalize_token(term)


def represented_source_fields(component: Any) -> list[str]:
    fields = list(_component_get(component, "source_fields", []) or [])
    source = str(_component_get(component, "source", "") or "")
    for field_name in SOURCE_FIELDS:
        if field_name in source:
            fields.append(field_name)
    if "linked requirement" in source:
        fields.append("linked requirement")
    if source.startswith("adjacent-skill rule"):
        fields.append("adjacent-skill rule")
    if "criterion description" in source:
        fields.append("criterion description")
    return _dedupe(fields)


def component_represents_source(component: Any, term: str, source_field: str) -> bool:
    if source_field not in represented_source_fields(component):
        return False
    source_terms = _component_get(component, "source_terms", {}) or {}
    candidates = list(source_terms.get(source_field, []) or [])
    candidates.append(str(_component_get(component, "term", "") or ""))
    return any(canonical_source_term(candidate) == canonical_source_term(term) for candidate in candidates)


def component_represents_requirement(component: Any, requirement_id: str) -> bool:
    requirement_ids = set(str(value) for value in (_component_get(component, "source_requirement_ids", []) or []))
    source = str(_component_get(component, "source", "") or "")
    return requirement_id in requirement_ids or requirement_id in source


@dataclass(frozen=True)
class RepresentationDiagnostic:
    term: str
    canonical_term: str
    source_field: str
    candidates: list[dict[str, Any]]

    def message(self, criterion_id: str) -> str:
        if not self.candidates:
            return (
                f"{criterion_id} lost {self.source_field}: {self.term} "
                f"(canonical={self.canonical_term}); no canonical-equivalent components"
            )
        return (
            f"{criterion_id} lost {self.source_field}: {self.term} "
            f"(canonical={self.canonical_term}); candidates={self.candidates}"
        )


class SourceRepresentationIndex:
    def __init__(self, components: list[Any]):
        self.components = components

    def term_is_represented(self, term: str, source_field: str) -> bool:
        return any(component_represents_source(component, term, source_field) for component in self.components)

    def requirement_is_represented(self, requirement_id: str) -> bool:
        return any(component_represents_requirement(component, requirement_id) for component in self.components)

    def missing_term_diagnostic(self, term: str, source_field: str) -> RepresentationDiagnostic:
        canonical = canonical_source_term(term)
        candidates = [
            _safe_component_summary(component)
            for component in self.components
            if canonical_source_term(str(_component_get(component, "term", "") or "")) == canonical
        ]
        return RepresentationDiagnostic(
            term=term,
            canonical_term=canonical,
            source_field=source_field,
            candidates=candidates,
        )


def _safe_component_summary(component: Any) -> dict[str, Any]:
    return {
        "term": _component_get(component, "term", ""),
        "class": _component_get(component, "classification", None) or _component_get(component, "class", None) or _component_get(component, "component_class", ""),
        "source": _component_get(component, "source", ""),
        "source_fields": represented_source_fields(component),
        "source_requirement_ids": _component_get(component, "source_requirement_ids", []) or [],
        "source_terms": _component_get(component, "source_terms", {}) or {},
        "matched": bool(_component_get(component, "matched", False)),
        "evidence_count": len(_component_get(component, "evidence", []) or []),
    }


def _component_get(component: Any, key: str, default: Any = None) -> Any:
    if isinstance(component, dict):
        return component.get(key, default)
    return getattr(component, key, default)


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        key = normalize_token(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output
