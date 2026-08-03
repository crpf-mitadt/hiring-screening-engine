"""CSV report writer."""

import csv
from pathlib import Path

from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def write_csv_report(report: CandidateEvaluationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "evaluation_id",
                "candidate_id",
                "job_id",
                "criterion_id",
                "criterion_name",
                "category",
                "weight",
                "linked_requirements",
                "status",
                "match_fraction",
                "weighted_score",
                "explanation",
                "component_assessment",
                "evidence_found",
                "missing_information",
                "verification_needed",
                "report_warnings",
            ],
        )
        writer.writeheader()
        for item in report.criterion_evaluations:
            writer.writerow(
                {
                    "evaluation_id": report.evaluation_id,
                    "candidate_id": report.candidate_id,
                    "job_id": report.job_id,
                    "criterion_id": item.criterion_id,
                    "criterion_name": item.criterion_name,
                    "category": item.category,
                    "weight": item.weight,
                    "linked_requirements": " | ".join(item.requirement_ids),
                    "status": item.status.value,
                    "match_fraction": item.match_fraction,
                    "weighted_score": item.weighted_score,
                    "explanation": item.explanation,
                    "component_assessment": _component_summary(item.component_assessment),
                    "evidence_found": " | ".join(item.evidence_found),
                    "missing_information": " | ".join(item.missing_information),
                    "verification_needed": " | ".join(item.verification_needed),
                    "report_warnings": " | ".join(report.warnings),
                }
            )
    return path


def _component_summary(assessment: dict) -> str:
    parts: list[str] = []
    parts.extend(_matched_component_parts(assessment))
    for label, key in (
        ("required/core", "required_core_components"),
        ("core", "supported_core_components"),
        ("preferred considered", "preferred_components"),
        ("preferred", "supported_preferred_components"),
        ("source adjacent considered", "source_declared_adjacent_components"),
        ("inferred adjacent", "inferred_adjacent_components"),
        ("adjacent", "supported_adjacent_components"),
        ("controlled", "controlled_equivalents_used"),
        ("dimensions considered", "evaluation_dimension_components"),
        ("evaluation dimensions", "supported_evaluation_dimensions"),
        ("missing core", "missing_core_components"),
        ("missing preferred", "missing_preferred_components"),
    ):
        values = assessment.get(key) or []
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    parts.extend(_missing_dimension_parts(assessment))
    return " | ".join(parts)


def _matched_component_parts(assessment: dict) -> list[str]:
    matched = []
    seen: set[tuple[str, str]] = set()
    for component in assessment.get("components") or []:
        if not component.get("matched"):
            continue
        term = str(component.get("term") or "").strip()
        if not term:
            continue
        classification = str(component.get("classification") or component.get("class") or "component").strip()
        key = (term, classification)
        if key in seen:
            continue
        seen.add(key)
        evidence_count = component.get("evidence_count")
        if evidence_count is None:
            evidence_count = len(component.get("evidence") or [])
        matched.append(f"{term} [{classification}; evidence={evidence_count}]")
    return [f"matched components: {', '.join(matched)}"] if matched else []


def _missing_dimension_parts(assessment: dict) -> list[str]:
    labels = assessment.get("evaluation_dimension_gap_labels") or {}
    grouped: dict[str, list[str]] = {}
    for term in assessment.get("missing_evaluation_dimensions") or []:
        label = labels.get(term) or _dimension_gap_label(str(assessment.get("category", "")))
        grouped.setdefault(label, []).append(term)
    return [f"{label}: {', '.join(values)}" for label, values in grouped.items()]


def _dimension_gap_label(category: str) -> str:
    if category == "skill":
        return "unverified technical dimension"
    if category == "experience":
        return "unverified experience dimension"
    if category == "leadership":
        return "unverified soft-skill dimension"
    return "unverified evaluation dimension"
