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
    for label, key in (
        ("core", "supported_core_components"),
        ("adjacent", "supported_adjacent_components"),
        ("controlled", "controlled_equivalents_used"),
        ("missing core", "missing_core_components"),
    ):
        values = assessment.get(key) or []
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    return " | ".join(parts)
