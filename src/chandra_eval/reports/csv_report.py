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
                "status",
                "match_fraction",
                "weighted_score",
                "explanation",
                "evidence_found",
                "missing_information",
                "verification_needed",
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
                    "status": item.status.value,
                    "match_fraction": item.match_fraction,
                    "weighted_score": item.weighted_score,
                    "explanation": item.explanation,
                    "evidence_found": " | ".join(item.evidence_found),
                    "missing_information": " | ".join(item.missing_information),
                    "verification_needed": " | ".join(item.verification_needed),
                }
            )
    return path
