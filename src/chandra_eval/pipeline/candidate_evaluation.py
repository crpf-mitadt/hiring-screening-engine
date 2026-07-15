"""Candidate evaluation pipeline."""

from pathlib import Path

from chandra_eval.ingestion.jd_document import load_structured_jd
from chandra_eval.ingestion.resume_document import load_candidate_profile
from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def evaluate_candidate_files(
    candidate_path: Path,
    jd_path: Path,
    output_dir: Path,
) -> CandidateEvaluationReport:
    candidate = load_candidate_profile(candidate_path)
    jd = load_structured_jd(jd_path)
    if candidate.scoring_safe_profile is None:
        raise RuntimeError("Candidate profile did not produce a scoring-safe profile")
    report = evaluate_candidate(candidate.scoring_safe_profile, jd)
    write_json_report(report, output_dir / "evaluation.json")
    write_csv_report(report, output_dir / "evaluation.csv")
    write_html_report(report, output_dir / "evaluation.html")
    return report
