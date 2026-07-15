"""Synthetic end-to-end demo runner."""

from pathlib import Path

from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files
from chandra_eval.pipeline.jd_structuring import structure_jd_text_file
from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def run_synthetic_pilot(root: Path, output_dir: Path) -> CandidateEvaluationReport:
    candidate_path = root / "candidate_profile.json"
    jd_text_path = root / "job_description.txt"
    structured_jd_path = output_dir / "structured_jd.json"
    structure_jd_text_file(jd_text_path, "job-synthetic-001", structured_jd_path)
    return evaluate_candidate_files(candidate_path, structured_jd_path, output_dir)
