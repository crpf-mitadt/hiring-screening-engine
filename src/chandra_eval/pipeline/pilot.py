"""Synthetic end-to-end demo runner."""

from pathlib import Path

from chandra_eval.ocr.full_document import write_ocr_artifacts
from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files
from chandra_eval.pipeline.jd_structuring import structure_jd_text_file
from chandra_eval.pipeline.resume_structuring import structure_resume_from_ocr
from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def run_synthetic_pilot(root: Path, output_dir: Path) -> CandidateEvaluationReport:
    ocr_dir = output_dir / "ocr" / "candidate-001"
    candidate_path = output_dir / "candidate_profile.json"
    jd_text_path = root / "job_description.txt"
    structured_jd_path = output_dir / "structured_jd.json"
    html = (root / "chandra_page.html").read_text(encoding="utf-8")
    write_ocr_artifacts(
        responses=[
            {
                "model": "chandra-ocr-2-q8",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": html},
                    }
                ],
            }
        ],
        output_dir=ocr_dir,
        source_pdf="synthetic-resume.pdf",
        model="chandra-ocr-2-q8",
    )
    structure_resume_from_ocr(
        ocr_dir,
        candidate_id="candidate-001",
        output_path=candidate_path,
        mock_llm=True,
    )
    structure_jd_text_file(
        jd_text_path,
        "job-synthetic-001",
        structured_jd_path,
        mock_llm=True,
    )
    return evaluate_candidate_files(candidate_path, structured_jd_path, output_dir)
