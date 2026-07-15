import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.ocr.full_document import write_ocr_artifacts
from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files
from chandra_eval.pipeline.jd_structuring import structure_jd_document_file
from chandra_eval.pipeline.resume_structuring import structure_resume_from_ocr


class PipelineContractTests(unittest.TestCase):
    def test_mock_pipeline_contract_runs_without_manual_file_rewrites(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocr_dir = root / "ocr" / "candidate-001"
            jd_path = root / "job.md"
            structured_jd = root / "jd.json"
            candidate_json = root / "candidate.json"
            evaluation_json = root / "evaluation.json"

            write_ocr_artifacts(
                responses=[
                    {
                        "model": "chandra-ocr-2-q8",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": '<p data-label="Text" data-bbox="[0,0,1,1]">Skills: Python, OCR, data validation</p>'
                                },
                            }
                        ],
                    }
                ],
                output_dir=ocr_dir,
                source_pdf="synthetic.pdf",
                model="chandra-ocr-2-q8",
            )
            jd_path.write_text(
                "Title: OCR Engineer\n"
                "Required skills: Python, OCR, data validation\n"
                "Preferred skills: Pydantic\n"
                "Minimum experience: 1 year\n",
                encoding="utf-8",
            )

            structure_jd_document_file(jd_path, "ocr-engineer", structured_jd, mock_llm=True)
            structure_resume_from_ocr(
                ocr_dir,
                candidate_id="candidate-001",
                output_path=candidate_json,
                mock_llm=True,
            )
            report = evaluate_candidate_files(candidate_json, structured_jd, evaluation_json)

            self.assertTrue(evaluation_json.is_file())
            self.assertTrue((root / "evaluation.csv").is_file())
            self.assertTrue((root / "evaluation.html").is_file())
            self.assertEqual(report.candidate_id, "candidate-001")
            self.assertTrue(report.human_review_required)
            saved = json.loads(evaluation_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["job_id"], "ocr-engineer")


if __name__ == "__main__":
    unittest.main()
