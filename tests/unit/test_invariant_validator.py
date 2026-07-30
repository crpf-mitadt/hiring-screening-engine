import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.schemas.jd import RequirementCategory, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import CandidateProfile


def _load_validator_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_evaluation_invariants.py"
    spec = importlib.util.spec_from_file_location("validate_evaluation_invariants", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InvariantValidatorTests(unittest.TestCase):
    def test_validator_reports_skipped_report_check_when_reports_are_not_supplied(self) -> None:
        validator = _load_validator_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = _sample_report()
            evaluation_path = write_json_report(report, root / "evaluation.json")

            code, output = self._run_validator(validator, ["--evaluation", str(evaluation_path)])

        self.assertEqual(code, 0)
        self.assertIn("SKIP: report consistency: report files not supplied", output)
        self.assertNotIn("PASS: report consistency", output)

    def test_validator_checks_supplied_reports_and_ocr_metadata(self) -> None:
        validator = _load_validator_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = _sample_report()
            evaluation_path = write_json_report(report, root / "evaluation.json")
            csv_path = write_csv_report(report, root / "evaluation.csv")
            html_path = write_html_report(report, root / "evaluation.html")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "raw_responses_persisted": False,
                        "artifact_contract_version": 2,
                        "pages": [{"page_number": 1, "finish_reason": "stop", "status": "completed"}],
                    }
                ),
                encoding="utf-8",
            )

            code, output = self._run_validator(
                validator,
                [
                    "--evaluation",
                    str(evaluation_path),
                    "--csv-report",
                    str(csv_path),
                    "--html-report",
                    str(html_path),
                    "--ocr-metadata",
                    str(metadata_path),
                ],
            )

        self.assertEqual(code, 0)
        self.assertIn("PASS: report consistency: checked CSV and HTML report consistency", output)
        self.assertIn("PASS: raw-response privacy: OCR metadata checked; raw OCR responses disabled", output)

    def test_validator_fails_supplied_unsafe_ocr_metadata_and_missing_report(self) -> None:
        validator = _load_validator_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = _sample_report()
            evaluation_path = write_json_report(report, root / "evaluation.json")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps({"raw_responses_persisted": True}),
                encoding="utf-8",
            )

            code, output = self._run_validator(
                validator,
                [
                    "--evaluation",
                    str(evaluation_path),
                    "--csv-report",
                    str(root / "missing.csv"),
                    "--ocr-metadata",
                    str(metadata_path),
                ],
            )

        self.assertEqual(code, 1)
        self.assertIn("FAIL: raw-response privacy: raw OCR responses persisted", output)
        self.assertIn("FAIL: report consistency: CSV report not found:", output)

    def _run_validator(self, validator, argv: list[str]) -> tuple[int, str]:
        old_argv = sys.argv
        output = io.StringIO()
        try:
            sys.argv = ["validate_evaluation_invariants.py", *argv]
            with contextlib.redirect_stdout(output):
                code = validator.main()
        finally:
            sys.argv = old_argv
        return code, output.getvalue()


def _sample_report():
    jd = StructuredJD(
        job_id="job",
        source_document_id="jd",
        job_title="Developer",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="python",
                name="Python",
                category=RequirementCategory.SKILL,
                description="Python",
                weight=100,
                required_skill_names=["Python"],
            )
        ],
    )
    candidate = CandidateProfile.model_validate(
        {
            "candidate_id": "candidate",
            "source_document_id": "resume",
            "skills": ["Python"],
        }
    )
    return evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-validator")
