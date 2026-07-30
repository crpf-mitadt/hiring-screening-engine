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

    def test_validator_fails_soft_skill_dimension_labelled_technical(self) -> None:
        validator = _load_validator_module()
        payload = _soft_skill_dimension_report().model_dump(mode="json")
        payload["gaps"] = [
            "unverified technical dimension: Soft Skills and Proactivity: Communication clarity",
            "unverified technical dimension: Soft Skills and Proactivity: Initiative",
        ]
        payload["missing_information"] = list(payload["gaps"])

        code, output = self._run_validator_payload(validator, payload)

        self.assertEqual(code, 1)
        self.assertIn("FAIL: dimension label correctness", output)
        self.assertIn("unverified soft-skill dimension: Soft Skills and Proactivity: Communication clarity", output)

    def test_validator_fails_experience_dimension_labelled_soft_skill(self) -> None:
        validator = _load_validator_module()
        payload = _experience_dimension_report().model_dump(mode="json")
        payload["gaps"] = [
            "unverified soft-skill dimension: Experience and Breadth: Years of experience",
            "unverified soft-skill dimension: Experience and Breadth: Tech stack versatility",
        ]
        payload["missing_information"] = list(payload["gaps"])

        code, output = self._run_validator_payload(validator, payload)

        self.assertEqual(code, 1)
        self.assertIn("FAIL: dimension label correctness", output)
        self.assertIn("unverified experience dimension: Experience and Breadth: Years of experience", output)

    def test_validator_fails_technical_dimension_labelled_soft_skill(self) -> None:
        validator = _load_validator_module()
        payload = _technical_dimension_report().model_dump(mode="json")
        payload["gaps"] = ["unverified soft-skill dimension: Engineering Quality: Code quality"]
        payload["missing_information"] = list(payload["gaps"])

        code, output = self._run_validator_payload(validator, payload)

        self.assertEqual(code, 1)
        self.assertIn("FAIL: dimension label correctness", output)
        self.assertIn("unverified technical dimension: Engineering Quality: Code quality", output)

    def test_validator_fails_unmatched_breadth_dimension_with_supported_components(self) -> None:
        validator = _load_validator_module()
        payload = _breadth_report(["React.js", "Node.js", "Express"]).model_dump(mode="json")
        criterion = payload["criterion_evaluations"][0]
        assessment = criterion["component_assessment"]
        for component in assessment["components"]:
            if component["term"] == "Tech stack versatility":
                component["matched"] = False
                component["missing"] = True
                component["evidence"] = []
                component["evidence_count"] = 0
                component["shared_evidence_with"] = []
        assessment["supported_evaluation_dimensions"] = []
        assessment["missing_evaluation_dimensions"] = ["Tech stack versatility"]
        payload["gaps"] = ["unverified technical dimension: Technical Breadth: Tech stack versatility"]
        payload["missing_information"] = list(payload["gaps"])

        code, output = self._run_validator_payload(validator, payload)

        self.assertEqual(code, 1)
        self.assertIn(
            "FAIL: semantic dimension duplicates: technical-breadth leaves equivalent tech-stack versatility dimension 'Tech stack versatility' unverified",
            output,
        )

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

    def _run_validator_payload(self, validator, payload: dict) -> tuple[int, str]:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return self._run_validator(validator, ["--evaluation", str(path)])


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


def _soft_skill_dimension_report():
    jd = StructuredJD(
        job_id="job",
        source_document_id="jd",
        job_title="Developer",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="soft",
                name="Soft Skills and Proactivity",
                category=RequirementCategory.SKILL,
                description="Communication, initiative, and ownership.",
                weight=100,
                requirement_ids=["req_soft_01"],
                evaluation_dimensions=["Communication clarity", "Initiative"],
            )
        ],
    )
    candidate = CandidateProfile.model_validate({"candidate_id": "candidate", "source_document_id": "resume"})
    return evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-soft")


def _experience_dimension_report():
    jd = StructuredJD(
        job_id="job",
        source_document_id="jd",
        job_title="Developer",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="experience",
                name="Experience and Breadth",
                category=RequirementCategory.EXPERIENCE,
                description="Experience and breadth.",
                weight=100,
                requirement_ids=["req_exp_01"],
                evaluation_dimensions=["Years of experience", "Tech stack versatility"],
            )
        ],
    )
    candidate = CandidateProfile.model_validate({"candidate_id": "candidate", "source_document_id": "resume"})
    return evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-experience")


def _technical_dimension_report():
    jd = StructuredJD(
        job_id="job",
        source_document_id="jd",
        job_title="Developer",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="technical",
                name="Engineering Quality",
                category=RequirementCategory.SKILL,
                description="Technical implementation quality.",
                weight=100,
                evaluation_dimensions=["Code quality"],
            )
        ],
    )
    candidate = CandidateProfile.model_validate({"candidate_id": "candidate", "source_document_id": "resume"})
    return evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-technical")


def _breadth_report(supported_terms: list[str]):
    jd = StructuredJD(
        job_id="job",
        source_document_id="jd",
        job_title="Developer",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="technical-breadth",
                name="Technical Breadth",
                category=RequirementCategory.SKILL,
                description="Modern JavaScript stack breadth.",
                weight=100,
                preferred_skill_names=supported_terms,
                evaluation_dimensions=["Tech stack versatility"],
            )
        ],
    )
    candidate = CandidateProfile.model_validate(
        {
            "candidate_id": "candidate",
            "source_document_id": "resume",
            "skills": supported_terms,
        }
    )
    return evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-breadth")
