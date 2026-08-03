import json
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

from chandra_eval.cli import main as cli_main
from chandra_eval.config import Settings
from chandra_eval.pipeline.match_workflow import (
    JD_EXTENSIONS,
    MatchWorkflowOptions,
    check_services,
    discover_supported_files,
    run_match_workflow,
    select_file,
)


def _candidate_json(path: Path, skill: str = "Python") -> None:
    path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-safe",
                "source_document_id": "candidate.json",
                "summary": f"{skill} engineer",
                "skills": [skill],
            }
        ),
        encoding="utf-8",
    )


def _load_matrix_script():
    spec = importlib.util.spec_from_file_location("validate_match_matrix", Path("scripts/validate_match_matrix.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MatchWorkflowTests(unittest.TestCase):
    def test_non_interactive_workflow_writes_safe_summaries(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jd = root / "Job description - Python Role.txt"
            resume = root / "person@example.com.json"
            output = root / "matches"
            jd.write_text("Python Role\nRequired skills: Python\n1 year experience", encoding="utf-8")
            _candidate_json(resume)

            result = run_match_workflow(
                MatchWorkflowOptions(
                    jd_path=jd,
                    resume_path=resume,
                    output_root=output,
                    candidate_id="candidate-workflow",
                    mock_llm=True,
                    skip_preflight=True,
                    quiet=True,
                ),
                settings=Settings(match_output_root=output),
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.report.candidate_id, "candidate-workflow")
            self.assertTrue(result.summary_json.is_file())
            self.assertTrue(result.summary_md.is_file())
            summary_text = result.summary_json.read_text(encoding="utf-8")
            self.assertNotIn("person@example.com", summary_text)
            summary = json.loads(summary_text)
            self.assertEqual(summary["human_review_required"], True)
            self.assertEqual(summary["raw_artifacts_persisted"]["chandra_raw_responses"], False)
            self.assertIn("evaluation_html", summary["generated_artifact_paths"])
            self.assertGreaterEqual(summary["final_score"], 0)

    def test_cli_non_interactive_invocation_completes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jd = root / "jd.txt"
            resume = root / "candidate.json"
            output = root / "matches"
            jd.write_text("Role\nRequired skills: Python", encoding="utf-8")
            _candidate_json(resume)

            code = cli_main(
                [
                    "match",
                    "--jd",
                    str(jd),
                    "--resume",
                    str(resume),
                    "--output-root",
                    str(output),
                    "--mock-llm",
                    "--skip-preflight",
                    "--quiet",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(list(output.glob("*/*/*/run-summary.json")))

    def test_cache_reuse_uses_content_identity_not_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "matches"
            resume = root / "candidate.json"
            _candidate_json(resume, "Python")
            jd_a = root / "a" / "job.txt"
            jd_b = root / "b" / "job.txt"
            jd_a.parent.mkdir()
            jd_b.parent.mkdir()
            jd_a.write_text("Role\nRequired skills: Python", encoding="utf-8")
            jd_b.write_text("Role\nRequired skills: Java", encoding="utf-8")

            first = run_match_workflow(
                MatchWorkflowOptions(
                    jd_path=jd_a,
                    resume_path=resume,
                    output_root=output,
                    mock_llm=True,
                    skip_preflight=True,
                    quiet=True,
                ),
                settings=Settings(match_output_root=output),
            )
            second = run_match_workflow(
                MatchWorkflowOptions(
                    jd_path=jd_b,
                    resume_path=resume,
                    output_root=output,
                    mock_llm=True,
                    skip_preflight=True,
                    quiet=True,
                ),
                settings=Settings(match_output_root=output),
            )

            self.assertEqual(first.report.criterion_evaluations[0].match_fraction, 1)
            self.assertLess(second.report.criterion_evaluations[0].match_fraction, 1)
            self.assertEqual(
                [stage.status for stage in second.stages if stage.name == "jd_structuring"],
                ["generated"],
            )

    def test_file_selection_handles_duplicates_search_and_cancel(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            first = root / "a" / "job.docx"
            second = root / "b" / "job.docx"
            first.write_text("a", encoding="utf-8")
            second.write_text("b", encoding="utf-8")
            files = discover_supported_files(root, JD_EXTENSIONS)
            output: list[str] = []
            choices = iter(["s", "b/", "1"])

            selected = select_file(
                title="Select a JD",
                root=root,
                files=files,
                input_func=lambda _: next(choices),
                print_func=output.append,
            )
            cancelled = select_file(
                title="Select a JD",
                root=root,
                files=files,
                input_func=lambda _: "q",
                print_func=lambda _: None,
            )

            self.assertEqual(selected, second)
            self.assertIsNone(cancelled)
            self.assertTrue(any("a/job.docx" in line for line in output))
            self.assertTrue(any("b/job.docx" in line for line in output))

    def test_preflight_failures_are_actionable(self) -> None:
        class Response:
            def raise_for_status(self):
                raise RuntimeError("down")

        with patch("requests.get", return_value=Response()):
            results = check_services(Settings())

        self.assertFalse(all(item[1] for item in results))
        self.assertIn("http", results[0][2])

    def test_matrix_runner_accounts_for_pass_pipeline_fail_and_validator_fail(self) -> None:
        matrix = _load_matrix_script()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "matrix"
            jd = root / "structured-jd.json"
            resume_ok = root / "resume-ok.json"
            resume_bad = root / "resume-bad.json"
            metadata = root / "metadata.json"
            jd.write_text(
                json.dumps(
                    {
                        "job_id": "job-shared",
                        "source_document_id": "jd",
                        "job_title": "Python Role",
                        "scoring_rubric": [
                            {
                                "criterion_id": "python",
                                "name": "Python",
                                "category": "skill",
                                "description": "Python",
                                "weight": 100,
                                "required_skill_names": ["Python"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _candidate_json(resume_ok, "Python")
            resume_bad.write_text("{not json", encoding="utf-8")
            metadata.write_text(json.dumps({"raw_responses_persisted": True}), encoding="utf-8")
            manifest = {
                "pairs": [
                    {
                        "test_name": "pass",
                        "jd_group": "shared-python",
                        "job_id": "job-shared",
                        "candidate_id": "candidate-pass",
                        "structured_jd": str(jd),
                        "structured_resume": str(resume_ok),
                    },
                    {
                        "test_name": "pipeline-fail",
                        "jd_group": "shared-python",
                        "job_id": "job-shared",
                        "candidate_id": "candidate-pipeline-fail",
                        "structured_jd": str(jd),
                        "structured_resume": str(resume_bad),
                    },
                    {
                        "test_name": "validator-fail",
                        "jd_group": "shared-python",
                        "job_id": "job-shared",
                        "candidate_id": "candidate-validator-fail",
                        "structured_jd": str(jd),
                        "structured_resume": str(resume_ok),
                        "ocr_metadata": str(metadata),
                    },
                ]
            }

            rows = matrix.run_matrix(manifest, output, skip_validator=True)
            matrix._write_summaries(rows, output)

            self.assertEqual([row["status"] for row in rows], ["PASS", "PIPELINE_FAIL", "VALIDATOR_FAIL"])
            self.assertEqual(rows[1]["evaluation_exit_status"], 1)
            self.assertFalse(rows[1]["evaluation_json"])
            self.assertIn("raw responses", rows[2]["error"])
            self.assertEqual(len({row["structured_jd_hash"] for row in rows if row["structured_jd_hash"]}), 1)
            self.assertTrue(all(row["shared_jd_hash_proven"] for row in rows))
            summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "match-matrix-comparison-v1")
            self.assertEqual(summary["status_counts"]["PASS"], 1)
            self.assertEqual(rows[0]["scoring_version"], "scoring-v2-evidence-discrete")
            self.assertEqual(rows[0]["eligibility"], "PASS")
            self.assertEqual(rows[0]["human_review_required"], True)
            csv_columns = (output / "comparison.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
            tsv_columns = (output / "comparison.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
            self.assertEqual(csv_columns, tsv_columns)

    def test_matrix_runner_marks_validator_traceback_without_fail_text(self) -> None:
        matrix = _load_matrix_script()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "matrix"
            jd = root / "structured-jd.json"
            resume = root / "resume.json"
            jd.write_text(
                json.dumps(
                    {
                        "job_id": "job",
                        "source_document_id": "jd",
                        "job_title": "Python Role",
                        "scoring_rubric": [
                            {
                                "criterion_id": "python",
                                "name": "Python",
                                "category": "skill",
                                "description": "Python",
                                "weight": 100,
                                "required_skill_names": ["Python"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _candidate_json(resume, "Python")
            manifest = {
                "pairs": [
                    {
                        "test_name": "validator-traceback",
                        "job_id": "job",
                        "candidate_id": "candidate",
                        "structured_jd": str(jd),
                        "structured_resume": str(resume),
                    }
                ]
            }

            with patch.object(
                matrix,
                "_run_validator",
                return_value=types.SimpleNamespace(returncode=2, stdout="", stderr="Traceback: FileNotFoundError"),
            ):
                rows = matrix.run_matrix(manifest, output)

            self.assertEqual(rows[0]["status"], "VALIDATOR_FAIL")
            self.assertEqual(rows[0]["validator_exit_status"], 2)
            self.assertIn("Traceback", rows[0]["error"])

    def test_matrix_manifest_validation_reports_path_and_duplicate_errors(self) -> None:
        matrix = _load_matrix_script()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "matrix"
            jd = root / "structured-jd.json"
            resume = root / "resume.json"
            jd.write_text("{}", encoding="utf-8")
            resume.write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": "match-matrix-manifest-v1",
                "pairs": [
                    {
                        "test_name": "one",
                        "jd_group": "group",
                        "job_id": "job",
                        "candidate_id": "candidate",
                        "structured_jd": str(jd),
                        "structured_resume": str(resume),
                    },
                    {
                        "test_name": "duplicate",
                        "jd_group": "group",
                        "job_id": "job",
                        "candidate_id": "candidate",
                        "structured_jd": str(root / "missing.json"),
                        "structured_resume": str(resume),
                    },
                ],
            }

            errors = matrix.validate_manifest(manifest, output)

            self.assertTrue(any("duplicate" in error for error in errors))
            self.assertTrue(any("not found" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
