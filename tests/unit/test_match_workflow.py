import json
from pathlib import Path
from tempfile import TemporaryDirectory
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


if __name__ == "__main__":
    unittest.main()
