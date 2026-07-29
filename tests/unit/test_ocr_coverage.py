from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from chandra_eval.config import Settings
from chandra_eval.ocr.full_document import ocr_pdf_pages
from chandra_eval.pipeline.match_workflow import MatchWorkflowOptions, run_match_workflow


class FakeChandraClient:
    def __init__(self):
        self.calls = 0

    def chat_ocr(self, image, prompt, max_tokens=4096):
        self.calls += 1
        return {
            "model": "chandra-ocr-2-q8",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            f'<p data-label="Text" data-bbox="[0,0,1,1]">'
                            f"Page {self.calls} skills: Python</p>"
                        )
                    },
                }
            ],
        }


class OcrCoverageTests(unittest.TestCase):
    def test_smoke_test_remains_first_page_only(self) -> None:
        source = Path("scripts/smoke_test_q8.py").read_text(encoding="utf-8")

        self.assertIn('"test_scope": "first page only"', source)
        self.assertIn('"page_range": "0"', source)

    def test_two_page_pdf_triggers_two_ocr_requests_by_default(self) -> None:
        client = FakeChandraClient()
        with patch("chandra_eval.ocr.full_document.pdf_page_count", return_value=2), patch(
            "chandra_eval.ocr.full_document.render_pdf_page", side_effect=["page1", "page2"]
        ), patch("chandra_eval.ocr.full_document.scale_to_fit", side_effect=lambda image: image):
            responses = ocr_pdf_pages(Path("resume.pdf"), client)

        self.assertEqual(len(responses), 2)
        self.assertEqual(client.calls, 2)

    def test_match_workflow_does_not_reuse_one_page_cache_for_full_document(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "matches"
            jd = root / "jd.txt"
            resume = root / "resume.pdf"
            jd.write_text("Role\nRequired skills: Python", encoding="utf-8")
            resume.write_bytes(b"fake pdf identity")
            calls: list[int | None] = []

            def fake_ocr(path, client, page_count=None, max_tokens=4096):
                calls.append(page_count)
                count = 2 if page_count is None else page_count
                return [
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        f'<p data-label="Text" data-bbox="[0,0,1,1]">'
                                        f"Skills: Python page {index}</p>"
                                    )
                                }
                            }
                        ]
                    }
                    for index in range(1, count + 1)
                ]

            with patch("chandra_eval.pipeline.match_workflow.pdf_page_count", return_value=2), patch(
                "chandra_eval.pipeline.match_workflow.ocr_pdf_pages", side_effect=fake_ocr
            ):
                with self.assertRaisesRegex(RuntimeError, "Partial resume OCR"):
                    run_match_workflow(
                        MatchWorkflowOptions(
                            jd_path=jd,
                            resume_path=resume,
                            output_root=output,
                            mock_llm=True,
                            skip_preflight=True,
                            quiet=True,
                            ocr_pages=1,
                        ),
                        settings=Settings(match_output_root=output),
                    )
                result = run_match_workflow(
                    MatchWorkflowOptions(
                        jd_path=jd,
                        resume_path=resume,
                        output_root=output,
                        mock_llm=True,
                        skip_preflight=True,
                        quiet=True,
                    ),
                    settings=Settings(match_output_root=output),
                )

            self.assertEqual(calls, [1, None])
            self.assertEqual(result.report.criterion_evaluations[0].match_fraction, 1.0)
            self.assertTrue((result.run_dir / "ocr" / "page-001.html").is_file())
            self.assertTrue((result.run_dir / "ocr" / "page-002.html").is_file())
            structured = (result.run_dir / "structured-resume.json").read_text(encoding="utf-8")
            self.assertIn('"source_pages": [\n    1,\n    2\n  ]', structured)
            source_map = (result.run_dir / "ocr" / "source-map.json").read_text(encoding="utf-8")
            self.assertLess(source_map.index("Skills: Python page 1"), source_map.index("Skills: Python page 2"))

    def test_explicit_partial_evaluation_is_marked_review_with_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "matches"
            jd = root / "jd.txt"
            resume = root / "resume.pdf"
            jd.write_text("Role\nRequired skills: Python", encoding="utf-8")
            resume.write_bytes(b"fake pdf identity")

            def fake_ocr(path, client, page_count=None, max_tokens=4096):
                return [
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '<p data-label="Text" data-bbox="[0,0,1,1]">Skills: Python</p>'
                                }
                            }
                        ]
                    }
                ]

            with patch("chandra_eval.pipeline.match_workflow.pdf_page_count", return_value=2), patch(
                "chandra_eval.pipeline.match_workflow.ocr_pdf_pages", side_effect=fake_ocr
            ):
                result = run_match_workflow(
                    MatchWorkflowOptions(
                        jd_path=jd,
                        resume_path=resume,
                        output_root=output,
                        mock_llm=True,
                        skip_preflight=True,
                        quiet=True,
                        ocr_pages=1,
                        allow_partial_evaluation=True,
                    ),
                    settings=Settings(match_output_root=output),
                )

            summary = (result.run_dir / "run-summary.md").read_text(encoding="utf-8")
            self.assertEqual(result.report.eligibility.status.value, "REVIEW")
            self.assertLessEqual(result.report.evidence_completeness, 0.5)
            self.assertIn("Partial resume OCR", summary)


if __name__ == "__main__":
    unittest.main()
