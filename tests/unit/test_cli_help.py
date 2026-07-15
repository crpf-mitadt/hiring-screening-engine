import subprocess
import sys
import unittest


class CliHelpTests(unittest.TestCase):
    def test_existing_and_pipeline_scripts_help(self) -> None:
        scripts = [
            "scripts/smoke_test_q8.py",
            "scripts/make_resume_contact_sheet.py",
            "scripts/run_full_resume_ocr.py",
            "scripts/extract_jd.py",
            "scripts/structure_resume.py",
            "scripts/evaluate_candidate.py",
            "scripts/build_report.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
