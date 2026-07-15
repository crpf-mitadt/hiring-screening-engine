from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.pipeline.pilot import run_synthetic_pilot


class SyntheticPilotTests(unittest.TestCase):
    def test_pilot_writes_reports(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_synthetic_pilot(Path("tests/fixtures/synthetic"), output_dir)

            self.assertEqual(report.eligibility.status.value, "PASS")
            self.assertTrue((output_dir / "evaluation.json").is_file())
            self.assertTrue((output_dir / "evaluation.csv").is_file())
            self.assertTrue((output_dir / "evaluation.html").is_file())


if __name__ == "__main__":
    unittest.main()
