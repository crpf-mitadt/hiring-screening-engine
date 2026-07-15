import unittest

from chandra_eval.ingestion.jd_document import extract_synthetic_jd
from chandra_eval.ingestion.resume_document import load_candidate_profile
from chandra_eval.scoring.engine import evaluate_candidate


class ScoringTests(unittest.TestCase):
    def test_synthetic_candidate_scores_and_preserves_human_review(self) -> None:
        candidate = load_candidate_profile("tests/fixtures/synthetic/candidate_profile.json")
        with open("tests/fixtures/synthetic/job_description.txt", encoding="utf-8") as handle:
            jd_text = handle.read()
        jd = extract_synthetic_jd(jd_text, "job-001", "synthetic-jd")

        report = evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-test")

        self.assertEqual(report.eligibility.status.value, "PASS")
        self.assertGreaterEqual(report.final_score, 80)
        self.assertTrue(report.human_review_required)
        self.assertIn("required-skills", report.strengths)


if __name__ == "__main__":
    unittest.main()
