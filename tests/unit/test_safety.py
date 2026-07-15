import unittest

from chandra_eval.ingestion.resume_document import load_candidate_profile


class SafetyTests(unittest.TestCase):
    def test_scoring_safe_profile_excludes_personal_information(self) -> None:
        candidate = load_candidate_profile("tests/fixtures/synthetic/candidate_profile.json")
        safe = candidate.scoring_safe_profile.model_dump()

        self.assertEqual(safe["candidate_id"], "candidate-001")
        self.assertNotIn("personal_information", safe)
        self.assertNotIn("email", str(safe).lower())
        self.assertNotIn("phone", str(safe).lower())
        self.assertNotIn("Synthetic Candidate", str(safe))


if __name__ == "__main__":
    unittest.main()
