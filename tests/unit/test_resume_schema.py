import unittest

from pydantic import ValidationError

from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.schemas.jd import ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import CandidateProfile


class ResumeSchemaTests(unittest.TestCase):
    def test_live_null_fields_validate(self) -> None:
        payload = {
            "candidate_id": "candidate-live",
            "source_document_id": "ocr-live",
            "extraction_method": "live_gemma",
            "summary": "Marketing candidate",
            "education": [
                {"degree_level": "bachelor", "field_of_study": "Marketing"},
                {"degree_level": "certification", "field_of_study": "Analytics"},
                {"degree_level": "certification", "field_of_study": "Admissions"},
                {"degree_level": "certification", "field_of_study": None},
            ],
            "experience": [
                {"title": "Coordinator", "organization": "Org", "employment_type": None},
                {"title": "Assistant", "organization": "Org", "employment_type": None},
            ],
            "skills": ["CRM"],
        }

        candidate = CandidateProfile.model_validate(payload)

        self.assertIsNone(candidate.education[3].field_of_study)
        self.assertIsNone(candidate.experience[0].employment_type)
        self.assertIsNone(candidate.experience[1].employment_type)

    def test_optional_missing_values_remain_null_and_null_lists_become_empty(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-null",
                "source_document_id": "ocr-null",
                "education": [{"degree_level": "unknown", "evidence": None}],
                "experience": [{"responsibilities": None, "skills_used": None}],
                "skills": None,
            }
        )

        self.assertIsNone(candidate.education[0].field_of_study)
        self.assertEqual(candidate.education[0].evidence, [])
        self.assertEqual(candidate.experience[0].responsibilities, [])
        self.assertEqual(candidate.skills, [])

    def test_required_structural_fields_and_enums_still_fail(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateProfile.model_validate({"source_document_id": "ocr"})
        with self.assertRaises(ValidationError):
            CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate-bad",
                    "source_document_id": "ocr",
                    "education": [{"degree_level": "invented"}],
                }
            )
        with self.assertRaises(ValidationError):
            CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate-bad-evidence",
                    "source_document_id": "ocr",
                    "education": [{"evidence": [{"source_type": "resume"}]}],
                }
            )

    def test_null_optional_values_do_not_create_scoring_matches(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-score",
                "source_document_id": "ocr",
                "experience": [
                    {
                        "title": None,
                        "organization": None,
                        "employment_type": None,
                        "start_date": None,
                        "end_date": None,
                    }
                ],
                "skills": [],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Role",
            minimum_experience_months=12,
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="experience",
                    name="Experience",
                    category="experience",
                    description="One year required",
                    weight=100,
                    mandatory=True,
                )
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-null")

        self.assertEqual(report.final_score, 0)
        self.assertIn("12 additional months of relevant experience", report.missing_information)


if __name__ == "__main__":
    unittest.main()
