import unittest

from pydantic import ValidationError

from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.scoring.evidence_completeness import evidence_completeness
from chandra_eval.schemas.common import EvidenceRef
from chandra_eval.schemas.evaluation import CriterionEvaluation, CriterionStatus, EligibilityStatus
from chandra_eval.schemas.jd import JDRequirement, RequirementCategory, RequirementType, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import CandidateProfile


def _evidence(text: str, source: str = "resume_ocr") -> EvidenceRef:
    return EvidenceRef(source_type=source, source_document_id="resume", exact_text=text, normalized_text=text)


class ScoringCorrectnessTests(unittest.TestCase):
    def test_discrete_fraction_validation(self) -> None:
        for value in (0, 0.25, 0.5, 0.75, 1):
            CriterionEvaluation(
                criterion_id=f"c-{value}",
                match_fraction=value,
                weighted_score=value * 10,
                status=CriterionStatus.MET if value == 1 else CriterionStatus.PARTIALLY_MET,
                explanation="ok",
            )
        for value in (0.083, 0.333, 0.071):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    CriterionEvaluation(
                        criterion_id="bad",
                        match_fraction=value,
                        weighted_score=1,
                        status=CriterionStatus.PARTIALLY_MET,
                        explanation="bad",
                    )

    def test_accepted_evidence_is_verification_not_missing_requirement(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["Python"],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Role",
            required_skills=["Python"],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="python",
                    name="Python",
                    category=RequirementCategory.SKILL,
                    description="Python",
                    weight=100,
                    accepted_evidence=["Resume", "Technical interview", "Academic transcripts"],
                )
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        criterion = report.criterion_evaluations[0]

        self.assertEqual(criterion.match_fraction, 1.0)
        self.assertNotIn("Resume", criterion.missing_information)
        self.assertNotIn("Technical interview", criterion.missing_information)
        self.assertIn("Technical interview", criterion.verification_needed)
        self.assertNotIn("Academic transcripts", report.gaps)

    def test_empty_education_and_ml_experience_do_not_satisfy_degree(self) -> None:
        paragraph = "Led enterprise governance, digital transformation, hyperautomation, Gen AI, ML, security, and IT infrastructure."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "education": [],
                "experience": [{"responsibilities": [paragraph], "evidence": [_evidence(paragraph).model_dump(mode="json")]}],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Research Assistant",
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="degree",
                    category=RequirementCategory.EDUCATION,
                    description="Pursuing or holding a Master's degree or holding a Bachelor's degree",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="degree-score",
                    name="Degree",
                    category=RequirementCategory.EDUCATION,
                    description="Pursuing or holding a Master's degree or holding a Bachelor's degree",
                    weight=100,
                )
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)

        self.assertEqual(report.criterion_evaluations[0].match_fraction, 0)
        self.assertEqual(report.eligibility.status, EligibilityStatus.REVIEW)
        self.assertEqual(report.eligibility.requirement_statuses["degree"], "not_found")
        self.assertNotIn("postgraduate education", str(report.model_dump()))

    def test_legitimate_degree_evidence_still_matches(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "education": [
                    {
                        "degree_level": "master",
                        "degree_name": "Master of Science",
                        "completion_status": "completed",
                        "evidence": [_evidence("Master of Science completed", "resume_ocr").model_dump(mode="json")],
                    }
                ],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Research Assistant",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="degree-score",
                    name="Degree",
                    category=RequirementCategory.EDUCATION,
                    description="Master's degree",
                    weight=100,
                )
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)

        self.assertGreater(report.criterion_evaluations[0].match_fraction, 0)
        self.assertGreater(len(report.criterion_evaluations[0].evidence), 0)

    def test_generic_ai_ml_does_not_leak_into_specific_technical_criteria(self) -> None:
        paragraph = "Oversaw broad AI and ML operational reporting for digital transformation."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [{"responsibilities": [paragraph], "evidence": [_evidence(paragraph).model_dump(mode="json")]}],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Research Assistant",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="theory",
                    name="AI/ML theoretical knowledge",
                    category=RequirementCategory.SKILL,
                    description="statistics, probability distributions, linear algebra, calculus",
                    weight=25,
                ),
                ScoringCriterion(
                    criterion_id="osint",
                    name="OSINT and data handling",
                    category=RequirementCategory.SKILL,
                    description="OSINT, scraping, unstructured data handling",
                    weight=25,
                ),
                ScoringCriterion(
                    criterion_id="llm",
                    name="Preferred LLM libraries",
                    category=RequirementCategory.SKILL,
                    description="transformers, LlamaCPP, Unsloth, Docker, databases, AWS, Azure, GCP",
                    weight=50,
                ),
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)

        self.assertEqual([item.match_fraction for item in report.criterion_evaluations], [0, 0, 0])
        self.assertEqual(report.evidence_completeness, 0)

    def test_evidence_completeness_accounts_for_dates_and_partial_ocr(self) -> None:
        with_evidence = CriterionEvaluation(
            criterion_id="one",
            match_fraction=1,
            weighted_score=10,
            status=CriterionStatus.MET,
            explanation="ok",
            evidence=[_evidence("Python")],
        )
        repeated = with_evidence.model_copy(update={"criterion_id": "two", "evidence": [_evidence("Python"), _evidence("Python")]})
        missing = CriterionEvaluation(
            criterion_id="three",
            match_fraction=0,
            weighted_score=0,
            status=CriterionStatus.NOT_FOUND,
            explanation="missing",
        )

        complete = evidence_completeness([with_evidence, repeated])
        degraded = evidence_completeness(
            [with_evidence, missing],
            date_warnings=["unclear date"],
            warnings=["Partial resume OCR: processed 1 of 2 pages."],
        )
        none = evidence_completeness([missing])

        self.assertEqual(complete, 1)
        self.assertLess(degraded, complete)
        self.assertEqual(none, 0)


if __name__ == "__main__":
    unittest.main()
