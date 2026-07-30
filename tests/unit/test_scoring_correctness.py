import unittest

from pydantic import ValidationError

from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.scoring.evidence_completeness import evidence_completeness
from chandra_eval.scoring.matching import build_candidate_evidence_index, match_terms
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

    def test_field_restricted_technical_degree_requires_level_and_field(self) -> None:
        requirement = JDRequirement(
            requirement_id="technical-degree",
            category=RequirementCategory.EDUCATION,
            description=(
                "Pursuing/holding a Master's degree OR holding a Bachelor's degree in "
                "Computer Science, Artificial Intelligence, Data Science, or a closely related technical field"
            ),
            requirement_type=RequirementType.MINIMUM,
            mandatory=True,
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Research Assistant",
            mandatory_requirements=[requirement],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="technical-degree-score",
                    name="Technical degree",
                    category=RequirementCategory.EDUCATION,
                    description=requirement.description,
                    weight=100,
                )
            ],
        )

        def score(degree_level: str, degree_name: str, field: str | None):
            candidate = CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate",
                    "source_document_id": "resume",
                    "education": [
                        {
                            "degree_level": degree_level,
                            "degree_name": degree_name,
                            "field_of_study": field,
                            "completion_status": "completed",
                            "evidence": [_evidence(f"{degree_name} {field or ''} completed").model_dump(mode="json")],
                        }
                    ],
                }
            )
            return evaluate_candidate(candidate.scoring_safe_profile, jd)

        mba = score("master", "MBA", "International Business")
        generic_master = score("master", "Master of Science", None)
        cs_bachelor = score("bachelor", "Bachelor of Engineering", "Computer Science")
        ds_master = score("master", "Master of Science", "Data Science")
        ml_master = score("master", "Master of Science", "Machine Learning")
        executive = score("certification", "Chief Digital Officer Program", "Leadership")

        self.assertEqual(mba.criterion_evaluations[0].match_fraction, 0)
        self.assertEqual(mba.eligibility.requirement_statuses["technical-degree"], "not_found")
        self.assertIn("Field mismatch", str(mba.eligibility.requirement_details))
        self.assertEqual(generic_master.criterion_evaluations[0].match_fraction, 0)
        self.assertEqual(cs_bachelor.criterion_evaluations[0].match_fraction, 1)
        self.assertEqual(ds_master.criterion_evaluations[0].match_fraction, 1)
        self.assertEqual(ml_master.criterion_evaluations[0].match_fraction, 1)
        self.assertEqual(executive.criterion_evaluations[0].match_fraction, 0)

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

    def test_short_skill_terms_use_boundaries_and_direct_evidence(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "education": [
                    {
                        "degree_level": "certification",
                        "degree_name": "Chief Digital Officer Program",
                        "institution": "Indian School of Business",
                        "evidence": [
                            _evidence("Chief Digital Officer Program").model_dump(mode="json"),
                            _evidence("Indian School of Business").model_dump(mode="json"),
                        ],
                    }
                ],
                "skills": ["GitHub", "SQL", "AI", "ML", "R", "C", "Go"],
            }
        )
        index = build_candidate_evidence_index(candidate.scoring_safe_profile)

        self.assertEqual(match_terms(["Git"], index, allowed_sources={"education"}).fraction, 0)
        self.assertEqual(match_terms(["Git"], index).fraction, 1)
        self.assertIn("GitHub", match_terms(["Git"], index).evidence[0].exact_text)
        for term in ("Git", "R", "C", "AI", "ML"):
            paragraph_candidate = CandidateProfile.model_validate(
                {
                    "candidate_id": f"candidate-{term}",
                    "source_document_id": "resume",
                    "experience": [
                        {
                            "responsibilities": [
                                "Digital digitization legitimate agility research mail claim compact goal"
                            ],
                            "evidence": [
                                _evidence(
                                    "Digital digitization legitimate agility research mail claim compact goal"
                                ).model_dump(mode="json")
                            ],
                        }
                    ],
                }
            )
            with self.subTest(term=term):
                self.assertEqual(match_terms([term], build_candidate_evidence_index(paragraph_candidate.scoring_safe_profile)).fraction, 0)

    def test_digital_program_does_not_score_git_or_evidence_completeness(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "education": [
                    {
                        "degree_level": "certification",
                        "degree_name": "Chief Digital Officer Program",
                        "institution": "Indian School of Business",
                        "evidence": [
                            _evidence("Chief Digital Officer Program").model_dump(mode="json"),
                            _evidence("Indian School of Business").model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Role",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="git",
                    name="Git",
                    category=RequirementCategory.SKILL,
                    description="Git version control",
                    weight=100,
                    required_skill_names=["Git"],
                )
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)

        self.assertEqual(report.final_score, 0)
        self.assertEqual(report.evidence_completeness, 0)
        self.assertEqual(report.criterion_evaluations[0].evidence, [])
        self.assertNotIn("Indian School of Business", str(report.model_dump()))

    def test_qualitative_experience_does_not_require_numeric_months(self) -> None:
        requirement = JDRequirement(
            requirement_id="ai-ml-experience",
            category=RequirementCategory.EXPERIENCE,
            description="Demonstrated experience through research publication, relevant internship, or job experience in AI/ML",
            requirement_type=RequirementType.MINIMUM,
            mandatory=True,
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Research Assistant",
            mandatory_requirements=[requirement],
        )
        operational = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "responsibilities": ["Oversaw broad AI and ML operational reporting for governance."],
                        "evidence": [_evidence("Oversaw broad AI and ML operational reporting for governance.").model_dump(mode="json")],
                    }
                ],
            }
        )
        direct = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "title": "AI Research Intern",
                        "responsibilities": ["Developed machine learning models for research experiments."],
                        "evidence": [_evidence("AI Research Intern developed machine learning models.").model_dump(mode="json")],
                    }
                ],
            }
        )

        operational_report = evaluate_candidate(operational.scoring_safe_profile, jd)
        direct_report = evaluate_candidate(direct.scoring_safe_profile, jd)

        self.assertEqual(operational_report.eligibility.requirement_statuses["ai-ml-experience"], "not_found")
        self.assertNotIn("no deterministic minimum value", str(operational_report.model_dump()).lower())
        self.assertEqual(direct_report.eligibility.requirement_statuses["ai-ml-experience"], "met")

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
