import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import ValidationError

from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files
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
                evidence=[] if value == 0 else [_evidence("supported")],
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

    def test_positive_weighted_score_requires_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            CriterionEvaluation(
                criterion_id="bad-positive",
                match_fraction=0.25,
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

    def test_supported_quantitative_experience_attaches_role_date_and_relevance_evidence(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "title": "Software Developer I",
                        "organization": "Synthetic Product Company",
                        "start_date": "January 2024",
                        "end_date": "Present",
                        "responsibilities": [
                            "Full-stack logistics systems, Node.js, Express.js, React.js, Vue.js, REST APIs, PostgreSQL and MySQL."
                        ],
                        "skills_used": ["Node.js", "React.js", "MySQL", "PostgreSQL"],
                        "evidence": [
                            _evidence("Software Developer I").model_dump(mode="json"),
                            _evidence("Synthetic Product Company").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence(
                                "Full-stack logistics systems, Node.js, Express.js, React.js, Vue.js, REST APIs, PostgreSQL and MySQL."
                            ).model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="experience",
                    name="Relevant full-stack experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    weight=100,
                    mandatory=True,
                )
            ],
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        criterion = report.criterion_evaluations[0]
        detail = report.eligibility.requirement_details[0]

        self.assertEqual(report.eligibility.requirement_statuses["experience"], "met")
        self.assertEqual(criterion.match_fraction, 1)
        self.assertEqual(criterion.weighted_score, 100)
        self.assertGreaterEqual(report.experience_summary["confirmed_months"], 30)
        self.assertGreaterEqual(len(criterion.evidence), 3)
        self.assertIn("Software Developer I", " ".join(item.exact_text for item in criterion.evidence))
        self.assertIn("January 2024 - Present", " ".join(item.exact_text for item in criterion.evidence))
        self.assertIn("Full-stack logistics systems", " ".join(item.exact_text for item in criterion.evidence))
        self.assertGreater(len(detail["evidence"]), 0)
        self.assertNotIn("No direct resume evidence", criterion.explanation)
        self.assertNotIn("Minimum 24 months", report.missing_information)
        self.assertGreater(report.evidence_completeness, 0)

    def test_duration_without_relevance_evidence_does_not_fully_satisfy_experience(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="experience",
                    name="Relevant full-stack experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    weight=100,
                    mandatory=True,
                )
            ],
        )
        unrelated = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "title": "Administrative Coordinator",
                        "start_date": "2024-01",
                        "end_date": "Present",
                        "responsibilities": ["Managed schedules and vendor documentation."],
                        "evidence": [
                            _evidence("Administrative Coordinator").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence("Managed schedules and vendor documentation.").model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        ambiguous = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "title": "Software Developer",
                        "start_date": "2024-01",
                        "end_date": "Present",
                        "responsibilities": ["Worked on internal systems."],
                        "evidence": [
                            _evidence("Software Developer").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence("Worked on internal systems.").model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )

        unrelated_report = evaluate_candidate(unrelated.scoring_safe_profile, jd)
        ambiguous_report = evaluate_candidate(ambiguous.scoring_safe_profile, jd)

        self.assertNotEqual(unrelated_report.eligibility.requirement_statuses["experience"], "met")
        self.assertEqual(unrelated_report.criterion_evaluations[0].weighted_score, 0)
        self.assertNotEqual(ambiguous_report.eligibility.requirement_statuses["experience"], "met")
        self.assertLess(ambiguous_report.criterion_evaluations[0].match_fraction, 1)

    def test_mysql_and_controlled_postgresql_database_equivalency(self) -> None:
        requirement = "Experience with MySQL or similar enterprise databases."
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="database",
                    category=RequirementCategory.SKILL,
                    description=requirement,
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                )
            ],
        )

        def candidate_with(skill_text: str) -> CandidateProfile:
            return CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate",
                    "source_document_id": "resume",
                    "skills": [skill_text],
                    "experience": [
                        {
                            "responsibilities": [skill_text],
                            "evidence": [_evidence(skill_text).model_dump(mode="json")],
                        }
                    ],
                    "projects": [skill_text],
                }
            )

        mysql = evaluate_candidate(candidate_with("Optimized PostgreSQL/MySQL database operations.").scoring_safe_profile, jd)
        postgres = evaluate_candidate(candidate_with("Implemented PostgreSQL database storage.").scoring_safe_profile, jd)
        generic = evaluate_candidate(candidate_with("Managed backend data storage.").scoring_safe_profile, jd)
        strict_jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="database",
                    category=RequirementCategory.SKILL,
                    description="MySQL only",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                )
            ],
        )
        strict_postgres = evaluate_candidate(candidate_with("PostgreSQL database implementation.").scoring_safe_profile, strict_jd)

        self.assertEqual(mysql.eligibility.requirement_statuses["database"], "met")
        self.assertIn("MySQL", str(mysql.eligibility.requirement_details))
        self.assertEqual(postgres.eligibility.requirement_statuses["database"], "met")
        self.assertIn("PostgreSQL", str(postgres.eligibility.requirement_details))
        self.assertEqual(generic.eligibility.requirement_statuses["database"], "not_found")
        self.assertEqual(strict_postgres.eligibility.requirement_statuses["database"], "not_found")

    def test_technical_stack_partial_credit_uses_independent_components(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="technical-stack",
                    name="Technical stack",
                    category=RequirementCategory.SKILL,
                    description="PHP; Laravel; WordPress; SQL/database management",
                    weight=40,
                    required_skill_names=["PHP", "Laravel", "WordPress", "SQL/database management"],
                ),
                ScoringCriterion(
                    criterion_id="other",
                    name="Other",
                    category=RequirementCategory.OTHER,
                    description="No-op",
                    weight=60,
                ),
            ],
        )
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["MySQL", "PostgreSQL"],
                "experience": [
                    {
                        "responsibilities": ["Optimized PostgreSQL/MySQL database operations for reliability and performance."],
                        "evidence": [
                            _evidence("Optimized PostgreSQL/MySQL database operations for reliability and performance.").model_dump(mode="json")
                        ],
                    }
                ],
            }
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        criterion = report.criterion_evaluations[0]

        self.assertEqual(criterion.match_fraction, 0.25)
        self.assertEqual(criterion.weighted_score, 10)
        self.assertIn("SQL/database management", criterion.explanation)
        self.assertIn("PHP", " ".join(criterion.missing_information))
        self.assertIn("Laravel", " ".join(criterion.missing_information))
        self.assertIn("WordPress", " ".join(criterion.missing_information))
        self.assertIn("PostgreSQL/MySQL database operations", " ".join(criterion.evidence_found))

    def test_technical_stack_classifies_core_adjacent_and_linked_database_evidence(self) -> None:
        db_text = "Optimized PostgreSQL/MySQL database operations for reliability and performance."
        js_text = "Built full-stack JavaScript and React.js REST API features."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["JavaScript", "React.js", "MySQL", "PostgreSQL"],
                "experience": [
                    {
                        "responsibilities": [db_text, js_text],
                        "evidence": [
                            _evidence(db_text).model_dump(mode="json"),
                            _evidence(js_text).model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        jd = self._full_stack_jd()

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        by_id = {item.criterion_id: item for item in report.criterion_evaluations}
        technical = by_id["crit_01"]
        db_requirement = next(
            item for item in report.eligibility.requirement_details if item["requirement_id"] == "req_tech_02"
        )

        self.assertEqual(db_requirement["status"], "met")
        self.assertEqual(technical.match_fraction, 0.5)
        self.assertIn("SQL/database management", technical.component_assessment["supported_core_components"])
        self.assertIn("JavaScript", technical.component_assessment["supported_adjacent_components"])
        self.assertIn("PHP", technical.component_assessment["missing_core_components"])
        self.assertIn("Laravel", technical.component_assessment["missing_core_components"])
        self.assertIn("WordPress", technical.component_assessment["missing_core_components"])
        self.assertIn("PostgreSQL/MySQL database operations", " ".join(technical.evidence_found))
        self.assertNotIn("JavaScript", " ".join(technical.component_assessment["supported_controlled_equivalents"]))
        self.assertEqual(
            len([item for item in technical.evidence if item.exact_text == db_text]),
            1,
        )

    def test_adjacent_javascript_does_not_satisfy_mandatory_core_terms(self) -> None:
        js_text = "Built full-stack JavaScript and React.js REST API features."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["JavaScript", "React.js"],
                "experience": [{"responsibilities": [js_text], "evidence": [_evidence(js_text).model_dump(mode="json")]}],
            }
        )
        report = evaluate_candidate(candidate.scoring_safe_profile, self._full_stack_jd())
        technical = {item.criterion_id: item for item in report.criterion_evaluations}["crit_01"]

        self.assertEqual(report.eligibility.requirement_statuses["req_tech_02"], "not_found")
        self.assertEqual(technical.match_fraction, 0.25)
        self.assertEqual(technical.component_assessment["supported_core_components"], [])
        self.assertIn("JavaScript", technical.component_assessment["supported_adjacent_components"])
        self.assertIn("PHP", technical.missing_information)
        self.assertIn("Laravel", technical.missing_information)
        self.assertIn("WordPress", technical.missing_information)
        self.assertIn("SQL/database management", technical.missing_information)

    def test_soft_and_additional_skill_criteria_are_isolated(self) -> None:
        react_text = "Built React.js dashboards with JavaScript."
        initiative_text = "Led an initiative to improve release quality and owned the rollout."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["React.js", "JavaScript"],
                "experience": [
                    {
                        "responsibilities": [react_text, initiative_text],
                        "evidence": [
                            _evidence(react_text).model_dump(mode="json"),
                            _evidence(initiative_text).model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, self._full_stack_jd())
        by_id = {item.criterion_id: item for item in report.criterion_evaluations}

        self.assertNotIn("React.js", by_id["crit_03"].explanation)
        self.assertNotIn("JavaScript", by_id["crit_03"].explanation)
        self.assertEqual(by_id["crit_03"].category, "leadership")
        self.assertIn("Ownership", by_id["crit_03"].component_assessment["supported_evaluation_dimensions"])
        self.assertIn("Proactivity", by_id["crit_03"].component_assessment["supported_evaluation_dimensions"])
        self.assertEqual(by_id["crit_04"].match_fraction, 0)
        self.assertIn("DevOps", by_id["crit_04"].component_assessment["missing_preferred_components"])
        self.assertIn("SEO", by_id["crit_04"].component_assessment["missing_preferred_components"])
        self.assertIn("UI/UX", by_id["crit_04"].component_assessment["missing_preferred_components"])
        self.assertEqual(by_id["crit_04"].missing_information, [])
        self.assertNotIn("Proactive Problem Solving", str(by_id["crit_04"].model_dump()))

    def test_criterion_identity_is_stable_when_rubric_order_is_shuffled(self) -> None:
        db_text = "Optimized PostgreSQL/MySQL database operations for reliability and performance."
        js_text = "Built full-stack JavaScript REST API features."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["JavaScript", "MySQL", "PostgreSQL"],
                "experience": [
                    {
                        "title": "Software Developer I",
                        "organization": "Synthetic Product Company",
                        "start_date": "2024-01",
                        "end_date": "Present",
                        "responsibilities": [
                            "Full-stack logistics systems with APIs.",
                            db_text,
                            js_text,
                        ],
                        "evidence": [
                            _evidence("Software Developer I").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence("Full-stack logistics systems with APIs.").model_dump(mode="json"),
                            _evidence(db_text).model_dump(mode="json"),
                            _evidence(js_text).model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        normal_jd = self._full_stack_jd()
        shuffled_jd = normal_jd.model_copy(update={"scoring_rubric": list(reversed(normal_jd.scoring_rubric))})

        normal = evaluate_candidate(candidate.scoring_safe_profile, normal_jd)
        shuffled = evaluate_candidate(candidate.scoring_safe_profile, shuffled_jd)
        normal_by_id = {item.criterion_id: item for item in normal.criterion_evaluations}
        shuffled_by_id = {item.criterion_id: item for item in shuffled.criterion_evaluations}

        self.assertEqual(set(normal_by_id), set(shuffled_by_id))
        for criterion_id in normal_by_id:
            with self.subTest(criterion_id=criterion_id):
                self.assertEqual(normal_by_id[criterion_id].match_fraction, shuffled_by_id[criterion_id].match_fraction)
                self.assertEqual(normal_by_id[criterion_id].missing_information, shuffled_by_id[criterion_id].missing_information)
                self.assertEqual(
                    normal_by_id[criterion_id].component_assessment,
                    shuffled_by_id[criterion_id].component_assessment,
                )

    def test_file_based_evaluation_preserves_source_rubric_fields(self) -> None:
        db_text = "Optimized PostgreSQL/MySQL database operations for reliability and performance."
        adjacent_text = "Built JavaScript, React.js, Node.js, Express.js, Vue.js and REST API features."
        initiative_text = "Owned release planning and proactively improved deployment quality."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["JavaScript", "React.js", "Node.js", "Express.js", "Vue.js", "MySQL", "PostgreSQL"],
                "experience": [
                    {
                        "title": "Software Developer I",
                        "organization": "Synthetic Product Company",
                        "start_date": "2024-01",
                        "end_date": "Present",
                        "responsibilities": [
                            "Full-stack logistics systems with REST APIs.",
                            db_text,
                            adjacent_text,
                            initiative_text,
                        ],
                        "evidence": [
                            _evidence("Software Developer I").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence("Full-stack logistics systems with REST APIs.").model_dump(mode="json"),
                            _evidence(db_text).model_dump(mode="json"),
                            _evidence(adjacent_text).model_dump(mode="json"),
                            _evidence(initiative_text).model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        jd = self._full_stack_jd()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = root / "structured-resume.json"
            jd_path = root / "structured-jd.json"
            output_path = root / "evaluation.json"
            candidate_path.write_text(json.dumps(candidate.model_dump(mode="json"), indent=2), encoding="utf-8")
            jd_path.write_text(json.dumps(jd.model_dump(mode="json"), indent=2), encoding="utf-8")

            report = evaluate_candidate_files(candidate_path, jd_path, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        source_by_id = {criterion.criterion_id: criterion for criterion in jd.scoring_rubric}
        result_by_id = {item.criterion_id: item for item in report.criterion_evaluations}
        json_by_id = {item["criterion_id"]: item for item in payload["criterion_evaluations"]}

        self.assertEqual(set(result_by_id), set(source_by_id))
        self.assertEqual(set(json_by_id), set(source_by_id))
        for criterion_id, source in source_by_id.items():
            with self.subTest(criterion_id=criterion_id):
                result = result_by_id[criterion_id]
                dumped = json_by_id[criterion_id]
                self.assertEqual(result.criterion_name, source.name)
                self.assertEqual(result.category, source.category.value)
                self.assertEqual(result.weight, source.weight)
                self.assertEqual(result.requirement_ids, source.requirement_ids)
                self.assertEqual(result.component_assessment.get("required_skill_names"), source.required_skill_names)
                self.assertEqual(result.component_assessment.get("preferred_skill_names"), source.preferred_skill_names)
                self.assertEqual(result.component_assessment.get("evaluation_dimensions"), source.evaluation_dimensions)
                self.assertEqual(dumped["criterion_name"], source.name)
                self.assertEqual(dumped["category"], source.category.value)
                self.assertEqual(dumped["weight"], source.weight)
                self.assertEqual(dumped["requirement_ids"], source.requirement_ids)
                if source.category.value in {"skill", "leadership"}:
                    self.assertIsInstance(dumped["component_assessment"], dict)
                    self.assertGreater(len(dumped["component_assessment"]["components"]), 0)

        technical = result_by_id["crit_01"]
        self.assertEqual(technical.requirement_ids, ["req_tech_01", "req_tech_02"])
        self.assertIn("SQL/database management", technical.component_assessment["supported_core_components"])
        self.assertIn("PostgreSQL", technical.component_assessment["supported_controlled_equivalents"])
        self.assertIn("JavaScript", technical.component_assessment["supported_adjacent_components"])
        self.assertNotIn("JavaScript", technical.component_assessment["supported_core_components"])
        self.assertIn("PHP", technical.component_assessment["missing_core_components"])
        self.assertIn("Laravel", technical.component_assessment["missing_core_components"])
        self.assertIn("WordPress", technical.component_assessment["missing_core_components"])
        self.assertEqual(result_by_id["crit_03"].criterion_name, "Soft Skills & Proactivity")
        self.assertEqual(result_by_id["crit_03"].component_assessment["evaluation_dimensions"], ["Communication", "Ownership", "Proactivity"])
        self.assertNotIn("React.js", str(result_by_id["crit_03"].component_assessment))
        self.assertEqual(result_by_id["crit_04"].criterion_name, "Additional Technical Skills")
        self.assertEqual(result_by_id["crit_04"].component_assessment["preferred_skill_names"], ["DevOps", "SEO", "UI/UX"])
        self.assertNotIn("Proactive Problem Solving", str(result_by_id["crit_04"].model_dump()))
        self.assertIn("Experience & Domain Fit", report.strengths)

    def test_controlled_database_equivalents_stay_with_database_owner(self) -> None:
        db_text = "Optimized PostgreSQL/MySQL database operations for reliability and performance."
        frontend_text = "Built React.js dashboards with HTML and CSS."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["MySQL", "PostgreSQL", "React.js", "HTML", "CSS"],
                "experience": [
                    {
                        "title": "Software Developer I",
                        "start_date": "2024-01",
                        "end_date": "Present",
                        "responsibilities": [
                            "Full-stack logistics systems with REST APIs.",
                            db_text,
                            frontend_text,
                        ],
                        "evidence": [
                            _evidence("Software Developer I").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence("Full-stack logistics systems with REST APIs.").model_dump(mode="json"),
                            _evidence(db_text).model_dump(mode="json"),
                            _evidence(frontend_text).model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        jd = self._live_shape_jd()

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        by_id = {item.criterion_id: item for item in report.criterion_evaluations}

        self.assertEqual(by_id["crit_01"].component_assessment["controlled_equivalent_components"], [])
        self.assertEqual(by_id["crit_01"].component_assessment["supported_controlled_equivalents"], [])
        self.assertEqual(by_id["crit_03"].component_assessment["controlled_equivalent_components"], [])
        self.assertEqual(by_id["crit_03"].component_assessment["supported_controlled_equivalents"], [])
        self.assertIn("SQL", by_id["crit_02"].component_assessment["supported_core_components"])
        self.assertIn("MySQL", by_id["crit_02"].component_assessment["supported_core_components"])
        self.assertIn("PostgreSQL", by_id["crit_02"].component_assessment["supported_controlled_equivalents"])

    def test_mixed_experience_database_criterion_uses_independent_components(self) -> None:
        jd = self._mixed_experience_database_jd()

        def score(responsibilities: list[str], *, start_date: str = "2024-01", title: str = "Software Developer I"):
            candidate = CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate",
                    "source_document_id": "resume",
                    "experience": [
                        {
                            "title": title,
                            "start_date": start_date,
                            "end_date": "Present",
                            "responsibilities": responsibilities,
                            "evidence": [
                                _evidence(title).model_dump(mode="json"),
                                _evidence(f"{start_date} - Present").model_dump(mode="json"),
                                *[_evidence(text).model_dump(mode="json") for text in responsibilities],
                            ],
                        }
                    ],
                }
            )
            return evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]

        full = score(["Full-stack systems with REST APIs.", "Optimized MySQL and SQL database operations."])
        no_database = score(["Full-stack systems with REST APIs."])
        no_experience = score(["Optimized MySQL and SQL database operations."], start_date="2026-01")
        unrelated = score(
            ["Administrative scheduling.", "Optimized MySQL and SQL database operations."],
            title="Administrative Coordinator",
        )

        self.assertEqual(full.match_fraction, 1)
        self.assertEqual([item["term"] for item in full.component_assessment["components"]], ["Relevant experience duration/relevance", "SQL", "MySQL"])
        self.assertLess(no_database.match_fraction, 1)
        self.assertIn("SQL", no_database.missing_information)
        self.assertIn("MySQL", no_database.missing_information)
        self.assertLess(no_experience.match_fraction, 1)
        self.assertIn("Relevant experience duration/relevance", no_experience.missing_information)
        self.assertLess(unrelated.match_fraction, 1)
        self.assertIn("Relevant experience duration/relevance", unrelated.missing_information)

    def test_postgresql_satisfies_explicit_equivalent_database_component(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_db",
                    category=RequirementCategory.SKILL,
                    description="Experience with MySQL or similar enterprise databases.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="database",
                    name="Database",
                    category=RequirementCategory.SKILL,
                    description="Database management",
                    weight=100,
                    requirement_ids=["req_db"],
                    required_skill_names=["SQL/database management"],
                )
            ],
        )
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "responsibilities": ["Implemented PostgreSQL database operations."],
                        "evidence": [_evidence("Implemented PostgreSQL database operations.").model_dump(mode="json")],
                    }
                ],
            }
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        criterion = report.criterion_evaluations[0]

        self.assertEqual(report.eligibility.requirement_statuses["req_db"], "met")
        self.assertEqual(criterion.match_fraction, 1)
        self.assertEqual(criterion.component_assessment["supported_controlled_equivalents"], ["PostgreSQL"])

    def test_preferred_skill_components_are_not_replaced_by_description(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="soft",
                    name="Soft Skills & Proactivity",
                    category=RequirementCategory.OTHER,
                    description="Communication, ownership, and proactive approach.",
                    weight=100,
                    preferred_skill_names=["Proactive Problem Solving"],
                )
            ],
        )
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "responsibilities": ["Worked with a team on routine delivery."],
                        "evidence": [_evidence("Worked with a team on routine delivery.").model_dump(mode="json")],
                    }
                ],
            }
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        criterion = report.criterion_evaluations[0]

        self.assertEqual(criterion.component_assessment["required_core_components"], [])
        self.assertEqual(criterion.component_assessment["preferred_components"], ["Proactive Problem Solving"])
        self.assertEqual(criterion.component_assessment["missing_preferred_components"], ["Proactive Problem Solving"])
        self.assertEqual(criterion.missing_information, [])
        self.assertNotIn("Communication, ownership", str(criterion.component_assessment))

    def test_shared_experience_qualification_reused_by_linked_mixed_criterion(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["Git"],
                "experience": [
                    {
                        "title": "Software Developer",
                        "organization": "Product Studio",
                        "start_date": "2023-01",
                        "end_date": "2025-07",
                        "responsibilities": ["Delivered full-stack development for REST API products."],
                        "evidence": [
                            _evidence("Software Developer").model_dump(mode="json"),
                            _evidence("January 2023 - July 2025").model_dump(mode="json"),
                            _evidence("Delivered full-stack development for REST API products.").model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )
        report = evaluate_candidate(candidate.scoring_safe_profile, self._experience_devops_jd())
        criterion = report.criterion_evaluations[0]
        detail = report.eligibility.requirement_details[0]
        experience_component = criterion.component_assessment["components"][0]

        self.assertEqual(detail["status"], "met")
        self.assertEqual(detail["confirmed_months"], 30)
        self.assertEqual(detail["required_months"], 24)
        self.assertTrue(experience_component["matched"])
        self.assertEqual(experience_component["term"], "Relevant experience duration/relevance")
        self.assertIn("req_exp_01", experience_component["source"])
        self.assertEqual(
            {item["exact_text"] for item in detail["evidence"]},
            {item["exact_text"] for item in experience_component["evidence"]},
        )
        self.assertIn("DevOps", criterion.component_assessment["missing_core_components"])
        self.assertEqual(criterion.component_assessment["preferred_components"], ["GitLab", "GitHub"])
        self.assertEqual(criterion.component_assessment["missing_preferred_components"], ["GitLab", "GitHub"])
        self.assertNotIn("GitHub", criterion.component_assessment["supported_preferred_components"])
        self.assertEqual(criterion.component_assessment["supported_adjacent_components"], ["Git version control"])

    def test_insufficient_and_uncertain_experience_are_consistent(self) -> None:
        jd = self._experience_devops_jd()

        def score(start_date: str | None, end_date: str | None):
            candidate = CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate",
                    "source_document_id": "resume",
                    "experience": [
                        {
                            "title": "Software Developer",
                            "organization": "Product Studio",
                            "start_date": start_date,
                            "end_date": end_date,
                            "responsibilities": ["Delivered full-stack development for REST API products."],
                            "evidence": [
                                _evidence("Software Developer").model_dump(mode="json"),
                                _evidence(str(start_date or "unknown")).model_dump(mode="json"),
                                _evidence("Delivered full-stack development for REST API products.").model_dump(mode="json"),
                            ],
                        }
                    ],
                }
            )
            report = evaluate_candidate(candidate.scoring_safe_profile, jd)
            return report, report.criterion_evaluations[0].component_assessment["components"][0]

        insufficient, insufficient_component = score("2025-01", "2025-12")
        uncertain, uncertain_component = score("2024-01", None)

        self.assertEqual(insufficient.eligibility.requirement_statuses["req_exp_01"], "not_met")
        self.assertFalse(insufficient_component["matched"])
        self.assertEqual(uncertain.eligibility.requirement_statuses["req_exp_01"], "unclear")
        self.assertEqual(uncertain.criterion_evaluations[0].status, CriterionStatus.UNCLEAR)
        self.assertFalse(uncertain_component["matched"])

    def test_git_github_and_gitlab_exact_matching_boundaries(self) -> None:
        def index_for(skills: list[str]):
            candidate = CandidateProfile.model_validate(
                {"candidate_id": "candidate", "source_document_id": "resume", "skills": skills}
            )
            return build_candidate_evidence_index(candidate.scoring_safe_profile)

        self.assertEqual(match_terms(["GitHub"], index_for(["Git"])).fraction, 0)
        self.assertEqual(match_terms(["GitHub"], index_for(["GitHub"])).fraction, 1)
        self.assertEqual(match_terms(["GitHub"], index_for(["github.com/example"])).fraction, 1)
        self.assertEqual(match_terms(["GitHub"], index_for(["GitLab"])).fraction, 0)
        self.assertEqual(match_terms(["GitLab"], index_for(["Git"])).fraction, 0)

        candidate = CandidateProfile.model_validate(
            {"candidate_id": "candidate", "source_document_id": "resume", "skills": ["Git"]}
        )
        jd = self._experience_devops_jd()
        criterion = evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]

        self.assertEqual(criterion.component_assessment["supported_preferred_components"], [])
        self.assertEqual(criterion.component_assessment["supported_adjacent_components"], ["Git version control"])

    def test_mixed_required_preferred_adjacent_completeness_and_strengths_are_separate(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Developer",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="mixed",
                    name="Mixed Components",
                    category=RequirementCategory.SKILL,
                    description="Python required, GitHub preferred, JavaScript adjacent.",
                    weight=100,
                    required_skill_names=["Python"],
                    preferred_skill_names=["GitHub"],
                    adjacent_skill_names=["JavaScript"],
                )
            ],
        )
        candidate = CandidateProfile.model_validate(
            {"candidate_id": "candidate", "source_document_id": "resume", "skills": ["Python", "JavaScript"]}
        )
        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        criterion = report.criterion_evaluations[0]
        diagnostics = report.evidence_completeness_diagnostics

        self.assertEqual(criterion.component_assessment["required_core_components"], ["Python"])
        self.assertEqual(criterion.component_assessment["preferred_components"], ["GitHub"])
        self.assertEqual(criterion.component_assessment["missing_core_components"], [])
        self.assertEqual(criterion.component_assessment["missing_preferred_components"], ["GitHub"])
        self.assertEqual(criterion.component_assessment["source_declared_adjacent_components"], ["JavaScript"])
        self.assertEqual(diagnostics["total_required_core_components"], 1)
        self.assertEqual(diagnostics["total_preferred_components"], 1)
        self.assertEqual(diagnostics["penalties"]["missing_core_components"], 0)
        self.assertEqual(diagnostics["penalties"]["missing_preferred_components"], 1)
        self.assertTrue(any(item.startswith("required/core:") for item in report.strengths))
        self.assertTrue(any(item.startswith("adjacent:") for item in report.strengths))
        self.assertTrue(any(item.startswith("preferred gap:") for item in report.gaps))

    def test_soft_skill_dimensions_require_dimension_specific_evidence(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Developer",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="soft",
                    name="Soft Skills",
                    category=RequirementCategory.LEADERSHIP,
                    description="Communication, ownership, proactivity, problem-solving.",
                    weight=100,
                    evaluation_dimensions=["Communication", "Ownership", "Proactivity", "Problem-solving"],
                )
            ],
        )

        def supported(responsibility: str) -> list[str]:
            candidate = CandidateProfile.model_validate(
                {
                    "candidate_id": "candidate",
                    "source_document_id": "resume",
                    "experience": [
                        {
                            "responsibilities": [responsibility],
                            "evidence": [_evidence(responsibility).model_dump(mode="json")],
                        }
                    ],
                }
            )
            criterion = evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]
            return criterion.component_assessment["supported_evaluation_dimensions"]

        self.assertNotIn("Communication", supported("Collaborated with teams on routine delivery."))
        self.assertNotIn("Ownership", supported("Developed a feature for the platform."))
        self.assertNotIn("Proactivity", supported("Implemented an API for the product."))
        self.assertIn("Ownership", supported("Owned the billing workstream and was accountable for delivery."))
        self.assertIn("Proactivity", supported("Initiated an automation that reduced manual deployment work."))
        self.assertIn("Communication", supported("Presented findings to stakeholders and authored documentation."))
        self.assertIn("Problem-solving", supported("Diagnosed the outage root cause and resolved the production issue."))

    def test_dimensions_do_not_delete_linked_required_preferred_or_adjacent_components(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Developer",
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_docker",
                    category=RequirementCategory.SKILL,
                    description="Docker is required.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="all-sources",
                    name="All Sources",
                    category=RequirementCategory.LEADERSHIP,
                    description="Docker, Python, GitHub, JavaScript, communication.",
                    weight=100,
                    requirement_ids=["req_docker"],
                    required_skill_names=["Python"],
                    preferred_skill_names=["GitHub"],
                    adjacent_skill_names=["JavaScript"],
                    evaluation_dimensions=["Communication"],
                )
            ],
        )
        communication = "Presented findings to stakeholders."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["Docker", "Python", "GitHub", "JavaScript"],
                "experience": [
                    {
                        "responsibilities": [communication],
                        "evidence": [_evidence(communication).model_dump(mode="json")],
                    }
                ],
            }
        )

        criterion = evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]
        assessment = criterion.component_assessment

        self.assertIn("Docker", assessment["required_core_components"])
        self.assertIn("Python", assessment["required_core_components"])
        self.assertEqual(assessment["preferred_components"], ["GitHub"])
        self.assertEqual(assessment["source_declared_adjacent_components"], ["JavaScript"])
        self.assertEqual(assessment["evaluation_dimension_components"], ["Communication"])
        self.assertIn("Docker", assessment["supported_core_components"])
        self.assertIn("Python", assessment["supported_core_components"])
        self.assertEqual(assessment["supported_preferred_components"], ["GitHub"])
        self.assertEqual(assessment["supported_adjacent_components"], ["JavaScript"])
        self.assertEqual(assessment["supported_evaluation_dimensions"], ["Communication"])
        self.assertEqual(criterion.match_fraction, 1)

    def test_component_source_combination_matrix_preserves_all_populated_fields(self) -> None:
        source_cases = [
            {"name": "required only", "required": ["Python"]},
            {"name": "preferred only", "preferred": ["GitHub"]},
            {"name": "adjacent only", "adjacent": ["JavaScript"]},
            {"name": "dimensions only", "dimensions": ["Communication"]},
            {"name": "linked only", "linked": True},
            {"name": "required preferred", "required": ["Python"], "preferred": ["GitHub"]},
            {"name": "required adjacent", "required": ["Python"], "adjacent": ["JavaScript"]},
            {"name": "required dimensions", "required": ["Python"], "dimensions": ["Communication"]},
            {"name": "preferred dimensions", "preferred": ["GitHub"], "dimensions": ["Communication"]},
            {"name": "linked required", "linked": True, "required": ["Python"]},
            {"name": "linked dimensions", "linked": True, "dimensions": ["Communication"]},
            {"name": "required preferred dimensions", "required": ["Python"], "preferred": ["GitHub"], "dimensions": ["Communication"]},
            {
                "name": "all non-experience sources",
                "linked": True,
                "required": ["Python"],
                "preferred": ["GitHub"],
                "adjacent": ["JavaScript"],
                "dimensions": ["Communication"],
            },
        ]
        communication = "Presented findings to stakeholders."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["Docker", "Python", "GitHub", "JavaScript"],
                "experience": [
                    {
                        "responsibilities": [communication],
                        "evidence": [_evidence(communication).model_dump(mode="json")],
                    }
                ],
            }
        )

        for index, case in enumerate(source_cases):
            with self.subTest(case=case["name"]):
                linked = bool(case.get("linked"))
                jd = StructuredJD(
                    job_id=f"job-{index}",
                    source_document_id="jd",
                    job_title="Developer",
                    mandatory_requirements=[
                        JDRequirement(
                            requirement_id="req_docker",
                            category=RequirementCategory.SKILL,
                            description="Docker is required.",
                            requirement_type=RequirementType.MINIMUM,
                            mandatory=True,
                        )
                    ]
                    if linked
                    else [],
                    scoring_rubric=[
                        ScoringCriterion(
                            criterion_id="criterion",
                            name="Criterion",
                            category=RequirementCategory.SKILL,
                            description=case["name"],
                            weight=100,
                            requirement_ids=["req_docker"] if linked else [],
                            required_skill_names=case.get("required", []),
                            preferred_skill_names=case.get("preferred", []),
                            adjacent_skill_names=case.get("adjacent", []),
                            evaluation_dimensions=case.get("dimensions", []),
                        )
                    ],
                )
                criterion = evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]
                assessment = criterion.component_assessment
                self.assertEqual(criterion.weighted_score, criterion.weight * criterion.match_fraction)
                for term in case.get("required", []):
                    self.assertIn(term, assessment["required_core_components"])
                for term in case.get("preferred", []):
                    self.assertIn(term, assessment["preferred_components"])
                    self.assertNotIn(term, assessment["required_core_components"])
                for term in case.get("adjacent", []):
                    self.assertIn(term, assessment["source_declared_adjacent_components"])
                    self.assertNotIn(term, assessment["required_core_components"])
                for term in case.get("dimensions", []):
                    self.assertIn(term, assessment["evaluation_dimension_components"])
                if linked:
                    self.assertIn("req_docker", assessment["represented_requirement_ids"])

    def test_serialization_order_duplicate_and_report_consistency_invariants(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Developer",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="mixed",
                    name="Mixed Components",
                    category=RequirementCategory.SKILL,
                    description="Python required, GitHub preferred, JavaScript adjacent.",
                    weight=100,
                    required_skill_names=["Python"],
                    preferred_skill_names=["GitHub"],
                    adjacent_skill_names=["JavaScript"],
                )
            ],
        )
        duplicate_python = _evidence("Python").model_dump(mode="json")
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["JavaScript", "Python", "GitHub", "UnrelatedSkill", "Python"],
                "experience": [
                    {
                        "responsibilities": ["Python"],
                        "evidence": [duplicate_python, duplicate_python],
                    }
                ],
            }
        )
        reloaded_jd = StructuredJD.model_validate_json(jd.model_dump_json())
        reloaded_candidate = CandidateProfile.model_validate_json(candidate.model_dump_json())

        report = evaluate_candidate(reloaded_candidate.scoring_safe_profile, reloaded_jd, evaluation_id="eval-matrix")
        report_again = evaluate_candidate(reloaded_candidate.scoring_safe_profile, reloaded_jd, evaluation_id="eval-matrix")

        self.assertEqual(
            report.model_dump(mode="json", exclude={"criterion_evaluations": {0: {"evidence": True}}})["criterion_evaluations"],
            report_again.model_dump(mode="json", exclude={"criterion_evaluations": {0: {"evidence": True}}})["criterion_evaluations"],
        )
        self.assertEqual(report.final_score, report_again.final_score)
        self.assertEqual(report.evidence_completeness, report_again.evidence_completeness)
        self.assertEqual(report.criterion_evaluations[0].match_fraction, 1)
        self.assertEqual(report.evidence_completeness_diagnostics["deduplicated_source_count"], 3)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            from chandra_eval.reports.csv_report import write_csv_report
            from chandra_eval.reports.html_report import write_html_report
            from chandra_eval.reports.json_report import write_json_report

            json_path = write_json_report(report, root / "evaluation.json")
            csv_path = write_csv_report(report, root / "evaluation.csv")
            html_path = write_html_report(report, root / "evaluation.html")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            csv_text = csv_path.read_text(encoding="utf-8")
            html_text = html_path.read_text(encoding="utf-8")

        self.assertIn("preferred_components", str(payload))
        self.assertIn("preferred considered: GitHub", csv_text)
        self.assertIn("source adjacent considered: JavaScript", html_text)

    def test_golden_synthetic_scenarios_cover_expected_invariant_outcomes(self) -> None:
        scenarios = [
            ("strong", ["PHP", "Laravel", "WordPress", "MySQL", "JavaScript"], "2023-01", "2025-07", 1.0),
            ("adjacent-stack", ["MySQL", "JavaScript", "React.js", "Node.js"], "2023-01", "2025-07", 0.5),
            ("experience-only", [], "2023-01", "2025-07", 0.25),
            ("unrelated", ["Excel"], "2020-01", "2025-07", 0.0),
        ]
        jd = self._golden_full_stack_jd()
        for name, skills, start, end, expected_technical_fraction in scenarios:
            with self.subTest(name=name):
                relevant = "Delivered full-stack development for REST API products."
                responsibilities = [relevant] if name != "unrelated" else ["Coordinated office administration."]
                candidate = CandidateProfile.model_validate(
                    {
                        "candidate_id": name,
                        "source_document_id": "resume",
                        "skills": skills,
                        "experience": [
                            {
                                "title": "Software Developer" if name != "unrelated" else "Office Administrator",
                                "start_date": start,
                                "end_date": end,
                                "responsibilities": responsibilities,
                                "evidence": [
                                    _evidence("Software Developer" if name != "unrelated" else "Office Administrator").model_dump(mode="json"),
                                    _evidence(f"{start} - {end}").model_dump(mode="json"),
                                    *[_evidence(text).model_dump(mode="json") for text in responsibilities],
                                ],
                            }
                        ],
                    }
                )
                report = evaluate_candidate(candidate.scoring_safe_profile, jd)
                technical = {item.criterion_id: item for item in report.criterion_evaluations}["technical"]
                self.assertEqual(technical.match_fraction, expected_technical_fraction)
                self.assertTrue(report.human_review_required)
                for criterion in report.criterion_evaluations:
                    if criterion.weighted_score > 0:
                        self.assertGreater(len(criterion.evidence), 0)

    def test_dimension_gap_labels_are_category_aware(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Role",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="technical",
                    name="Technical Dimension",
                    category=RequirementCategory.SKILL,
                    description="Security review",
                    weight=50,
                    evaluation_dimensions=["Security review"],
                ),
                ScoringCriterion(
                    criterion_id="soft",
                    name="Soft Dimension",
                    category=RequirementCategory.LEADERSHIP,
                    description="Empathy",
                    weight=50,
                    evaluation_dimensions=["Empathy"],
                ),
            ],
        )
        candidate = CandidateProfile.model_validate(
            {"candidate_id": "candidate", "source_document_id": "resume"}
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)

        self.assertIn("unverified technical dimension: Technical Dimension: Security review", report.gaps)
        self.assertIn("unverified soft-skill dimension: Soft Dimension: Empathy", report.gaps)

    def test_shared_experience_evidence_satisfies_years_of_experience_dimension(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_exp_01",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="experience",
                    name="Experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Relevant years of experience.",
                    weight=100,
                    requirement_ids=["req_exp_01"],
                    evaluation_dimensions=["Years of experience"],
                )
            ],
        )
        evidence = [
            _evidence("Software Developer").model_dump(mode="json"),
            _evidence("January 2023 - July 2025").model_dump(mode="json"),
            _evidence("Delivered full-stack development with REST APIs.").model_dump(mode="json"),
        ]
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "title": "Software Developer",
                        "start_date": "2023-01",
                        "end_date": "2025-07",
                        "responsibilities": ["Delivered full-stack development with REST APIs."],
                        "evidence": evidence,
                    }
                ],
            }
        )

        criterion = evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]
        dimension = next(
            item for item in criterion.component_assessment["components"] if item["term"] == "Years of experience"
        )

        self.assertTrue(dimension["matched"])
        self.assertEqual(dimension["shared_evidence_with"], ["Relevant experience duration/relevance"])
        self.assertIn("Years of experience", criterion.component_assessment["supported_evaluation_dimensions"])
        self.assertNotIn("Years of experience", criterion.component_assessment["missing_evaluation_dimensions"])
        self.assertEqual(criterion.match_fraction, 1)

    def test_supported_technical_breadth_satisfies_versatility_dimension(self) -> None:
        jd = StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Developer",
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="technical-breadth",
                    name="Technical Breadth",
                    category=RequirementCategory.SKILL,
                    description="Modern JavaScript stack breadth.",
                    weight=100,
                    required_skill_names=["React.js", "Node.js", "Express.js"],
                    evaluation_dimensions=["Tech stack versatility"],
                )
            ],
        )
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["React.js", "Node.js", "Express.js"],
            }
        )

        criterion = evaluate_candidate(candidate.scoring_safe_profile, jd).criterion_evaluations[0]
        dimension = next(
            item for item in criterion.component_assessment["components"] if item["term"] == "Tech stack versatility"
        )

        self.assertTrue(dimension["matched"])
        self.assertIn("technical dimension rule v1: tech-stack-versatility", dimension["source"])
        self.assertIn("Tech stack versatility", criterion.component_assessment["supported_evaluation_dimensions"])
        self.assertEqual(criterion.match_fraction, 1)

    def test_report_gaps_use_one_structured_entry_per_missing_component(self) -> None:
        jd = self._full_stack_jd()
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "skills": ["MySQL", "JavaScript"],
            }
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, jd)
        normalized = [item.lower() for item in report.gaps]

        self.assertIn("mandatory/core gap: Technical Stack Proficiency: PHP".lower(), normalized)
        self.assertNotIn("php", normalized)
        self.assertEqual(len(normalized), len(set(normalized)))
        self.assertTrue(all(item.startswith(("mandatory/core gap:", "preferred gap:", "unverified ")) for item in report.gaps))

    def test_met_experience_requirement_uses_explanation_not_ambiguity(self) -> None:
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate",
                "source_document_id": "resume",
                "experience": [
                    {
                        "title": "Software Developer I",
                        "organization": "Synthetic Product Company",
                        "start_date": "2024-01",
                        "end_date": "Present",
                        "responsibilities": ["Full-stack logistics systems with REST APIs."],
                        "evidence": [
                            _evidence("Software Developer I").model_dump(mode="json"),
                            _evidence("Synthetic Product Company").model_dump(mode="json"),
                            _evidence("January 2024 - Present").model_dump(mode="json"),
                            _evidence("Full-stack logistics systems with REST APIs.").model_dump(mode="json"),
                        ],
                    }
                ],
            }
        )

        report = evaluate_candidate(candidate.scoring_safe_profile, self._full_stack_jd())
        detail = next(item for item in report.eligibility.requirement_details if item["requirement_id"] == "req_exp_01")

        self.assertEqual(detail["status"], "met")
        self.assertEqual(detail["ambiguity"], [])
        self.assertIn("confirmed months", detail["explanation"])

    def test_duplicate_criterion_ids_are_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            StructuredJD(
                job_id="job",
                source_document_id="jd",
                job_title="Role",
                scoring_rubric=[
                    ScoringCriterion(
                        criterion_id="duplicate",
                        name="One",
                        category=RequirementCategory.SKILL,
                        description="Python",
                        weight=50,
                    ),
                    ScoringCriterion(
                        criterion_id="duplicate",
                        name="Two",
                        category=RequirementCategory.SKILL,
                        description="JavaScript",
                        weight=50,
                    ),
                ],
            )

    def test_component_to_discrete_mapping_covers_all_valid_fractions(self) -> None:
        components = ["PHP", "Laravel", "WordPress", "SQL/database management"]
        expected_by_count = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1}
        for count, expected in expected_by_count.items():
            with self.subTest(count=count):
                skills = components[:count]
                if "SQL/database management" in skills:
                    skills = [*skills[:-1], "MySQL"]
                candidate = CandidateProfile.model_validate(
                    {
                        "candidate_id": "candidate",
                        "source_document_id": "resume",
                        "skills": skills,
                    }
                )
                jd = StructuredJD(
                    job_id="job",
                    source_document_id="jd",
                    job_title="Full Stack Developer",
                    scoring_rubric=[
                        ScoringCriterion(
                            criterion_id="technical-stack",
                            name="Technical stack",
                            category=RequirementCategory.SKILL,
                            description="PHP; Laravel; WordPress; SQL/database management",
                            weight=100,
                            required_skill_names=components,
                        )
                    ],
                )
                report = evaluate_candidate(candidate.scoring_safe_profile, jd)
                self.assertEqual(report.criterion_evaluations[0].match_fraction, expected)

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

    def _full_stack_jd(self) -> StructuredJD:
        return StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_tech_01",
                    category=RequirementCategory.SKILL,
                    description="Experience with PHP, Laravel and WordPress.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
                JDRequirement(
                    requirement_id="req_tech_02",
                    category=RequirementCategory.SKILL,
                    description="Experience with MySQL or similar enterprise databases.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
                JDRequirement(
                    requirement_id="req_exp_01",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                ),
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="crit_01",
                    name="Technical Stack Proficiency",
                    category=RequirementCategory.SKILL,
                    description="PHP; Laravel; WordPress; SQL/database management",
                    weight=40,
                    requirement_ids=["req_tech_01", "req_tech_02"],
                    required_skill_names=["PHP", "Laravel", "WordPress", "SQL/database management"],
                ),
                ScoringCriterion(
                    criterion_id="crit_02",
                    name="Experience & Domain Fit",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    weight=30,
                    requirement_ids=["req_exp_01"],
                ),
                ScoringCriterion(
                    criterion_id="crit_03",
                    name="Soft Skills & Proactivity",
                    category=RequirementCategory.LEADERSHIP,
                    description="Communication, ownership, and proactivity",
                    weight=20,
                    evaluation_dimensions=["Communication", "Ownership", "Proactivity"],
                ),
                ScoringCriterion(
                    criterion_id="crit_04",
                    name="Additional Technical Skills",
                    category=RequirementCategory.SKILL,
                    description="DevOps, SEO, UI/UX",
                    weight=10,
                    preferred_skill_names=["DevOps", "SEO", "UI/UX"],
                ),
            ],
        )

    def _live_shape_jd(self) -> StructuredJD:
        return StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_tech_01",
                    category=RequirementCategory.SKILL,
                    description="Experience with PHP, Laravel and WordPress.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
                JDRequirement(
                    requirement_id="req_exp_01",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                ),
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="crit_01",
                    name="Core Technical Stack",
                    category=RequirementCategory.SKILL,
                    description="PHP, Laravel, WordPress, JavaScript",
                    weight=40,
                    requirement_ids=["req_tech_01"],
                    required_skill_names=["PHP", "Laravel", "WordPress", "JavaScript"],
                ),
                ScoringCriterion(
                    criterion_id="crit_02",
                    name="Experience & Database Management",
                    category=RequirementCategory.EXPERIENCE,
                    description="Relevant years of experience and MySQL/SQL knowledge.",
                    weight=30,
                    requirement_ids=["req_exp_01"],
                    required_skill_names=["SQL", "MySQL"],
                ),
                ScoringCriterion(
                    criterion_id="crit_03",
                    name="Frontend & UI/UX",
                    category=RequirementCategory.SKILL,
                    description="Frontend implementation and UI/UX.",
                    weight=20,
                    required_skill_names=["HTML", "CSS", "React.js", "UI/UX Design"],
                ),
                ScoringCriterion(
                    criterion_id="crit_04",
                    name="Soft Skills & Proactivity",
                    category=RequirementCategory.OTHER,
                    description="Communication, ownership, and proactive approach.",
                    weight=10,
                    preferred_skill_names=["Proactive Problem Solving"],
                ),
            ],
        )

    def _mixed_experience_database_jd(self) -> StructuredJD:
        return StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_exp",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                )
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="exp-db",
                    name="Experience & Database Management",
                    category=RequirementCategory.EXPERIENCE,
                    description="Relevant years of experience and MySQL/SQL knowledge.",
                    weight=100,
                    requirement_ids=["req_exp"],
                    required_skill_names=["SQL", "MySQL"],
                )
            ],
        )

    def _experience_devops_jd(self) -> StructuredJD:
        return StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_exp_01",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                ),
                JDRequirement(
                    requirement_id="req_tech_03",
                    category=RequirementCategory.SKILL,
                    description="Familiarity with DevOps practices and tools for continuous integration and deployment.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="crit_02",
                    name="Experience and DevOps",
                    category=RequirementCategory.EXPERIENCE,
                    description="Relevant experience, DevOps, GitLab and GitHub.",
                    weight=100,
                    requirement_ids=["req_exp_01", "req_tech_03"],
                    required_skill_names=["DevOps"],
                    preferred_skill_names=["GitLab", "GitHub"],
                )
            ],
        )

    def _golden_full_stack_jd(self) -> StructuredJD:
        return StructuredJD(
            job_id="job",
            source_document_id="jd",
            job_title="Full Stack Developer",
            minimum_experience_months=24,
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req_exp",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                ),
                JDRequirement(
                    requirement_id="req_db",
                    category=RequirementCategory.SKILL,
                    description="Experience with MySQL or similar enterprise databases.",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
            ],
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="technical",
                    name="Technical Stack",
                    category=RequirementCategory.SKILL,
                    description="PHP, Laravel, WordPress, SQL/database management.",
                    weight=50,
                    requirement_ids=["req_db"],
                    required_skill_names=["PHP", "Laravel", "WordPress", "SQL/database management"],
                ),
                ScoringCriterion(
                    criterion_id="experience",
                    name="Experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Minimum 24 months of relevant full-stack development experience.",
                    weight=50,
                    requirement_ids=["req_exp"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
