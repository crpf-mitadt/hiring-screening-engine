import unittest

from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.scoring.experience import calculate_experience_duration
from chandra_eval.schemas.common import EvidenceRef, EvidenceStatus
from chandra_eval.schemas.evaluation import EligibilityStatus
from chandra_eval.schemas.jd import JDRequirement, RequirementCategory, RequirementType, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import CandidateProfile


def _evidence(text: str, block: int) -> EvidenceRef:
    return EvidenceRef(
        source_type="resume_ocr",
        source_document_id="sanitized-live-resume",
        page_number=1,
        block_index=block,
        exact_text=text,
        normalized_text=text,
        evidence_status=EvidenceStatus.SUPPORTED,
    )


class LiveScoringRegressionTests(unittest.TestCase):
    def test_sanitized_live_fixture_scores_from_structured_evidence(self) -> None:
        office = "Created client-ready presentations and reports using MS Excel, Word, and PowerPoint."
        crm = "Managed lead databases, tracked sales opportunities, and maintained CRM records for pipeline management."
        marketing = "Led lead generation, client acquisition, social-media marketing, campaign analysis, and stakeholder engagement."
        candidate = CandidateProfile.model_validate(
            {
                "candidate_id": "candidate-marketing-001",
                "source_document_id": "sanitized-live-resume",
                "summary": "Marketing research and business-development candidate.",
                "education": [
                    {
                        "degree_level": "master",
                        "degree_name": "MBA",
                        "field_of_study": None,
                        "completion_status": "completed",
                        "graduation_year": 2023,
                        "evidence": [_evidence("MBA completed in 2023", 1).model_dump(mode="json")],
                    },
                    {
                        "degree_level": "bachelor",
                        "degree_name": "BA",
                        "field_of_study": "Economics",
                        "completion_status": "completed",
                    },
                ],
                "experience": [
                    {
                        "title": "Marketing Research Analyst",
                        "organization": "Org A",
                        "employment_type": None,
                        "start_date": "2023-02",
                        "end_date": "2024-10",
                        "responsibilities": [office, crm, marketing],
                        "skills_used": ["Lead Generation", "CRM Management", "Social Media Marketing"],
                        "evidence": [
                            _evidence(office, 10).model_dump(mode="json"),
                            _evidence(crm, 11).model_dump(mode="json"),
                            _evidence(marketing, 12).model_dump(mode="json"),
                        ],
                    },
                    {
                        "title": "Marketing Research Consultant",
                        "organization": "Org B",
                        "industry_or_domain": None,
                        "employment_type": None,
                        "start_date": "2025-01",
                        "end_date": "202",
                        "responsibilities": ["Business-development support and proposal development."],
                        "evidence": [_evidence("January 2025 - October 202", 20).model_dump(mode="json")],
                    },
                ],
                "skills": [
                    "Lead Generation",
                    "Client Acquisition",
                    "CRM Management",
                    "Sales Pipeline Management",
                    "Social Media Marketing",
                    "Campaign Analysis",
                    "MS Excel",
                    "PowerPoint",
                ],
            }
        )
        jd = StructuredJD(
            job_id="marketing-coordinator",
            source_document_id="sanitized-jd",
            job_title="Marketing Coordinator",
            mandatory_requirements=[
                JDRequirement(
                    requirement_id="req-education",
                    category=RequirementCategory.EDUCATION,
                    description="Graduate/Post-Graduate in Marketing, Mass Communication, or related field",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
                JDRequirement(
                    requirement_id="req-experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="2-4 years of educational marketing, admissions, or field-marketing experience",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                    minimum_value=24,
                    unit="months",
                ),
                JDRequirement(
                    requirement_id="req-language",
                    category=RequirementCategory.LANGUAGE,
                    description="English, Hindi, and Marathi communication",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
                JDRequirement(
                    requirement_id="req-office",
                    category=RequirementCategory.SKILL,
                    description="MS Office proficiency, specifically Excel and PowerPoint",
                    requirement_type=RequirementType.MINIMUM,
                    mandatory=True,
                ),
            ],
            preferred_requirements=[
                JDRequirement(
                    requirement_id="pref-upsc",
                    category=RequirementCategory.DOMAIN,
                    description="UPSC/Competitive Exam Sector Knowledge",
                    requirement_type=RequirementType.PREFERRED,
                    preferred=True,
                )
            ],
            required_skills=[
                "Marketing Campaign Management",
                "Lead Generation",
                "Content Creation",
                "Event Planning",
                "CRM Management",
                "MS Office",
                "Social Media Management",
                "Multilingual Communication",
            ],
            preferred_skills=["UPSC/Competitive Exam Sector Knowledge", "Immediate availability"],
            minimum_experience_months=24,
            scoring_rubric=[
                ScoringCriterion(
                    criterion_id="education-experience",
                    name="Education & Relevant Experience",
                    category=RequirementCategory.EXPERIENCE,
                    description="Education and relevant marketing experience",
                    weight=30,
                    mandatory=True,
                ),
                ScoringCriterion(
                    criterion_id="marketing-outreach",
                    name="Marketing & Outreach Expertise",
                    category=RequirementCategory.RESPONSIBILITY,
                    description="Campaign management, lead generation, outreach, event management, client acquisition, stakeholder engagement",
                    weight=25,
                ),
                ScoringCriterion(
                    criterion_id="technical-crm",
                    name="Technical Savvy & CRM",
                    category=RequirementCategory.SKILL,
                    description="MS Excel, Excel, MS PowerPoint, PowerPoint, MS Office, CRM, database management, pipeline tracking",
                    weight=20,
                ),
                ScoringCriterion(
                    criterion_id="communication",
                    name="Multilingual Communication",
                    category=RequirementCategory.LANGUAGE,
                    description="English, Hindi, and Marathi communication",
                    weight=15,
                    mandatory=True,
                ),
                ScoringCriterion(
                    criterion_id="preferred-sector",
                    name="Preferred UPSC Sector Knowledge",
                    category=RequirementCategory.DOMAIN,
                    description="UPSC/Competitive Exam Sector Knowledge",
                    weight=10,
                ),
            ],
        )

        duration = calculate_experience_duration(candidate.scoring_safe_profile.experience)
        report = evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="eval-live-sanitized")

        self.assertEqual(duration.confirmed_months, 20)
        self.assertEqual(report.experience_summary["confirmed_months"], 20)
        self.assertGreater(len(report.experience_summary["uncertain_ranges"]), 0)
        self.assertNotEqual(report.eligibility.status, EligibilityStatus.FAIL)
        self.assertIn(report.eligibility.requirement_statuses["req-language"], {"not_found", "unclear"})
        self.assertIn(report.eligibility.requirement_statuses["req-experience"], {"unclear", "not_met"})
        self.assertGreater(report.final_score, 0)
        self.assertEqual(
            report.score_before_gates,
            round(sum(item.weighted_score for item in report.criterion_evaluations), 3),
        )
        self.assertGreater(report.evidence_completeness, 0)

        by_id = {item.criterion_id: item for item in report.criterion_evaluations}
        self.assertGreater(by_id["technical-crm"].match_fraction, 0)
        self.assertGreater(len(by_id["technical-crm"].evidence), 0)
        self.assertGreater(by_id["marketing-outreach"].match_fraction, 0)
        self.assertGreater(len(by_id["marketing-outreach"].evidence), 0)
        self.assertEqual(by_id["preferred-sector"].match_fraction, 0)
        self.assertNotIn("UPSC", " ".join(by_id["marketing-outreach"].missing_information))
        self.assertNotIn("UPSC", " ".join(by_id["technical-crm"].missing_information))
        self.assertIn("MS Office", by_id["technical-crm"].explanation)


if __name__ == "__main__":
    unittest.main()
