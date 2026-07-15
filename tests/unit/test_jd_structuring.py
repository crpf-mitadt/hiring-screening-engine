from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from chandra_eval.pipeline.jd_structuring import structure_jd_document_file


class FakeGemmaClient:
    def __init__(self, content: str):
        self.content = content

    def chat(self, messages, max_tokens=4096):
        self.messages = messages
        return self.content


class JDStructuringTests(unittest.TestCase):
    def test_live_gemma_response_is_validated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "jd.txt"
            output_path = root / "jd.json"
            input_path.write_text("Marketing Coordinator", encoding="utf-8")
            payload = {
                "job_id": "ignored",
                "source_document_id": "ignored",
                "extraction_method": "live_gemma",
                "job_title": "Marketing Coordinator",
                "department": "Marketing",
                "role_summary": "Coordinate campaigns.",
                "responsibilities": ["Campaign reporting"],
                "mandatory_requirements": [
                    {
                        "requirement_id": "req-1",
                        "category": "skill",
                        "description": "SEO",
                        "requirement_type": "minimum",
                        "mandatory": True,
                        "preferred": False,
                    }
                ],
                "preferred_requirements": [
                    {
                        "requirement_id": "pref-1",
                        "category": "skill",
                        "description": "Analytics",
                        "requirement_type": "preferred",
                        "mandatory": False,
                        "preferred": True,
                    }
                ],
                "required_skills": ["SEO"],
                "preferred_skills": ["Analytics"],
                "minimum_experience_months": 24,
                "domain": "Marketing",
                "seniority": "Coordinator",
                "scoring_rubric": [
                    {
                        "criterion_id": "skills",
                        "name": "Skills",
                        "category": "skill",
                        "description": "Required skills",
                        "weight": 60,
                        "mandatory": True,
                    },
                    {
                        "criterion_id": "experience",
                        "name": "Experience",
                        "category": "experience",
                        "description": "Minimum experience",
                        "weight": 40,
                        "mandatory": True,
                    },
                ],
            }

            jd = structure_jd_document_file(
                input_path,
                "marketing-coordinator",
                output_path,
                gemma_client=FakeGemmaClient("Here is JSON:\n" + json.dumps(payload)),
            )

            self.assertEqual(jd.extraction_method.value, "live_gemma")
            self.assertEqual(jd.job_id, "marketing-coordinator")
            self.assertEqual(jd.required_skills, ["SEO"])
            self.assertEqual(sum(item.weight for item in jd.scoring_rubric), 100)

    def test_bad_weight_total_fails_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "jd.txt"
            input_path.write_text("Role", encoding="utf-8")
            payload = {
                "job_id": "job",
                "source_document_id": "jd.txt",
                "job_title": "Role",
                "scoring_rubric": [
                    {
                        "criterion_id": "bad",
                        "name": "Bad",
                        "category": "skill",
                        "description": "Bad",
                        "weight": 10,
                    }
                ],
            }

            with self.assertRaises(ValueError):
                structure_jd_document_file(
                    input_path,
                    "job",
                    root / "out.json",
                    gemma_client=FakeGemmaClient(json.dumps(payload)),
                )


if __name__ == "__main__":
    unittest.main()
