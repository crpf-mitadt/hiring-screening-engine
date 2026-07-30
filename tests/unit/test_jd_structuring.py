from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from chandra_eval.pipeline.jd_structuring import structure_jd_document_file


class FakeGemmaClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def chat_response(self, messages, max_tokens=4096, response_format=None):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        return {
            "model": "gemma4-12b",
            "choices": [{"finish_reason": "stop", "message": {"content": self.content}}],
        }


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

            client = FakeGemmaClient("Here is JSON:\n" + json.dumps(payload))
            jd = structure_jd_document_file(
                input_path,
                "marketing-coordinator",
                output_path,
                gemma_client=client,
            )

            self.assertEqual(jd.extraction_method.value, "live_gemma")
            self.assertEqual(jd.job_id, "marketing-coordinator")
            self.assertEqual(jd.required_skills, ["SEO"])
            self.assertEqual(sum(item.weight for item in jd.scoring_rubric), 100)
            self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})

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

    def test_postprocess_moves_contextual_required_skill_to_adjacent(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "jd.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "Full Stack Developer",
                        "Must be proficient in PHP, Laravel and WordPress.",
                        "JavaScript experience is helpful.",
                        "Responsibilities include work on JavaScript projects.",
                    ]
                ),
                encoding="utf-8",
            )
            payload = {
                "job_id": "job",
                "source_document_id": "jd.txt",
                "job_title": "Full Stack Developer",
                "mandatory_requirements": [
                    {
                        "requirement_id": "req-php",
                        "category": "skill",
                        "description": "Must be proficient in PHP, Laravel and WordPress.",
                        "requirement_type": "minimum",
                        "mandatory": True,
                    }
                ],
                "required_skills": ["PHP", "Laravel", "WordPress", "JavaScript"],
                "scoring_rubric": [
                    {
                        "criterion_id": "core",
                        "name": "Core stack",
                        "category": "skill",
                        "description": "Core stack",
                        "weight": 100,
                        "requirement_ids": ["req-php"],
                        "required_skill_names": ["PHP", "Laravel", "WordPress", "JavaScript"],
                    }
                ],
            }

            jd = structure_jd_document_file(input_path, "job", root / "out.json", gemma_client=FakeGemmaClient(json.dumps(payload)))
            criterion = jd.scoring_rubric[0]

            self.assertEqual(criterion.required_skill_names, ["PHP", "Laravel", "WordPress"])
            self.assertIn("JavaScript", criterion.adjacent_skill_names)
            self.assertIn("moved 'JavaScript'", " ".join(jd.extraction_warnings))

    def test_required_javascript_remains_core_when_explicitly_mandatory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "jd.txt"
            input_path.write_text("Required: JavaScript. Must be proficient in PHP.", encoding="utf-8")
            payload = {
                "job_id": "job",
                "source_document_id": "jd.txt",
                "job_title": "Frontend Developer",
                "mandatory_requirements": [
                    {
                        "requirement_id": "req-js",
                        "category": "skill",
                        "description": "Required: JavaScript.",
                        "requirement_type": "minimum",
                        "mandatory": True,
                    }
                ],
                "required_skills": ["JavaScript", "PHP"],
                "scoring_rubric": [
                    {
                        "criterion_id": "core",
                        "name": "Core stack",
                        "category": "skill",
                        "description": "Core stack",
                        "weight": 100,
                        "requirement_ids": ["req-js"],
                        "required_skill_names": ["JavaScript", "PHP"],
                    }
                ],
            }

            jd = structure_jd_document_file(input_path, "job", root / "out.json", gemma_client=FakeGemmaClient(json.dumps(payload)))
            criterion = jd.scoring_rubric[0]

            self.assertIn("JavaScript", criterion.required_skill_names)

    def test_unlinked_mandatory_requirement_gets_rubric_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "jd.txt"
            input_path.write_text("Must know PHP. Experience with MySQL or similar databases.", encoding="utf-8")
            payload = {
                "job_id": "job",
                "source_document_id": "jd.txt",
                "job_title": "Full Stack Developer",
                "mandatory_requirements": [
                    {
                        "requirement_id": "req_php",
                        "category": "skill",
                        "description": "Must know PHP.",
                        "requirement_type": "minimum",
                        "mandatory": True,
                    },
                    {
                        "requirement_id": "req_db",
                        "category": "skill",
                        "description": "Experience with MySQL or similar databases.",
                        "requirement_type": "minimum",
                        "mandatory": True,
                    },
                ],
                "required_skills": ["PHP", "MySQL"],
                "scoring_rubric": [
                    {
                        "criterion_id": "core",
                        "name": "Core stack",
                        "category": "skill",
                        "description": "Core stack",
                        "weight": 100,
                        "requirement_ids": ["req_php"],
                        "required_skill_names": ["PHP"],
                    }
                ],
            }

            jd = structure_jd_document_file(input_path, "job", root / "out.json", gemma_client=FakeGemmaClient(json.dumps(payload)))

            self.assertIn("req_db", " ".join(jd.extraction_warnings))


if __name__ == "__main__":
    unittest.main()
