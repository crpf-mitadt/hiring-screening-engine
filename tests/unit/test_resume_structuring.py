import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.pipeline.resume_structuring import structure_resume_from_ocr


class FakeGemmaClient:
    def __init__(self, content: str):
        self.content = content

    def chat(self, messages, max_tokens=4096):
        self.messages = messages
        return self.content


class ResumeStructuringTests(unittest.TestCase):
    def test_ocr_directory_mock_builds_candidate_with_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocr_dir = root / "ocr"
            ocr_dir.mkdir()
            (ocr_dir / "source-map.json").write_text(
                json.dumps(
                    [
                        {
                            "page_number": 1,
                            "block_index": 0,
                            "data_label": "Text",
                            "data_bbox": [0.1, 0.1, 0.9, 0.2],
                            "exact_text": "Skills: Python, OCR",
                            "normalized_text": "Skills: Python, OCR",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            candidate = structure_resume_from_ocr(
                ocr_dir,
                candidate_id="candidate-001",
                output_path=root / "candidate.json",
                mock_llm=True,
            )

            self.assertEqual(candidate.extraction_method.value, "mock_gemma")
            self.assertEqual(candidate.skills, ["OCR", "Python"])
            self.assertEqual(candidate.experience[0].evidence[0].page_number, 1)
            self.assertEqual(candidate.scoring_safe_profile.candidate_id, "candidate-001")

    def test_raw_response_json_live_fake_gemma_and_json_repair(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            raw.write_text(
                json.dumps(
                    [
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": '<p data-label="Text" data-bbox="[0,0,1,1]">Skills: Python</p>'
                                    }
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            payload = {
                "candidate_id": "ignored",
                "source_document_id": "ignored",
                "extraction_method": "live_gemma",
                "summary": "Python OCR engineer",
                "skills": ["Python"],
                "source_pages": [1],
            }
            candidate = structure_resume_from_ocr(
                raw,
                candidate_id="candidate-002",
                output_path=root / "candidate.json",
                gemma_client=FakeGemmaClient("Result:\n" + json.dumps(payload) + "\nDone"),
            )

            self.assertEqual(candidate.candidate_id, "candidate-002")
            self.assertEqual(candidate.extraction_method.value, "live_gemma")
            self.assertEqual(candidate.skills, ["Python"])

    def test_missing_ocr_content_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.json"
            empty.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                structure_resume_from_ocr(
                    empty,
                    candidate_id="candidate-003",
                    output_path=root / "candidate.json",
                    mock_llm=True,
                )


if __name__ == "__main__":
    unittest.main()
