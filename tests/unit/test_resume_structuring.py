import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.config import Settings
from chandra_eval.exceptions import APICompatibilityError, ModelResponseError
from chandra_eval.pipeline.resume_structuring import structure_resume_from_ocr


class FakeGemmaClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat_response(self, messages, max_tokens=4096, response_format=None):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        if not self.payloads:
            return {"choices": [{"message": {"content": ""}}]}
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _message_payload(content, finish_reason="stop") -> dict:
    return {
        "model": "gemma4-12b",
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"completion_tokens": 10},
    }


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
                gemma_client=FakeGemmaClient(
                    [_message_payload("Result:\n" + json.dumps(payload) + "\nDone")]
                ),
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

    def test_recovers_role_dates_from_nearby_heading_blocks_without_leakage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocr_dir = root / "ocr"
            ocr_dir.mkdir()
            (ocr_dir / "source-map.json").write_text(
                json.dumps(
                    [
                        {
                            "page_number": 1,
                            "block_index": 1,
                            "data_label": "Text",
                            "exact_text": "Global Director",
                            "normalized_text": "Global Director",
                        },
                        {
                            "page_number": 1,
                            "block_index": 2,
                            "data_label": "Text",
                            "exact_text": "05/2021 - 08/2023",
                            "normalized_text": "05/2021 - 08/2023",
                        },
                        {
                            "page_number": 1,
                            "block_index": 3,
                            "data_label": "Text",
                            "exact_text": "GBS India Site Lead",
                            "normalized_text": "GBS India Site Lead",
                        },
                        {
                            "page_number": 1,
                            "block_index": 4,
                            "data_label": "Text",
                            "exact_text": "07/2018 - 04/2021",
                            "normalized_text": "07/2018 - 04/2021",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            payload = {
                "candidate_id": "candidate-dates",
                "source_document_id": "ocr",
                "extraction_method": "live_gemma",
                "experience": [
                    {"title": "Global Director", "start_date": None, "end_date": None},
                    {"title": "GBS India Site Lead", "start_date": None, "end_date": None},
                ],
            }

            candidate = structure_resume_from_ocr(
                ocr_dir,
                candidate_id="candidate-dates",
                output_path=root / "candidate.json",
                gemma_client=FakeGemmaClient([_message_payload(json.dumps(payload))]),
            )

            self.assertEqual(candidate.experience[0].start_date, "2021-05")
            self.assertEqual(candidate.experience[0].end_date, "2023-08")
            self.assertEqual(candidate.experience[1].start_date, "2018-07")
            self.assertEqual(candidate.experience[1].end_date, "2021-04")
            self.assertEqual(candidate.experience[0].evidence[-1].block_index, 2)
            self.assertEqual(candidate.experience[1].evidence[-1].block_index, 4)

    def test_ambiguous_three_component_date_is_repaired_from_nearby_ocr_or_left_unknown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocr_dir = root / "ocr"
            ocr_dir.mkdir()
            (ocr_dir / "source-map.json").write_text(
                json.dumps(
                    [
                        {
                            "page_number": 1,
                            "block_index": 1,
                            "data_label": "Text",
                            "exact_text": "Program Manager",
                            "normalized_text": "Program Manager",
                        },
                        {
                            "page_number": 1,
                            "block_index": 2,
                            "data_label": "Text",
                            "exact_text": "08 / 2017 - 07 / 2018",
                            "normalized_text": "08 / 2017 - 07 / 2018",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            payload = {
                "candidate_id": "candidate-dates",
                "source_document_id": "ocr",
                "extraction_method": "live_gemma",
                "experience": [
                    {
                        "title": "Program Manager",
                        "start_date": "08/07/17",
                        "end_date": "2018-07",
                        "start_date_precision": "month",
                    }
                ],
            }

            candidate = structure_resume_from_ocr(
                ocr_dir,
                candidate_id="candidate-dates",
                output_path=root / "candidate.json",
                gemma_client=FakeGemmaClient([_message_payload(json.dumps(payload))]),
            )

            self.assertEqual(candidate.experience[0].start_date, "2017-08")
            self.assertEqual(candidate.experience[0].start_date_precision.value, "month")
            self.assertNotIn("08/07/17", candidate.experience[0].start_date or "")

    def test_empty_schema_response_retries_with_response_format(self) -> None:
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
                "candidate_id": "candidate-004",
                "source_document_id": "raw.json",
                "extraction_method": "live_gemma",
                "summary": "Python engineer",
                "skills": ["Python"],
            }
            client = FakeGemmaClient(
                [
                    _message_payload(""),
                    _message_payload(json.dumps(payload)),
                ]
            )
            candidate = structure_resume_from_ocr(
                raw,
                candidate_id="candidate-004",
                output_path=root / "candidate.json",
                gemma_client=client,
                settings=Settings(llm_max_retries=2, gemma_resume_max_tokens=8192),
            )

            self.assertEqual(candidate.skills, ["Python"])
            self.assertIsNotNone(client.calls[0]["response_format"])
            self.assertIsNotNone(client.calls[1]["response_format"])
            self.assertEqual(client.calls[0]["max_tokens"], 8192)

    def test_repair_request_succeeds_after_invalid_json(self) -> None:
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
                "candidate_id": "candidate-005",
                "source_document_id": "raw.json",
                "extraction_method": "live_gemma",
                "summary": "Python engineer",
                "skills": ["Python"],
            }
            client = FakeGemmaClient(
                [
                    _message_payload("{bad"),
                    _message_payload("{bad"),
                    _message_payload(json.dumps(payload)),
                ]
            )
            candidate = structure_resume_from_ocr(
                raw,
                candidate_id="candidate-005",
                output_path=root / "candidate.json",
                gemma_client=client,
                settings=Settings(llm_max_retries=2),
            )

            self.assertEqual(candidate.skills, ["Python"])
            self.assertIn("malformed", client.calls[2]["messages"][1]["content"])
            self.assertIsNotNone(client.calls[2]["response_format"])

    def test_schema_validation_repair_preserves_existing_json_and_omits_ocr(self) -> None:
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
            invalid_payload = {
                "candidate_id": "candidate-008",
                "source_document_id": "raw.json",
                "extraction_method": "live_gemma",
                "summary": "Valid summary",
                "skills": ["Python"],
                "education": [{"degree_level": "not-a-degree"}],
            }
            repaired_payload = {
                **invalid_payload,
                "education": [{"degree_level": "unknown", "field_of_study": None}],
            }
            client = FakeGemmaClient(
                [
                    _message_payload(json.dumps(invalid_payload)),
                    _message_payload(json.dumps(repaired_payload)),
                ]
            )

            candidate = structure_resume_from_ocr(
                raw,
                candidate_id="candidate-008",
                output_path=root / "candidate.json",
                gemma_client=client,
                settings=Settings(llm_max_retries=1),
            )

            repair_prompt = client.calls[1]["messages"][1]["content"]
            self.assertEqual(candidate.summary, "Valid summary")
            self.assertEqual(candidate.skills, ["Python"])
            self.assertEqual(candidate.education[0].degree_level.value, "unknown")
            self.assertIn("Existing JSON object", repair_prompt)
            self.assertIn("education.0.degree_level", repair_prompt)
            self.assertNotIn("OCR text with source references", repair_prompt)
            self.assertNotIn("Skills: Python</p>", repair_prompt)
            self.assertIsNotNone(client.calls[1]["response_format"])

    def test_response_format_removed_only_for_api_compatibility_error(self) -> None:
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
                "candidate_id": "candidate-009",
                "source_document_id": "raw.json",
                "extraction_method": "live_gemma",
                "summary": "Python engineer",
                "skills": ["Python"],
            }
            client = FakeGemmaClient(
                [
                    APICompatibilityError("unsupported response_format"),
                    _message_payload(json.dumps(payload)),
                ]
            )

            candidate = structure_resume_from_ocr(
                raw,
                candidate_id="candidate-009",
                output_path=root / "candidate.json",
                gemma_client=client,
                settings=Settings(llm_max_retries=1),
            )

            self.assertEqual(candidate.skills, ["Python"])
            self.assertIsNotNone(client.calls[0]["response_format"])
            self.assertIsNone(client.calls[1]["response_format"])

    def test_retries_exhausted_writes_safe_metadata_not_raw_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            output = root / "candidate.json"
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
            client = FakeGemmaClient([_message_payload("")])

            with self.assertRaises(ModelResponseError) as context:
                structure_resume_from_ocr(
                    raw,
                    candidate_id="candidate-006",
                    output_path=output,
                    gemma_client=client,
                    settings=Settings(llm_max_retries=0, pipeline_debug_raw_responses=False),
                )

            self.assertIn("response_metadata", str(context.exception))
            self.assertIn("requested_max_tokens", str(context.exception))
            metadata_path = root / "resume-structuring-error-metadata.json"
            self.assertTrue(metadata_path.is_file())
            metadata_text = metadata_path.read_text(encoding="utf-8")
            self.assertIn("content_length", metadata_text)
            self.assertNotIn('<p data-label="Text"', metadata_text)
            self.assertFalse((root / "structured-resume-gemma-response.json").exists())

    def test_raw_response_written_only_when_debug_enabled_and_length_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            output = root / "candidate.json"
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
            client = FakeGemmaClient([_message_payload('{"candidate_id": "x"}', finish_reason="length")])

            with self.assertRaises(ModelResponseError) as context:
                structure_resume_from_ocr(
                    raw,
                    candidate_id="candidate-007",
                    output_path=output,
                    gemma_client=client,
                    settings=Settings(llm_max_retries=0, pipeline_debug_raw_responses=True),
                )

            self.assertIn("finish_reason=length", str(context.exception))
            raw_path = root / "structured-resume-gemma-response.json"
            self.assertTrue(raw_path.is_file())
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(raw_payload[0]["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
