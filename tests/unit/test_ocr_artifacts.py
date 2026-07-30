import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from chandra_eval.ocr.full_document import extract_response_content, ocr_pdf_pages, write_ocr_artifacts


def _response(html: str, page: int) -> dict:
    return {
        "model": "chandra-ocr-2-q8",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": html},
            }
        ],
        "usage": {"prompt_tokens": page},
        "timings": {"prompt_ms": page},
    }


class RetryClient:
    def __init__(self, finish_reasons):
        self.finish_reasons = list(finish_reasons)
        self.calls: list[int] = []

    def chat_ocr(self, image, prompt, max_tokens=4096):
        self.calls.append(max_tokens)
        finish_reason = self.finish_reasons.pop(0)
        return {
            "model": "chandra-ocr-2-q8",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": f'<p data-label="Text" data-bbox="[0,0,1,1]">tokens {max_tokens}</p>'
                    },
                }
            ],
            "usage": {"completion_tokens": max_tokens},
        }


class OcrArtifactTests(unittest.TestCase):
    def test_writes_html_metadata_document_source_map_without_raw_by_default(self) -> None:
        responses = [
            _response('<p data-label="Text" data-bbox="[0,0,1,1]">Page one</p>', 1),
            _response('<p data-label="Text" data-bbox="[0,0,1,1]">Page two</p>', 2),
        ]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            metadata = write_ocr_artifacts(
                responses=responses,
                output_dir=output,
                source_pdf="resume.pdf",
                model="chandra-ocr-2-q8",
                source_page_count=2,
                processed_page_numbers=[1, 2],
            )

            self.assertFalse((output / "raw-responses.json").is_file())
            self.assertTrue((output / "page-001.html").is_file())
            self.assertTrue((output / "page-002-metadata.json").is_file())
            document = json.loads((output / "document.json").read_text(encoding="utf-8"))
            source_map = json.loads((output / "source-map.json").read_text(encoding="utf-8"))
            self.assertEqual(document["page_count"], 2)
            self.assertEqual(document["source_page_count"], 2)
            self.assertEqual(document["processed_page_count"], 2)
            self.assertEqual(document["processed_page_numbers"], [1, 2])
            self.assertFalse(document["partial_ocr"])
            self.assertFalse(document["partial"])
            self.assertEqual(document["partial_reason"], "complete")
            self.assertEqual(document["completed_page_numbers"], [1, 2])
            self.assertEqual(document["truncated_page_numbers"], [])
            self.assertEqual(document["failed_page_numbers"], [])
            self.assertFalse(metadata["raw_responses_persisted"])
            self.assertEqual(source_map[0]["exact_text"], "Page one")
            self.assertEqual(source_map[1]["page_number"], 2)

    def test_writes_raw_response_only_when_debug_enabled_and_marks_partial(self) -> None:
        responses = [_response('<p data-label="Text" data-bbox="[0,0,1,1]">Page one</p>', 1)]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            metadata = write_ocr_artifacts(
                responses=responses,
                output_dir=output,
                source_pdf="resume.pdf",
                model="chandra-ocr-2-q8",
                source_page_count=2,
                processed_page_numbers=[1],
                debug_raw_responses=True,
            )

            self.assertTrue((output / "raw-responses.json").is_file())
            self.assertTrue(metadata["raw_responses_persisted"])
            self.assertTrue(metadata["partial_ocr"])
            self.assertEqual(metadata["completed_page_numbers"], [1])

    def test_finish_reason_length_marks_page_and_document_partial(self) -> None:
        response = _response('<p data-label="Text" data-bbox="[0,0,1,1]">Cut off</p>', 1)
        response["choices"][0]["finish_reason"] = "length"
        response["usage"]["completion_tokens"] = 4096
        with TemporaryDirectory() as tmp:
            metadata = write_ocr_artifacts(
                responses=[response],
                output_dir=Path(tmp),
                source_pdf="resume.pdf",
                model="chandra-ocr-2-q8",
                source_page_count=1,
                processed_page_numbers=[1],
            )

            self.assertTrue(metadata["partial"])
            self.assertEqual(metadata["partial_reason"], "truncated pages: 1")
            self.assertEqual(metadata["completed_page_numbers"], [])
            self.assertEqual(metadata["truncated_page_numbers"], [1])
            self.assertEqual(metadata["failed_page_numbers"], [])
            self.assertIn("finish_reason=length", metadata["page_level_warnings"][0])

    def test_truncated_page_is_retried_and_successful_retry_replaces_result(self) -> None:
        client = RetryClient(["length", "stop"])
        with patch("chandra_eval.ocr.full_document.pdf_page_count", return_value=1), patch(
            "chandra_eval.ocr.full_document.render_pdf_page", return_value="page"
        ), patch("chandra_eval.ocr.full_document.scale_to_fit", side_effect=lambda image: image):
            responses = ocr_pdf_pages(
                Path("resume.pdf"),
                client,
                max_tokens=4096,
                retry_max_tokens=8192,
            )

        self.assertEqual(client.calls, [4096, 8192])
        self.assertEqual(responses[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(responses[0]["usage"]["completion_tokens"], 8192)

    def test_invalid_and_empty_response_shapes_are_warned(self) -> None:
        content, warnings = extract_response_content({"choices": []})
        self.assertEqual(content, "")
        self.assertIn("invalid_response_shape", warnings[0])

        content, warnings = extract_response_content(_response("", 1))
        self.assertEqual(content, "")
        self.assertIn("empty_ocr_content", warnings)


if __name__ == "__main__":
    unittest.main()
