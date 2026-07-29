import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.ocr.full_document import extract_response_content, write_ocr_artifacts


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

    def test_invalid_and_empty_response_shapes_are_warned(self) -> None:
        content, warnings = extract_response_content({"choices": []})
        self.assertEqual(content, "")
        self.assertIn("invalid_response_shape", warnings[0])

        content, warnings = extract_response_content(_response("", 1))
        self.assertEqual(content, "")
        self.assertIn("empty_ocr_content", warnings)


if __name__ == "__main__":
    unittest.main()
