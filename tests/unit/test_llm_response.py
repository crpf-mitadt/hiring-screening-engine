import unittest

from chandra_eval.exceptions import ModelResponseError
from chandra_eval.llm.response import extract_assistant_text, safe_response_metadata
from chandra_eval.llm.structured_output import parse_json_object


def _payload(message: dict, finish_reason: str = "stop") -> dict:
    return {
        "model": "gemma4-12b",
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"completion_tokens": 10},
    }


class LlmResponseTests(unittest.TestCase):
    def test_extracts_normal_string_and_content_parts(self) -> None:
        self.assertEqual(
            extract_assistant_text(_payload({"content": '{"ok": true}'})).text,
            '{"ok": true}',
        )
        self.assertEqual(
            extract_assistant_text(
                _payload({"content": [{"type": "text", "text": '{"ok": true}'}]})
            ).text,
            '{"ok": true}',
        )

    def test_empty_and_null_content_are_actionable(self) -> None:
        for content in ("", None):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ModelResponseError, "metadata="):
                    extract_assistant_text(_payload({"content": content}))

    def test_reasoning_content_fallback_only_when_json_present(self) -> None:
        text = extract_assistant_text(
            _payload({"content": "", "reasoning_content": "<think>x</think>{\"ok\": true}"})
        ).text
        self.assertIn('{"ok": true}', text)

        with self.assertRaises(ModelResponseError):
            extract_assistant_text(_payload({"content": "", "reasoning_content": "thinking only"}))

    def test_safe_metadata_omits_content(self) -> None:
        metadata = safe_response_metadata(
            _payload({"content": "private resume text", "reasoning_content": "private"})
        )

        self.assertEqual(metadata["content_length"], len("private resume text"))
        self.assertNotIn("private resume text", str(metadata))
        self.assertEqual(metadata["reasoning_content_length"], len("private"))

    def test_parse_json_variants(self) -> None:
        cases = [
            '\ufeff{"a": 1}',
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            'Here is the JSON: {"a": 1}',
            '<think>reasoning</think>\n{"a": 1}',
            [{"type": "text", "text": '{"a": 1}'}],
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_json_object(value), {"a": 1})

    def test_malformed_and_multiple_json_objects_fail(self) -> None:
        with self.assertRaises(ModelResponseError):
            parse_json_object('{"a":')
        with self.assertRaises(ModelResponseError):
            parse_json_object('{"a": 1} {"b": 2}')


if __name__ == "__main__":
    unittest.main()
