import json
import unittest
from unittest.mock import patch

from chandra_eval.config import Settings
from chandra_eval.exceptions import APICompatibilityError
from chandra_eval.llm.gemma_client import GemmaClient


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "{}", payload: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        return self._payload


class GemmaClientTests(unittest.TestCase):
    def test_structured_requests_disable_thinking(self) -> None:
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["body"] = json
            return FakeResponse()

        with patch("requests.post", fake_post):
            GemmaClient(Settings()).chat_response(
                [{"role": "user", "content": "Return JSON"}],
                response_format={"type": "json_object"},
            )

        self.assertEqual(
            captured["body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})

    def test_response_format_compatibility_error_is_explicit(self) -> None:
        def fake_post(url, headers, json, timeout):
            return FakeResponse(
                status_code=400,
                text=json_module.dumps({"error": "unsupported response_format"}),
            )

        json_module = json
        with patch("requests.post", fake_post):
            with self.assertRaises(APICompatibilityError):
                GemmaClient(Settings(llm_max_retries=0)).chat_response(
                    [{"role": "user", "content": "Return JSON"}],
                    response_format={"type": "json_object"},
                )


if __name__ == "__main__":
    unittest.main()
