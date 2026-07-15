"""Small direct client for Gemma's OpenAI-compatible chat endpoint."""

import time
from typing import Any

import requests

from chandra_eval.config import Settings
from chandra_eval.exceptions import APICompatibilityError, ModelResponseError
from chandra_eval.llm.response import extract_assistant_text


class GemmaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat_response(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = str(self.settings.gemma_base_url).rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.gemma_api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.settings.gemma_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None:
            body["response_format"] = response_format
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.settings.llm_request_timeout_seconds,
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError as error:
                    if response_format is not None and _is_response_format_compatibility_error(response):
                        raise APICompatibilityError(
                            f"Gemma API rejected response_format: HTTP {response.status_code}: "
                            f"{response.text[:500]}"
                        ) from error
                    raise
                return response.json()
            except APICompatibilityError:
                raise
            except Exception as error:
                last_error = error
                if attempt < self.settings.llm_max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise ModelResponseError(f"Gemma request failed: {last_error}") from last_error

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 4096) -> str:
        payload = self.chat_response(messages, max_tokens=max_tokens)
        return extract_assistant_text(payload).text


def _is_response_format_compatibility_error(response: requests.Response) -> bool:
    text = response.text.lower()
    return any(
        marker in text
        for marker in (
            "unknown parameter",
            "unsupported response_format",
            "unsupported json_schema",
            "invalid response_format",
            "response_format is not supported",
        )
    )
