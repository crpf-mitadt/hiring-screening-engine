"""Small direct client for Gemma's OpenAI-compatible chat endpoint."""

import time
from typing import Any

import requests

from chandra_eval.config import Settings
from chandra_eval.exceptions import ModelResponseError


class GemmaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 4096) -> str:
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
        }
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.settings.llm_request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                return _extract_content(payload)
            except Exception as error:
                last_error = error
                if attempt < self.settings.llm_max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise ModelResponseError(f"Gemma request failed: {last_error}") from last_error


def _extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelResponseError("Gemma response did not contain message content") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)
