"""Configurable Chandra OCR client for full-document pilots."""

import base64
import io
from typing import Any

import requests
from PIL import Image

from chandra_eval.config import Settings


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ChandraClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat_ocr(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> dict[str, Any]:
        url = str(self.settings.chandra_base_url).rstrip("/") + "/v1/chat/completions"
        body = {
            "model": self.settings.chandra_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_png(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = requests.post(
            url,
            json=body,
            timeout=self.settings.llm_request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
