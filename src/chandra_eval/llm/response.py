"""OpenAI-compatible assistant response extraction."""

from dataclasses import dataclass
import json
from typing import Any

from chandra_eval.exceptions import ModelResponseError


@dataclass(frozen=True)
class AssistantText:
    text: str
    metadata: dict[str, Any]


def extract_assistant_text(payload: dict[str, Any]) -> AssistantText:
    metadata = safe_response_metadata(payload)
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelResponseError(
            "LLM response did not contain choices[0].message; "
            f"metadata={json.dumps(metadata, sort_keys=True)}"
        ) from error

    text = _content_to_text(message.get("content"))
    if not text.strip():
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and _contains_json_object(reasoning):
            text = reasoning

    if not text.strip():
        raise ModelResponseError(
            "LLM response contained empty assistant content; "
            f"metadata={json.dumps(metadata, sort_keys=True)}"
        )

    return AssistantText(text=text, metadata=metadata)


def safe_response_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    choice = None
    message: dict[str, Any] = {}
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        raw_message = choice.get("message")
        if isinstance(raw_message, dict):
            message = raw_message

    content = message.get("content")
    reasoning = message.get("reasoning_content")
    return {
        "model": payload.get("model"),
        "finish_reason": choice.get("finish_reason") if choice else None,
        "usage": payload.get("usage", {}),
        "message_keys": sorted(message.keys()),
        "content_type": type(content).__name__,
        "content_length": len(content) if isinstance(content, str) else _content_parts_length(content),
        "reasoning_content_type": type(reasoning).__name__,
        "reasoning_content_length": len(reasoning) if isinstance(reasoning, str) else 0,
        "has_tool_calls": bool(message.get("tool_calls")),
        "has_refusal": bool(message.get("refusal")),
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {None, "text"} and item.get("text"):
                parts.append(str(item["text"]))
            elif item.get("content"):
                parts.append(str(item["content"]))
        return "\n".join(parts)
    return ""


def _content_parts_length(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return len(_content_to_text(content))


def _contains_json_object(text: str) -> bool:
    return "{" in text and "}" in text
