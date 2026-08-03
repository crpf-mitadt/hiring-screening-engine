"""Structured output parsing helpers."""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from chandra_eval.exceptions import ModelResponseError

ModelT = TypeVar("ModelT", bound=BaseModel)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str | list[dict[str, Any]]) -> dict:
    text = _coerce_text(text)
    text = _THINK_RE.sub("", text.lstrip("\ufeff")).strip()
    match = _JSON_BLOCK_RE.search(text)
    payload = match.group(1).strip() if match else _extract_single_json_object_text(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ModelResponseError(f"Model did not return valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ModelResponseError("Model JSON response must be an object")
    return parsed


def parse_model(text: str | list[dict[str, Any]], model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(parse_json_object(text))


def _coerce_text(value: str | list[dict[str, Any]]) -> str:
    if isinstance(value, str):
        return value
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {None, "text"} and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts)


def _extract_single_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        return stripped

    depth = 0
    in_string = False
    escape = False
    for index, character in enumerate(stripped[start:], start=start):
        if escape:
            escape = False
            continue
        if character == "\\":
            escape = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                payload = stripped[start : index + 1]
                _reject_extra_json_objects(stripped[index + 1 :], payload)
                return payload
    return stripped


def _reject_extra_json_objects(remainder: str, payload: str) -> None:
    trailing = remainder.strip()
    if not trailing:
        return
    if "{" in trailing and "}" in trailing:
        raise ModelResponseError("Model response contained multiple JSON objects")
