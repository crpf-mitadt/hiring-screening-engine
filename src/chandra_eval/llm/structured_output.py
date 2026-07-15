"""Structured output parsing helpers."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel

from chandra_eval.exceptions import ModelResponseError

ModelT = TypeVar("ModelT", bound=BaseModel)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict:
    match = _JSON_BLOCK_RE.search(text)
    payload = match.group(1) if match else _extract_json_object_text(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ModelResponseError(f"Model did not return valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ModelResponseError("Model JSON response must be an object")
    return parsed


def parse_model(text: str, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(parse_json_object(text))


def _extract_json_object_text(text: str) -> str:
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
                return stripped[start : index + 1]
    return stripped
