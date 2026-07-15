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
    payload = match.group(1) if match else text
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ModelResponseError(f"Model did not return valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ModelResponseError("Model JSON response must be an object")
    return parsed


def parse_model(text: str, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(parse_json_object(text))
