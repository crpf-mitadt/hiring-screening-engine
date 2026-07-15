"""Normalization helpers for deterministic matching."""

import re


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", value.lower()).strip()


def contains_equivalent(haystack: list[str], needle: str, equivalents: list[str] | None = None) -> bool:
    normalized = {normalize_token(item) for item in haystack}
    options = [needle, *(equivalents or [])]
    return any(normalize_token(option) in normalized for option in options)
