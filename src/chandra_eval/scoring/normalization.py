"""Normalization helpers for deterministic matching."""

import re


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("&", " and "))


def normalize_token(value: str) -> str:
    text = value.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    words = [word[:-1] if len(word) > 3 and word.endswith("s") else word for word in text.split()]
    return " ".join(words)


def contains_equivalent(haystack: list[str], needle: str, equivalents: list[str] | None = None) -> bool:
    normalized = {normalize_token(item) for item in haystack}
    options = [needle, *(equivalents or [])]
    return any(normalize_token(option) in normalized for option in options)
