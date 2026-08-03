"""Skill scoring helpers."""

from chandra_eval.scoring.normalization import normalize_token


def skill_match_fraction(required: list[str], available: list[str]) -> float:
    if not required:
        return 1.0
    available_norm = {normalize_token(skill) for skill in available}
    matched = sum(1 for skill in required if normalize_token(skill) in available_norm)
    return matched / len(required)


def missing_skills(required: list[str], available: list[str]) -> list[str]:
    available_norm = {normalize_token(skill) for skill in available}
    return [skill for skill in required if normalize_token(skill) not in available_norm]
