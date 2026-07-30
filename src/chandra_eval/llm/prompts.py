"""Prompts for future Gemma-backed structuring and criterion review."""

RESUME_STRUCTURING_SYSTEM_PROMPT = (
    "Extract only job-related resume facts into the requested JSON schema. "
    "Do not infer protected traits. Do not include contact details in the "
    "scoring-safe profile. Use null for missing optional facts, not empty "
    "strings or invented values."
)

JD_STRUCTURING_SYSTEM_PROMPT = (
    "Extract job requirements, mandatory gates, preferred qualifications, "
    "skills, and a weighted scoring rubric into the requested JSON schema. "
    "Put a skill in required_skill_names only when mandatory/core wording or "
    "a linked mandatory requirement explicitly requires it. Put helpful, "
    "preferred, optional, exposure, desirable, responsibility-context, or "
    "transferable technologies in preferred_skill_names or adjacent_skill_names, "
    "not required_skill_names. Preserve requirement_ids for every criterion "
    "that evaluates a mandatory requirement."
)

CRITERION_SCORING_SYSTEM_PROMPT = (
    "Compare the scoring-safe candidate profile with one job criterion. "
    "Return evidence-grounded JSON only. Do not perform final arithmetic."
)
