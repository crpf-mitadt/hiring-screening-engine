"""Prompts for future Gemma-backed structuring and criterion review."""

RESUME_STRUCTURING_SYSTEM_PROMPT = (
    "Extract only job-related resume facts into the requested JSON schema. "
    "Do not infer protected traits. Do not include contact details in the "
    "scoring-safe profile."
)

JD_STRUCTURING_SYSTEM_PROMPT = (
    "Extract job requirements, mandatory gates, preferred qualifications, "
    "skills, and a weighted scoring rubric into the requested JSON schema."
)

CRITERION_SCORING_SYSTEM_PROMPT = (
    "Compare the scoring-safe candidate profile with one job criterion. "
    "Return evidence-grounded JSON only. Do not perform final arithmetic."
)
