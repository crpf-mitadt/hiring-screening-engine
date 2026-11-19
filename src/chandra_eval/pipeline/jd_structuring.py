"""JD structuring pipeline."""

from pathlib import Path
import re

from chandra_eval.config import Settings, load_settings
from chandra_eval.ingestion.jd_document import (
    extract_synthetic_jd,
    ingest_jd_document,
    write_structured_jd,
)
from chandra_eval.llm.gemma_client import GemmaClient
from chandra_eval.llm.prompts import JD_STRUCTURING_SYSTEM_PROMPT
from chandra_eval.llm.response import extract_assistant_text
from chandra_eval.llm.structured_output import parse_model
from chandra_eval.schemas.common import ExtractionMethod
from chandra_eval.schemas.jd import StructuredJD


def structure_jd_document_file(
    path: Path,
    job_id: str,
    output_path: Path,
    *,
    mock_llm: bool = False,
    settings: Settings | None = None,
    gemma_client: GemmaClient | None = None,
) -> StructuredJD:
    document = ingest_jd_document(path)
    if mock_llm:
        jd = extract_synthetic_jd(
            document.text,
            job_id=job_id,
            source_document_id=document.source_document_id,
        )
        jd.extraction_method = ExtractionMethod.MOCK_GEMMA
        jd.extraction_warnings = [
            *jd.extraction_warnings,
            "Generated with explicit mock LLM mode; not a live Gemma extraction.",
        ]
    else:
        client = gemma_client or GemmaClient(settings or load_settings())
        jd = _extract_jd_with_gemma(
            document_text=document.text,
            job_id=job_id,
            source_document_id=document.source_document_id,
            client=client,
        )
    jd = _postprocess_jd_rubric(jd, document.text)
    write_structured_jd(jd, output_path)
    return jd


def structure_jd_text_file(
    path: Path,
    job_id: str,
    output_path: Path,
    *,
    mock_llm: bool = False,
    settings: Settings | None = None,
    gemma_client: GemmaClient | None = None,
) -> StructuredJD:
    return structure_jd_document_file(
        path,
        job_id,
        output_path,
        mock_llm=mock_llm,
        settings=settings,
        gemma_client=gemma_client,
    )


def _extract_jd_with_gemma(
    *,
    document_text: str,
    job_id: str,
    source_document_id: str,
    client: GemmaClient,
) -> StructuredJD:
    schema_hint = StructuredJD.model_json_schema()
    payload = client.chat_response(
        [
            {"role": "system", "content": JD_STRUCTURING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Create a StructuredJD JSON object. "
                    f"Set job_id to {job_id!r}, source_document_id to {source_document_id!r}, "
                    "and extraction_method to 'live_gemma'. "
                    "The scoring_rubric weights must sum to 100. "
                    "Return JSON only, matching this schema:\n"
                    f"{schema_hint}\n\n"
                    "Job description text:\n"
                    f"{document_text}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    content = extract_assistant_text(payload).text
    jd = parse_model(content, StructuredJD)
    jd.job_id = job_id
    jd.source_document_id = source_document_id
    jd.extraction_method = ExtractionMethod.LIVE_GEMMA
    return jd


def _postprocess_jd_rubric(jd: StructuredJD, document_text: str) -> StructuredJD:
    if not jd.scoring_rubric:
        return jd
    warnings = list(jd.extraction_warnings)
    mandatory_skill_terms = _mandatory_skill_terms(jd)
    linked_requirement_ids = {
        requirement_id
        for criterion in jd.scoring_rubric
        for requirement_id in criterion.requirement_ids
    }
    mandatory_ids = {requirement.requirement_id for requirement in jd.mandatory_requirements}
    orphaned = sorted(mandatory_ids - linked_requirement_ids)
    if orphaned:
        warnings.append(
            "Rubric quality warning: mandatory requirements are not linked to any scoring criterion: "
            + ", ".join(orphaned)
        )

    rubric = []
    for criterion in jd.scoring_rubric:
        required: list[str] = []
        adjacent = list(criterion.adjacent_skill_names)
        preferred = list(criterion.preferred_skill_names)
        for skill in criterion.required_skill_names:
            classification = _skill_classification(skill, criterion, jd, document_text, mandatory_skill_terms)
            if classification == "required/core":
                required.append(skill)
            elif classification == "preferred":
                preferred.append(skill)
                warnings.append(
                    f"Rubric quality warning: moved {skill!r} from required_skill_names to preferred_skill_names "
                    f"for {criterion.criterion_id}; source wording is preferred/helpful rather than mandatory."
                )
            else:
                adjacent.append(skill)
                warnings.append(
                    f"Rubric quality warning: moved {skill!r} from required_skill_names to adjacent_skill_names "
                    f"for {criterion.criterion_id}; source wording is contextual/transferable, not mandatory."
                )
        rubric.append(
            criterion.model_copy(
                update={
                    "required_skill_names": _dedupe(required),
                    "preferred_skill_names": _dedupe(preferred),
                    "adjacent_skill_names": _dedupe(adjacent),
                }
            )
        )
    return jd.model_copy(update={"scoring_rubric": rubric, "extraction_warnings": _dedupe(warnings)})


def _mandatory_skill_terms(jd: StructuredJD) -> set[str]:
    terms: set[str] = set()
    for requirement in jd.mandatory_requirements:
        text = requirement.description
        if requirement.category.value != "skill":
            continue
        for skill in [*jd.required_skills, *requirement.accepted_equivalents]:
            if _contains_word(text, skill):
                terms.add(_norm(skill))
        for phrase in _technology_phrases(text):
            if _has_required_wording(text, phrase):
                terms.add(_norm(phrase))
    return terms


def _skill_classification(
    skill: str,
    criterion,
    jd: StructuredJD,
    document_text: str,
    mandatory_skill_terms: set[str],
) -> str:
    skill_key = _norm(skill)
    if skill_key in mandatory_skill_terms:
        return "required/core"
    for requirement in jd.mandatory_requirements:
        if requirement.requirement_id in criterion.requirement_ids and _contains_word(requirement.description, skill):
            return "required/core" if _has_required_wording(requirement.description, skill) else "adjacent/transferable"
    snippets = _snippets_containing(document_text, skill)
    if any(_has_required_wording(snippet, skill) for snippet in snippets):
        return "required/core"
    if any(_has_preferred_priority_wording(snippet, skill) for snippet in snippets):
        return "preferred"
    if any(_has_preferred_wording(snippet, skill) for snippet in snippets):
        return "adjacent/transferable"
    return "adjacent/transferable"


def _has_required_wording(text: str, skill: str) -> bool:
    pattern = rf"\b(must|required|mandatory|core|proficient|expertise|strong)\b[^.\n;]*\b{re.escape(skill)}\b"
    reverse = rf"\b{re.escape(skill)}\b[^.\n;]*\b(required|mandatory|must|core)\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE) or re.search(reverse, text, flags=re.IGNORECASE))


def _has_preferred_wording(text: str, skill: str) -> bool:
    pattern = rf"\b(preferred|optional|helpful|beneficial|desirable|familiarity|exposure|nice to have|additional)\b[^.\n;]*\b{re.escape(skill)}\b"
    reverse = rf"\b{re.escape(skill)}\b[^.\n;]*\b(preferred|optional|helpful|beneficial|desirable|familiarity|exposure|nice to have|additional)\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE) or re.search(reverse, text, flags=re.IGNORECASE))


def _has_preferred_priority_wording(text: str, skill: str) -> bool:
    pattern = rf"\b(preferred|optional|beneficial|desirable|nice to have|additional)\b[^.\n;]*\b{re.escape(skill)}\b"
    reverse = rf"\b{re.escape(skill)}\b[^.\n;]*\b(preferred|optional|beneficial|desirable|nice to have|additional)\b"
    familiarity_helpful = rf"\bfamiliarity\b[^.\n;]*\b{re.escape(skill)}\b[^.\n;]*\b(helpful|preferred|beneficial|desirable)\b"
    familiarity_required = rf"\bfamiliarity\b[^.\n;]*\b{re.escape(skill)}\b[^.\n;]*\b(required|mandatory|must)\b"
    return bool(
        re.search(pattern, text, flags=re.IGNORECASE)
        or re.search(reverse, text, flags=re.IGNORECASE)
        or (
            re.search(familiarity_helpful, text, flags=re.IGNORECASE)
            and not re.search(familiarity_required, text, flags=re.IGNORECASE)
        )
    )


def _snippets_containing(text: str, skill: str) -> list[str]:
    return [line.strip() for line in re.split(r"[\n.]", text) if _contains_word(line, skill)]


def _technology_phrases(text: str) -> list[str]:
    return [
        phrase.strip(" .")
        for phrase in re.split(r",|;|/|\band\b|\bor\b", text, flags=re.IGNORECASE)
        if phrase.strip() and any(character.isupper() for character in phrase)
    ]


def _contains_word(text: str, skill: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9+#.]){re.escape(skill)}(?![A-Za-z0-9+#.])", text, flags=re.IGNORECASE))


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", value.lower()).strip()


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        key = _norm(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output
