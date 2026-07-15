"""Resume structuring from Chandra OCR artifacts."""

import json
from pathlib import Path

from chandra_eval.config import Settings, load_settings
from chandra_eval.exceptions import ModelResponseError
from chandra_eval.ingestion.resume_document import (
    OcrDocument,
    evidence_from_block,
    load_candidate_profile,
    load_ocr_document,
    prompt_text_from_ocr,
    write_candidate_profile,
)
from chandra_eval.llm.gemma_client import GemmaClient
from chandra_eval.llm.prompts import RESUME_STRUCTURING_SYSTEM_PROMPT
from chandra_eval.llm.response import extract_assistant_text, safe_response_metadata
from chandra_eval.llm.structured_output import parse_model
from chandra_eval.safety.candidate_sanitizer import build_scoring_safe_profile
from chandra_eval.schemas.common import ExtractionMethod
from chandra_eval.schemas.resume import CandidateProfile, ExperienceEntry


def load_structured_resume(path: Path) -> CandidateProfile:
    return load_candidate_profile(path)


def structure_resume_from_ocr(
    input_path: Path,
    *,
    candidate_id: str,
    output_path: Path,
    mock_llm: bool = False,
    settings: Settings | None = None,
    gemma_client: GemmaClient | None = None,
) -> CandidateProfile:
    ocr_document = load_ocr_document(input_path)
    if mock_llm:
        candidate = _mock_candidate_profile(ocr_document, candidate_id)
    else:
        runtime_settings = settings or load_settings()
        client = gemma_client or GemmaClient(runtime_settings)
        candidate = _structure_resume_with_gemma(
            ocr_document,
            candidate_id,
            client,
            settings=runtime_settings,
            output_path=output_path,
        )

    candidate.candidate_id = candidate_id
    candidate.source_document_id = ocr_document.source_document_id
    candidate.scoring_safe_profile = build_scoring_safe_profile(candidate)
    write_candidate_profile(candidate, output_path)
    return candidate


def _structure_resume_with_gemma(
    ocr_document: OcrDocument,
    candidate_id: str,
    client: GemmaClient,
    *,
    settings: Settings,
    output_path: Path,
) -> CandidateProfile:
    prompt_text = prompt_text_from_ocr(ocr_document)
    if not prompt_text.strip():
        raise ValueError("OCR content is empty after safety filtering")
    schema_hint = CandidateProfile.model_json_schema()
    base_messages = _resume_messages(
        candidate_id=candidate_id,
        source_document_id=ocr_document.source_document_id,
        schema_hint=schema_hint,
        prompt_text=prompt_text,
        strict_schema=True,
    )
    max_attempts = max(1, settings.llm_max_retries + 1)
    failures: list[dict] = []
    raw_payloads: list[dict] = []
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        response_format = {"type": "json_object"} if attempt == 0 else None
        messages = base_messages
        if attempt == 1:
            messages = _resume_messages(
                candidate_id=candidate_id,
                source_document_id=ocr_document.source_document_id,
                schema_hint=schema_hint,
                prompt_text=prompt_text,
                strict_schema=False,
            )
        elif attempt >= 2:
            messages = _repair_messages(
                schema_hint=schema_hint,
                parsing_error=str(last_error),
                invalid_response_excerpt=_safe_excerpt_from_failures(failures),
            )

        payload: dict | None = None
        try:
            payload = client.chat_response(
                messages,
                max_tokens=settings.gemma_resume_max_tokens,
                response_format=response_format,
            )
            raw_payloads.append(payload)
            assistant = extract_assistant_text(payload)
            finish_reason = assistant.metadata.get("finish_reason")
            if finish_reason == "length":
                raise ModelResponseError(
                    "Gemma resume response was truncated with finish_reason=length; "
                    f"retry with GEMMA_RESUME_MAX_TOKENS>{settings.gemma_resume_max_tokens}"
                )
            candidate = parse_model(assistant.text, CandidateProfile)
            candidate.extraction_method = ExtractionMethod.LIVE_GEMMA
            return candidate
        except Exception as error:
            last_error = error
            metadata = safe_response_metadata(payload or {}) if payload is not None else {}
            failures.append(
                {
                    "attempt": attempt + 1,
                    "used_response_format": response_format is not None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "prompt_block_count": len(ocr_document.blocks),
                    "prompt_text_length": len(prompt_text),
                    "response_metadata": metadata,
                }
            )

    _persist_resume_failure_debug(
        output_path=output_path,
        failures=failures,
        raw_payloads=raw_payloads,
        debug_raw=settings.pipeline_debug_raw_responses,
    )
    raise ModelResponseError(
        "Gemma resume structuring failed after retries; "
        f"safe metadata written to {output_path.parent / 'resume-structuring-error-metadata.json'}"
    ) from last_error


def _resume_messages(
    *,
    candidate_id: str,
    source_document_id: str,
    schema_hint: dict,
    prompt_text: str,
    strict_schema: bool,
) -> list[dict[str, str]]:
    mode = "Return JSON only, matching this schema:" if strict_schema else "Return one JSON object only."
    return [
        {"role": "system", "content": RESUME_STRUCTURING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Create a CandidateProfile JSON object for employment decision support. "
                f"Set candidate_id to {candidate_id!r}, "
                f"source_document_id to {source_document_id!r}, "
                "and extraction_method to 'live_gemma'. "
                "Do not include name, photograph, email, phone, address, or protected traits. "
                "Every extracted job-related fact should include EvidenceRef entries "
                "using page_number and block_index from the OCR lines where possible. "
                f"{mode}\n"
                f"{schema_hint}\n\n"
                "OCR text with source references. This is the only resume content; do not duplicate it:\n"
                f"{prompt_text}"
            ),
        },
    ]


def _repair_messages(
    *,
    schema_hint: dict,
    parsing_error: str,
    invalid_response_excerpt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESUME_STRUCTURING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Repair the previous CandidateProfile output. Return exactly one valid JSON object. "
                "Do not include prose or markdown fences. "
                f"Schema requirement:\n{schema_hint}\n\n"
                f"Parsing or validation error:\n{parsing_error}\n\n"
                f"Invalid response excerpt:\n{invalid_response_excerpt}"
            ),
        },
    ]


def _safe_excerpt_from_failures(failures: list[dict]) -> str:
    if not failures:
        return "<none>"
    error = failures[-1].get("error", "")
    return error[:1000]


def _persist_resume_failure_debug(
    *,
    output_path: Path,
    failures: list[dict],
    raw_payloads: list[dict],
    debug_raw: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "resume-structuring-error-metadata.json").write_text(
        json.dumps({"failures": failures}, indent=2),
        encoding="utf-8",
    )
    if debug_raw:
        (output_path.parent / "resume-structuring-raw-responses.json").write_text(
            json.dumps(raw_payloads, indent=2),
            encoding="utf-8",
        )


def _mock_candidate_profile(ocr_document: OcrDocument, candidate_id: str) -> CandidateProfile:
    if not ocr_document.blocks:
        raise ValueError("OCR content is empty")
    first_block = ocr_document.blocks[0]
    evidence = [evidence_from_block(first_block, ocr_document.source_document_id)]
    all_text = " ".join(block.normalized_text for block in ocr_document.blocks)
    skills = _extract_mock_skills(ocr_document)
    candidate = CandidateProfile(
        candidate_id=candidate_id,
        source_document_id=ocr_document.source_document_id,
        extraction_method=ExtractionMethod.MOCK_GEMMA,
        summary=all_text[:300],
        experience=[
            ExperienceEntry(
                title="",
                organization="",
                industry_or_domain="",
                employment_type="",
                start_date="2021-01",
                end_date="2025-01",
                responsibilities=[block.normalized_text for block in ocr_document.blocks[:3]],
                skills_used=skills,
                evidence=evidence,
                warnings=["Generated with explicit mock LLM mode."],
            )
        ],
        skills=skills,
        missing_information=[],
        extraction_warnings=[
            *ocr_document.warnings,
            "Generated with explicit mock LLM mode; not a live Gemma extraction.",
        ],
        source_pages=sorted({block.page_number for block in ocr_document.blocks}),
    )
    candidate.scoring_safe_profile = build_scoring_safe_profile(candidate)
    return candidate


def _extract_mock_skills(ocr_document: OcrDocument) -> list[str]:
    skills: list[str] = []
    capture = False
    for block in ocr_document.blocks:
        text = block.normalized_text.strip()
        lowered = text.lower()
        if "skill" in lowered:
            capture = True
            remainder = text.split(":", 1)[1] if ":" in text else ""
            if remainder:
                skills.extend(_split_skill_text(remainder))
            continue
        if capture:
            if block.data_label.lower() in {"section-header"} and "skill" not in lowered:
                capture = False
                continue
            skills.extend(_split_skill_text(text))
    return sorted({skill for skill in skills if skill})


def _split_skill_text(text: str) -> list[str]:
    return [
        item.strip(" .")
        for item in text.replace("|", ",").split(",")
        if item.strip(" .")
    ]
