"""Resume structuring from Chandra OCR artifacts."""

from pathlib import Path

from chandra_eval.config import Settings, load_settings
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
        client = gemma_client or GemmaClient(settings or load_settings())
        candidate = _structure_resume_with_gemma(ocr_document, candidate_id, client)

    candidate.candidate_id = candidate_id
    candidate.source_document_id = ocr_document.source_document_id
    candidate.scoring_safe_profile = build_scoring_safe_profile(candidate)
    write_candidate_profile(candidate, output_path)
    return candidate


def _structure_resume_with_gemma(
    ocr_document: OcrDocument,
    candidate_id: str,
    client: GemmaClient,
) -> CandidateProfile:
    prompt_text = prompt_text_from_ocr(ocr_document)
    if not prompt_text.strip():
        raise ValueError("OCR content is empty after safety filtering")
    schema_hint = CandidateProfile.model_json_schema()
    content = client.chat(
        [
            {"role": "system", "content": RESUME_STRUCTURING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Create a CandidateProfile JSON object for employment decision support. "
                    f"Set candidate_id to {candidate_id!r}, "
                    f"source_document_id to {ocr_document.source_document_id!r}, "
                    "and extraction_method to 'live_gemma'. "
                    "Do not include name, photograph, email, phone, address, or protected traits. "
                    "Every extracted job-related fact should include EvidenceRef entries "
                    "using page_number and block_index from the OCR lines where possible. "
                    "Return JSON only, matching this schema:\n"
                    f"{schema_hint}\n\n"
                    "OCR text with source references:\n"
                    f"{prompt_text}"
                ),
            },
        ]
    )
    candidate = parse_model(content, CandidateProfile)
    candidate.extraction_method = ExtractionMethod.LIVE_GEMMA
    return candidate


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
