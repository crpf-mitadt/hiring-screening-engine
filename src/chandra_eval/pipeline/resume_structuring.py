"""Resume structuring from Chandra OCR artifacts."""

import json
import re
import time
from pathlib import Path

from pydantic import ValidationError

from chandra_eval.config import Settings, load_settings
from chandra_eval.exceptions import (
    APICompatibilityError,
    EmptyModelResponseError,
    FinalResponseTruncatedError,
    ModelResponseError,
    StructuredExtractionError,
)
from chandra_eval.ingestion.resume_document import (
    OcrBlock,
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
from chandra_eval.llm.structured_output import parse_json_object
from chandra_eval.safety.candidate_sanitizer import build_scoring_safe_profile
from chandra_eval.schemas.common import DatePrecision, ExtractionMethod
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
    candidate = _recover_experience_dates(candidate, ocr_document)
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
    messages = _resume_messages(
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
    request_kind = "full_extraction"
    response_format: dict | None = {"type": "json_object"}
    response_format_fallback_used = False

    for attempt in range(max_attempts):
        payload: dict | None = None
        metadata: dict = {}
        json_parse_success = False
        validation_success = False
        validation_errors: list[dict] = []
        started_at = time.perf_counter()
        try:
            payload = client.chat_response(
                messages,
                max_tokens=settings.gemma_resume_max_tokens,
                response_format=response_format,
            )
            raw_payloads.append(payload)
            assistant = extract_assistant_text(payload)
            metadata = assistant.metadata
            finish_reason = metadata.get("finish_reason")
            if finish_reason == "length":
                raise FinalResponseTruncatedError(
                    "Gemma resume response was truncated with finish_reason=length; "
                    "keeping GEMMA_RESUME_MAX_TOKENS=8192 unless final JSON truncation is proven"
                )
            parsed = parse_json_object(assistant.text)
            json_parse_success = True
            try:
                candidate = CandidateProfile.model_validate(parsed)
                validation_success = True
            except ValidationError as error:
                validation_errors = _concise_validation_errors(error)
                last_error = error
                failures.append(
                    _attempt_failure(
                        attempt=attempt,
                        max_attempts=max_attempts,
                        request_kind=request_kind,
                        response_format=response_format,
                        response_format_fallback_used=response_format_fallback_used,
                        settings=settings,
                        ocr_document=ocr_document,
                        prompt_text=prompt_text,
                        error_type="SchemaValidationError",
                        error="CandidateProfile validation failed",
                        metadata=metadata,
                        elapsed_seconds=time.perf_counter() - started_at,
                        json_parse_success=json_parse_success,
                        validation_success=validation_success,
                        validation_errors=validation_errors,
                    )
                )
                if attempt + 1 >= max_attempts:
                    break
                messages = _schema_validation_repair_messages(
                    existing_json=parsed,
                    validation_errors=validation_errors,
                    schema_hint=schema_hint,
                )
                request_kind = "schema_validation_repair"
                response_format = {"type": "json_object"}
                continue
            candidate.extraction_method = ExtractionMethod.LIVE_GEMMA
            return candidate
        except APICompatibilityError as error:
            last_error = error
            failures.append(
                _attempt_failure(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    request_kind=request_kind,
                    response_format=response_format,
                    response_format_fallback_used=response_format_fallback_used,
                    settings=settings,
                    ocr_document=ocr_document,
                    prompt_text=prompt_text,
                    error_type="APICompatibilityError",
                    error=str(error),
                    metadata=metadata,
                    elapsed_seconds=time.perf_counter() - started_at,
                    json_parse_success=json_parse_success,
                    validation_success=validation_success,
                    validation_errors=validation_errors,
                )
            )
            if response_format is not None and attempt + 1 < max_attempts:
                response_format = None
                response_format_fallback_used = True
                continue
            break
        except Exception as error:
            last_error = error
            metadata = metadata or (safe_response_metadata(payload or {}) if payload is not None else {})
            error_type = _classified_error_type(error)
            failures.append(
                _attempt_failure(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    request_kind=request_kind,
                    response_format=response_format,
                    response_format_fallback_used=response_format_fallback_used,
                    settings=settings,
                    ocr_document=ocr_document,
                    prompt_text=prompt_text,
                    error_type=error_type,
                    error=_safe_error_message(error, error_type),
                    metadata=metadata,
                    elapsed_seconds=time.perf_counter() - started_at,
                    json_parse_success=json_parse_success,
                    validation_success=validation_success,
                    validation_errors=validation_errors,
                )
            )
            if attempt + 1 >= max_attempts:
                break
            if isinstance(error, EmptyModelResponseError):
                request_kind = "full_extraction"
                response_format = {"type": "json_object"}
                continue
            if isinstance(error, FinalResponseTruncatedError):
                break
            messages = _malformed_json_repair_messages(
                invalid_response_excerpt=_safe_model_output_excerpt(
                    extract_assistant_text(payload).text if payload else ""
                ),
                parsing_error=_safe_error_message(error, error_type),
                schema_hint=schema_hint,
            )
            request_kind = "malformed_json_repair"
            response_format = {"type": "json_object"}

    _persist_resume_failure_debug(
        output_path=output_path,
        failures=failures,
        raw_payloads=raw_payloads,
        debug_raw=settings.pipeline_debug_raw_responses,
    )
    failure_summary = json.dumps(
        {
            "failures": failures,
            "raw_response_path": (
                str(output_path.parent / "structured-resume-gemma-response.json")
                if settings.pipeline_debug_raw_responses
                else None
            ),
        },
        sort_keys=True,
    )
    raise StructuredExtractionError(
        "Gemma resume structuring failed after retries; "
        f"safe metadata written to {output_path.parent / 'resume-structuring-error-metadata.json'}; "
        f"{failure_summary}"
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
                "Extract only facts supported by the supplied OCR. Use null when an optional fact "
                "is unavailable, omitted, unreadable, or genuinely ambiguous. Never invent a value "
                "to satisfy the schema. Never replace missing facts with empty strings. Use empty "
                "lists only when the source contains no items for a collection. Use an unknown enum "
                "only when that exact enum member exists in the schema. "
                "For each experience entry, preserve visibly associated role dates from headings, "
                "nearby lines, adjacent blocks, and date blocks as well as responsibility paragraphs. "
                "When a role heading is followed by a date range line, attach that date range to that "
                "same role only. Do not copy dates from neighboring roles. Capture common "
                "date formats such as MM/YYYY, MM-YYYY, YYYY-MM, MMM YYYY, Month YYYY, YYYY, Present, "
                "and Current, including spaces around separators. Use "
                "start_date_precision/end_date_precision='month' for month/year dates, "
                "and set currently_active=true when the end date is Present or Current. Include the "
                "date line as evidence for the role when page_number and block_index are available. "
                "Do not interpret ambiguous three-component slash dates such as 08/07/17 unless the "
                "nearby OCR context deterministically supplies a month/year range. "
                "Every extracted job-related fact should include EvidenceRef entries "
                "using page_number and block_index from the OCR lines where possible. "
                f"{mode}\n"
                f"{schema_hint}\n\n"
                "OCR text with source references. This is the only resume content; do not duplicate it:\n"
                f"{prompt_text}"
            ),
        },
    ]


def _schema_validation_repair_messages(
    *,
    existing_json: dict,
    validation_errors: list[dict],
    schema_hint: dict,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESUME_STRUCTURING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "The supplied JSON has already been extracted from the resume. Correct only fields "
                "implicated by the validation errors. Retain every unrelated value unchanged. Do not "
                "add unsupported resume facts. Do not rewrite, summarize, or improve valid content. "
                "Use null for unavailable optional facts. Return the complete corrected JSON object "
                "with no commentary outside the structured response.\n\n"
                f"CandidateProfile JSON schema:\n{schema_hint}\n\n"
                f"Validation errors:\n{json.dumps(validation_errors, indent=2)}\n\n"
                f"Existing JSON object:\n{json.dumps(existing_json, indent=2)}"
            ),
        },
    ]


def _malformed_json_repair_messages(
    *,
    invalid_response_excerpt: str,
    parsing_error: str,
    schema_hint: dict,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESUME_STRUCTURING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Repair malformed CandidateProfile JSON syntax or structure only. Return exactly "
                "one complete JSON object. Do not include prose or markdown fences.\n\n"
                f"CandidateProfile JSON schema:\n{schema_hint}\n\n"
                f"Parsing error:\n{parsing_error}\n\n"
                f"Invalid response excerpt:\n{invalid_response_excerpt}"
            ),
        },
    ]


def _attempt_failure(
    *,
    attempt: int,
    max_attempts: int,
    request_kind: str,
    response_format: dict | None,
    response_format_fallback_used: bool,
    settings: Settings,
    ocr_document: OcrDocument,
    prompt_text: str,
    error_type: str,
    error: str,
    metadata: dict,
    elapsed_seconds: float,
    json_parse_success: bool,
    validation_success: bool,
    validation_errors: list[dict],
) -> dict:
    repaired_paths = [item["path"] for item in validation_errors]
    return {
        "operation": "candidate_profile_structuring",
        "request_kind": request_kind,
        "attempt": attempt + 1,
        "max_attempts": max_attempts,
        "model_alias": settings.gemma_model,
        "used_response_format": response_format is not None,
        "response_format_requested": response_format is not None,
        "response_format_fallback_used": response_format_fallback_used,
        "enable_thinking_false_requested": True,
        "requested_max_tokens": settings.gemma_resume_max_tokens,
        "error_type": error_type,
        "error": error,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "prompt_block_count": len(ocr_document.blocks),
        "prompt_text_length": len(prompt_text),
        "json_parse_success": json_parse_success,
        "validation_success": validation_success,
        "validation_error_count": len(validation_errors),
        "repaired_field_paths": repaired_paths,
        "truncation_classification": (
            "final_response_truncated" if error_type == "FinalResponseTruncatedError" else None
        ),
        "response_metadata": metadata,
    }


def _concise_validation_errors(error: ValidationError) -> list[dict]:
    output: list[dict] = []
    for item in error.errors():
        path = ".".join(str(part) for part in item.get("loc", ()))
        record = {
            "path": path,
            "type": item.get("type"),
            "message": item.get("msg"),
        }
        input_value = item.get("input")
        if input_value is None or isinstance(input_value, (str, int, float, bool)):
            record["input"] = input_value
        else:
            record["input_type"] = type(input_value).__name__
        output.append(record)
    return output


def _classified_error_type(error: Exception) -> str:
    if isinstance(error, EmptyModelResponseError):
        return "EmptyModelResponseError"
    if isinstance(error, FinalResponseTruncatedError):
        return "FinalResponseTruncatedError"
    if isinstance(error, ModelResponseError):
        return "MalformedJSONError"
    return type(error).__name__


def _safe_error_message(error: Exception, error_type: str) -> str:
    if error_type in {"MalformedJSONError", "EmptyModelResponseError", "FinalResponseTruncatedError"}:
        return str(error)
    return f"{error_type}: see safe metadata"


def _safe_model_output_excerpt(text: str, limit: int = 1200) -> str:
    return text[:limit]


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
        (output_path.parent / "structured-resume-gemma-response.json").write_text(
            json.dumps(raw_payloads, indent=2),
            encoding="utf-8",
        )


_DATE_TOKEN = r"(?:\d{1,2}\s*[-/]\s*\d{4}|\d{4}\s*-\s*\d{1,2}|[A-Za-z]{3,9}\s+\d{4}|\d{4}|Present|Current|Now)"
_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_TOKEN})\s*(?:-|–|—|to)\s*(?P<end>{_DATE_TOKEN})",
    flags=re.IGNORECASE,
)


def _recover_experience_dates(candidate: CandidateProfile, ocr_document: OcrDocument) -> CandidateProfile:
    if not candidate.experience or not ocr_document.blocks:
        return candidate
    recovered: list[ExperienceEntry] = []
    for entry in candidate.experience:
        recovered.append(_recover_entry_dates(entry, ocr_document))
    candidate.experience = recovered
    candidate.extraction_warnings = sorted(
        set(
            [
                *candidate.extraction_warnings,
                *[
                    warning
                    for entry in recovered
                    for warning in entry.warnings
                    if "date" in warning.lower() or "truncated" in warning.lower()
                ],
            ]
        )
    )
    return candidate


def _recover_entry_dates(entry: ExperienceEntry, ocr_document: OcrDocument) -> ExperienceEntry:
    needs_repair = (
        entry.start_date is None
        or (entry.end_date is None and not entry.currently_active)
        or entry.start_date_precision == DatePrecision.UNKNOWN
        or entry.end_date_precision == DatePrecision.UNKNOWN and entry.end_date is not None
    )
    if not needs_repair:
        return entry
    anchors = _entry_anchor_blocks(entry, ocr_document)
    if not anchors:
        return entry
    candidate_range = _nearest_date_range(anchors, ocr_document)
    if candidate_range is None:
        return _entry_with_warning(entry, "Role date remains uncertain; no nearby OCR date range was found.")
    start, end, block = candidate_range
    updates = {
        "start_date": start,
        "end_date": end,
        "currently_active": end is None,
        "start_date_precision": DatePrecision.MONTH if start and len(start) == 7 else DatePrecision.YEAR,
        "end_date_precision": DatePrecision.UNKNOWN if end is None else (DatePrecision.MONTH if len(end) == 7 else DatePrecision.YEAR),
        "warnings": [
            warning
            for warning in entry.warnings
            if "Ambiguous start_date" not in warning and "Ambiguous end_date" not in warning
        ],
    }
    evidence = list(entry.evidence)
    if not any(item.page_number == block.page_number and item.block_index == block.block_index for item in evidence):
        evidence.append(evidence_from_block(block, ocr_document.source_document_id))
    updates["evidence"] = evidence
    return entry.model_copy(update=updates)


def _entry_with_warning(entry: ExperienceEntry, warning: str) -> ExperienceEntry:
    return entry.model_copy(update={"warnings": sorted(set([*entry.warnings, warning]))})


def _entry_anchor_blocks(entry: ExperienceEntry, ocr_document: OcrDocument) -> list[OcrBlock]:
    by_location = {(block.page_number, block.block_index): block for block in ocr_document.blocks}
    anchors: list[OcrBlock] = []
    for evidence in entry.evidence:
        block = by_location.get((evidence.page_number, evidence.block_index))
        if block is not None:
            anchors.append(block)
    needles = [
        _anchor_text(entry.title),
        _anchor_text(entry.organization),
    ]
    for block in ocr_document.blocks:
        normalized = block.normalized_text.lower()
        if any(needle and needle in normalized for needle in needles):
            anchors.append(block)
    return _dedupe_blocks(anchors)


def _anchor_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _nearest_date_range(anchors: list[OcrBlock], ocr_document: OcrDocument) -> tuple[str, str | None, OcrBlock] | None:
    best: tuple[int, int, str, str | None, OcrBlock] | None = None
    blocks_by_page: dict[int, list[OcrBlock]] = {}
    for block in ocr_document.blocks:
        blocks_by_page.setdefault(block.page_number, []).append(block)
    for anchor in anchors:
        for block in blocks_by_page.get(anchor.page_number, []):
            distance = block.block_index - anchor.block_index
            if distance < -1 or distance > 4:
                continue
            parsed = _extract_date_range(block.normalized_text)
            if parsed is None:
                continue
            start, end = parsed
            ranking = (abs(distance), 0 if distance >= 0 else 1, start, end, block)
            if best is None or ranking[:4] < best[:4]:
                best = ranking
    if best is None:
        return None
    return best[2], best[3], best[4]


def _extract_date_range(text: str) -> tuple[str, str | None] | None:
    match = _DATE_RANGE_RE.search(text)
    if not match:
        return None
    start = _normalize_date_token(match.group("start"))
    end = _normalize_date_token(match.group("end"))
    if start is None or end == "AMBIGUOUS":
        return None
    return start, None if end == "CURRENT" else end


def _normalize_date_token(value: str) -> str | None:
    text = re.sub(r"\s*([-/])\s*", r"\1", value.strip())
    lowered = text.lower().strip(".")
    if lowered in {"present", "current", "now"}:
        return "CURRENT"
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text):
        return "AMBIGUOUS"
    parsed = ExperienceEntry(start_date=text).start_date
    return parsed


def _dedupe_blocks(blocks: list[OcrBlock]) -> list[OcrBlock]:
    output: list[OcrBlock] = []
    seen: set[tuple[int, int]] = set()
    for block in blocks:
        key = (block.page_number, block.block_index)
        if key in seen:
            continue
        seen.add(key)
        output.append(block)
    return output


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
                title=None,
                organization=None,
                industry_or_domain=None,
                employment_type=None,
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
