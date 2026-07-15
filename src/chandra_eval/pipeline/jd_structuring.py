"""JD structuring pipeline."""

from pathlib import Path

from chandra_eval.config import Settings, load_settings
from chandra_eval.ingestion.jd_document import (
    extract_synthetic_jd,
    ingest_jd_document,
    write_structured_jd,
)
from chandra_eval.llm.gemma_client import GemmaClient
from chandra_eval.llm.prompts import JD_STRUCTURING_SYSTEM_PROMPT
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
    content = client.chat(
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
        ]
    )
    jd = parse_model(content, StructuredJD)
    jd.job_id = job_id
    jd.source_document_id = source_document_id
    jd.extraction_method = ExtractionMethod.LIVE_GEMMA
    return jd
