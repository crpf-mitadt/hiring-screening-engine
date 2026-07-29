"""Full-document OCR orchestration placeholder for Tyrone pilots."""

import json
from pathlib import Path
from typing import Any

from chandra.prompts import OCR_LAYOUT_PROMPT
from chandra.model.util import scale_to_fit
import pypdfium2 as pdfium

from chandra_eval.ingestion.chandra_html import parse_chandra_html
from chandra_eval.ocr.chandra_client import ChandraClient
from chandra_eval.ocr.rendering import render_pdf_page


def pdf_page_count(path: Path) -> int:
    document = pdfium.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def ocr_pdf_pages(
    path: Path,
    client: ChandraClient,
    page_count: int | None = None,
    max_tokens: int = 4096,
) -> list[dict]:
    source_page_count = pdf_page_count(path)
    processed_count = source_page_count if page_count is None else min(page_count, source_page_count)
    results: list[dict] = []
    for page_index in range(processed_count):
        image = scale_to_fit(render_pdf_page(path, page_index=page_index))
        results.append(client.chat_ocr(image, OCR_LAYOUT_PROMPT, max_tokens=max_tokens))
    return results


def extract_response_content(response: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content", "")
    except (KeyError, IndexError, TypeError) as error:
        return "", [f"invalid_response_shape: {type(error).__name__}: {error}"]

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        text = "\n".join(parts)
    else:
        text = str(content)

    if not text.strip():
        warnings.append("empty_ocr_content")
    return text, warnings


def write_ocr_artifacts(
    *,
    responses: list[dict[str, Any]],
    output_dir: Path,
    source_pdf: str | None = None,
    model: str | None = None,
    source_page_count: int | None = None,
    processed_page_numbers: list[int] | None = None,
    debug_raw_responses: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_raw_responses:
        (output_dir / "raw-responses.json").write_text(
            json.dumps(responses, indent=2),
            encoding="utf-8",
        )

    document_pages: list[dict[str, Any]] = []
    source_map: list[dict[str, Any]] = []
    page_metadata: list[dict[str, Any]] = []
    page_numbers = processed_page_numbers or list(range(1, len(responses) + 1))

    for index, response in zip(page_numbers, responses, strict=False):
        html, warnings = extract_response_content(response)
        html_path = output_dir / f"page-{index:03d}.html"
        html_path.write_text(html, encoding="utf-8")

        choice = _first_choice(response)
        metadata = {
            "page_number": index,
            "model": response.get("model", model),
            "finish_reason": choice.get("finish_reason") if choice else None,
            "usage": response.get("usage", {}),
            "timings": response.get("timings", {}),
            "warnings": warnings,
            "html": html_path.name,
        }
        (output_dir / f"page-{index:03d}-metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        page_metadata.append(metadata)

        blocks = []
        if html.strip():
            for block in parse_chandra_html(html):
                block_record = {
                    "page_number": index,
                    "block_index": block.index,
                    "data_label": block.label,
                    "data_bbox": block.bbox,
                    "exact_text": block.text,
                    "normalized_text": block.normalized_text,
                    "html": block.html,
                }
                blocks.append(block_record)
                source_map.append(
                    {
                        "page_number": index,
                        "block_index": block.index,
                        "data_label": block.label,
                        "data_bbox": block.bbox,
                        "exact_text": block.text,
                        "normalized_text": block.normalized_text,
                    }
                )

        document_pages.append(
            {
                "page_number": index,
                "html": html_path.name,
                "blocks": blocks,
                "warnings": warnings,
            }
        )

    document = {
        "source_pdf": source_pdf,
        "source_page_count": source_page_count,
        "processed_page_count": len(responses),
        "processed_page_numbers": page_numbers,
        "partial_ocr": source_page_count is not None and len(responses) < source_page_count,
        "page_count": len(responses),
        "pages": document_pages,
    }
    metadata = {
        "source_pdf": source_pdf,
        "model": model,
        "source_page_count": source_page_count,
        "processed_page_count": len(responses),
        "processed_page_numbers": page_numbers,
        "partial_ocr": source_page_count is not None and len(responses) < source_page_count,
        "raw_responses_persisted": debug_raw_responses,
        "page_count": len(responses),
        "pages": page_metadata,
        "artifact_contract_version": 1,
    }

    (output_dir / "document.json").write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    (output_dir / "source-map.json").write_text(
        json.dumps(source_map, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _first_choice(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None
