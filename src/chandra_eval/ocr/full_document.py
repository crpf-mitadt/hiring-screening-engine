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
    retry_truncated: bool = True,
    retry_max_tokens: int = 8192,
) -> list[dict]:
    source_page_count = pdf_page_count(path)
    processed_count = source_page_count if page_count is None else min(page_count, source_page_count)
    results: list[dict] = []
    for page_index in range(processed_count):
        image = scale_to_fit(render_pdf_page(path, page_index=page_index))
        budgets = [max_tokens]
        if retry_truncated and retry_max_tokens > max_tokens:
            budgets.append(retry_max_tokens)
        attempts: list[dict[str, Any]] = []
        final_response: dict[str, Any] | None = None
        for budget in budgets:
            response = client.chat_ocr(image, OCR_LAYOUT_PROMPT, max_tokens=budget)
            attempts.append(
                {
                    "max_tokens": budget,
                    "finish_reason": _finish_reason(response),
                    "usage": response.get("usage", {}),
                }
            )
            final_response = response
            if _finish_reason(response) != "length":
                break
        if final_response is None:
            continue
        final_response = dict(final_response)
        final_response["_chandra_eval_attempts"] = attempts
        results.append(final_response)
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
        finish_reason = choice.get("finish_reason") if choice else None
        page_status, page_warnings = _page_status(finish_reason, warnings)
        warnings = [*warnings, *page_warnings]
        metadata = {
            "page_number": index,
            "model": response.get("model", model),
            "finish_reason": finish_reason,
            "status": page_status,
            "usage": response.get("usage", {}),
            "timings": response.get("timings", {}),
            "warnings": warnings,
            "retry_attempts": response.get("_chandra_eval_attempts", []),
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
                "status": page_status,
                "warnings": warnings,
            }
        )

    completed_page_numbers = [item["page_number"] for item in page_metadata if item["status"] == "completed"]
    truncated_page_numbers = [item["page_number"] for item in page_metadata if item["status"] == "truncated"]
    failed_page_numbers = [item["page_number"] for item in page_metadata if item["status"] == "failed"]
    partial = bool(
        (source_page_count is not None and len(responses) < source_page_count)
        or truncated_page_numbers
        or failed_page_numbers
    )
    partial_reason = _partial_reason(
        source_page_count=source_page_count,
        processed_count=len(responses),
        truncated_page_numbers=truncated_page_numbers,
        failed_page_numbers=failed_page_numbers,
    ) or "complete"
    page_level_warnings = [
        f"page {item['page_number']}: {warning}"
        for item in page_metadata
        for warning in item.get("warnings", [])
    ]
    requested_max_tokens = None
    if page_metadata:
        attempts = page_metadata[0].get("retry_attempts") or []
        if attempts:
            requested_max_tokens = attempts[0].get("max_tokens")
    document = {
        "source_pdf": source_pdf,
        "source_page_count": source_page_count,
        "processed_page_count": len(responses),
        "processed_page_numbers": page_numbers,
        "completed_page_numbers": completed_page_numbers,
        "truncated_page_numbers": truncated_page_numbers,
        "failed_page_numbers": failed_page_numbers,
        "partial": partial,
        "partial_reason": partial_reason,
        "partial_ocr": partial,
        "page_level_warnings": page_level_warnings,
        "requested_max_tokens": requested_max_tokens,
        "page_count": len(responses),
        "pages": document_pages,
    }
    metadata = {
        "source_pdf": source_pdf,
        "model": model,
        "source_page_count": source_page_count,
        "processed_page_count": len(responses),
        "processed_page_numbers": page_numbers,
        "completed_page_numbers": completed_page_numbers,
        "truncated_page_numbers": truncated_page_numbers,
        "failed_page_numbers": failed_page_numbers,
        "partial": partial,
        "partial_reason": partial_reason,
        "partial_ocr": partial,
        "page_level_warnings": page_level_warnings,
        "requested_max_tokens": requested_max_tokens,
        "raw_responses_persisted": debug_raw_responses,
        "page_count": len(responses),
        "pages": page_metadata,
        "artifact_contract_version": 2,
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


def _finish_reason(response: dict[str, Any]) -> str | None:
    choice = _first_choice(response)
    return choice.get("finish_reason") if choice else None


def _page_status(finish_reason: str | None, content_warnings: list[str]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if finish_reason == "length":
        warnings.append("finish_reason=length: OCR page was truncated; retry with a larger OCR token budget.")
        return "truncated", warnings
    if any(warning.startswith("invalid_response_shape") or warning == "empty_ocr_content" for warning in content_warnings):
        return "failed", warnings
    return "completed", warnings


def _partial_reason(
    *,
    source_page_count: int | None,
    processed_count: int,
    truncated_page_numbers: list[int],
    failed_page_numbers: list[int],
) -> str | None:
    reasons: list[str] = []
    if source_page_count is not None and processed_count < source_page_count:
        reasons.append(f"processed {processed_count} of {source_page_count} source pages")
    if truncated_page_numbers:
        reasons.append(f"truncated pages: {', '.join(str(page) for page in truncated_page_numbers)}")
    if failed_page_numbers:
        reasons.append(f"failed pages: {', '.join(str(page) for page in failed_page_numbers)}")
    return "; ".join(reasons) if reasons else None
