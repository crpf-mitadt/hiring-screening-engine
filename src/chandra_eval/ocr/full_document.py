"""Full-document OCR orchestration placeholder for Tyrone pilots."""

from pathlib import Path

from chandra.prompts import OCR_LAYOUT_PROMPT
from chandra.model.util import scale_to_fit

from chandra_eval.ocr.chandra_client import ChandraClient
from chandra_eval.ocr.rendering import render_pdf_page


def ocr_pdf_pages(
    path: Path,
    client: ChandraClient,
    page_count: int,
    max_tokens: int = 4096,
) -> list[dict]:
    results: list[dict] = []
    for page_index in range(page_count):
        image = scale_to_fit(render_pdf_page(path, page_index=page_index))
        results.append(client.chat_ocr(image, OCR_LAYOUT_PROMPT, max_tokens=max_tokens))
    return results
