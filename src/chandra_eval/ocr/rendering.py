"""Shared rendering helpers for future OCR commands."""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def render_pdf_page(path: Path, page_index: int = 0, scale: float = 1.0) -> Image.Image:
    document = pdfium.PdfDocument(str(path))
    try:
        if len(document) <= page_index:
            raise RuntimeError(f"PDF does not contain page index {page_index}")
        page = document[page_index]
        try:
            bitmap = page.render(scale=scale)
            try:
                return bitmap.to_pil().convert("RGB").copy()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()
