#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.config import load_settings
from chandra_eval.ocr.chandra_client import ChandraClient
from chandra_eval.ocr.full_document import ocr_pdf_pages, write_ocr_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configurable full-document Chandra OCR.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("outputs/full-resume-ocr"))
    args = parser.parse_args()

    settings = load_settings()
    args.output.mkdir(parents=True, exist_ok=True)
    client = ChandraClient(settings)
    responses = ocr_pdf_pages(args.pdf, client, page_count=args.pages, max_tokens=args.max_tokens)
    metadata = write_ocr_artifacts(
        responses=responses,
        output_dir=args.output,
        source_pdf=str(args.pdf),
        model=settings.chandra_model,
    )
    print(f"OCR artifacts: {args.output}")
    print(f"Raw responses: {args.output / 'raw-responses.json'}")
    print(f"Pages: {metadata['page_count']}")


if __name__ == "__main__":
    main()
