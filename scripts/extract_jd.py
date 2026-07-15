#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.pipeline.jd_structuring import structure_jd_document_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a structured JD JSON file from TXT, MD, DOCX, or PDF."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--job-id", default="job-001")
    parser.add_argument("--output", type=Path, default=Path("outputs/structured-jd.json"))
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock extraction for offline tests instead of live Gemma.",
    )
    args = parser.parse_args()

    jd = structure_jd_document_file(
        args.input,
        args.job_id,
        args.output,
        mock_llm=args.mock_llm,
    )
    print(f"Structured JD: {args.output}")
    print(f"Extraction method: {jd.extraction_method.value}")
    print(f"Mandatory requirements: {len(jd.mandatory_requirements)}")


if __name__ == "__main__":
    main()
