#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.pipeline.jd_structuring import structure_jd_document_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a synthetic structured JD JSON file from TXT, MD, or DOCX."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--job-id", default="job-001")
    parser.add_argument("--output", type=Path, default=Path("outputs/structured-jd.json"))
    args = parser.parse_args()

    jd = structure_jd_document_file(args.input, args.job_id, args.output)
    print(f"Structured JD: {args.output}")
    print(f"Mandatory requirements: {len(jd.mandatory_requirements)}")


if __name__ == "__main__":
    main()
