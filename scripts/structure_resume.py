#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.pipeline.resume_structuring import structure_resume_from_ocr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure a candidate profile from Chandra OCR artifacts."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/candidate-profile.json"))
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic mock extraction for offline tests instead of live Gemma.",
    )
    args = parser.parse_args()

    candidate = structure_resume_from_ocr(
        args.input,
        candidate_id=args.candidate_id,
        output_path=args.output,
        mock_llm=args.mock_llm,
    )
    print(f"Candidate profile: {args.output}")
    print(f"Extraction method: {candidate.extraction_method.value}")
    print(f"Scoring-safe candidate id: {candidate.scoring_safe_profile.candidate_id}")


if __name__ == "__main__":
    main()
