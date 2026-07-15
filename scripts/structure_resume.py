#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.ingestion.resume_document import load_candidate_profile, write_candidate_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and normalize a structured candidate profile JSON file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/candidate-profile.json"))
    args = parser.parse_args()

    candidate = load_candidate_profile(args.input)
    write_candidate_profile(candidate, args.output)
    print(f"Candidate profile: {args.output}")
    print(f"Scoring-safe candidate id: {candidate.scoring_safe_profile.candidate_id}")


if __name__ == "__main__":
    main()
