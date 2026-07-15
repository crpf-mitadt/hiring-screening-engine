#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.pipeline.pilot import run_synthetic_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline synthetic resume/JD matching pilot.")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic-pilot"))
    args = parser.parse_args()

    report = run_synthetic_pilot(args.fixtures, args.output)
    print(f"Pilot output: {args.output}")
    print(f"Eligibility: {report.eligibility.status.value}")
    print(f"Final score: {report.final_score:.3f}")
    print("Human review required: true")


if __name__ == "__main__":
    main()
