#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a structured candidate against a structured JD.")
    parser.add_argument("--candidate", "--resume", dest="candidate", type=Path, required=True)
    parser.add_argument("--jd", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/evaluation.json"))
    args = parser.parse_args()

    report = evaluate_candidate_files(args.candidate, args.jd, args.output)
    output_path = args.output if args.output.suffix.lower() == ".json" else args.output / "evaluation.json"
    print(f"Evaluation: {output_path}")
    print(f"Eligibility: {report.eligibility.status.value}")
    print(f"Final score: {report.final_score:.3f}")
    print("Human review required: true")


if __name__ == "__main__":
    main()
