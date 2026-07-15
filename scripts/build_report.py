#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON, CSV, and HTML reports from evaluation JSON.")
    parser.add_argument("evaluation_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/report"))
    args = parser.parse_args()

    report = CandidateEvaluationReport.model_validate_json(
        args.evaluation_json.read_text(encoding="utf-8")
    )
    write_json_report(report, args.output / "evaluation.json")
    write_csv_report(report, args.output / "evaluation.csv")
    write_html_report(report, args.output / "evaluation.html")
    print(f"Reports: {args.output}")


if __name__ == "__main__":
    main()
