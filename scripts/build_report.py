#!/usr/bin/env python3

import argparse
from pathlib import Path

from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build selected JSON, CSV, and HTML reports from evaluation JSON or a pipeline directory."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true", help="Write JSON report.")
    parser.add_argument("--csv", action="store_true", help="Write CSV report.")
    parser.add_argument("--html", action="store_true", help="Write HTML report.")
    args = parser.parse_args()

    evaluation_json = args.input / "evaluation.json" if args.input.is_dir() else args.input
    report = CandidateEvaluationReport.model_validate_json(
        evaluation_json.read_text(encoding="utf-8")
    )
    output = args.output or (args.input if args.input.is_dir() else evaluation_json.parent)
    selected = args.json or args.csv or args.html
    write_json = args.json or not selected
    write_csv = args.csv or not selected
    write_html = args.html or not selected
    if write_json:
        write_json_report(report, output / "evaluation.json")
    if write_csv:
        write_csv_report(report, output / "evaluation.csv")
    if write_html:
        write_html_report(report, output / "evaluation.html")
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
