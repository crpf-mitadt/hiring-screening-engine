"""Command-line entry point for chandra-eval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from chandra_eval.config import load_settings
from chandra_eval.pipeline.match_workflow import (
    JD_EXTENSIONS,
    RESUME_EXTENSIONS,
    MatchWorkflowOptions,
    discover_supported_files,
    run_match_workflow,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chandra-eval")
    subparsers = parser.add_subparsers(dest="command")
    _add_match_parser(subparsers)
    args = parser.parse_args(argv)
    if args.command == "match":
        return _run_match(args)
    parser.print_help()
    return 0


def _add_match_parser(subparsers) -> None:
    parser = subparsers.add_parser("match", description="Run an end-to-end resume/JD match.")
    parser.add_argument("--jd", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-dir", type=Path, help="List or batch extension point for resume directories.")
    parser.add_argument("--candidate-id")
    parser.add_argument("--job-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true", help="Regenerate all stages.")
    parser.add_argument("--reuse-existing", action="store_true", default=True, help="Reuse valid cached artifacts.")
    parser.add_argument("--no-cache", action="store_true", help="Disable artifact cache reads and writes.")
    parser.add_argument("--list-jds", action="store_true")
    parser.add_argument("--list-resumes", action="store_true")
    parser.add_argument("--search")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock extraction where structuring is needed.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip live service checks.")
    parser.add_argument("--ocr-pages", type=int)
    parser.add_argument("--ocr-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--allow-partial-evaluation",
        action="store_true",
        help="Allow scoring from intentionally partial OCR; eligibility will be REVIEW.",
    )


def _run_match(args) -> int:
    settings = load_settings()
    if args.list_jds:
        for path in discover_supported_files(settings.jd_input_dir, JD_EXTENSIONS, args.search):
            print(path)
        return 0
    if args.list_resumes:
        root = args.resume_dir or settings.resume_input_dir
        for path in discover_supported_files(root, RESUME_EXTENSIONS, args.search):
            print(path)
        return 0
    try:
        result = run_match_workflow(
            MatchWorkflowOptions(
                jd_path=args.jd,
                resume_path=args.resume,
                candidate_id=args.candidate_id,
                job_id=args.job_id,
                output_root=args.output_root,
                force=args.force,
                reuse_existing=args.reuse_existing,
                no_cache=args.no_cache,
                mock_llm=args.mock_llm,
                skip_preflight=args.skip_preflight,
                quiet=args.quiet,
                json_summary=args.json_summary,
                ocr_pages=args.ocr_pages,
                ocr_max_tokens=args.ocr_max_tokens,
                allow_partial_evaluation=args.allow_partial_evaluation,
            ),
            settings=settings,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 130 if result is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
