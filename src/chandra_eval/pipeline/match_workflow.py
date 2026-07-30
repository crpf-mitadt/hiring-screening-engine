"""End-to-end resume/JD matching workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable, Iterable

import requests

from chandra_eval import __version__
from chandra_eval.config import Settings, load_settings
from chandra_eval.exceptions import ScannedPdfRequiresOcrError
from chandra_eval.ingestion.jd_document import ingest_jd_document
from chandra_eval.ingestion.resume_document import (
    load_candidate_profile,
    prompt_text_from_ocr,
    load_ocr_document,
    write_candidate_profile,
)
from chandra_eval.llm.prompt_versions import JD_STRUCTURING_PROMPT_VERSION, RESUME_STRUCTURING_PROMPT_VERSION
from chandra_eval.ocr.chandra_client import ChandraClient
from chandra_eval.ocr.full_document import ocr_pdf_pages, pdf_page_count, write_ocr_artifacts
from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files
from chandra_eval.pipeline.jd_structuring import structure_jd_document_file
from chandra_eval.pipeline.resume_structuring import structure_resume_from_ocr
from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.schemas.evaluation import CandidateEvaluationReport, EligibilityStatus


JD_EXTENSIONS = {".docx", ".txt", ".md", ".pdf"}
RESUME_EXTENSIONS = {".pdf", ".json", ".html", ".htm"}
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_PHONE_RE = re.compile(r"(\+\d[\d\s().-]{7,}\d|\b\d{3}[\s().-]\d{3}[\s().-]\d{4}\b)")


@dataclass
class StageRecord:
    name: str
    status: str
    elapsed_seconds: float
    path: str | None = None
    warning: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "path": self.path,
            "warning": self.warning,
        }


@dataclass
class MatchWorkflowOptions:
    jd_path: Path | None = None
    resume_path: Path | None = None
    candidate_id: str | None = None
    job_id: str | None = None
    output_root: Path | None = None
    force: bool = False
    reuse_existing: bool = True
    no_cache: bool = False
    mock_llm: bool = False
    skip_preflight: bool = False
    quiet: bool = False
    json_summary: bool = False
    ocr_pages: int | None = None
    ocr_max_tokens: int = 4096
    allow_partial_evaluation: bool = False


@dataclass
class MatchWorkflowResult:
    report: CandidateEvaluationReport
    run_dir: Path
    summary_json: Path
    summary_md: Path
    stages: list[StageRecord] = field(default_factory=list)


def discover_supported_files(root: Path, extensions: set[str], search: str | None = None) -> list[Path]:
    if not root.exists():
        return []
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if search:
        needle = search.lower()
        files = [path for path in files if needle in str(path.relative_to(root)).lower()]
    return sorted(files, key=lambda path: str(path.relative_to(root)).lower())


def select_file(
    *,
    title: str,
    root: Path,
    files: list[Path],
    allow_back: bool = False,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> Path | None:
    filtered = files
    while True:
        print_func("")
        print_func(title)
        print_func("=" * len(title))
        if not filtered:
            print_func(f"No supported files found under {root}.")
        duplicate_names = _duplicate_names(filtered)
        for index, path in enumerate(filtered, start=1):
            label = str(path.relative_to(root)) if path.name in duplicate_names else path.name
            print_func(f"[{index}] {label}")
        print_func("[S] Search")
        if allow_back:
            print_func("[B] Back")
        print_func("[Q] Quit")
        choice = input_func("Selection: ").strip()
        if not choice:
            continue
        lowered = choice.lower()
        if lowered == "q":
            return None
        if lowered == "b" and allow_back:
            return Path("__BACK__")
        if lowered == "s":
            query = input_func("Search: ").strip().lower()
            filtered = [path for path in files if query in str(path.relative_to(root)).lower()]
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(filtered):
                return filtered[index - 1]
        print_func("Invalid selection.")


def run_match_workflow(
    options: MatchWorkflowOptions,
    *,
    settings: Settings | None = None,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> MatchWorkflowResult | None:
    settings = settings or load_settings()
    output_root = options.output_root or settings.match_output_root
    started_total = time.perf_counter()
    warnings: list[str] = []

    if options.jd_path is None or options.resume_path is None:
        if not options.skip_preflight and not options.mock_llm:
            _print_preflight(settings, print_func)
        selected = _interactive_select(options, settings, input_func, print_func)
        if selected is None:
            print_func("Cancelled.")
            return None
        jd_path, resume_path = selected
    else:
        jd_path, resume_path = options.jd_path, options.resume_path

    if not options.skip_preflight and not options.mock_llm:
        _preflight_or_raise(settings)

    jd_path = jd_path.expanduser().resolve()
    resume_path = resume_path.expanduser().resolve()
    if not jd_path.exists():
        raise FileNotFoundError(f"JD input not found: {jd_path}")
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume input not found: {resume_path}")

    source_hashes = {
        "jd": _path_identity_hash(jd_path),
        "resume": _path_identity_hash(resume_path),
    }
    job_id = options.job_id or _slug(jd_path.stem)
    candidate_id = options.candidate_id or f"candidate-{source_hashes['resume'][:12]}"
    run_dir = _unique_run_dir(output_root, job_id, candidate_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_root = output_root / ".cache"
    use_cache = not options.no_cache
    read_cache = use_cache and options.reuse_existing and not options.force
    stages: list[StageRecord] = []

    jd_structured = run_dir / "structured-jd.json"
    resume_structured = run_dir / "structured-resume.json"
    evaluation_path = run_dir / "evaluation.json"

    jd_text_path = run_dir / "jd-source.txt"
    _timed_stage(
        stages,
        "jd_ingestion",
        lambda: _ingest_jd_stage(jd_path, jd_text_path, settings, options),
    )

    jd_cache_key = _cache_key(
        "jd-structure",
        source_hashes["jd"],
        settings.gemma_model,
        JD_STRUCTURING_PROMPT_VERSION,
        str(options.mock_llm),
    )
    _cached_file_stage(
        stages,
        "jd_structuring",
        cache_root / jd_cache_key / "structured-jd.json",
        jd_structured,
        read_cache,
        use_cache,
        lambda: structure_jd_document_file(
            jd_text_path if jd_text_path.exists() else jd_path,
            job_id,
            jd_structured,
            mock_llm=options.mock_llm,
            settings=settings,
        ),
    )

    ocr_input: Path
    if resume_path.is_dir():
        ocr_input = resume_path
        stages.append(StageRecord("resume_ocr", "reused", 0.0, str(_safe_path(resume_path))))
    elif resume_path.suffix.lower() == ".json" and _looks_like_candidate_profile(resume_path):
        candidate = load_candidate_profile(resume_path)
        candidate.candidate_id = candidate_id
        if candidate.scoring_safe_profile is not None:
            candidate.scoring_safe_profile.candidate_id = candidate_id
        write_candidate_profile(candidate, resume_structured)
        stages.append(StageRecord("resume_ocr", "skipped", 0.0, warning="Structured candidate JSON supplied."))
        stages.append(StageRecord("resume_structuring", "reused", 0.0, str(_safe_path(resume_path))))
        ocr_input = resume_path
    elif resume_path.suffix.lower() == ".pdf":
        source_page_count = pdf_page_count(resume_path)
        processed_page_count = source_page_count if options.ocr_pages is None else min(options.ocr_pages, source_page_count)
        processed_page_numbers = list(range(1, processed_page_count + 1))
        ocr_cache_key = _cache_key(
            "resume-ocr",
            source_hashes["resume"],
            settings.chandra_model,
            f"source-pages={source_page_count}",
            f"processed-pages={','.join(str(page) for page in processed_page_numbers)}",
            f"ocr-max-tokens={options.ocr_max_tokens}",
            f"ocr-retry-max-tokens={settings.chandra_ocr_retry_max_tokens}",
            "completion-aware-v2",
            f"debug-raw={settings.pipeline_debug_raw_responses}",
        )
        ocr_cache_path = cache_root / ocr_cache_key
        ocr_dir = run_dir / "ocr"
        _cached_dir_stage(
            stages,
            "resume_ocr",
            ocr_cache_path,
            ocr_dir,
            read_cache,
            use_cache,
            lambda: _ocr_resume_pdf(
                resume_path,
                ocr_dir,
                settings,
                options,
                source_page_count=source_page_count,
                processed_page_numbers=processed_page_numbers,
            ),
        )
        ocr_metadata = _load_ocr_metadata(ocr_dir)
        if _ocr_cache_requires_regeneration(ocr_metadata):
            shutil.rmtree(ocr_dir)
            _ocr_resume_pdf(
                resume_path,
                ocr_dir,
                settings,
                options,
                source_page_count=source_page_count,
                processed_page_numbers=processed_page_numbers,
            )
            ocr_metadata = _load_ocr_metadata(ocr_dir)
            if use_cache:
                if ocr_cache_path.exists():
                    shutil.rmtree(ocr_cache_path)
                shutil.copytree(ocr_dir, ocr_cache_path)
            stages.append(
                StageRecord(
                    "resume_ocr_cache_validation",
                    "regenerated",
                    0.0,
                    warning="OCR cache/artifact was incomplete or lacked completion-aware metadata.",
                )
            )
        if ocr_metadata.get("partial_ocr") or ocr_metadata.get("partial"):
            warning = _ocr_partial_warning(ocr_metadata, document_label="resume")
            warnings.append(warning)
            if not (options.allow_partial_evaluation or settings.allow_partial_evaluation):
                raise RuntimeError(
                    f"{warning} Re-run without --ocr-pages, raise --ocr-max-tokens/"
                    "CHANDRA_OCR_RETRY_MAX_TOKENS, or pass --allow-partial-evaluation."
                )
        ocr_input = ocr_dir
    else:
        ocr_input = resume_path
        stages.append(StageRecord("resume_ocr", "skipped", 0.0, warning="Resume input is an OCR artifact."))

    if not resume_structured.exists():
        resume_cache_key = _cache_key(
            "resume-structure",
            source_hashes["resume"],
            settings.gemma_model,
            RESUME_STRUCTURING_PROMPT_VERSION,
            str(options.mock_llm),
        )
        _cached_file_stage(
            stages,
            "resume_structuring",
            cache_root / resume_cache_key / "structured-resume.json",
            resume_structured,
            read_cache,
            use_cache,
            lambda: structure_resume_from_ocr(
                ocr_input,
                candidate_id=candidate_id,
                output_path=resume_structured,
                mock_llm=options.mock_llm,
                settings=settings,
            ),
        )

    report = _timed_stage(
        stages,
        "evaluation",
        lambda: evaluate_candidate_files(resume_structured, jd_structured, evaluation_path),
    )
    if options.mock_llm:
        report = report.model_copy(update={"mock_evaluation": True})
        write_json_report(report, evaluation_path)
    if warnings:
        report = _apply_workflow_warnings(report, warnings)
        write_json_report(report, evaluation_path)
    _verify_arithmetic(report)
    _timed_stage(
        stages,
        "reports",
        lambda: _write_reports(report, run_dir),
    )

    total_elapsed = time.perf_counter() - started_total
    summary = _build_summary(
        report=report,
        run_dir=run_dir,
        stages=stages,
        settings=settings,
        jd_path=jd_path,
        resume_path=resume_path,
        source_hashes=source_hashes,
        total_elapsed=total_elapsed,
        warnings=warnings,
    )
    summary_json = run_dir / "run-summary.json"
    summary_md = run_dir / "run-summary.md"
    (run_dir / "run-manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    if not options.quiet:
        _print_terminal_summary(summary, print_func)
    if options.json_summary:
        print_func(json.dumps(summary, indent=2))
    return MatchWorkflowResult(report=report, run_dir=run_dir, summary_json=summary_json, summary_md=summary_md, stages=stages)


def _interactive_select(
    options: MatchWorkflowOptions,
    settings: Settings,
    input_func: Callable[[str], str],
    print_func: Callable[[str], None],
) -> tuple[Path, Path] | None:
    print_func("")
    print_func("Resume-JD Matching")
    print_func("==================")
    jd_files = discover_supported_files(settings.jd_input_dir, JD_EXTENSIONS)
    resume_files = discover_supported_files(settings.resume_input_dir, RESUME_EXTENSIONS)
    while True:
        jd = options.jd_path or select_file(
            title="Select a JD",
            root=settings.jd_input_dir,
            files=jd_files,
            input_func=input_func,
            print_func=print_func,
        )
        if jd is None:
            return None
        resume = options.resume_path or select_file(
            title="Select a resume",
            root=settings.resume_input_dir,
            files=resume_files,
            allow_back=True,
            input_func=input_func,
            print_func=print_func,
        )
        if resume is None:
            return None
        if resume.name == "__BACK__":
            continue
        return jd, resume


def _ingest_jd_stage(
    jd_path: Path,
    jd_text_path: Path,
    settings: Settings,
    options: MatchWorkflowOptions,
) -> None:
    try:
        document = ingest_jd_document(jd_path)
        jd_text_path.write_text(document.text, encoding="utf-8")
    except ScannedPdfRequiresOcrError:
        if options.mock_llm:
            raise
        ocr_dir = jd_text_path.parent / "jd-ocr"
        responses = ocr_pdf_pages(
            jd_path,
            ChandraClient(settings),
            page_count=options.ocr_pages,
            max_tokens=options.ocr_max_tokens,
            retry_max_tokens=settings.chandra_ocr_retry_max_tokens,
        )
        source_pages = pdf_page_count(jd_path)
        metadata = write_ocr_artifacts(
            responses=responses,
            output_dir=ocr_dir,
            source_pdf=str(jd_path),
            model=settings.chandra_model,
            source_page_count=source_pages,
            processed_page_numbers=list(range(1, len(responses) + 1)),
            debug_raw_responses=settings.pipeline_debug_raw_responses,
        )
        if metadata.get("partial") and not (options.allow_partial_evaluation or settings.allow_partial_evaluation):
            raise RuntimeError(_ocr_partial_warning(metadata, document_label="JD"))
        ocr_document = load_ocr_document(ocr_dir)
        jd_text_path.write_text(prompt_text_from_ocr(ocr_document), encoding="utf-8")


def _ocr_resume_pdf(
    resume_path: Path,
    output_dir: Path,
    settings: Settings,
    options: MatchWorkflowOptions,
    *,
    source_page_count: int,
    processed_page_numbers: list[int],
) -> None:
    responses = ocr_pdf_pages(
        resume_path,
        ChandraClient(settings),
        page_count=options.ocr_pages,
        max_tokens=options.ocr_max_tokens,
        retry_max_tokens=settings.chandra_ocr_retry_max_tokens,
    )
    write_ocr_artifacts(
        responses=responses,
        output_dir=output_dir,
        source_pdf=str(resume_path),
        model=settings.chandra_model,
        source_page_count=source_page_count,
        processed_page_numbers=processed_page_numbers,
        debug_raw_responses=settings.pipeline_debug_raw_responses,
    )


def _timed_stage(stages: list[StageRecord], name: str, fn):
    started = time.perf_counter()
    try:
        result = fn()
    except Exception:
        stages.append(StageRecord(name, "failed", time.perf_counter() - started))
        raise
    stages.append(StageRecord(name, "generated", time.perf_counter() - started))
    return result


def _cached_file_stage(
    stages: list[StageRecord],
    name: str,
    cache_path: Path,
    output_path: Path,
    read_cache: bool,
    write_cache: bool,
    fn,
) -> None:
    started = time.perf_counter()
    if read_cache and cache_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, output_path)
        stages.append(StageRecord(name, "reused", time.perf_counter() - started, str(_safe_path(cache_path))))
        return
    fn()
    if write_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, cache_path)
    stages.append(StageRecord(name, "generated", time.perf_counter() - started, str(_safe_path(output_path))))


def _cached_dir_stage(
    stages: list[StageRecord],
    name: str,
    cache_path: Path,
    output_path: Path,
    read_cache: bool,
    write_cache: bool,
    fn,
) -> None:
    started = time.perf_counter()
    if read_cache and cache_path.is_dir():
        shutil.copytree(cache_path, output_path)
        stages.append(StageRecord(name, "reused", time.perf_counter() - started, str(_safe_path(cache_path))))
        return
    fn()
    if write_cache:
        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.copytree(output_path, cache_path)
    stages.append(StageRecord(name, "generated", time.perf_counter() - started, str(_safe_path(output_path))))


def _write_reports(report: CandidateEvaluationReport, run_dir: Path) -> None:
    write_json_report(report, run_dir / "evaluation.json")
    write_csv_report(report, run_dir / "evaluation.csv")
    write_html_report(report, run_dir / "evaluation.html")


def _load_ocr_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _ocr_cache_requires_regeneration(metadata: dict) -> bool:
    if not metadata:
        return True
    if int(metadata.get("artifact_contract_version", 0) or 0) < 2:
        return True
    required = (
        "partial",
        "partial_reason",
        "completed_page_numbers",
        "truncated_page_numbers",
        "failed_page_numbers",
        "page_level_warnings",
    )
    if any(key not in metadata for key in required):
        return True
    return bool(metadata.get("truncated_page_numbers") or metadata.get("failed_page_numbers"))


def _ocr_partial_warning(metadata: dict, *, document_label: str) -> str:
    attempted = metadata.get("processed_page_count")
    source = metadata.get("source_page_count")
    completed = len(metadata.get("completed_page_numbers", []))
    truncated = metadata.get("truncated_page_numbers", [])
    failed = metadata.get("failed_page_numbers", [])
    reason = metadata.get("partial_reason") or "OCR page extraction incomplete"
    details = [f"Pages attempted: {attempted} / {source}", f"Pages completed: {completed} / {source}"]
    if truncated:
        details.append(f"Truncated pages: {', '.join(str(page) for page in truncated)}")
    if failed:
        details.append(f"Failed pages: {', '.join(str(page) for page in failed)}")
    retry_summary = _ocr_retry_attempt_summary(metadata)
    if retry_summary:
        details.append(f"Retry attempts: {retry_summary}")
    return f"Partial {document_label} OCR: {reason}. {'; '.join(details)}."


def _ocr_retry_attempt_summary(metadata: dict) -> str:
    parts: list[str] = []
    for page in metadata.get("pages", []):
        attempts = page.get("retry_attempts") or []
        if len(attempts) <= 1:
            continue
        attempt_text = ", ".join(
            f"{item.get('max_tokens')}:{item.get('finish_reason')}" for item in attempts
        )
        parts.append(f"page {page.get('page_number')} [{attempt_text}]")
    return "; ".join(parts)


def _apply_workflow_warnings(report: CandidateEvaluationReport, warnings: list[str]) -> CandidateEvaluationReport:
    eligibility = report.eligibility
    if warnings:
        eligibility = eligibility.model_copy(
            update={
                "status": EligibilityStatus.REVIEW,
                "explanation": f"{eligibility.explanation} Workflow warnings require human review.",
            }
        )
    return report.model_copy(
        update={
            "eligibility": eligibility,
            "warnings": sorted(set([*report.warnings, *warnings])),
            "evidence_completeness": min(report.evidence_completeness, 0.5),
        }
    )


def _build_summary(
    *,
    report: CandidateEvaluationReport,
    run_dir: Path,
    stages: list[StageRecord],
    settings: Settings,
    jd_path: Path,
    resume_path: Path,
    source_hashes: dict[str, str],
    total_elapsed: float,
    warnings: list[str],
) -> dict:
    return {
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "pipeline_version": __version__,
        "candidate_id": report.candidate_id,
        "job_id": report.job_id,
        "selected_sources": {
            "jd": str(_safe_path(jd_path)),
            "resume": str(_safe_path(resume_path)),
        },
        "source_file_hashes": source_hashes,
        "ocr_backend": "chandra",
        "chandra_model": settings.chandra_model,
        "gemma_model": settings.gemma_model,
        "token_budgets": {
            "gemma_resume_max_tokens": settings.gemma_resume_max_tokens,
            "chandra_ocr_initial_max_tokens": _ocr_page_coverage(run_dir).get("requested_max_tokens"),
            "chandra_ocr_retry_max_tokens": settings.chandra_ocr_retry_max_tokens,
        },
        "enable_thinking_false_requested": True,
        "response_format_usage": "json_object unless explicit API compatibility fallback",
        "ocr_page_coverage": _ocr_page_coverage(run_dir),
        "stages": {stage.name: stage.as_dict() for stage in stages},
        "total_elapsed_seconds": round(total_elapsed, 3),
        "extraction_methods": _extraction_methods(run_dir),
        "extraction_warnings": _extraction_warnings(run_dir),
        "confirmed_experience_months": report.experience_summary.get("confirmed_months", 0),
        "uncertain_experience_ranges": report.experience_summary.get("uncertain_ranges", []),
        "required_skill_matches": _criterion_match_summary(report, preferred=False),
        "preferred_skill_matches": _criterion_match_summary(report, preferred=True),
        "mandatory_requirement_statuses": report.eligibility.requirement_statuses,
        "mandatory_requirement_details": report.eligibility.requirement_details,
        "criterion_scores": [
            {
                "criterion_id": item.criterion_id,
                "status": item.status.value,
                "match_fraction": item.match_fraction,
                "weighted_score": item.weighted_score,
                "evidence_count": len(item.evidence),
                "missing_information": item.missing_information,
                "ambiguity": item.ambiguity,
            }
            for item in report.criterion_evaluations
        ],
        "score_before_gates": report.score_before_gates,
        "final_score": report.final_score,
        "maximum_score": report.maximum_score,
        "evidence_completeness": report.evidence_completeness,
        "eligibility": report.eligibility.status.value,
        "recommendation_label": report.recommendation_label,
        "human_review_required": report.human_review_required,
        "generated_artifact_paths": {
            "run_dir": str(_safe_path(run_dir)),
            "structured_jd": str(_safe_path(run_dir / "structured-jd.json")),
            "structured_resume": str(_safe_path(run_dir / "structured-resume.json")),
            "evaluation_json": str(_safe_path(run_dir / "evaluation.json")),
            "evaluation_csv": str(_safe_path(run_dir / "evaluation.csv")),
            "evaluation_html": str(_safe_path(run_dir / "evaluation.html")),
            "run_summary_json": str(_safe_path(run_dir / "run-summary.json")),
            "run_summary_md": str(_safe_path(run_dir / "run-summary.md")),
        },
        "errors": [],
        "warnings": warnings,
        "raw_debug_persistence_enabled": settings.pipeline_debug_raw_responses,
        "raw_artifacts_persisted": _raw_artifacts_persisted(run_dir),
        "human_review_disclaimer": report.human_review_disclaimer,
    }


def _summary_markdown(summary: dict) -> str:
    mandatory_rows = "\n".join(
        f"| {item.get('description', item.get('requirement_id', ''))} | {item.get('status', 'unclear')} | "
        f"{_summary_requirement_evidence(item)} | "
        f"{'; '.join(item.get('ambiguity', [])) or item.get('missing_evidence', '') or 'None'} |"
        for item in summary.get("mandatory_requirement_details", [])
    ) or "| None | not_applicable | No direct evidence recorded. | None |"
    criterion_rows = "\n".join(
        f"| {item['criterion_id']} | {item['match_fraction']:.3f} | {item['weighted_score']:.3f} | {item['status']} |"
        for item in summary["criterion_scores"]
    )
    stages = "\n".join(
        f"* {stage['name']}: {stage['status']}"
        for stage in summary["stages"].values()
    )
    ocr = summary["ocr_page_coverage"]
    return f"""# Resume-JD Match Summary

Run ID: {summary['run_id']}
Candidate: {summary['candidate_id']}
Job: {summary['job_id']}
Git commit: {summary['git_commit']}
Models: Chandra={summary['chandra_model']}, Gemma={summary['gemma_model']}

## Pipeline

{stages}

## Result

* Eligibility: {summary['eligibility']}
* Score: {summary['final_score']} / {summary['maximum_score']}
* Evidence completeness: {summary['evidence_completeness']}
* Human review required: {'yes' if summary['human_review_required'] else 'no'}
* OCR pages attempted: {ocr.get('processed_page_count')} / {ocr.get('source_page_count')}
* OCR pages completed: {len(ocr.get('completed_page_numbers', []))} / {ocr.get('source_page_count')}
* OCR truncated pages: {', '.join(str(page) for page in ocr.get('truncated_page_numbers', [])) or 'None'}
* OCR failed pages: {', '.join(str(page) for page in ocr.get('failed_page_numbers', [])) or 'None'}

## Mandatory requirements

| Requirement | Status | Evidence | Notes |
| --- | --- | --- | --- |
{mandatory_rows}

## Criterion scores

| Criterion | Match | Weighted score | Status |
| --- | ---: | ---: | --- |
{criterion_rows}

## Strengths

See criterion scores with positive evidence counts.

## Gaps

See criterion missing_information fields in run-summary.json.

## Ambiguities

Uncertain experience ranges: {len(summary['uncertain_experience_ranges'])}

## Warnings

{chr(10).join(f'* {warning}' for warning in summary['warnings']) or '* None'}

## Artifacts

* HTML report: {summary['generated_artifact_paths']['evaluation_html']}
* JSON summary: {summary['generated_artifact_paths']['run_summary_json']}
* Markdown summary: {summary['generated_artifact_paths']['run_summary_md']}

{summary['human_review_disclaimer']}
"""


def _print_terminal_summary(summary: dict, print_func: Callable[[str], None]) -> None:
    print_func("")
    print_func("=" * 50)
    print_func("MATCH COMPLETE")
    print_func("=" * 14)
    print_func(f"Candidate ID: {summary['candidate_id']}")
    print_func(f"Job: {summary['job_id']}")
    print_func("")
    print_func(f"Eligibility: {summary['eligibility']}")
    print_func(f"Score: {summary['final_score']} / {summary['maximum_score']}")
    print_func(f"Evidence completeness: {summary['evidence_completeness']}")
    print_func(f"Human review required: {'yes' if summary['human_review_required'] else 'no'}")
    print_func("")
    print_func(f"Confirmed experience: {summary['confirmed_experience_months']} months")
    print_func(f"Uncertain experience ranges: {len(summary['uncertain_experience_ranges'])}")
    print_func("")
    print_func("Reports:")
    print_func(f"* {summary['generated_artifact_paths']['evaluation_html']}")
    print_func(f"* {summary['generated_artifact_paths']['run_summary_md']}")
    print_func(f"* {summary['generated_artifact_paths']['run_summary_json']}")
    print_func("")
    print_func(summary["human_review_disclaimer"])


def _print_preflight(settings: Settings, print_func: Callable[[str], None]) -> None:
    print_func("Checking services...")
    for name, ok, detail in check_services(settings):
        mark = "OK" if ok else "ERROR"
        print_func(f"{mark}: {name}: {detail}")


def _preflight_or_raise(settings: Settings) -> None:
    failures = [item for item in check_services(settings) if not item[1]]
    if failures:
        name, _, detail = failures[0]
        raise RuntimeError(f"{name} server is not reachable. {detail}")


def check_services(settings: Settings) -> list[tuple[str, bool, str]]:
    return [
        _check_service("Chandra OCR", str(settings.chandra_base_url).rstrip("/") + "/v1/models"),
        _check_service("Gemma", str(settings.gemma_base_url).rstrip("/") + "/models"),
    ]


def _check_service(name: str, url: str) -> tuple[str, bool, str]:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return name, True, "healthy"
    except Exception as error:
        return name, False, f"{url} ({type(error).__name__}: {error})"


def _looks_like_candidate_profile(path: Path) -> bool:
    try:
        load_candidate_profile(path)
        return True
    except Exception:
        return False


def _verify_arithmetic(report: CandidateEvaluationReport) -> None:
    expected = round(sum(item.weighted_score for item in report.criterion_evaluations), 3)
    if round(report.score_before_gates, 3) != expected:
        raise ValueError(f"Score arithmetic mismatch: {report.score_before_gates} != {expected}")
    if round(report.final_score, 3) != round(min(100.0, report.score_before_gates), 3):
        raise ValueError("Final score does not match documented uncapped score policy")


def _path_identity_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(item.read_bytes())
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _unique_run_dir(output_root: Path, job_id: str, candidate_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / _slug(job_id) / _slug(candidate_id) / timestamp
    if not base.exists():
        return base
    for index in range(2, 100):
        candidate = base.with_name(f"{timestamp}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate unique run directory")


def _slug(value: str) -> str:
    text = _EMAIL_RE.sub("redacted", value.lower())
    text = _PHONE_RE.sub("redacted", text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:80] or "item"


def _safe_path(path: Path) -> Path | str:
    text = str(path)
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    return text


def _duplicate_names(paths: Iterable[Path]) -> set[str]:
    counts: dict[str, int] = {}
    for path in paths:
        counts[path.name] = counts.get(path.name, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _extraction_methods(run_dir: Path) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for name, path in {
        "jd": run_dir / "structured-jd.json",
        "resume": run_dir / "structured-resume.json",
    }.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            output[name] = payload.get("extraction_method")
        except Exception:
            output[name] = None
    return output


def _extraction_warnings(run_dir: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for name, path in {
        "jd": run_dir / "structured-jd.json",
        "resume": run_dir / "structured-resume.json",
    }.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            output[name] = payload.get("extraction_warnings", [])
        except Exception:
            output[name] = []
    return output


def _criterion_match_summary(report: CandidateEvaluationReport, *, preferred: bool) -> list[dict]:
    needle = "preferred" if preferred else "required"
    return [
        {
            "criterion_id": item.criterion_id,
            "match_fraction": item.match_fraction,
            "evidence_count": len(item.evidence),
        }
        for item in report.criterion_evaluations
        if needle in item.criterion_id.lower() or needle in item.explanation.lower()
    ]


def _ocr_page_coverage(run_dir: Path) -> dict:
    metadata = _load_ocr_metadata(run_dir / "ocr")
    return {
        "source_page_count": metadata.get("source_page_count"),
        "processed_page_count": metadata.get("processed_page_count"),
        "processed_page_numbers": metadata.get("processed_page_numbers", []),
        "completed_page_numbers": metadata.get("completed_page_numbers", []),
        "truncated_page_numbers": metadata.get("truncated_page_numbers", []),
        "failed_page_numbers": metadata.get("failed_page_numbers", []),
        "partial": metadata.get("partial", False),
        "partial_reason": metadata.get("partial_reason"),
        "partial_ocr": metadata.get("partial_ocr", False),
        "page_level_warnings": metadata.get("page_level_warnings", []),
        "requested_max_tokens": metadata.get("requested_max_tokens"),
        "page_retry_attempts": {
            str(item.get("page_number")): item.get("retry_attempts", [])
            for item in metadata.get("pages", [])
        },
    }


def _raw_artifacts_persisted(run_dir: Path) -> dict:
    metadata = _load_ocr_metadata(run_dir / "ocr")
    return {
        "chandra_raw_responses": bool(metadata.get("raw_responses_persisted", False)),
        "gemma_raw_responses": (run_dir / "structured-resume-gemma-response.json").is_file(),
    }


def _summary_requirement_evidence(item: dict) -> str:
    evidence = item.get("evidence") or []
    if not evidence:
        return "No direct evidence recorded."
    texts = []
    for evidence_item in evidence[:3]:
        text = evidence_item.get("normalized_text") or evidence_item.get("exact_text") or ""
        page = evidence_item.get("page_number")
        block = evidence_item.get("block_index")
        location = f" page {page} block {block}" if page is not None or block is not None else ""
        texts.append(f"{text[:80]}{location}")
    return "<br>".join(texts)
