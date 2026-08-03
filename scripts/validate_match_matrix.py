#!/usr/bin/env python3
"""Run deterministic resume/JD evaluation matrices from reusable artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from chandra_eval.pipeline.candidate_evaluation import evaluate_candidate_files


PASS = "PASS"
PIPELINE_FAIL = "PIPELINE_FAIL"
VALIDATOR_FAIL = "VALIDATOR_FAIL"
MATRIX_MANIFEST_VERSION = "match-matrix-manifest-v1"
MATRIX_COMPARISON_VERSION = "match-matrix-comparison-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a matrix of reusable structured JD/resume matches.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument("--skip-validator", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest paths and planned outputs without evaluating.")
    args = parser.parse_args()

    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_errors = validate_manifest(manifest, args.output_dir)
    if args.dry_run:
        summary = {
            "schema_version": MATRIX_COMPARISON_VERSION,
            "dry_run": True,
            "manifest_errors": manifest_errors,
            "pair_count": len(manifest.get("pairs", [])),
        }
        (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1 if manifest_errors else 0
    if manifest_errors:
        for error in manifest_errors:
            print(f"MANIFEST_FAIL: {error}", file=sys.stderr)
        return 1
    rows = run_matrix(manifest, args.output_dir, skip_validator=args.skip_validator)
    _write_summaries(rows, args.output_dir)
    return 1 if any(row["status"] != PASS for row in rows) else 0


def run_matrix(manifest: dict[str, Any], output_dir: Path, *, skip_validator: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jd_cache: dict[str, Path] = {}
    for index, pair in enumerate(manifest.get("pairs", []), start=1):
        test_name = pair.get("test_name") or f"pair-{index}"
        job_id = pair.get("job_id") or "job"
        candidate_id = pair.get("candidate_id") or f"candidate-{index}"
        run_dir = output_dir / _slug(job_id) / _slug(candidate_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        row = _base_row(test_name, job_id, candidate_id, run_dir)
        try:
            jd_path = _prepare_structured_jd(pair, run_dir, jd_cache)
            resume_path = _prepare_structured_resume(pair, run_dir)
            row["structured_jd_hash"] = _file_sha256(jd_path)
            row["structured_resume_hash"] = _file_sha256(resume_path)
            evaluation_path = run_dir / "evaluation.json"
            report = evaluate_candidate_files(resume_path, jd_path, evaluation_path)
            row["evaluation_exit_status"] = 0
            row["evaluation_json"] = str(evaluation_path)
            row["scoring_version"] = report.scoring_version
            row["final_score"] = report.final_score
            row["eligibility"] = report.eligibility.status.value
            row["evidence_completeness"] = report.evidence_completeness
            row["recommendation_label"] = report.recommendation_label
            row["human_review_required"] = report.human_review_required
        except Exception as error:
            row["status"] = PIPELINE_FAIL
            row["evaluation_exit_status"] = 1
            row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            continue

        evaluation_path = Path(row["evaluation_json"])
        if not evaluation_path.is_file():
            row["status"] = PIPELINE_FAIL
            row["evaluation_exit_status"] = 1
            row["error"] = "evaluation.json was not created"
            rows.append(row)
            continue

        privacy_error = _current_raw_response_privacy_error(run_dir, pair)
        if privacy_error:
            row["status"] = VALIDATOR_FAIL
            row["validator_exit_status"] = 1
            row["error"] = privacy_error
            rows.append(row)
            continue

        if skip_validator:
            row["status"] = PASS
            rows.append(row)
            continue

        validator = _run_validator(
            evaluation_path=evaluation_path,
            jd_path=run_dir / "structured-jd.json",
            resume_path=run_dir / "structured-resume.json",
            ocr_metadata=_optional_path(pair.get("ocr_metadata")),
            csv_report=run_dir / "evaluation.csv",
            html_report=run_dir / "evaluation.html",
        )
        row["validator_exit_status"] = validator.returncode
        row["validator_output"] = (validator.stdout + validator.stderr).strip()
        if validator.returncode != 0:
            row["status"] = VALIDATOR_FAIL
            row["error"] = row["validator_output"] or "validator exited non-zero"
        else:
            row["status"] = PASS
        rows.append(row)
    _record_shared_jd_proofs(rows, manifest)
    return rows


def _base_row(test_name: str, job_id: str, candidate_id: str, run_dir: Path) -> dict[str, Any]:
    return {
        "test_name": test_name,
        "job_id": job_id,
        "candidate_id": candidate_id,
        "status": "",
        "evaluation_exit_status": "",
        "validator_exit_status": "",
        "structured_jd_hash": "",
        "structured_resume_hash": "",
        "shared_jd_hash_proven": "",
        "scoring_version": "",
        "final_score": "",
        "eligibility": "",
        "evidence_completeness": "",
        "recommendation_label": "",
        "human_review_required": "",
        "run_dir": str(run_dir),
        "evaluation_json": "",
        "error": "",
    }


def _prepare_structured_jd(pair: dict[str, Any], run_dir: Path, jd_cache: dict[str, Path]) -> Path:
    source = _required_path(pair, "structured_jd")
    cache_key = pair.get("jd_group") or str(source.resolve())
    if cache_key not in jd_cache:
        jd_cache[cache_key] = source
    target = run_dir / "structured-jd.json"
    shutil.copy2(jd_cache[cache_key], target)
    return target


def _prepare_structured_resume(pair: dict[str, Any], run_dir: Path) -> Path:
    source = _required_path(pair, "structured_resume")
    target = run_dir / "structured-resume.json"
    shutil.copy2(source, target)
    return target


def _run_validator(
    *,
    evaluation_path: Path,
    jd_path: Path,
    resume_path: Path,
    ocr_metadata: Path | None,
    csv_report: Path,
    html_report: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/validate_evaluation_invariants.py",
        "--evaluation",
        str(evaluation_path),
        "--jd",
        str(jd_path),
        "--structured-resume",
        str(resume_path),
        "--csv-report",
        str(csv_report),
        "--html-report",
        str(html_report),
    ]
    if ocr_metadata is not None:
        command.extend(["--ocr-metadata", str(ocr_metadata)])
    return subprocess.run(command, capture_output=True, text=True)


def _current_raw_response_privacy_error(run_dir: Path, pair: dict[str, Any]) -> str:
    metadata_path = _optional_path(pair.get("ocr_metadata"))
    if metadata_path is not None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as error:
            return f"OCR metadata could not be read: {type(error).__name__}: {error}"
        if metadata.get("raw_responses_persisted"):
            return "Current OCR metadata says raw responses were persisted while debug is false."
    raw_files = sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and "raw-response" in path.name.lower()
    )
    if raw_files:
        return "Current run contains unexpected raw response files: " + ", ".join(str(path) for path in raw_files)
    return ""


def _record_shared_jd_proofs(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row, pair in zip(rows, manifest.get("pairs", []), strict=False):
        key = pair.get("jd_group") or row["structured_jd_hash"]
        groups.setdefault(str(key), []).append(row)
    for group_rows in groups.values():
        hashes = {row["structured_jd_hash"] for row in group_rows if row["structured_jd_hash"]}
        proven = len(group_rows) < 2 or len(hashes) == 1
        for row in group_rows:
            row["shared_jd_hash_proven"] = proven


def _write_summaries(rows: list[dict[str, Any]], output_dir: Path) -> None:
    summary = {
        "schema_version": MATRIX_COMPARISON_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "status_counts": {status: len([row for row in rows if row["status"] == status]) for status in (PASS, PIPELINE_FAIL, VALIDATOR_FAIL)},
    }
    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = list(_base_row("", "", "", Path("")).keys())
    for name, delimiter in (("comparison.csv", ","), ("comparison.tsv", "\t")):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})


def validate_manifest(manifest: dict[str, Any], output_dir: Path) -> list[str]:
    errors: list[str] = []
    version = manifest.get("schema_version", MATRIX_MANIFEST_VERSION)
    if version != MATRIX_MANIFEST_VERSION:
        errors.append(f"unsupported manifest schema_version: {version}")
    seen_pairs: set[tuple[str, str, str]] = set()
    planned_outputs: set[Path] = set()
    for index, pair in enumerate(manifest.get("pairs", []), start=1):
        label = pair.get("test_name") or f"pair-{index}"
        try:
            jd_path = _required_path(pair, "structured_jd")
            if not jd_path.name.endswith(".json"):
                errors.append(f"{label}: structured_jd must be JSON")
        except Exception as error:
            errors.append(f"{label}: {error}")
        try:
            _required_path(pair, "structured_resume")
        except Exception as error:
            errors.append(f"{label}: {error}")
        metadata = pair.get("ocr_metadata")
        if metadata and not Path(metadata).expanduser().is_file():
            errors.append(f"{label}: ocr_metadata not found: {metadata}")
        job_id = pair.get("job_id") or "job"
        candidate_id = pair.get("candidate_id") or f"candidate-{index}"
        key = (str(pair.get("jd_group") or pair.get("structured_jd") or ""), str(job_id), str(candidate_id))
        if key in seen_pairs:
            errors.append(f"{label}: duplicate jd_group/job_id/candidate_id")
        seen_pairs.add(key)
        run_dir = output_dir / _slug(str(job_id)) / _slug(str(candidate_id))
        if run_dir in planned_outputs:
            errors.append(f"{label}: duplicate planned output {run_dir}")
        planned_outputs.add(run_dir)
    if not manifest.get("pairs"):
        errors.append("manifest must contain at least one pair")
    return errors


def _required_path(pair: dict[str, Any], key: str) -> Path:
    value = pair.get(key)
    if not value:
        raise ValueError(f"matrix pair requires {key}")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-") or "item"


if __name__ == "__main__":
    raise SystemExit(main())
