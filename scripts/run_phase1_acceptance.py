#!/usr/bin/env python3
"""Run synthetic Phase 1 acceptance checks for the evaluation engine."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Callable

from chandra_eval.ocr.full_document import write_ocr_artifacts
from chandra_eval.pipeline.pilot import run_synthetic_pilot
from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.scoring.engine import evaluate_candidate
from chandra_eval.scoring.provenance import SourceRepresentationIndex
from chandra_eval.schemas.common import EvidenceRef
from chandra_eval.schemas.jd import JDRequirement, RequirementCategory, RequirementType, ScoringCriterion, StructuredJD
from chandra_eval.schemas.resume import CandidateProfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 synthetic acceptance checks.")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1-acceptance"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and args.force:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []

    def check(name: str, fn: Callable[[], Any]) -> Any:
        try:
            result = fn()
            checks.append({"name": name, "status": "PASS", "detail": result if isinstance(result, str) else ""})
            return result
        except Exception as error:
            checks.append({"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})
            return None

    pilot = check("synthetic pilot", lambda: _pilot(args.output / "synthetic-pilot"))
    check("strong golden case", lambda: _score_case("strong", ["Python", "SQL"], expected_score=100))
    check("partial adjacent-stack case", lambda: _score_case("partial", ["Python"], expected_score=50))
    check("weak unrelated case", lambda: _score_case("weak", ["Accounting"], expected_score=0))
    check("experience-only case", lambda: _experience_case())
    check("ambiguous-date case", lambda: _ambiguous_date_case())
    check("partial OCR privacy case", lambda: _partial_ocr_case(args.output / "partial-ocr"))
    check("soft-skill evidence case", lambda: _soft_skill_case(with_evidence=True))
    check("evidence-less trigger case", lambda: _soft_skill_case(with_evidence=False))
    check("required/preferred/adjacent overlap", lambda: _data_science_overlap_case(matched=True))
    check("Data Science merged-provenance matrix", lambda: _data_science_overlap_case(matched=False))
    if pilot:
        check("report consistency validation", lambda: _report_consistency(Path(pilot)))
        check("privacy validation", lambda: _privacy_validation(Path(pilot)))
    check("matrix mixed-status test", lambda: _matrix_case(args.output / "matrix"))
    check("matrix runner PASS case", lambda: _matrix_status_case(args.output / "matrix", "pass", "PASS"))
    check("matrix runner PIPELINE_FAIL case", lambda: _matrix_status_case(args.output / "matrix", "pipeline-fail", "PIPELINE_FAIL"))
    check("matrix runner VALIDATOR_FAIL case", lambda: _matrix_status_case(args.output / "matrix", "validator-fail", "VALIDATOR_FAIL"))
    check("shared-JD hash comparison", lambda: _shared_jd_hash_case(args.output / "matrix"))
    check("serialization determinism", lambda: _determinism_case())
    check("source-order determinism", lambda: _source_order_case())
    check("criterion-order determinism", lambda: _criterion_order_case())
    check("scoring-repeat determinism", lambda: _repeat_scoring_case())

    summary = {
        "schema_version": "phase1-acceptance-v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "total_checks": len(checks),
        "passed": len([item for item in checks if item["status"] == "PASS"]),
        "failed": len([item for item in checks if item["status"] == "FAIL"]),
        "checks": checks,
    }
    (args.output / "phase1-acceptance-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "phase1-acceptance-summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def _pilot(output: Path) -> str:
    report = run_synthetic_pilot(Path("tests/fixtures/synthetic"), output)
    if report.final_score <= 0:
        raise AssertionError("synthetic pilot produced no score")
    return str(output)


def _score_case(name: str, skills: list[str], *, expected_score: float) -> str:
    report = evaluate_candidate(_candidate(name, skills).scoring_safe_profile, _simple_jd())
    if report.final_score != expected_score:
        raise AssertionError(f"{name} score {report.final_score} != {expected_score}")
    return f"score={report.final_score}"


def _experience_case() -> str:
    jd = StructuredJD(
        job_id="job-exp",
        source_document_id="jd",
        job_title="Experience Role",
        minimum_experience_months=24,
        mandatory_requirements=[
            JDRequirement(
                requirement_id="req_exp_01",
                category=RequirementCategory.EXPERIENCE,
                description="Minimum 24 months of relevant Python experience.",
                requirement_type=RequirementType.MINIMUM,
                mandatory=True,
                minimum_value=24,
                unit="months",
            )
        ],
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="exp",
                name="Years of experience",
                category=RequirementCategory.EXPERIENCE,
                description="Minimum 24 months of relevant Python experience.",
                weight=100,
                requirement_ids=["req_exp_01"],
                evaluation_dimensions=["Years of experience"],
            )
        ],
    )
    text = "Python developer January 2023 to January 2026."
    candidate = CandidateProfile.model_validate(
        {
            "candidate_id": "exp",
            "source_document_id": "resume",
            "experience": [
                {
                    "title": "Python Developer",
                    "start_date": "2023-01",
                    "end_date": "2026-01",
                    "responsibilities": ["Built Python services."],
                    "skills_used": ["Python"],
                    "evidence": [_evidence(text).model_dump(mode="json"), _evidence("Built Python services.").model_dump(mode="json")],
                }
            ],
        }
    )
    report = evaluate_candidate(candidate.scoring_safe_profile, jd)
    if report.final_score <= 0 or report.eligibility.status.value not in {"PASS", "REVIEW"}:
        raise AssertionError("experience case did not score")
    return f"score={report.final_score}"


def _ambiguous_date_case() -> str:
    candidate = CandidateProfile.model_validate(
        {
            "candidate_id": "ambiguous",
            "source_document_id": "resume",
            "experience": [{"start_date": "01/02/2020", "end_date": "2022", "skills_used": ["Python"]}],
        }
    )
    report = evaluate_candidate(candidate.scoring_safe_profile, _simple_jd())
    if not report.experience_summary.get("date_warnings"):
        raise AssertionError("ambiguous date warning missing")
    return "ambiguous date retained"


def _partial_ocr_case(output: Path) -> str:
    metadata = write_ocr_artifacts(
        responses=[{"model": "synthetic", "choices": [{"finish_reason": "length", "message": {"content": "<p>x</p>"}}]}],
        output_dir=output,
        source_pdf="synthetic.pdf",
        model="synthetic",
        source_page_count=2,
        processed_page_numbers=[1],
        debug_raw_responses=False,
    )
    if metadata.get("raw_responses_persisted"):
        raise AssertionError("raw responses persisted unexpectedly")
    return "partial OCR metadata safe"


def _soft_skill_case(*, with_evidence: bool) -> str:
    jd = StructuredJD(
        job_id="soft",
        source_document_id="jd",
        job_title="Role",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="soft",
                name="Communication",
                category=RequirementCategory.LEADERSHIP,
                description="Communication",
                weight=100,
                evaluation_dimensions=["Communication"],
            )
        ],
    )
    payload = {"candidate_id": "soft", "source_document_id": "resume"}
    if with_evidence:
        payload["experience"] = [{"responsibilities": ["Presented findings to stakeholders."], "evidence": [_evidence("Presented findings to stakeholders.").model_dump(mode="json")]}]
    else:
        payload["summary"] = "Presented findings to stakeholders."
    report = evaluate_candidate(CandidateProfile.model_validate(payload).scoring_safe_profile, jd)
    expected = 100 if with_evidence else 0
    if report.final_score != expected:
        raise AssertionError(f"soft skill score {report.final_score} != {expected}")
    return f"score={report.final_score}"


def _data_science_overlap_case(*, matched: bool) -> str:
    jd = StructuredJD(
        job_id="academic",
        source_document_id="jd",
        job_title="Assistant Professor",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="crit_001",
                name="Academic Breadth",
                category=RequirementCategory.SKILL,
                description="Data Science, AI, ML, Teaching, Research, Publications, Communication.",
                weight=100,
                required_skill_names=["Data-Science", "Artificial Intelligence"],
                preferred_skill_names=["Machine Learning", "Research"],
                adjacent_skill_names=["Data Science", "Publications"],
                evaluation_dimensions=["Communication", "Teaching"],
            )
        ],
    )
    candidate = _candidate("academic", ["Data Science", "Artificial Intelligence", "Machine Learning", "Research", "Publications", "Teaching"] if matched else ["Accounting"])
    report = evaluate_candidate(candidate.scoring_safe_profile, jd)
    components = report.criterion_evaluations[0].component_assessment["components"]
    index = SourceRepresentationIndex(components)
    if not index.term_is_represented("Data Science", "adjacent_skill_names"):
        raise AssertionError("Data Science adjacent provenance missing")
    canonical = [component for component in components if "adjacent_skill_names" in component["source_fields"] and "required_skill_names" in component["source_fields"]]
    if not canonical:
        raise AssertionError("merged required+adjacent component missing")
    return f"score={report.final_score}"


def _report_consistency(pilot_output: Path) -> str:
    report = json.loads((pilot_output / "evaluation.json").read_text(encoding="utf-8"))
    csv_text = (pilot_output / "evaluation.csv").read_text(encoding="utf-8")
    html_text = (pilot_output / "evaluation.html").read_text(encoding="utf-8")
    if str(report["final_score"]) not in html_text:
        raise AssertionError("HTML missing final score")
    for criterion in report["criterion_evaluations"]:
        if criterion["criterion_id"] not in csv_text or criterion["criterion_id"] not in html_text:
            raise AssertionError(f"reports missing {criterion['criterion_id']}")
    return "reports agree with evaluation identifiers"


def _privacy_validation(pilot_output: Path) -> str:
    metadata = json.loads((pilot_output / "ocr" / "candidate-001" / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("raw_responses_persisted"):
        raise AssertionError("raw OCR responses persisted")
    return "raw responses disabled"


def _matrix_case(output: Path) -> str:
    matrix = _load_matrix_module()
    fixtures = output / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    jd_path = fixtures / "structured-jd.json"
    resume_path = fixtures / "resume.json"
    bad_resume = fixtures / "bad-resume.json"
    metadata = fixtures / "unsafe-metadata.json"
    jd_path.write_text(_simple_jd().model_dump_json(), encoding="utf-8")
    resume_path.write_text(_candidate("matrix", ["Python", "SQL"]).model_dump_json(), encoding="utf-8")
    bad_resume.write_text("{not json", encoding="utf-8")
    metadata.write_text(json.dumps({"raw_responses_persisted": True}), encoding="utf-8")
    manifest = {
        "schema_version": "match-matrix-manifest-v1",
        "pairs": [
            {"test_name": "pass", "jd_group": "shared", "job_id": "job", "candidate_id": "pass", "structured_jd": str(jd_path), "structured_resume": str(resume_path)},
            {"test_name": "pipeline-fail", "jd_group": "shared", "job_id": "job", "candidate_id": "pipeline", "structured_jd": str(jd_path), "structured_resume": str(bad_resume)},
            {"test_name": "validator-fail", "jd_group": "shared", "job_id": "job", "candidate_id": "validator", "structured_jd": str(jd_path), "structured_resume": str(resume_path), "ocr_metadata": str(metadata)},
        ],
    }
    rows = matrix.run_matrix(manifest, output / "runs", skip_validator=True)
    matrix._write_summaries(rows, output / "runs")
    statuses = [row["status"] for row in rows]
    if statuses != ["PASS", "PIPELINE_FAIL", "VALIDATOR_FAIL"]:
        raise AssertionError(statuses)
    return "mixed statuses accounted"


def _shared_jd_hash_case(matrix_output: Path) -> str:
    summary_path = matrix_output / "runs" / "comparison.json"
    if not summary_path.is_file():
        _matrix_case(matrix_output)
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["rows"]
    hashes = {row["structured_jd_hash"] for row in rows if row["structured_jd_hash"]}
    if len(hashes) != 1 or not all(row["shared_jd_hash_proven"] for row in rows):
        raise AssertionError("shared JD hash proof failed")
    return next(iter(hashes))


def _matrix_status_case(matrix_output: Path, test_name: str, expected_status: str) -> str:
    summary_path = matrix_output / "runs" / "comparison.json"
    if not summary_path.is_file():
        _matrix_case(matrix_output)
    rows = json.loads(summary_path.read_text(encoding="utf-8"))["rows"]
    row = next(item for item in rows if item["test_name"] == test_name)
    if row["status"] != expected_status:
        raise AssertionError(f"{test_name} status {row['status']} != {expected_status}")
    return expected_status


def _determinism_case() -> str:
    report = evaluate_candidate(_candidate("det", ["Python", "SQL"]).scoring_safe_profile, _simple_jd(), evaluation_id="fixed")
    dumped = report.model_dump(mode="json")
    reloaded = type(report).model_validate_json(report.model_dump_json()).model_dump(mode="json")
    if dumped != reloaded:
        raise AssertionError("serialization changed report")
    return "serialization stable"


def _source_order_case() -> str:
    jd_a = _simple_jd()
    jd_b = _simple_jd()
    jd_b.scoring_rubric[0].required_skill_names = list(reversed(jd_b.scoring_rubric[0].required_skill_names))
    candidate = _candidate("order", ["Python", "SQL"])
    if evaluate_candidate(candidate.scoring_safe_profile, jd_a).final_score != evaluate_candidate(candidate.scoring_safe_profile, jd_b).final_score:
        raise AssertionError("source order changed score")
    return "source order stable"


def _criterion_order_case() -> str:
    jd_a = StructuredJD(
        job_id="job-order",
        source_document_id="jd",
        job_title="Order Role",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="python",
                name="Python",
                category=RequirementCategory.SKILL,
                description="Python",
                weight=50,
                required_skill_names=["Python"],
            ),
            ScoringCriterion(
                criterion_id="sql",
                name="SQL",
                category=RequirementCategory.SKILL,
                description="SQL",
                weight=50,
                required_skill_names=["SQL"],
            ),
        ],
    )
    jd_b = jd_a.model_copy(update={"scoring_rubric": list(reversed(jd_a.scoring_rubric))})
    candidate = _candidate("criterion-order", ["Python", "SQL"])
    first = evaluate_candidate(candidate.scoring_safe_profile, jd_a, evaluation_id="fixed")
    second = evaluate_candidate(candidate.scoring_safe_profile, jd_b, evaluation_id="fixed")
    comparable_first = sorted(first.model_dump(mode="json")["criterion_evaluations"], key=lambda item: item["criterion_id"])
    comparable_second = sorted(second.model_dump(mode="json")["criterion_evaluations"], key=lambda item: item["criterion_id"])
    if comparable_first != comparable_second or first.final_score != second.final_score:
        raise AssertionError("criterion order changed scoring")
    return "criterion order stable"


def _repeat_scoring_case() -> str:
    candidate = _candidate("repeat", ["Python", "SQL"])
    jd = _simple_jd()
    first = evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="fixed")
    second = evaluate_candidate(candidate.scoring_safe_profile, jd, evaluation_id="fixed")
    if first.model_dump(mode="json") != second.model_dump(mode="json"):
        raise AssertionError("repeat scoring changed output")
    return "repeat stable"


def _simple_jd() -> StructuredJD:
    return StructuredJD(
        job_id="job-simple",
        source_document_id="jd",
        job_title="Python Role",
        scoring_rubric=[
            ScoringCriterion(
                criterion_id="skills",
                name="Skills",
                category=RequirementCategory.SKILL,
                description="Python and SQL",
                weight=100,
                required_skill_names=["Python", "SQL"],
            )
        ],
    )


def _candidate(candidate_id: str, skills: list[str]) -> CandidateProfile:
    return CandidateProfile.model_validate({"candidate_id": candidate_id, "source_document_id": "resume", "skills": skills})


def _evidence(text: str) -> EvidenceRef:
    return EvidenceRef(source_type="resume_ocr", source_document_id="resume", exact_text=text, normalized_text=text)


def _load_matrix_module():
    path = Path(__file__).resolve().parent / "validate_match_matrix.py"
    spec = importlib.util.spec_from_file_location("validate_match_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Phase 1 Acceptance", "", f"Status: {summary['status']}", f"Checks: {summary['passed']} / {summary['total_checks']} passed", ""]
    for item in summary["checks"]:
        lines.append(f"- {item['status']}: {item['name']} {item['detail']}".rstrip())
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
