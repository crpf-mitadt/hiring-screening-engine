#!/usr/bin/env python3
"""Validate privacy-safe evaluation invariants for generated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate resume/JD evaluation invariants.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--jd", type=Path)
    parser.add_argument("--structured-resume", type=Path)
    parser.add_argument("--ocr-metadata", type=Path)
    parser.add_argument("--csv-report", type=Path)
    parser.add_argument("--html-report", type=Path)
    args = parser.parse_args()

    evaluation = _load_json(args.evaluation)
    jd = _load_json(args.jd) if args.jd else None
    resume = _load_json(args.structured_resume) if args.structured_resume else None
    ocr_metadata = _load_json(args.ocr_metadata) if args.ocr_metadata else None
    checks = _checks(evaluation, jd=jd, resume=resume, ocr_metadata=ocr_metadata, csv_report=args.csv_report, html_report=args.html_report)

    failed = False
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}: {detail}")
        failed = failed or not ok
    return 1 if failed else 0


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _checks(
    evaluation: dict[str, Any],
    *,
    jd: dict[str, Any] | None,
    resume: dict[str, Any] | None,
    ocr_metadata: dict[str, Any] | None,
    csv_report: Path | None,
    html_report: Path | None,
) -> list[tuple[str, bool, str]]:
    criteria = evaluation.get("criterion_evaluations", [])
    checks: list[tuple[str, bool, str]] = []
    checks.append(_source_rubric_preserved(criteria, jd))
    checks.append(_component_sources_represented(criteria))
    checks.append(_component_classes_correct(criteria))
    checks.append(_matched_components_have_evidence(criteria))
    checks.append(_positive_criteria_have_evidence(criteria))
    checks.append(_criterion_evidence_union(criteria))
    checks.append(_linked_experience_consistent(evaluation))
    checks.append(_git_github_boundary(criteria))
    checks.append(_allowed_fractions_and_arithmetic(criteria))
    checks.append(_completeness_reconciles(evaluation))
    checks.append(_strengths_gaps_classified(evaluation))
    checks.append(_privacy_safe_resume(resume))
    checks.append(_raw_response_privacy(ocr_metadata))
    checks.append(_report_consistency(evaluation, csv_report, html_report))
    return checks


def _source_rubric_preserved(criteria: list[dict], jd: dict[str, Any] | None) -> tuple[str, bool, str]:
    if jd is None:
        return "source rubric preservation", True, "structured JD not supplied; skipped"
    source = {item["criterion_id"]: item for item in jd.get("scoring_rubric", [])}
    result = {item["criterion_id"]: item for item in criteria}
    if set(source) != set(result):
        return "source rubric preservation", False, "criterion IDs differ"
    for criterion_id, criterion in source.items():
        item = result[criterion_id]
        for key in ("name", "weight", "requirement_ids"):
            result_key = "criterion_name" if key == "name" else key
            if item.get(result_key) != criterion.get(key):
                return "source rubric preservation", False, f"{criterion_id} {key} differs"
        if item.get("category") != criterion.get("category"):
            return "source rubric preservation", False, f"{criterion_id} category differs"
    return "source rubric preservation", True, f"{len(criteria)} criteria preserved"


def _component_sources_represented(criteria: list[dict]) -> tuple[str, bool, str]:
    for item in criteria:
        assessment = item.get("component_assessment") or {}
        components = assessment.get("components", [])
        for field, source in (
            ("required_skill_names", "required_skill_names"),
            ("preferred_skill_names", "preferred_skill_names"),
            ("adjacent_skill_names", "adjacent_skill_names"),
            ("evaluation_dimensions", "evaluation_dimensions"),
        ):
            terms = {str(term).lower() for term in assessment.get(field, [])}
            represented = {
                str(component.get("term", "")).lower()
                for component in components
                if source in str(component.get("source", ""))
            }
            if not terms.issubset(represented):
                return "component source representation", False, f"{item['criterion_id']} lost {field}"
    return "component source representation", True, "all populated fields represented"


def _component_classes_correct(criteria: list[dict]) -> tuple[str, bool, str]:
    expected = {
        "required_skill_names": "required/core",
        "preferred_skill_names": "preferred",
        "adjacent_skill_names": "adjacent/transferable",
        "evaluation_dimensions": "evaluation_dimension",
    }
    for item in criteria:
        for component in (item.get("component_assessment") or {}).get("components", []):
            source = str(component.get("source", ""))
            classification = component.get("classification")
            for marker, expected_class in expected.items():
                if marker in source and classification != expected_class:
                    return "component class correctness", False, f"{item['criterion_id']} {source} is {classification}"
            if source.startswith("adjacent-skill rule") and classification != "adjacent/transferable":
                return "component class correctness", False, f"{item['criterion_id']} inferred adjacent class differs"
    return "component class correctness", True, "component source classes are consistent"


def _matched_components_have_evidence(criteria: list[dict]) -> tuple[str, bool, str]:
    for item in criteria:
        for component in (item.get("component_assessment") or {}).get("components", []):
            if component.get("matched") and not component.get("evidence"):
                return "matched component evidence", False, f"{item['criterion_id']} {component.get('term')} lacks evidence"
    return "matched component evidence", True, "all matched components have evidence"


def _positive_criteria_have_evidence(criteria: list[dict]) -> tuple[str, bool, str]:
    for item in criteria:
        if item.get("weighted_score", 0) > 0 and not item.get("evidence"):
            return "positive criterion evidence", False, f"{item['criterion_id']} has score without evidence"
    return "positive criterion evidence", True, "all positive criteria have evidence"


def _criterion_evidence_union(criteria: list[dict]) -> tuple[str, bool, str]:
    for item in criteria:
        criterion_keys = {_evidence_key(evidence) for evidence in item.get("evidence", [])}
        component_keys = {
            _evidence_key(evidence)
            for component in (item.get("component_assessment") or {}).get("components", [])
            if component.get("matched")
            for evidence in component.get("evidence", [])
        }
        if criterion_keys != component_keys:
            return "criterion evidence union", False, f"{item['criterion_id']} evidence differs from matched components"
    return "criterion evidence union", True, "criterion evidence equals matched component evidence"


def _linked_experience_consistent(evaluation: dict[str, Any]) -> tuple[str, bool, str]:
    details = {item.get("requirement_id"): item for item in evaluation.get("eligibility", {}).get("requirement_details", [])}
    for criterion in evaluation.get("criterion_evaluations", []):
        for component in (criterion.get("component_assessment") or {}).get("components", []):
            source = str(component.get("source", ""))
            for requirement_id, detail in details.items():
                if requirement_id and requirement_id in source and "exp" in requirement_id.lower() and detail.get("status") == "met" and not component.get("matched"):
                    return "linked experience consistency", False, f"{criterion['criterion_id']} contradicts {requirement_id}"
    return "linked experience consistency", True, "no linked experience contradictions"


def _git_github_boundary(criteria: list[dict]) -> tuple[str, bool, str]:
    for item in criteria:
        for component in (item.get("component_assessment") or {}).get("components", []):
            if str(component.get("term", "")).lower() == "github":
                for evidence in component.get("evidence", []):
                    notes = str(evidence.get("notes", "")).lower()
                    if "exact token/phrase 'git'" in notes:
                        return "Git/GitHub boundary", False, f"{item['criterion_id']} labels Git as exact GitHub"
    return "Git/GitHub boundary", True, "plain Git is not exact GitHub evidence"


def _allowed_fractions_and_arithmetic(criteria: list[dict]) -> tuple[str, bool, str]:
    allowed = {0, 0.25, 0.5, 0.75, 1}
    for item in criteria:
        fraction = item.get("match_fraction")
        if fraction not in allowed:
            return "fraction and arithmetic", False, f"{item['criterion_id']} fraction {fraction} is not allowed"
        if round(item.get("weighted_score", 0), 6) != round(item.get("weight", 0) * fraction, 6):
            return "fraction and arithmetic", False, f"{item['criterion_id']} weighted score mismatch"
    return "fraction and arithmetic", True, "all fractions and scores are valid"


def _completeness_reconciles(evaluation: dict[str, Any]) -> tuple[str, bool, str]:
    diagnostics = evaluation.get("evidence_completeness_diagnostics", {})
    totals = {"required/core": 0, "preferred": 0, "adjacent/transferable": 0, "evaluation_dimension": 0}
    supported = dict.fromkeys(totals, 0)
    for item in evaluation.get("criterion_evaluations", []):
        for component in (item.get("component_assessment") or {}).get("components", []):
            classification = component.get("classification")
            if classification in totals:
                totals[classification] += 1
                if component.get("matched"):
                    supported[classification] += 1
    checks = (
        (totals["required/core"], diagnostics.get("total_required_core_components")),
        (supported["required/core"], diagnostics.get("supported_required_core_components")),
        (totals["preferred"], diagnostics.get("total_preferred_components")),
        (supported["preferred"], diagnostics.get("supported_preferred_components")),
        (totals["adjacent/transferable"], diagnostics.get("total_adjacent_components_considered")),
        (supported["adjacent/transferable"], diagnostics.get("supported_adjacent_components")),
        (totals["evaluation_dimension"], diagnostics.get("total_evaluation_dimension_components")),
        (supported["evaluation_dimension"], diagnostics.get("supported_evaluation_dimension_components")),
    )
    if any(left != right for left, right in checks):
        return "completeness reconciliation", False, "component totals do not match diagnostics"
    return "completeness reconciliation", True, "diagnostic totals reconcile"


def _strengths_gaps_classified(evaluation: dict[str, Any]) -> tuple[str, bool, str]:
    strengths = evaluation.get("strengths", [])
    gaps = evaluation.get("gaps", [])
    allowed_strength_prefixes = ("required/core:", "preferred:", "adjacent:")
    allowed_gap_prefixes = ("mandatory/core gap:", "preferred gap:", "unverified soft-skill dimension:")
    if any(":" in item and not item.startswith(allowed_strength_prefixes) for item in strengths):
        return "strengths/gaps classification", False, "unexpected structured strength prefix"
    if any("gap:" in item and not item.startswith(allowed_gap_prefixes) for item in gaps):
        return "strengths/gaps classification", False, "unexpected gap prefix"
    return "strengths/gaps classification", True, "strengths and gaps use structured classes"


def _privacy_safe_resume(resume: dict[str, Any] | None) -> tuple[str, bool, str]:
    if resume is None:
        return "scoring-safe privacy", True, "structured resume not supplied; skipped"
    scoring = resume.get("scoring_safe_profile") or {}
    prohibited = {"personal_information", "name", "email", "phone", "address", "photo"}
    leaked = prohibited & set(scoring)
    if leaked:
        return "scoring-safe privacy", False, f"scoring_safe_profile contains {sorted(leaked)}"
    return "scoring-safe privacy", True, "scoring-safe profile has no prohibited top-level personal fields"


def _raw_response_privacy(ocr_metadata: dict[str, Any] | None) -> tuple[str, bool, str]:
    if ocr_metadata is None:
        return "raw-response privacy", True, "OCR metadata not supplied; skipped"
    if ocr_metadata.get("raw_responses_persisted"):
        return "raw-response privacy", False, "raw OCR responses persisted"
    return "raw-response privacy", True, "raw OCR responses disabled"


def _report_consistency(evaluation: dict[str, Any], csv_report: Path | None, html_report: Path | None) -> tuple[str, bool, str]:
    expected_terms = {
        component.get("term")
        for item in evaluation.get("criterion_evaluations", [])
        for component in (item.get("component_assessment") or {}).get("components", [])
        if component.get("matched")
    }
    if csv_report and csv_report.is_file():
        rows = list(csv.DictReader(csv_report.open(encoding="utf-8")))
        csv_text = " ".join(row.get("component_assessment", "") for row in rows)
        if not all(term in csv_text for term in expected_terms if term):
            return "report consistency", False, "CSV report is missing matched component terms"
    if html_report and html_report.is_file():
        html_text = html_report.read_text(encoding="utf-8")
        if not all(term in html_text for term in expected_terms if term):
            return "report consistency", False, "HTML report is missing matched component terms"
    return "report consistency", True, "supplied reports include matched component terms"


def _evidence_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("source_type"),
        item.get("source_document_id"),
        item.get("page_number"),
        item.get("block_index"),
        item.get("normalized_text") or item.get("exact_text"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
