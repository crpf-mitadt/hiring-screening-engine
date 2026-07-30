#!/usr/bin/env python3
"""Validate privacy-safe evaluation invariants for generated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

CheckResult = tuple[str, bool | None, str]


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
        status = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        print(f"{status}: {name}: {detail}")
        failed = failed or ok is False
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
) -> list[CheckResult]:
    criteria = evaluation.get("criterion_evaluations", [])
    checks: list[CheckResult] = []
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
    checks.append(_dimension_labels_correct(evaluation))
    checks.append(_semantic_dimension_duplicates(evaluation))
    checks.append(_duplicate_gap_entries(evaluation))
    checks.append(_privacy_safe_resume(resume))
    checks.append(_raw_response_privacy(ocr_metadata))
    checks.append(_report_consistency(evaluation, csv_report, html_report))
    return checks


def _source_rubric_preserved(criteria: list[dict], jd: dict[str, Any] | None) -> tuple[str, bool, str]:
    if jd is None:
        return "source rubric preservation", None, "structured JD not supplied"
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
    allowed_gap_prefixes = (
        "mandatory/core gap:",
        "preferred gap:",
        "unverified technical dimension:",
        "unverified experience dimension:",
        "unverified soft-skill dimension:",
        "unverified evaluation dimension:",
    )
    if any(":" in item and not item.startswith(allowed_strength_prefixes) for item in strengths):
        return "strengths/gaps classification", False, "unexpected structured strength prefix"
    if any(":" in item and not item.startswith(allowed_gap_prefixes) for item in gaps):
        return "strengths/gaps classification", False, "unexpected gap prefix"
    return "strengths/gaps classification", True, "strengths and gaps use structured classes"


def _dimension_labels_correct(evaluation: dict[str, Any]) -> tuple[str, bool, str]:
    criteria_by_name = {item.get("criterion_name"): item for item in evaluation.get("criterion_evaluations", [])}
    for gap in evaluation.get("gaps", []):
        if not gap.startswith("unverified ") or " dimension:" not in gap:
            continue
        parts = gap.split(": ", 2)
        if len(parts) != 3:
            return "dimension label correctness", False, f"malformed dimension gap: {gap}"
        prefix, criterion_name, _term = parts
        criterion = criteria_by_name.get(criterion_name)
        if criterion is None:
            return "dimension label correctness", False, f"dimension gap references unknown criterion {criterion_name}"
        expected = _expected_dimension_prefix(str(criterion.get("category", "")))
        if prefix != expected:
            return "dimension label correctness", False, f"{criterion_name} uses '{prefix}', expected '{expected}'"
    return "dimension label correctness", True, "dimension gaps use category-aware labels"


def _semantic_dimension_duplicates(evaluation: dict[str, Any]) -> tuple[str, bool, str]:
    for criterion in evaluation.get("criterion_evaluations", []):
        assessment = criterion.get("component_assessment") or {}
        components = assessment.get("components", [])
        for component in components:
            if (component.get("classification") or component.get("class")) != "evaluation_dimension":
                continue
            if component.get("matched"):
                continue
            term = str(component.get("term", ""))
            term_norm = _normalize(term)
            if _is_experience_dimension(term_norm) and _has_supported_experience_component(components):
                return (
                    "semantic dimension duplicates",
                    False,
                    f"{criterion['criterion_id']} leaves equivalent experience dimension '{term}' unverified",
                )
            if _is_tech_stack_versatility_dimension(term_norm) and _supported_tech_stack_terms(components) >= 2:
                return (
                    "semantic dimension duplicates",
                    False,
                    f"{criterion['criterion_id']} leaves equivalent tech-stack versatility dimension '{term}' unverified",
                )
            if _is_communication_dimension(term_norm) and _has_supported_communication_component(components):
                note = _normalize(str(component.get("note") or component.get("explanation") or ""))
                if "stricter communication dimension" not in note:
                    return (
                        "semantic dimension duplicates",
                        False,
                        f"{criterion['criterion_id']} leaves communication dimension '{term}' unverified without stricter-dimension explanation",
                    )
    return "semantic dimension duplicates", True, "no unsupported dimension duplicates found"


def _duplicate_gap_entries(evaluation: dict[str, Any]) -> tuple[str, bool, str]:
    seen: set[tuple[str, str, str]] = set()
    for gap in evaluation.get("gaps", []):
        parsed = _parse_gap(gap)
        key = (parsed["kind"], _normalize(parsed["criterion"]), _normalize(parsed["term"]))
        if key in seen:
            return "duplicate gap entries", False, f"duplicate gap entry for {parsed['criterion']}: {parsed['term']}"
        seen.add(key)
        if parsed["kind"] == "raw":
            structured_term = _normalize(parsed["term"])
            for other in evaluation.get("gaps", []):
                other_parsed = _parse_gap(other)
                if other_parsed["kind"] != "raw" and _normalize(other_parsed["term"]) == structured_term:
                    return "duplicate gap entries", False, f"raw and structured gap both mention {parsed['term']}"
    return "duplicate gap entries", True, "gap entries are semantically unique"


def _privacy_safe_resume(resume: dict[str, Any] | None) -> tuple[str, bool, str]:
    if resume is None:
        return "scoring-safe privacy", None, "structured resume not supplied"
    scoring = resume.get("scoring_safe_profile") or {}
    prohibited = {"personal_information", "name", "email", "phone", "address", "photo"}
    leaked = prohibited & set(scoring)
    if leaked:
        return "scoring-safe privacy", False, f"scoring_safe_profile contains {sorted(leaked)}"
    return "scoring-safe privacy", True, "scoring-safe profile has no prohibited top-level personal fields"


def _raw_response_privacy(ocr_metadata: dict[str, Any] | None) -> tuple[str, bool, str]:
    if ocr_metadata is None:
        return "raw-response privacy", None, "OCR metadata not supplied"
    if ocr_metadata.get("raw_responses_persisted"):
        return "raw-response privacy", False, "raw OCR responses persisted"
    leaked_keys = _find_raw_response_keys(ocr_metadata)
    if leaked_keys:
        return "raw-response privacy", False, f"OCR metadata contains raw response keys: {sorted(leaked_keys)}"
    return "raw-response privacy", True, "OCR metadata checked; raw OCR responses disabled"


def _report_consistency(evaluation: dict[str, Any], csv_report: Path | None, html_report: Path | None) -> tuple[str, bool, str]:
    if csv_report is None and html_report is None:
        return "report consistency", None, "report files not supplied"
    expected_terms = {
        component.get("term")
        for item in evaluation.get("criterion_evaluations", [])
        for component in (item.get("component_assessment") or {}).get("components", [])
        if component.get("matched")
    }
    checked: list[str] = []
    if csv_report:
        if not csv_report.is_file():
            return "report consistency", False, f"CSV report not found: {csv_report}"
        with csv_report.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        csv_text = " ".join(row.get("component_assessment", "") for row in rows)
        if not all(term in csv_text for term in expected_terms if term):
            return "report consistency", False, "CSV report is missing matched component terms"
        checked.append("CSV")
    if html_report:
        if not html_report.is_file():
            return "report consistency", False, f"HTML report not found: {html_report}"
        html_text = html_report.read_text(encoding="utf-8")
        if not all(term in html_text for term in expected_terms if term):
            return "report consistency", False, "HTML report is missing matched component terms"
        checked.append("HTML")
    return "report consistency", True, f"checked {' and '.join(checked)} report consistency"


def _evidence_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("source_type"),
        item.get("source_document_id"),
        item.get("page_number"),
        item.get("block_index"),
        item.get("normalized_text") or item.get("exact_text"),
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _expected_dimension_prefix(category: str) -> str:
    if category == "skill":
        return "unverified technical dimension"
    if category == "experience":
        return "unverified experience dimension"
    if category == "leadership":
        return "unverified soft-skill dimension"
    return "unverified evaluation dimension"


def _is_experience_dimension(normalized: str) -> bool:
    return ("year" in normalized or "duration" in normalized) and "experience" in normalized


def _is_tech_stack_versatility_dimension(normalized: str) -> bool:
    return (
        "tech stack versatility" in normalized
        or "technical versatility" in normalized
        or ("versatility" in normalized and "stack" in normalized)
        or "technical breadth" in normalized
    )


def _is_communication_dimension(normalized: str) -> bool:
    return "communication" in normalized


def _has_supported_experience_component(components: list[dict]) -> bool:
    for component in components:
        if not component.get("matched") or not component.get("evidence"):
            continue
        text = _normalize(
            " ".join(
                [
                    str(component.get("term", "")),
                    str(component.get("source", "")),
                    str(component.get("note", "")),
                ]
            )
        )
        if "experience" in text or "confirmed months" in text:
            return True
    return False


def _supported_tech_stack_terms(components: list[dict]) -> int:
    supported = set()
    for component in components:
        if not component.get("matched"):
            continue
        text = _normalize(
            " ".join([str(component.get("term", "")), json.dumps(component.get("evidence", []))])
        )
        if "react" in text:
            supported.add("react")
        if "node" in text:
            supported.add("node")
        if "express" in text:
            supported.add("express")
    return len(supported)


def _has_supported_communication_component(components: list[dict]) -> bool:
    for component in components:
        if not component.get("matched") or not component.get("evidence"):
            continue
        text = _normalize(
            " ".join(
                [
                    str(component.get("term", "")),
                    str(component.get("source", "")),
                    str(component.get("note", "")),
                ]
            )
        )
        if "communication" in text:
            return True
    return False


def _parse_gap(gap: str) -> dict[str, str]:
    parts = gap.split(": ", 2)
    if len(parts) == 3 and (
        parts[0].endswith("gap")
        or parts[0].startswith("unverified ")
    ):
        return {"kind": parts[0], "criterion": parts[1], "term": parts[2]}
    return {"kind": "raw", "criterion": "", "term": gap}


def _find_raw_response_keys(value: Any, *, path: str = "") -> set[str]:
    raw_keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in {"choices", "message", "content"}:
                raw_keys.add(child_path)
            raw_keys.update(_find_raw_response_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            raw_keys.update(_find_raw_response_keys(child, path=f"{path}[{index}]"))
    return raw_keys


if __name__ == "__main__":
    raise SystemExit(main())
