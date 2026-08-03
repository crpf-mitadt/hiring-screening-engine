import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from chandra_eval.reports.csv_report import write_csv_report
from chandra_eval.reports.html_report import write_html_report
from chandra_eval.reports.json_report import write_json_report
from chandra_eval.schemas.common import EvidenceRef
from chandra_eval.schemas.evaluation import CandidateEvaluationReport


FIXTURE_ROOT = Path("tests/fixtures/live_regressions")


def _load_validator_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_evaluation_invariants.py"
    spec = importlib.util.spec_from_file_location("validate_evaluation_invariants", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReportContractTests(unittest.TestCase):
    def test_sanitized_admission_report_shapes_render_all_matched_terms(self) -> None:
        validator = _load_validator_module()
        fixture_names = [
            "admission-counsellor-report-candidate-001-shape.json",
            "admission-counsellor-report-candidate-002-shape.json",
            "admission-counsellor-report-candidate-003-shape.json",
        ]

        for fixture_name in fixture_names:
            with self.subTest(fixture=fixture_name):
                fixture = _load_fixture(fixture_name)
                report = _report_from_fixture(fixture)
                with TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    evaluation_path = write_json_report(report, root / "evaluation.json")
                    csv_path = write_csv_report(report, root / "evaluation.csv")
                    html_path = write_html_report(report, root / "evaluation.html")

                    code, output = _run_validator(
                        validator,
                        [
                            "--evaluation",
                            str(evaluation_path),
                            "--csv-report",
                            str(csv_path),
                            "--html-report",
                            str(html_path),
                        ],
                    )
                    html_text = validator._human_visible_html_text(html_path.read_text(encoding="utf-8"))
                    csv_text = csv_path.read_text(encoding="utf-8")

                self.assertEqual(code, 0, output)
                self.assertIn("PASS: report consistency: checked CSV and HTML report consistency", output)
                self.assertEqual(report.final_score, fixture["final_score"])
                self.assertEqual(report.eligibility.status.value, fixture["eligibility"])
                self.assertEqual(report.evidence_completeness, fixture["evidence_completeness"])
                for term in _matched_terms(report):
                    self.assertIn(validator._normalize_report_text(term), validator._normalize_report_text(html_text))
                    self.assertIn(validator._normalize_report_text(term), validator._normalize_report_text(csv_text))

    def test_report_contract_covers_all_component_classes_and_shared_evidence(self) -> None:
        report = _report_from_fixture(_load_fixture("admission-counsellor-report-candidate-001-shape.json"))
        components = [
            component
            for criterion in report.criterion_evaluations
            for component in criterion.component_assessment["components"]
        ]
        classes = {component["classification"] for component in components}

        self.assertEqual(
            {
                "required/core",
                "preferred",
                "adjacent/transferable",
                "evaluation_dimension",
                "controlled equivalent",
            },
            classes,
        )
        shared = next(component for component in components if component["term"] == "Communication clarity")
        self.assertTrue(shared["matched"])
        self.assertEqual(shared["shared_evidence_with"], ["Parent & student engagement"])

        with TemporaryDirectory() as tmp:
            html_text = write_html_report(report, Path(tmp) / "evaluation.html").read_text(encoding="utf-8")

        self.assertIn("Communication clarity", html_text)
        self.assertIn("Parent &amp; student engagement", html_text)

    def test_html_report_contract_decodes_entities_unicode_and_angle_brackets(self) -> None:
        validator = _load_validator_module()
        report = _report_from_fixture(_load_fixture("admission-counsellor-report-candidate-003-shape.json"))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_path = write_json_report(report, root / "evaluation.json")
            csv_path = write_csv_report(report, root / "evaluation.csv")
            html_path = write_html_report(report, root / "evaluation.html")
            raw_html = html_path.read_text(encoding="utf-8")
            visible_html = validator._human_visible_html_text(raw_html)
            code, output = _run_validator(
                validator,
                [
                    "--evaluation",
                    str(evaluation_path),
                    "--csv-report",
                    str(csv_path),
                    "--html-report",
                    str(html_path),
                ],
            )

        self.assertEqual(code, 0, output)
        self.assertIn("Calls &amp; follow-up", raw_html)
        self.assertIn("Policy &lt;compliance&gt;", raw_html)
        self.assertIn("Calls & follow-up", visible_html)
        self.assertIn("Policy <compliance>", visible_html)
        self.assertIn("O\u2019Level counselling", visible_html)
        self.assertIn("R\u00e9sum\u00e9 review / documentation", visible_html)

    def test_unmatched_terms_are_not_required_as_matched_report_terms(self) -> None:
        validator = _load_validator_module()
        report = _report_from_fixture(_load_fixture("admission-counsellor-report-candidate-003-shape.json"))
        unmatched = "Unmatched hidden term & value"

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_path = write_json_report(report, root / "evaluation.json")
            csv_path = write_csv_report(report, root / "evaluation.csv")
            html_path = write_html_report(report, root / "evaluation.html")
            html_path.write_text(html_path.read_text(encoding="utf-8").replace("Unmatched hidden term &amp; value", ""), encoding="utf-8")
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace(unmatched, ""), encoding="utf-8")
            code, output = _run_validator(
                validator,
                [
                    "--evaluation",
                    str(evaluation_path),
                    "--csv-report",
                    str(csv_path),
                    "--html-report",
                    str(html_path),
                ],
            )

        self.assertEqual(code, 0, output)
        self.assertNotIn("missing matched component terms", output)

    def test_duplicate_matched_component_terms_require_one_visible_entry(self) -> None:
        validator = _load_validator_module()
        report = _report_from_fixture(_load_fixture("admission-counsellor-report-candidate-001-shape.json"))
        payload = report.model_dump(mode="json")
        assessment = payload["criterion_evaluations"][0]["component_assessment"]
        duplicate = dict(assessment["components"][0])
        duplicate["source_fields"] = ["required_skill_names", "linked requirement"]
        duplicate["source"] = "linked requirement req_duplicate"
        duplicate["source_requirement_ids"] = ["req_duplicate"]
        assessment["components"].append(duplicate)
        assessment["total_duplicate_note"] = "sanitized duplicate merged-provenance shape"
        payload["evidence_completeness_diagnostics"] = _completeness_diagnostics(payload["criterion_evaluations"])
        report = CandidateEvaluationReport.model_validate(payload)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_path = write_json_report(report, root / "evaluation.json")
            csv_path = write_csv_report(report, root / "evaluation.csv")
            html_path = write_html_report(report, root / "evaluation.html")
            code, output = _run_validator(
                validator,
                [
                    "--evaluation",
                    str(evaluation_path),
                    "--csv-report",
                    str(csv_path),
                    "--html-report",
                    str(html_path),
                ],
            )
            html_text = html_path.read_text(encoding="utf-8")

        self.assertEqual(code, 0, output)
        self.assertEqual(html_text.count("Admissions counselling [required/core; evidence=1]"), 1)

    def test_validator_reports_missing_human_visible_terms(self) -> None:
        validator = _load_validator_module()
        report = _report_from_fixture(_load_fixture("admission-counsellor-report-candidate-001-shape.json"))
        missing_term = "CRM (lead-tracking)"

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_path = write_json_report(report, root / "evaluation.json")
            csv_path = write_csv_report(report, root / "evaluation.csv")
            html_path = write_html_report(report, root / "evaluation.html")
            html_path.write_text(html_path.read_text(encoding="utf-8").replace(missing_term, ""), encoding="utf-8")
            code, output = _run_validator(
                validator,
                [
                    "--evaluation",
                    str(evaluation_path),
                    "--csv-report",
                    str(csv_path),
                    "--html-report",
                    str(html_path),
                ],
            )

        self.assertEqual(code, 1)
        self.assertIn("FAIL: report consistency", output)
        self.assertIn(f"HTML report is missing matched component terms: {missing_term}", output)


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _report_from_fixture(fixture: dict) -> CandidateEvaluationReport:
    criteria = [_criterion_from_fixture(item) for item in fixture["criteria"]]
    return CandidateEvaluationReport.model_validate(
        {
            "evaluation_id": fixture["case_id"],
            "candidate_id": fixture["case_id"],
            "job_id": "admission-counsellor-fresh-batch",
            "eligibility": {
                "status": fixture["eligibility"],
                "checked_requirements": [],
                "failed_requirements": [],
                "ambiguous_requirements": [],
                "requirement_statuses": {},
                "requirement_details": [],
                "explanation": "Sanitized report-facing regression fixture.",
            },
            "criterion_evaluations": criteria,
            "score_before_gates": fixture["final_score"],
            "final_score": fixture["final_score"],
            "maximum_score": 100.0,
            "evidence_completeness": fixture["evidence_completeness"],
            "evidence_completeness_diagnostics": _completeness_diagnostics(criteria),
            "experience_summary": {},
            "strengths": _strengths(criteria),
            "gaps": [],
            "missing_information": [],
            "recommendation_label": "weak_match_for_human_review",
        }
    )


def _criterion_from_fixture(item: dict) -> dict:
    matched_components = [_component_from_fixture(item["criterion_id"], component, matched=True) for component in item.get("components", [])]
    unmatched_components = [_component_from_fixture(item["criterion_id"], component, matched=False) for component in item.get("unmatched_components", [])]
    components = [*matched_components, *unmatched_components]
    evidence = _dedupe_evidence(
        evidence_item
        for component in components
        if component["matched"]
        for evidence_item in component["evidence"]
    )
    assessment = _assessment_from_components(item, components)
    return {
        "criterion_id": item["criterion_id"],
        "criterion_name": item["criterion_name"],
        "category": item["category"],
        "weight": item["weight"],
        "requirement_ids": [],
        "match_fraction": item["match_fraction"],
        "weighted_score": item["weighted_score"],
        "status": "partially_met" if item["weighted_score"] else "not_found",
        "explanation": "Sanitized report-facing criterion.",
        "evidence": evidence,
        "missing_information": [],
        "ambiguity": [],
        "verification_needed": [],
        "candidate_requirement": item["criterion_name"],
        "evidence_found": [f"{component['term']} supported" for component in components if component["matched"]],
        "component_assessment": assessment,
    }


def _component_from_fixture(criterion_id: str, item: dict, *, matched: bool) -> dict:
    source_fields = list(item.get("source_fields") or [])
    source_requirement_ids = list(item.get("source_requirement_ids") or [])
    evidence_count = int(item.get("evidence_count") or 0) if matched else 0
    evidence = [_evidence(criterion_id, index).model_dump(mode="json") for index in range(evidence_count)]
    return {
        "term": item["term"],
        "source": item["source"],
        "source_fields": source_fields,
        "source_requirement_ids": source_requirement_ids,
        "source_terms": {field: [item["term"]] for field in source_fields},
        "class": item["class"],
        "classification": item["class"],
        "matched": matched,
        "missing": not matched,
        "evidence_count": evidence_count,
        "evidence": evidence,
        "explanation": "",
        "note": "",
        "shared_evidence_with": list(item.get("shared_evidence_with") or []),
    }


def _assessment_from_components(item: dict, components: list[dict]) -> dict:
    required = [component["term"] for component in components if component["classification"] == "required/core"]
    preferred = [component["term"] for component in components if component["classification"] == "preferred"]
    adjacent = [component["term"] for component in components if component["classification"] == "adjacent/transferable"]
    dimensions = [component["term"] for component in components if component["classification"] == "evaluation_dimension"]
    controlled = [component["term"] for component in components if component["classification"] == "controlled equivalent"]
    return {
        "criterion_id": item["criterion_id"],
        "criterion_name": item["criterion_name"],
        "category": item["category"],
        "weight": item["weight"],
        "requirement_ids": [],
        "represented_requirement_ids": [],
        "required_skill_names": [component["term"] for component in components if "required_skill_names" in component["source_fields"]],
        "preferred_skill_names": [component["term"] for component in components if component["source_fields"] == ["preferred_skill_names"]],
        "adjacent_skill_names": [component["term"] for component in components if component["source_fields"] == ["adjacent_skill_names"]],
        "evaluation_dimensions": dimensions,
        "evaluation_dimension_domains": {term: "technical" for term in dimensions},
        "evaluation_dimension_gap_labels": {term: "unverified technical dimension" for term in dimensions},
        "source_criterion_hash": f"sanitized-{item['criterion_id']}",
        "required_core_components": required,
        "preferred_components": preferred,
        "controlled_equivalent_components": controlled,
        "adjacent_components_considered": adjacent,
        "source_declared_adjacent_components": list(adjacent),
        "inferred_adjacent_components": [],
        "adjacent_inference_rules": [],
        "evaluation_dimension_components": dimensions,
        "components": components,
        "supported_core_components": [component["term"] for component in components if component["classification"] == "required/core" and component["matched"]],
        "supported_required_components": [component["term"] for component in components if component["classification"] == "required/core" and component["matched"]],
        "missing_core_components": [component["term"] for component in components if component["classification"] == "required/core" and not component["matched"]],
        "supported_preferred_components": [component["term"] for component in components if component["classification"] == "preferred" and component["matched"]],
        "missing_preferred_components": [component["term"] for component in components if component["classification"] == "preferred" and not component["matched"]],
        "supported_adjacent_components": [component["term"] for component in components if component["classification"] == "adjacent/transferable" and component["matched"]],
        "supported_evaluation_dimensions": [component["term"] for component in components if component["classification"] == "evaluation_dimension" and component["matched"]],
        "missing_evaluation_dimensions": [component["term"] for component in components if component["classification"] == "evaluation_dimension" and not component["matched"]],
        "supported_controlled_equivalents": list(controlled),
        "controlled_equivalents_used": list(controlled),
        "evidence_scopes": ["in_scope" for _ in _dedupe_evidence(
            evidence_item
            for component in components
            if component["matched"]
            for evidence_item in component["evidence"]
        )],
    }


def _completeness_diagnostics(criteria: list[dict]) -> dict:
    totals = {
        "required/core": 0,
        "preferred": 0,
        "adjacent/transferable": 0,
        "evaluation_dimension": 0,
    }
    supported = {key: 0 for key in totals}
    evidence_keys = set()
    for criterion in criteria:
        for component in criterion["component_assessment"]["components"]:
            classification = component["classification"]
            if classification in totals:
                totals[classification] += 1
                if component["matched"]:
                    supported[classification] += 1
            for evidence in component.get("evidence") or []:
                evidence_keys.add(_evidence_key(evidence))
    return {
        "total_required_core_components": totals["required/core"],
        "supported_required_core_components": supported["required/core"],
        "total_preferred_components": totals["preferred"],
        "supported_preferred_components": supported["preferred"],
        "total_adjacent_components_considered": totals["adjacent/transferable"],
        "supported_adjacent_components": supported["adjacent/transferable"],
        "total_evaluation_dimension_components": totals["evaluation_dimension"],
        "supported_evaluation_dimension_components": supported["evaluation_dimension"],
        "deduplicated_source_count": len(evidence_keys),
    }


def _strengths(criteria: list[dict]) -> list[str]:
    output = []
    for criterion in criteria:
        for component in criterion["component_assessment"]["components"]:
            if not component["matched"]:
                continue
            if component["classification"] == "required/core":
                output.append(f"required/core: {criterion['criterion_name']}: {component['term']}")
            elif component["classification"] == "preferred":
                output.append(f"preferred: {criterion['criterion_name']}: {component['term']}")
            elif component["classification"] == "adjacent/transferable":
                output.append(f"adjacent: {criterion['criterion_name']}: {component['term']}")
    return output


def _matched_terms(report: CandidateEvaluationReport) -> list[str]:
    return [
        component["term"]
        for criterion in report.criterion_evaluations
        for component in criterion.component_assessment["components"]
        if component["matched"]
    ]


def _dedupe_evidence(items) -> list[dict]:
    output = []
    seen = set()
    for item in items:
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _evidence(criterion_id: str, index: int) -> EvidenceRef:
    text = f"sanitized supported evidence {criterion_id} {index}"
    return EvidenceRef(
        source_type="sanitized_report_fixture",
        source_document_id="sanitized-report",
        exact_text=text,
        normalized_text=text,
    )


def _evidence_key(item: dict) -> tuple:
    return (
        item.get("source_type"),
        item.get("source_document_id"),
        item.get("page_number"),
        item.get("block_index"),
        item.get("normalized_text") or item.get("exact_text"),
    )


def _run_validator(validator, argv: list[str]) -> tuple[int, str]:
    old_argv = sys.argv
    output = io.StringIO()
    try:
        sys.argv = ["validate_evaluation_invariants.py", *argv]
        with contextlib.redirect_stdout(output):
            code = validator.main()
    finally:
        sys.argv = old_argv
    return code, output.getvalue()
