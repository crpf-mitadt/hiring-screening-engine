"""Small human-readable HTML report writer."""

from html import escape
from pathlib import Path

from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def write_html_report(report: CandidateEvaluationReport, path: Path) -> Path:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(item.criterion_id)}</td>"
        f"<td>{escape(item.criterion_name)}</td>"
        f"<td>{escape(item.category)}</td>"
        f"<td>{item.weight:.3f}</td>"
        f"<td>{escape('; '.join(item.requirement_ids) or 'None')}</td>"
        f"<td>{escape(item.status.value)}</td>"
        f"<td>{item.match_fraction:.3f}</td>"
        f"<td>{item.weighted_score:.3f}</td>"
        f"<td>{escape(item.explanation)}</td>"
        f"<td>{escape(_component_summary(item.component_assessment))}</td>"
        f"<td>{escape('; '.join(item.evidence_found) or 'No direct evidence recorded.')}</td>"
        f"<td>{escape('; '.join(item.verification_needed) or 'None')}</td>"
        "</tr>"
        for item in report.criterion_evaluations
    )
    requirement_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(item.get('description', item.get('requirement_id', ''))))}</td>"
        f"<td>{escape(str(item.get('status', 'unclear')))}</td>"
        f"<td>{escape(_requirement_evidence_text(item))}</td>"
        f"<td>{escape(item.get('explanation') or '; '.join(item.get('ambiguity', [])) or item.get('missing_evidence', '') or 'None')}</td>"
        "</tr>"
        for item in report.eligibility.requirement_details
    )
    warning_items = "".join(f"<li>{escape(warning)}</li>" for warning in report.warnings)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(report.evaluation_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.45rem; text-align: left; }}
    .notice {{ border: 1px solid #996; background: #fffbe6; padding: 0.75rem; }}
  </style>
</head>
<body>
  <h1>Candidate Evaluation Report</h1>
  <p><strong>Candidate:</strong> {escape(report.candidate_id)}</p>
  <p><strong>Job:</strong> {escape(report.job_id)}</p>
  <p><strong>Eligibility:</strong> {escape(report.eligibility.status.value)}</p>
  <p><strong>Final score:</strong> {report.final_score:.3f} / {report.maximum_score:.1f}</p>
  <p><strong>Evidence completeness:</strong> {report.evidence_completeness:.3f}</p>
  <p><strong>Scoring version:</strong> {escape(report.scoring_version)}</p>
  <p><strong>Mock evaluation:</strong> {str(report.mock_evaluation).lower()}</p>
  <p><strong>Recommendation label:</strong> {escape(report.recommendation_label)}</p>
  <div class="notice">{escape(report.human_review_disclaimer)}</div>
  <h2>Warnings</h2>
  <ul>{warning_items or '<li>None</li>'}</ul>
  <h2>Mandatory Requirements</h2>
  <table>
    <thead><tr><th>Requirement</th><th>Status</th><th>Evidence</th><th>Notes</th></tr></thead>
    <tbody>{requirement_rows}</tbody>
  </table>
  <h2>Criteria</h2>
  <table>
    <thead><tr><th>Criterion</th><th>Name</th><th>Category</th><th>Weight</th><th>Linked Requirements</th><th>Status</th><th>Match</th><th>Score</th><th>Explanation</th><th>Components</th><th>Evidence</th><th>Verification</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _requirement_evidence_text(item: dict) -> str:
    evidence = item.get("evidence") or []
    if not evidence:
        return item.get("missing_evidence") or "No direct evidence recorded."
    texts = []
    for evidence_item in evidence:
        text = evidence_item.get("normalized_text") or evidence_item.get("exact_text") or ""
        page = evidence_item.get("page_number")
        block = evidence_item.get("block_index")
        suffix = f" (page {page}, block {block})" if page is not None or block is not None else ""
        texts.append(f"{text[:160]}{suffix}")
    return "; ".join(texts)


def _component_summary(assessment: dict) -> str:
    parts: list[str] = []
    for label, key in (
        ("required/core", "required_core_components"),
        ("core", "supported_core_components"),
        ("preferred considered", "preferred_components"),
        ("preferred", "supported_preferred_components"),
        ("source adjacent considered", "source_declared_adjacent_components"),
        ("inferred adjacent", "inferred_adjacent_components"),
        ("adjacent", "supported_adjacent_components"),
        ("controlled", "controlled_equivalents_used"),
        ("dimensions considered", "evaluation_dimension_components"),
        ("evaluation dimensions", "supported_evaluation_dimensions"),
        ("missing core", "missing_core_components"),
        ("missing preferred", "missing_preferred_components"),
        ("unverified dimensions", "missing_evaluation_dimensions"),
    ):
        values = assessment.get(key) or []
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    return " | ".join(parts) or "No in-scope components matched."
