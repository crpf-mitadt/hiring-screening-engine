"""Small human-readable HTML report writer."""

from html import escape
from pathlib import Path

from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def write_html_report(report: CandidateEvaluationReport, path: Path) -> Path:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(item.criterion_id)}</td>"
        f"<td>{escape(item.status.value)}</td>"
        f"<td>{item.match_fraction:.3f}</td>"
        f"<td>{item.weighted_score:.3f}</td>"
        f"<td>{escape(item.explanation)}</td>"
        "</tr>"
        for item in report.criterion_evaluations
    )
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
  <p><strong>Recommendation label:</strong> {escape(report.recommendation_label)}</p>
  <div class="notice">{escape(report.human_review_disclaimer)}</div>
  <h2>Criteria</h2>
  <table>
    <thead><tr><th>Criterion</th><th>Status</th><th>Match</th><th>Score</th><th>Explanation</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
