"""JSON report writer."""

import json
from pathlib import Path

from chandra_eval.schemas.evaluation import CandidateEvaluationReport


def write_json_report(report: CandidateEvaluationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path
