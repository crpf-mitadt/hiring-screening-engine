"""Evidence completeness metrics."""

from chandra_eval.schemas.common import EvidenceStatus
from chandra_eval.schemas.evaluation import CriterionEvaluation


def evidence_completeness(
    evaluations: list[CriterionEvaluation],
    *,
    date_warnings: list[str] | None = None,
    warnings: list[str] | None = None,
) -> float:
    if not evaluations:
        return 0.0
    supported = 0
    for evaluation in evaluations:
        if any(
            evidence.evidence_status
            in {EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED}
            for evidence in evaluation.evidence
        ):
            supported += 1
    criterion_score = supported / len(evaluations)
    if criterion_score == 0:
        return 0.0
    date_score = 0.5 if date_warnings else 1.0
    warning_text = " ".join(warnings or []).lower()
    ocr_score = 0.5 if "partial" in warning_text and "ocr" in warning_text else 1.0
    return round((criterion_score * 0.7) + (date_score * 0.15) + (ocr_score * 0.15), 3)
