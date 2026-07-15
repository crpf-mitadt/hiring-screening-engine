"""Evidence completeness metrics."""

from chandra_eval.schemas.common import EvidenceStatus
from chandra_eval.schemas.evaluation import CriterionEvaluation


def evidence_completeness(evaluations: list[CriterionEvaluation]) -> float:
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
    return round(supported / len(evaluations), 3)
