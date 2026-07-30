"""Evidence completeness metrics."""

from typing import Any

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
    diagnostics = evidence_completeness_diagnostics(evaluations, date_warnings=date_warnings, warnings=warnings)
    criterion_score = diagnostics["component_support_score"]
    if criterion_score == 0:
        return 0.0
    date_score = 0.5 if date_warnings else 1.0
    warning_text = " ".join(warnings or []).lower()
    ocr_score = 0.5 if "partial" in warning_text and "ocr" in warning_text else 1.0
    return round((criterion_score * 0.7) + (date_score * 0.15) + (ocr_score * 0.15), 3)


def evidence_completeness_diagnostics(
    evaluations: list[CriterionEvaluation],
    *,
    date_warnings: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    supported = 0
    source_keys: set[tuple] = set()
    total_core = 0
    supported_core = 0
    total_adjacent = 0
    supported_adjacent = 0
    per_criterion: list[dict[str, Any]] = []
    criterion_scores: list[float] = []
    for evaluation in evaluations:
        in_scope_supported = [
            evidence
            for evidence in evaluation.evidence
            if evidence.evidence_status in {EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED}
        ]
        if in_scope_supported:
            supported += 1
        for evidence in in_scope_supported:
            source_keys.add(
                (
                    evidence.source_type,
                    evidence.source_document_id,
                    evidence.page_number,
                    evidence.block_index,
                    evidence.normalized_text or evidence.exact_text,
                )
            )
        components = evaluation.component_assessment.get("components", [])
        criterion_core = 0
        criterion_supported_core = 0
        criterion_adjacent = 0
        criterion_supported_adjacent = 0
        for component in components:
            component_class = component.get("class", "")
            if component_class in {"required/core", "preferred/core"}:
                total_core += 1
                criterion_core += 1
                if component.get("matched"):
                    supported_core += 1
                    criterion_supported_core += 1
            elif component_class == "adjacent/transferable":
                total_adjacent += 1
                criterion_adjacent += 1
                if component.get("matched"):
                    supported_adjacent += 1
                    criterion_supported_adjacent += 1
        denominator = criterion_core + (criterion_adjacent * 0.5)
        if denominator:
            score = (criterion_supported_core + (criterion_supported_adjacent * 0.5)) / denominator
        else:
            score = 1.0 if in_scope_supported else 0.0
        criterion_scores.append(score)
        per_criterion.append(
            {
                "criterion_id": evaluation.criterion_id,
                "total_required_components": criterion_core,
                "supported_required_components": criterion_supported_core,
                "total_adjacent_components": criterion_adjacent,
                "supported_adjacent_components": criterion_supported_adjacent,
                "controlled_equivalent_components": evaluation.component_assessment.get("controlled_equivalent_components", []),
                "supported_controlled_equivalents": evaluation.component_assessment.get("supported_controlled_equivalents", []),
                "deduplicated_source_count": len(
                    {
                        (
                            evidence.source_type,
                            evidence.source_document_id,
                            evidence.page_number,
                            evidence.block_index,
                            evidence.normalized_text or evidence.exact_text,
                        )
                        for evidence in in_scope_supported
                    }
                ),
                "component_support_score": round(score, 3),
            }
        )
    warning_text = " ".join(warnings or []).lower()
    return {
        "total_criteria": len(evaluations),
        "criteria_with_supported_evidence": supported,
        "component_support_score": round(sum(criterion_scores) / len(criterion_scores), 3) if criterion_scores else 0.0,
        "per_criterion": per_criterion,
        "total_required_core_components": total_core,
        "supported_required_core_components": supported_core,
        "total_adjacent_components_considered": total_adjacent,
        "supported_adjacent_components": supported_adjacent,
        "deduplicated_source_count": len(source_keys),
        "date_warning_count": len(date_warnings or []),
        "date_score": 0.5 if date_warnings else 1.0,
        "ocr_score": 0.5 if "partial" in warning_text and "ocr" in warning_text else 1.0,
        "penalties": {
            "missing_core_components": max(0, total_core - supported_core),
            "duplicate_sources_removed": sum(len(item.evidence) for item in evaluations) - len(source_keys),
        },
    }
