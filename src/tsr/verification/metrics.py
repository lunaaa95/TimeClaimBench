"""Evaluation metrics for claim-level time-series rationale verification."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def interval_iou(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    a0, a1 = a
    b0, b1 = b
    if a1 < a0 or b1 < b0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    return float(inter / union) if union > 0 else 0.0


def claim_faithfulness(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for label in labels if label == "supported") / len(labels)


def unsupported_claim_rate(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for label in labels if label == "unsupported") / len(labels)


def partial_support_rate(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for label in labels if label == "partial") / len(labels)


def precision_recall_f1(y_true: Sequence[str], y_pred: Sequence[str], label: str) -> Tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[Sequence[str]] = None) -> float:
    labels = list(labels or ["supported", "partial", "unsupported"])
    if not y_true:
        return 0.0
    return sum(precision_recall_f1(y_true, y_pred, label)[2] for label in labels) / len(labels)


def repair_success_rate(before_labels: Sequence[str], after_labels: Sequence[str]) -> float:
    invalid = [i for i, label in enumerate(before_labels) if label != "supported"]
    if not invalid:
        return 0.0
    fixed = sum(1 for i in invalid if i < len(after_labels) and after_labels[i] == "supported")
    return fixed / len(invalid)


def chronology_violation_rate(error_types: Sequence[Optional[str]]) -> float:
    if not error_types:
        return 0.0
    return sum(1 for err in error_types if err == "chronology_violation") / len(error_types)


def claims_per_explanation(n_claims: int, n_explanations: int) -> float:
    if n_explanations <= 0:
        return 0.0
    return n_claims / n_explanations


def claim_retention_rate(original_claim_count: int, repaired_claim_count: int) -> float:
    if original_claim_count <= 0:
        return 0.0
    return repaired_claim_count / original_claim_count


def supported_claim_preservation(records: Sequence[dict]) -> float:
    """Approximate semantic preservation using verifier-grounded supported claims.

    A supported original claim is counted as preserved when at least one repaired
    supported claim is grounded by an overlapping support event.
    """

    total_supported = 0
    preserved = 0
    for record in records:
        repaired_events = {
            event_id
            for result in record.get("repaired_verification_results", [])
            if result.get("label") == "supported"
            for event_id in result.get("support_events", [])
        }
        for result in record.get("verification_results", []):
            if result.get("label") != "supported":
                continue
            total_supported += 1
            if set(result.get("support_events", [])) & repaired_events:
                preserved += 1
    if total_supported <= 0:
        return 0.0
    return preserved / total_supported


def summarize_generation_cost(records: Sequence[dict]) -> Dict[str, float]:
    total_cost = sum(float(record.get("estimated_cost_usd", 0.0) or 0.0) for record in records)
    input_tokens = sum(int(record.get("usage", {}).get("input_tokens", 0) or 0) for record in records)
    output_tokens = sum(int(record.get("usage", {}).get("output_tokens", 0) or 0) for record in records)
    total_tokens = sum(int(record.get("usage", {}).get("total_tokens", 0) or 0) for record in records)
    n = len(records)
    return {
        "generation_cost_usd": total_cost,
        "cost_per_explanation_usd": total_cost / n if n else 0.0,
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(total_tokens),
        "tokens_per_explanation": total_tokens / n if n else 0.0,
    }


def summarize_verification(results: Iterable[dict]) -> Dict[str, float]:
    labels = [result.get("label") for result in results]
    errors = [result.get("error_type") for result in results]
    return {
        "claim_faithfulness": claim_faithfulness(labels),
        "unsupported_claim_rate": unsupported_claim_rate(labels),
        "partial_support_rate": partial_support_rate(labels),
        "chronology_violation_rate": chronology_violation_rate(errors),
        "n_claims": float(len(labels)),
    }


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    labels = ["supported", "partial", "unsupported"]
    report: Dict[str, float] = {"macro_f1": macro_f1(y_true, y_pred, labels)}
    for label in labels:
        precision, recall, f1 = precision_recall_f1(y_true, y_pred, label)
        report[f"{label}_precision"] = precision
        report[f"{label}_recall"] = recall
        report[f"{label}_f1"] = f1
    report["n"] = float(len(y_true))
    report["label_distribution_true"] = dict(Counter(y_true))  # type: ignore[assignment]
    report["label_distribution_pred"] = dict(Counter(y_pred))  # type: ignore[assignment]
    return report
