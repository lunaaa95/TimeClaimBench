#!/usr/bin/env python3
"""Validate internal arithmetic and cross-file consistency of released aggregates.

The human-study raw inputs are intentionally not distributed.  This script does
not recreate bootstrap samples or annotations; it verifies every statistic that
is recoverable from the public aggregate counts and cross-checks the automatic
diagnostics against their tabular summaries.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/analysis/camera_ready"
LABELS = ("supported", "partial", "unsupported")


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def agreement_and_kappa(matrix: list[list[int]]) -> tuple[float, float]:
    n = sum(sum(row) for row in matrix)
    observed = sum(matrix[index][index] for index in range(len(matrix))) / n
    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix))]
    expected = sum(row * column for row, column in zip(row_totals, column_totals)) / (n * n)
    kappa = (observed - expected) / (1.0 - expected)
    return observed, kappa


def macro_f1(matrix: list[list[int]]) -> float:
    scores = []
    for index in range(len(matrix)):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(matrix)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(matrix)) if column != index)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def validate_event_recovery() -> None:
    rows = read_csv("event_extractor_recovery.csv")
    summary = read_json("event_extractor_recovery.json")
    n_gold = sum(int(row["n_gold"]) for row in rows)
    n_recovered = sum(int(row["recovered"]) for row in rows)
    if (n_gold, n_recovered) != (400, 394):
        raise AssertionError(f"event recovery counts: expected 394/400, found {n_recovered}/{n_gold}")
    if summary["n_series"] != n_gold or summary["n_recovered"] != n_recovered:
        raise AssertionError("event recovery CSV and JSON differ")
    close(summary["micro_recall"], n_recovered / n_gold, "event micro recall")
    close(
        summary["macro_mean_best_iou"],
        sum(float(row["mean_best_iou"]) for row in rows) / len(rows),
        "macro mean best IoU",
    )


def validate_shape_coverage() -> None:
    rows = read_csv("synthetic_real_shape_coverage.csv")
    summary = read_json("synthetic_real_shape_coverage.json")
    if len(rows) != 8:
        raise AssertionError(f"expected eight shape features, found {len(rows)}")
    mean_coverage = sum(float(row["real_coverage"]) for row in rows) / len(rows)
    close(mean_coverage, 0.845625, "mean feature coverage")
    close(summary["mean_feature_coverage"], mean_coverage, "shape CSV/JSON coverage")
    if (summary["n_synthetic"], summary["n_real"]) != (400, 200):
        raise AssertionError("unexpected synthetic/real diagnostic sample sizes")


def validate_gold_reference() -> None:
    rows = {row["prompt"]: row for row in read_csv("gold_reference_prompt_summary.csv")}
    summary = read_json("gold_reference_sensitivity.json")
    direct = rows["direct"]
    event_grounded = rows["event_grounded"]
    expected_deltas = {
        ("all", "faithfulness"): float(event_grounded["gold_faith_all"]) - float(direct["gold_faith_all"]),
        ("gold_type", "faithfulness"): float(event_grounded["gold_faith_gold_type"]) - float(direct["gold_faith_gold_type"]),
        ("gold_type", "grounded"): float(event_grounded["gold_grounded_gold_type"]) - float(direct["gold_grounded_gold_type"]),
    }
    bootstrap = {
        (row["panel"], row["metric"]): row
        for row in summary["bootstrap_event_grounded_vs_direct"]
    }
    for key, expected in expected_deltas.items():
        close(float(bootstrap[key]["delta"]), expected, f"gold-reference delta {key}")
    if float(event_grounded["gold_faith_all"]) >= float(direct["gold_faith_all"]):
        raise AssertionError("expected all-claim strict support rank reversal under sparse gold reference")


def validate_claim_audit() -> None:
    rows = read_csv("claim_audit_90_confusion_matrices.csv")
    matrices: dict[str, list[list[int]]] = {}
    for comparison in {row["comparison"] for row in rows}:
        subset = [row for row in rows if row["comparison"] == comparison]
        subset.sort(key=lambda row: LABELS.index(row["row_label"]))
        matrices[comparison] = [
            [int(row[f"column_{label}"]) for label in LABELS]
            for row in subset
        ]
    summary = read_json("claim_audit_90.json")
    human_exact, human_kappa = agreement_and_kappa(matrices["annotator_1_vs_annotator_2"])
    close(human_exact, summary["human_human"]["exact_agreement"], "human-human agreement")
    close(human_kappa, summary["human_human"]["cohen_kappa"], "human-human kappa")
    verifier = matrices["human_consensus_vs_verifier"]
    verifier_exact, verifier_kappa = agreement_and_kappa(verifier)
    target = summary["verifier_vs_exact_consensus"]
    close(verifier_exact, target["exact_agreement"], "verifier agreement")
    close(verifier_kappa, target["cohen_kappa"], "verifier kappa")
    close(macro_f1(verifier), target["macro_f1"], "verifier macro-F1")


def validate_preference_and_failures() -> None:
    rows = {row["comparison"]: row for row in read_csv("human_preference_40.csv")}
    summary = read_json("human_preference_40.json")
    for comparison, baseline in (
        ("event_grounded_vs_direct", "direct"),
        ("event_grounded_vs_cot", "cot"),
    ):
        row = rows[comparison]
        result = summary[comparison]
        close(
            float(row["mean_rating_delta_faithfulness"]),
            float(result["mean_faithfulness"][f"delta_event_grounded_minus_{baseline}"]),
            f"{comparison} faithfulness delta",
        )
        close(
            float(row["mean_rating_delta_informativeness"]),
            float(result["mean_informativeness"][f"delta_event_grounded_minus_{baseline}"]),
            f"{comparison} informativeness delta",
        )
        overall_count = sum(
            int(row[column])
            for column in (
                "event_grounded_win_count_overall",
                "baseline_win_count_overall",
                "tie_count_overall",
            )
        )
        if overall_count != 80:
            raise AssertionError(f"{comparison}: expected 80 overall judgments, found {overall_count}")

    automatic = {row["prompt"]: row for row in read_csv("human_preference_same_sample_automatic.csv")}
    for prompt, expected in summary["same_sample_operational_verifier_strict_support"].items():
        row = automatic[prompt]
        rate = int(row["supported_count"]) / int(row["n_claims"])
        close(rate, expected, f"same-sample operational support for {prompt}")

    diagnostics = read_json("period4_failure_diagnostics.json")
    period = diagnostics["period_4_text_diagnostic"]
    if sum(period["period_4_count_by_designed_pattern"].values()) != period["event_grounded_answers_asserting_period_4"]:
        raise AssertionError("period-4 pattern counts do not sum to the reported total")
    mismatch = diagnostics["consensus_supported_but_operationally_non_supported"]
    if sum(mismatch["stored_verifier_error_code_counts"].values()) != mismatch["n"]:
        raise AssertionError("verifier mismatch codes do not sum to the reported total")


def main() -> None:
    validate_event_recovery()
    validate_shape_coverage()
    validate_gold_reference()
    validate_claim_audit()
    validate_preference_and_failures()
    print("All camera-ready aggregate diagnostics passed.")
    print("event recovery: 394/400; shape coverage: 0.845625")
    print("claim audit: 76 exact-consensus claims; preference study: 40 series")


if __name__ == "__main__":
    main()
