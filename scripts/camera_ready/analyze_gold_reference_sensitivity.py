#!/usr/bin/env python3
"""Re-verify saved generations against sparse designed-event graphs.

The script makes no model calls.  It changes only the evidence reference while
retaining the released claim parser, verifier rules, and saved generations.
Because the gold graphs contain one designed primary event rather than an
exhaustive annotation of every valid local pattern, the output is a reference-
sensitivity diagnostic rather than independent human truth.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/analysis/camera_ready"

sys.path.insert(0, str(ROOT / "src"))

from tsr.claims.claim_schema import Claim  # noqa: E402
from tsr.events.event_graph import EventGraph  # noqa: E402
from tsr.utils.io import read_jsonl  # noqa: E402
from tsr.verification.verifier import (  # noqa: E402
    VerifierConfig,
    candidate_event_types,
    verify_claim,
)


LABELS = ["supported", "partial", "unsupported"]
PROMPTS = ["direct", "cot", "structured", "event_grounded"]
N_BOOTSTRAP = 5_000
SEED = 2680


def parse_run_name(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix("_verification.jsonl")
    provider, model, prompt = stem.split("__")
    return provider, model, prompt


def proportions(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {label: float("nan") for label in LABELS}
    return {label: counts[label] / total for label in LABELS}


def mean_run_metric(
    runs: dict[tuple[str, str, str], dict[str, dict[str, Counter[str]]]],
    prompt: str,
    panel: str,
    label: str,
) -> float:
    values = []
    for key, per_series in runs.items():
        if key[2] != prompt:
            continue
        counts: Counter[str] = Counter()
        for panels in per_series.values():
            counts.update(panels[panel])
        values.append(proportions(counts)[label])
    return float(np.nanmean(values))


def mean_run_grounded(
    runs: dict[tuple[str, str, str], dict[str, dict[str, Counter[str]]]],
    prompt: str,
    panel: str,
) -> float:
    return 1.0 - mean_run_metric(runs, prompt, panel, "unsupported")


def aggregate_panel_counts(
    per_series: dict[str, dict[str, Counter[str]]],
    panel: str,
    sampled_ids: Iterable[str] | None = None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    ids = sampled_ids if sampled_ids is not None else per_series.keys()
    for series_id in ids:
        counts.update(per_series[series_id][panel])
    return counts


def counter_metric(counts: Counter[str], metric: str) -> float:
    props = proportions(counts)
    if metric == "faithfulness":
        return props["supported"]
    if metric == "unsupported":
        return props["unsupported"]
    if metric == "grounded":
        return 1.0 - props["unsupported"]
    raise ValueError(metric)


def paired_prompt_bootstrap(
    runs: dict[tuple[str, str, str], dict[str, dict[str, Counter[str]]]],
    panel: str,
    metric: str,
    treatment: str = "event_grounded",
    baseline: str = "direct",
) -> dict[str, float | int | str]:
    treatment_keys = {(key[0], key[1]): key for key in runs if key[2] == treatment}
    baseline_keys = {(key[0], key[1]): key for key in runs if key[2] == baseline}
    shared_models = sorted(set(treatment_keys) & set(baseline_keys))
    seed_offset = {
        ("all", "faithfulness"): 11,
        ("all", "unsupported"): 12,
        ("all", "grounded"): 13,
        ("gold_type", "faithfulness"): 21,
        ("gold_type", "unsupported"): 22,
        ("gold_type", "grounded"): 23,
    }[(panel, metric)]
    rng = np.random.default_rng(SEED + seed_offset)

    exact_model_diffs = []
    for model_key in shared_models:
        treatment_series = runs[treatment_keys[model_key]]
        baseline_series = runs[baseline_keys[model_key]]
        shared_ids = sorted(set(treatment_series) & set(baseline_series))
        t_value = counter_metric(aggregate_panel_counts(treatment_series, panel, shared_ids), metric)
        b_value = counter_metric(aggregate_panel_counts(baseline_series, panel, shared_ids), metric)
        exact_model_diffs.append(t_value - b_value)

    samples = np.zeros(N_BOOTSTRAP)
    for boot_idx in range(N_BOOTSTRAP):
        model_diffs = []
        for model_key in shared_models:
            treatment_series = runs[treatment_keys[model_key]]
            baseline_series = runs[baseline_keys[model_key]]
            shared_ids = np.asarray(sorted(set(treatment_series) & set(baseline_series)))
            sampled_ids = rng.choice(shared_ids, size=len(shared_ids), replace=True)
            t_value = counter_metric(aggregate_panel_counts(treatment_series, panel, sampled_ids), metric)
            b_value = counter_metric(aggregate_panel_counts(baseline_series, panel, sampled_ids), metric)
            model_diffs.append(t_value - b_value)
        samples[boot_idx] = float(np.nanmean(model_diffs))

    finite = samples[np.isfinite(samples)]
    lower, upper = np.quantile(finite, [0.025, 0.975])
    p_value = min(1.0, 2.0 * min(float(np.mean(finite <= 0)), float(np.mean(finite >= 0))))
    return {
        "panel": panel,
        "metric": metric,
        "treatment": treatment,
        "baseline": baseline,
        "n_models": len(shared_models),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "delta": float(np.nanmean(exact_model_diffs)),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_value": p_value,
    }


def kendall_tau(values_a: dict[str, float], values_b: dict[str, float]) -> float:
    concordant = 0
    discordant = 0
    prompts = list(PROMPTS)
    for i, left in enumerate(prompts):
        for right in prompts[i + 1 :]:
            sign_a = np.sign(values_a[left] - values_a[right])
            sign_b = np.sign(values_b[left] - values_b[right])
            if sign_a == 0 or sign_b == 0:
                continue
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def run_gold_graph_analysis() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    result_dir = ROOT / "outputs" / "predictions" / "full_synthetic_400"
    gold_graphs = {
        item["series_id"]: EventGraph.from_json(item)
        for item in read_jsonl(ROOT / "data" / "processed" / "test_gold_graphs.jsonl")
    }
    config = VerifierConfig.from_yaml(ROOT / "configs" / "verifier.yaml")

    extracted_runs: dict[tuple[str, str, str], dict[str, dict[str, Counter[str]]]] = {}
    gold_runs: dict[tuple[str, str, str], dict[str, dict[str, Counter[str]]]] = {}
    for path in sorted(result_dir.glob("*_verification.jsonl")):
        run_key = parse_run_name(path)
        extracted_runs[run_key] = {}
        gold_runs[run_key] = {}
        for record in read_jsonl(path):
            series_id = str(record["id"])
            graph = gold_graphs[series_id]
            gold_types = {event.type for event in graph.events}
            extracted_panels = {"all": Counter(), "gold_type": Counter()}
            gold_panels = {"all": Counter(), "gold_type": Counter()}
            extracted_by_id = {
                result["claim_id"]: result for result in record.get("verification_results", [])
            }

            for raw_claim in record.get("claims", []):
                claim = Claim(**raw_claim)
                extracted = extracted_by_id[claim.claim_id]
                gold = verify_claim(claim, graph, config).model_dump()
                eligible = bool(set(candidate_event_types(claim)) & gold_types)
                extracted_panels["all"][extracted["label"]] += 1
                gold_panels["all"][gold["label"]] += 1
                if eligible:
                    extracted_panels["gold_type"][extracted["label"]] += 1
                    gold_panels["gold_type"][gold["label"]] += 1
            extracted_runs[run_key][series_id] = extracted_panels
            gold_runs[run_key][series_id] = gold_panels

    prompt_rows = []
    for prompt in PROMPTS:
        all_n = sum(
            sum(panels["all"].values())
            for key, per_series in gold_runs.items()
            if key[2] == prompt
            for panels in per_series.values()
        )
        eligible_n = sum(
            sum(panels["gold_type"].values())
            for key, per_series in gold_runs.items()
            if key[2] == prompt
            for panels in per_series.values()
        )
        prompt_rows.append(
            {
                "prompt": prompt,
                "n_claims": all_n,
                "n_gold_type_claims": eligible_n,
                "gold_type_share": eligible_n / all_n,
                "extracted_faith_all": mean_run_metric(extracted_runs, prompt, "all", "supported"),
                "gold_faith_all": mean_run_metric(gold_runs, prompt, "all", "supported"),
                "gold_unsupported_all": mean_run_metric(gold_runs, prompt, "all", "unsupported"),
                "extracted_faith_gold_type": mean_run_metric(
                    extracted_runs, prompt, "gold_type", "supported"
                ),
                "gold_faith_gold_type": mean_run_metric(gold_runs, prompt, "gold_type", "supported"),
                "gold_grounded_gold_type": mean_run_grounded(gold_runs, prompt, "gold_type"),
            }
        )

    prompt_df = pd.DataFrame(prompt_rows)
    prompt_df.to_csv(OUT / "gold_reference_prompt_summary.csv", index=False)

    extracted_all = dict(zip(prompt_df["prompt"], prompt_df["extracted_faith_all"]))
    gold_all = dict(zip(prompt_df["prompt"], prompt_df["gold_faith_all"]))
    extracted_restricted = dict(zip(prompt_df["prompt"], prompt_df["extracted_faith_gold_type"]))
    gold_restricted = dict(zip(prompt_df["prompt"], prompt_df["gold_faith_gold_type"]))
    bootstrap = [
        paired_prompt_bootstrap(gold_runs, panel, metric)
        for panel in ["all", "gold_type"]
        for metric in ["faithfulness", "unsupported", "grounded"]
    ]
    summary = {
        "source": "8,000 saved generation records; no new model calls",
        "all_claim_tau_extracted_vs_gold": kendall_tau(extracted_all, gold_all),
        "gold_type_tau_extracted_vs_gold": kendall_tau(extracted_restricted, gold_restricted),
        "prompt_order_extracted_all": sorted(PROMPTS, key=extracted_all.get, reverse=True),
        "prompt_order_gold_all": sorted(PROMPTS, key=gold_all.get, reverse=True),
        "prompt_order_extracted_gold_type": sorted(
            PROMPTS, key=extracted_restricted.get, reverse=True
        ),
        "prompt_order_gold_gold_type": sorted(PROMPTS, key=gold_restricted.get, reverse=True),
        "bootstrap_event_grounded_vs_direct": bootstrap,
        "interpretation_constraint": (
            "Synthetic gold graphs contain designed reference events, not exhaustive local evidence; "
            "gold-type restriction only removes claims whose parsed type cannot match a gold event."
        ),
    }
    (OUT / "gold_reference_sensitivity.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"table": prompt_rows, "summary": summary}


def main() -> None:
    gold = run_gold_graph_analysis()
    print("Gold-graph prompt summary")
    print(pd.DataFrame(gold["table"]).to_string(index=False))
    print("\nGold-reference ranking sensitivity")
    print(json.dumps({
        key: value
        for key, value in gold["summary"].items()
        if key.startswith("prompt_order") or key.endswith("tau_extracted_vs_gold")
    }, indent=2))


if __name__ == "__main__":
    main()
