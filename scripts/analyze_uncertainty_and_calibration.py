#!/usr/bin/env python3
"""Bootstrap prompt uncertainty and verifier calibration tables."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import read_jsonl, write_json, write_text


LABELS = ["supported", "partial", "unsupported"]
PROMPT_ORDER = ["direct", "cot", "structured", "event_grounded"]
PROMPT_DISPLAY = {
    "direct": "Direct",
    "cot": "CoT",
    "structured": "Structured",
    "event_grounded": "Event-grounded",
}


@dataclass(frozen=True)
class Counts:
    supported: int
    partial: int
    unsupported: int

    @property
    def total(self) -> int:
        return self.supported + self.partial + self.unsupported


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def _latex_token(value: str) -> str:
    return value.replace("_", r"\_")


def _ci(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _metric(counts: Counts, metric: str) -> float:
    if counts.total <= 0:
        return 0.0
    if metric == "faithfulness":
        return counts.supported / counts.total
    if metric == "unsupported":
        return counts.unsupported / counts.total
    if metric == "partial":
        return counts.partial / counts.total
    raise ValueError(f"Unsupported metric: {metric}")


def _sum_counts(values: list[Counts]) -> Counts:
    return Counts(
        supported=sum(item.supported for item in values),
        partial=sum(item.partial for item in values),
        unsupported=sum(item.unsupported for item in values),
    )


def _parse_run_name(path: Path) -> tuple[str, str, str]:
    stem = path.name[: -len("_verification.jsonl")]
    provider, model, prompt = stem.split("__")
    return provider, model, prompt


def load_verification_counts(result_dir: Path) -> dict[tuple[str, str, str], dict[str, Counts]]:
    runs: dict[tuple[str, str, str], dict[str, Counts]] = {}
    for path in sorted(result_dir.glob("*_verification.jsonl")):
        provider, model, prompt = _parse_run_name(path)
        per_series: dict[str, Counts] = {}
        for record in read_jsonl(path):
            tally = {label: 0 for label in LABELS}
            for result in record.get("verification_results", []):
                label = result.get("label")
                if label in tally:
                    tally[label] += 1
            per_series[str(record["id"])] = Counts(
                supported=tally["supported"],
                partial=tally["partial"],
                unsupported=tally["unsupported"],
            )
        runs[(provider, model, prompt)] = per_series
    if not runs:
        raise FileNotFoundError(f"No verification JSONL files found under {result_dir}")
    return runs


def model_keys_for_prompt(runs: dict[tuple[str, str, str], dict[str, Counts]], prompt: str) -> list[tuple[str, str, str]]:
    keys = [key for key in runs if key[2] == prompt]
    return sorted(keys)


def exact_prompt_metric(runs: dict[tuple[str, str, str], dict[str, Counts]], prompt: str, metric: str) -> float:
    values = []
    for key in model_keys_for_prompt(runs, prompt):
        values.append(_metric(_sum_counts(list(runs[key].values())), metric))
    return float(np.mean(values)) if values else 0.0


def bootstrap_prompt_metric(
    runs: dict[tuple[str, str, str], dict[str, Counts]],
    prompt: str,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    keys = model_keys_for_prompt(runs, prompt)
    if not keys:
        raise ValueError(f"No runs found for prompt: {prompt}")
    samples = np.zeros(n_bootstrap)
    series_ids_by_key = {key: sorted(runs[key]) for key in keys}
    for boot_idx in range(n_bootstrap):
        model_values = []
        for key in keys:
            series_ids = series_ids_by_key[key]
            sampled_ids = rng.choice(series_ids, size=len(series_ids), replace=True)
            model_values.append(_metric(_sum_counts([runs[key][series_id] for series_id in sampled_ids]), metric))
        samples[boot_idx] = float(np.mean(model_values))
    return samples


def bootstrap_prompt_difference(
    runs: dict[tuple[str, str, str], dict[str, Counts]],
    treatment_prompt: str,
    baseline_prompt: str,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, np.ndarray]:
    treatment_keys = {(provider, model): key for key in runs for provider, model, prompt in [key] if prompt == treatment_prompt}
    baseline_keys = {(provider, model): key for key in runs for provider, model, prompt in [key] if prompt == baseline_prompt}
    shared_model_keys = sorted(set(treatment_keys) & set(baseline_keys))
    if not shared_model_keys:
        raise ValueError(f"No paired model runs for {treatment_prompt} vs {baseline_prompt}")

    exact_diffs = []
    samples = np.zeros(n_bootstrap)
    for provider_model in shared_model_keys:
        treatment_key = treatment_keys[provider_model]
        baseline_key = baseline_keys[provider_model]
        shared_series = sorted(set(runs[treatment_key]) & set(runs[baseline_key]))
        exact_diffs.append(
            _metric(_sum_counts([runs[treatment_key][series_id] for series_id in shared_series]), metric)
            - _metric(_sum_counts([runs[baseline_key][series_id] for series_id in shared_series]), metric)
        )

    for boot_idx in range(n_bootstrap):
        model_diffs = []
        for provider_model in shared_model_keys:
            treatment_key = treatment_keys[provider_model]
            baseline_key = baseline_keys[provider_model]
            shared_series = sorted(set(runs[treatment_key]) & set(runs[baseline_key]))
            sampled_ids = rng.choice(shared_series, size=len(shared_series), replace=True)
            treatment_metric = _metric(_sum_counts([runs[treatment_key][series_id] for series_id in sampled_ids]), metric)
            baseline_metric = _metric(_sum_counts([runs[baseline_key][series_id] for series_id in sampled_ids]), metric)
            model_diffs.append(treatment_metric - baseline_metric)
        samples[boot_idx] = float(np.mean(model_diffs))
    return float(np.mean(exact_diffs)), samples


def two_sided_bootstrap_p(samples: np.ndarray) -> float:
    p_lower = float(np.mean(samples <= 0.0))
    p_upper = float(np.mean(samples >= 0.0))
    return min(1.0, 2.0 * min(p_lower, p_upper))


def build_bootstrap_outputs(
    runs: dict[tuple[str, str, str], dict[str, Counts]],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    prompt_rows = []
    prompt_samples: dict[tuple[str, str], np.ndarray] = {}
    for prompt in PROMPT_ORDER:
        row: dict[str, Any] = {"prompt": prompt, "n_model_runs": len(model_keys_for_prompt(runs, prompt))}
        for metric in ["faithfulness", "unsupported"]:
            samples = bootstrap_prompt_metric(runs, prompt, metric, rng, n_bootstrap)
            prompt_samples[(prompt, metric)] = samples
            lower, upper = _ci(samples)
            row[f"{metric}_mean"] = exact_prompt_metric(runs, prompt, metric)
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper
        prompt_rows.append(row)

    comparison_rows = []
    for baseline_prompt in ["direct", "cot", "structured"]:
        row = {"comparison": f"event_grounded_vs_{baseline_prompt}", "baseline_prompt": baseline_prompt}
        for metric in ["faithfulness", "unsupported"]:
            exact_diff, samples = bootstrap_prompt_difference(
                runs,
                treatment_prompt="event_grounded",
                baseline_prompt=baseline_prompt,
                metric=metric,
                rng=rng,
                n_bootstrap=n_bootstrap,
            )
            lower, upper = _ci(samples)
            row[f"delta_{metric}"] = exact_diff
            row[f"delta_{metric}_ci_low"] = lower
            row[f"delta_{metric}_ci_high"] = upper
            row[f"delta_{metric}_p"] = two_sided_bootstrap_p(samples)
        comparison_rows.append(row)

    summary = {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "unit": "series-level resampling within each fixed model run; metrics averaged over five models",
        "prompt_ci": prompt_rows,
        "prompt_comparisons": comparison_rows,
    }
    return pd.DataFrame(prompt_rows), pd.DataFrame(comparison_rows), summary


def read_human_audit(adjudicated_path: Path, audit_full_path: Path) -> pd.DataFrame:
    adjudicated = pd.read_csv(adjudicated_path, dtype=str).fillna("")
    audit_full = pd.read_csv(audit_full_path, dtype=str).fillna("")
    merged = adjudicated.merge(
        audit_full[["audit_id", "auto_score", "provider", "verifier_rationale"]],
        on="audit_id",
        how="left",
        validate="one_to_one",
    )
    merged["auto_score"] = pd.to_numeric(merged["auto_score"], errors="coerce")
    return merged


def calibration_by_label(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        subset = df[df["auto_label"] == label]
        n = int(len(subset))
        if n == 0:
            continue
        final_counts = {final_label: int((subset["final_label"] == final_label).sum()) for final_label in LABELS}
        if label == "supported":
            gate_consistent = final_counts["supported"] / n
        else:
            gate_consistent = (final_counts["partial"] + final_counts["unsupported"]) / n
        rows.append(
            {
                "auto_label": label,
                "n": n,
                "final_supported_rate": final_counts["supported"] / n,
                "final_partial_rate": final_counts["partial"] / n,
                "final_unsupported_rate": final_counts["unsupported"] / n,
                "exact_match_rate": final_counts[label] / n,
                "gate_consistent_rate": gate_consistent,
            }
        )
    return pd.DataFrame(rows)


def calibration_by_error_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    non_supported = df[df["auto_label"] != "supported"].copy()
    non_supported["auto_error_type"] = non_supported["auto_error_type"].replace("", "none")
    for error_type, subset in sorted(non_supported.groupby("auto_error_type"), key=lambda item: (-len(item[1]), item[0])):
        n = int(len(subset))
        rows.append(
            {
                "auto_error_type": error_type,
                "n": n,
                "final_supported_rate": float((subset["final_label"] == "supported").mean()),
                "final_partial_rate": float((subset["final_label"] == "partial").mean()),
                "final_unsupported_rate": float((subset["final_label"] == "unsupported").mean()),
                "non_supported_acceptance_rate": float((subset["final_label"] != "supported").mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_by_score_bin(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[df["auto_score"].notna()].copy()
    bins = [
        (0.0, 0.25, "[0.00,0.25)"),
        (0.25, 0.50, "[0.25,0.50)"),
        (0.50, 0.75, "[0.50,0.75)"),
        (0.75, 0.90, "[0.75,0.90)"),
        (0.90, 1.01, "[0.90,1.00]"),
    ]
    for low, high, label in bins:
        if high >= 1.01:
            subset = valid[(valid["auto_score"] >= low) & (valid["auto_score"] <= 1.0)]
        else:
            subset = valid[(valid["auto_score"] >= low) & (valid["auto_score"] < high)]
        n = int(len(subset))
        if n == 0:
            continue
        rows.append(
            {
                "score_bin": label,
                "n": n,
                "mean_auto_score": float(subset["auto_score"].mean()),
                "final_supported_rate": float((subset["final_label"] == "supported").mean()),
                "final_partial_rate": float((subset["final_label"] == "partial").mean()),
                "final_unsupported_rate": float((subset["final_label"] == "unsupported").mean()),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    by_label = calibration_by_label(df)
    by_error_type = calibration_by_error_type(df)
    by_score_bin = calibration_by_score_bin(df)
    summary = {
        "n": int(len(df)),
        "by_label": by_label.to_dict(orient="records"),
        "by_error_type": by_error_type.to_dict(orient="records"),
        "by_score_bin": by_score_bin.to_dict(orient="records"),
    }
    return by_label, by_error_type, by_score_bin, summary


def make_bootstrap_table(prompt_ci: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    ci_rows = []
    for _, row in prompt_ci.iterrows():
        prompt = str(row["prompt"])
        ci_rows.append(
            f"{PROMPT_DISPLAY[prompt]} & "
            f"{_fmt(row['faithfulness_mean'])} [{_fmt(row['faithfulness_ci_low'])}, {_fmt(row['faithfulness_ci_high'])}] & "
            f"{_fmt(row['unsupported_mean'])} [{_fmt(row['unsupported_ci_low'])}, {_fmt(row['unsupported_ci_high'])}] \\\\"
        )
    comparison_rows = []
    for _, row in comparisons.iterrows():
        baseline = str(row["baseline_prompt"])
        comparison_rows.append(
            f"Event-grounded $-$ {PROMPT_DISPLAY[baseline]} & "
            f"{_fmt(row['delta_faithfulness'])} [{_fmt(row['delta_faithfulness_ci_low'])}, {_fmt(row['delta_faithfulness_ci_high'])}] & "
            f"{_fmt_p(row['delta_faithfulness_p'])} & "
            f"{_fmt(row['delta_unsupported'])} [{_fmt(row['delta_unsupported_ci_low'])}, {_fmt(row['delta_unsupported_ci_high'])}] & "
            f"{_fmt_p(row['delta_unsupported_p'])} \\\\"
        )
    return rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrr}}
\toprule
Prompt & Faith. 95\% CI $\uparrow$ & Unsup. 95\% CI $\downarrow$ \\
\midrule
{chr(10).join(ci_rows)}
\bottomrule
\end{{tabular}}
}}
\vspace{{0.5em}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
Paired comparison & $\Delta$Faith. & $p$ & $\Delta$Unsup. & $p$ \\
\midrule
{chr(10).join(comparison_rows)}
\bottomrule
\end{{tabular}}
}}
\caption{{Bootstrap uncertainty for prompt-level full synthetic results. We resample series within each fixed model run, compute claim-level rates per model, average over the five models, and use paired resampling for event-grounded comparisons. Negative $\Delta$Unsup. means fewer unsupported claims for event-grounded prompting.}}
\label{{tab:bootstrap-uncertainty}}
\end{{table}}
"""


def make_calibration_table(by_label: pd.DataFrame) -> str:
    rows = []
    label_display = {"supported": "Supported", "partial": "Partial", "unsupported": "Unsupported"}
    for _, row in by_label.iterrows():
        rows.append(
            f"{label_display[str(row['auto_label'])]} & {int(row['n'])} & "
            f"{_fmt(row['final_supported_rate'])} & {_fmt(row['final_partial_rate'])} & "
            f"{_fmt(row['final_unsupported_rate'])} & {_fmt(row['gate_consistent_rate'])} \\\\"
        )
    return rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Verifier label & N & Human sup. & Human part. & Human unsup. & Gate-consistent \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
}}
\caption{{Verifier calibration on the adjudicated 180-claim audit sample. Human columns show the final adjudicated label distribution within each automatic verifier label. Gate-consistent counts verifier-supported claims as accepted only when humans also mark them supported, and counts verifier-partial/unsupported claims as accepted when humans mark them non-supported.}}
\label{{tab:verifier-calibration}}
\end{{table}}
"""


def make_calibration_appendix_table(by_error_type: pd.DataFrame, by_score_bin: pd.DataFrame) -> str:
    error_rows = []
    for _, row in by_error_type.iterrows():
        error_type = _latex_token(str(row["auto_error_type"]))
        error_rows.append(
            f"{error_type} & {int(row['n'])} & "
            f"{_fmt(row['final_supported_rate'])} & {_fmt(row['final_partial_rate'])} & "
            f"{_fmt(row['final_unsupported_rate'])} & {_fmt(row['non_supported_acceptance_rate'])} \\\\"
        )
    score_rows = []
    for _, row in by_score_bin.iterrows():
        score_bin = rf"\texttt{{{row['score_bin']}}}"
        score_rows.append(
            f"{score_bin} & {int(row['n'])} & {_fmt(row['mean_auto_score'])} & "
            f"{_fmt(row['final_supported_rate'])} & {_fmt(row['final_partial_rate'])} & {_fmt(row['final_unsupported_rate'])} \\\\"
        )
    return rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Verifier error type & N & Human sup. & Human part. & Human unsup. & Non-sup. accept \\
\midrule
{chr(10).join(error_rows)}
\bottomrule
\end{{tabular}}
}}
\caption{{Human calibration of verifier non-supported error types. The audit sample is balanced by verifier label, so these rates are diagnostic of threshold behavior rather than natural corpus prevalence.}}
\label{{tab:verifier-error-calibration}}
\end{{table}}

\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Score bin & N & Mean score & Human sup. & Human part. & Human unsup. \\
\midrule
{chr(10).join(score_rows)}
\bottomrule
\end{{tabular}}
}}
\caption{{Verifier score calibration on adjudicated human-audit claims. Scores are produced by the rule verifier before human labels are used.}}
\label{{tab:verifier-score-calibration}}
\end{{table}}
"""


def write_outputs(
    prompt_ci: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap_summary: dict[str, Any],
    by_label: pd.DataFrame,
    by_error_type: pd.DataFrame,
    by_score_bin: pd.DataFrame,
    calibration_summary: dict[str, Any],
    output_dir: Path,
    table_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    prompt_ci.to_csv(output_dir / "bootstrap_prompt_ci.csv", index=False)
    comparisons.to_csv(output_dir / "bootstrap_prompt_comparisons.csv", index=False)
    by_label.to_csv(output_dir / "verifier_calibration_by_label.csv", index=False)
    by_error_type.to_csv(output_dir / "verifier_calibration_by_error_type.csv", index=False)
    by_score_bin.to_csv(output_dir / "verifier_calibration_by_score_bin.csv", index=False)
    write_json(
        output_dir / "uncertainty_and_calibration.json",
        {
            "bootstrap": bootstrap_summary,
            "calibration": calibration_summary,
        },
    )

    bootstrap_tex = make_bootstrap_table(prompt_ci, comparisons)
    calibration_tex = make_calibration_table(by_label)
    calibration_appendix_tex = make_calibration_appendix_table(by_error_type, by_score_bin)
    for base in [ROOT / "outputs" / "tables", table_dir]:
        write_text(base / "bootstrap_uncertainty.tex", bootstrap_tex)
        write_text(base / "verifier_calibration.tex", calibration_tex)
        write_text(base / "verifier_calibration_appendix.tex", calibration_appendix_tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("outputs/predictions/full_synthetic_400"))
    parser.add_argument("--adjudicated", type=Path, default=Path("outputs/human_audit/human_audit_adjudicated_labels.csv"))
    parser.add_argument("--audit-full", type=Path, default=Path("outputs/human_audit/human_audit_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    runs = load_verification_counts(_resolve(args.result_dir))
    prompt_ci, comparisons, bootstrap_summary = build_bootstrap_outputs(
        runs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    audit_df = read_human_audit(_resolve(args.adjudicated), _resolve(args.audit_full))
    by_label, by_error_type, by_score_bin, calibration_summary = build_calibration_outputs(audit_df)
    write_outputs(
        prompt_ci,
        comparisons,
        bootstrap_summary,
        by_label,
        by_error_type,
        by_score_bin,
        calibration_summary,
        _resolve(args.output_dir),
        _resolve(args.table_dir),
    )

    print(
        json.dumps(
            {
                "bootstrap_rows": len(prompt_ci),
                "comparison_rows": len(comparisons),
                "calibration_rows": len(by_label),
                "outputs": str(_resolve(args.output_dir).relative_to(ROOT)),
                "tables": [
                    "outputs/tables/bootstrap_uncertainty.tex",
                    "outputs/tables/verifier_calibration.tex",
                    "outputs/tables/verifier_calibration_appendix.tex",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
