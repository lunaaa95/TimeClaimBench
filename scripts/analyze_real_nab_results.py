#!/usr/bin/env python3
"""Compute Real-NAB extension metrics and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.verification.metrics import interval_iou
from tsr.utils.io import read_jsonl, write_text


ANOMALY_LIKE_TYPES = {"anomaly", "sharp_rise", "sharp_drop", "change_point"}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _display_model(provider: str, model: str) -> str:
    display = {
        ("openai", "gpt-4.1-mini"): "GPT-4.1-mini",
        ("deepseek", "deepseek-v4-flash"): "DeepSeek-V4-Flash",
        ("gemini", "gemini-3-flash-preview"): "Gemini-3-Flash",
        ("anthropic", "claude-sonnet-4-5-20250929"): "Claude Sonnet 4.5",
        ("openai", "gpt-5.4-mini"): "GPT-5.4-mini",
        ("deterministic", "event-verbalizer"): "Event verbalizer",
    }
    return display.get((provider, model), f"{provider} {model}")


def _parse_run_name(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix("_verification.jsonl")
    provider, model, prompt = stem.split("__")
    return provider, model.replace("_", "."), prompt


def _as_interval(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if value[0] is None or value[1] is None:
        return None
    return int(value[0]), int(value[1])


def _gold_intervals(sample: dict) -> list[tuple[int, int]]:
    return [
        (int(event["start"]), int(event["end"]))
        for event in sample.get("events", [])
        if event.get("type") == "anomaly"
    ]


def _predicted_intervals(record: dict, claim_types: set[str]) -> list[tuple[int, int]]:
    claims = {claim.get("claim_id"): claim for claim in record.get("claims", [])}
    intervals: list[tuple[int, int]] = []
    for result in record.get("verification_results", []):
        claim = claims.get(result.get("claim_id"), {})
        if claim.get("claim_type") not in claim_types:
            continue
        if result.get("label") == "unsupported":
            continue
        interval = _as_interval(result.get("support_interval")) or _as_interval(claim.get("support_interval"))
        if interval is not None:
            intervals.append(interval)
    return intervals


def _nab_label_metrics(records: list[dict], samples: dict[str, dict]) -> dict:
    ious: list[float] = []
    center_distances: list[float] = []
    consistent = 0
    positives = 0
    normals = 0
    positive_hits = 0
    normal_false_anomalies = 0
    predicted_anomaly_like_windows = 0
    predicted_anomaly_like_hits = 0
    for record in records:
        sample = samples[record["id"]]
        gold = _gold_intervals(sample)
        predicted_anomaly_like = _predicted_intervals(record, ANOMALY_LIKE_TYPES)
        predicted_explicit_anomaly = _predicted_intervals(record, {"anomaly"})
        if predicted_anomaly_like:
            predicted_anomaly_like_windows += 1
        if gold:
            positives += 1
            pairs = [(pred_interval, gold_interval) for pred_interval in predicted_anomaly_like for gold_interval in gold]
            max_iou = max((interval_iou(pred_interval, gold_interval) for pred_interval, gold_interval in pairs), default=0.0)
            best_center_distance = min(
                (
                    abs(((pred[0] + pred[1]) / 2.0) - ((gold_interval[0] + gold_interval[1]) / 2.0))
                    / max(1.0, len(sample.get("series", [])) - 1.0)
                    for pred, gold_interval in pairs
                ),
                default=1.0,
            )
            ious.append(max_iou)
            center_distances.append(best_center_distance)
            hit = int(max_iou > 0.0)
            positive_hits += hit
            predicted_anomaly_like_hits += hit if predicted_anomaly_like else 0
            consistent += hit
        else:
            normals += 1
            false_anomaly = int(len(predicted_explicit_anomaly) > 0)
            normal_false_anomalies += false_anomaly
            consistent += int(not false_anomaly)
    total = positives + normals
    return {
        "nab_positive_windows": positives,
        "nab_normal_windows": normals,
        "nab_anomaly_interval_iou": sum(ious) / len(ious) if ious else 0.0,
        "nab_label_consistency": consistent / total if total else 0.0,
        "nab_positive_hit_rate": positive_hits / positives if positives else 0.0,
        "nab_normal_false_anomaly_rate": normal_false_anomalies / normals if normals else 0.0,
        "nab_anomaly_window_precision": predicted_anomaly_like_hits / predicted_anomaly_like_windows if predicted_anomaly_like_windows else 0.0,
        "nab_center_distance": sum(center_distances) / len(center_distances) if center_distances else 0.0,
    }


def _load_metric(path: Path) -> dict:
    metrics_path = path.with_name(path.name.replace("_verification.jsonl", "_metrics.json"))
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _rows(output_dir: Path, dataset_path: Path) -> list[dict]:
    samples = {sample["id"]: sample for sample in read_jsonl(dataset_path)}
    rows = []
    for verification_path in sorted(output_dir.glob("*_verification.jsonl")):
        records = read_jsonl(verification_path)
        metrics = _load_metric(verification_path)
        before = metrics.get("before_repair", {})
        cost = metrics.get("cost", {})
        provider, model, prompt = _parse_run_name(verification_path)
        nab = _nab_label_metrics(records, samples)
        rows.append(
            {
                "provider": provider,
                "model": model,
                "display_model": _display_model(provider, model),
                "prompt": prompt,
                "n_records": len(records),
                "faithfulness": before.get("claim_faithfulness", 0.0),
                "unsupported_rate": before.get("unsupported_claim_rate", 0.0),
                "partial_rate": before.get("partial_support_rate", 0.0),
                "claims_per_explanation": before.get("claims_per_explanation", 0.0),
                "cost_per_explanation_usd": cost.get("cost_per_explanation_usd", 0.0),
                **nab,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "provider",
        "model",
        "prompt",
        "n_records",
        "faithfulness",
        "unsupported_rate",
        "partial_rate",
        "claims_per_explanation",
        "nab_positive_windows",
        "nab_normal_windows",
        "nab_anomaly_interval_iou",
        "nab_label_consistency",
        "nab_positive_hit_rate",
        "nab_normal_false_anomaly_rate",
        "nab_anomaly_window_precision",
        "nab_center_distance",
        "cost_per_explanation_usd",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _latex_table(rows: list[dict]) -> str:
    if not rows:
        body = "-- & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
    else:
        rows = sorted(rows, key=lambda row: row["display_model"])
        body = "\n".join(
            (
                f"{row['display_model']} & {row['n_records']} & {row['faithfulness']:.3f} "
                f"& {row['unsupported_rate']:.3f} & {row['partial_rate']:.3f} "
                f"& {row['claims_per_explanation']:.2f} & {row['nab_positive_hit_rate']:.3f} "
                f"& {row['nab_normal_false_anomaly_rate']:.3f} & {row['nab_anomaly_window_precision']:.3f} "
                f"& {row['nab_anomaly_interval_iou']:.3f} & {row['nab_center_distance']:.3f} "
                f"& \\${row['cost_per_explanation_usd']:.5f} \\\\"
            )
            for row in rows
        )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{lrrrrrrrrrrr}
\toprule
Model & Windows & Faith. $\uparrow$ & Unsup. $\downarrow$ & Partial & Claims/Expl. & Pos. Hit $\uparrow$ & Explicit FA $\downarrow$ & Anom.-like Prec. $\uparrow$ & IoU $\uparrow$ & Center Dist. $\downarrow$ & Cost/Expl. $\downarrow$ \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Real-NAB diagnostic results on fixed-length windows from the real subsets of the Numenta Anomaly Benchmark. Pos. Hit is the fraction of positive windows where a supported or partially supported anomaly-like claim overlaps the NAB anomaly window. Explicit FA is the false-anomaly rate on normal windows for explicit anomaly claims. Anom.-like Prec. is computed over windows containing any supported or partially supported anomaly-like claim, including anomaly, sharp rise/drop, and change-point claims; it is therefore a broad localization precision rather than ordinary explicit-anomaly precision. IoU measures interval overlap on positive examples, and Center Dist. is normalized center distance on positive examples. The deterministic event verbalizer is a no-API template baseline over the same extracted event graph.}
\label{tab:real-nab-results}
\end{table*}
""" % body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/predictions/real_nab")
    parser.add_argument("--dataset", default="data/real/nab_windows.jsonl")
    parser.add_argument("--csv", default="outputs/analysis/real_nab_results.csv")
    parser.add_argument("--table", default="outputs/tables/real_nab_results.tex")
    args = parser.parse_args()

    rows = _rows(_resolve(args.output_dir), _resolve(args.dataset))
    _write_csv(_resolve(args.csv), rows)
    table = _latex_table(rows)
    write_text(_resolve(args.table), table)
    write_text(ROOT / "outputs" / "tables" / "real_nab_results.tex", table)
    print(f"Wrote {len(rows)} Real-NAB result rows.")
    print(f"CSV: {args.csv}")
    print(f"Table: {args.table}")


if __name__ == "__main__":
    main()
