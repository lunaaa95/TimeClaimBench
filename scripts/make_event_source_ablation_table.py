#!/usr/bin/env python3
"""Create the gold-event vs extracted-event ablation table."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import write_text


OUTPUT_DIR = ROOT / "outputs" / "predictions" / "event_source_ablation_400"
GOLD_DIR = ROOT / "outputs" / "predictions" / "gold_events_400"

RUNS = [
    ("openai", "gpt-4.1-mini", "GPT-4.1-mini"),
    ("deepseek", "deepseek-v4-flash", "DeepSeek-V4-Flash"),
]


def _safe_model(model: str) -> str:
    return model.replace(".", "_")


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows() -> list[dict]:
    rows = []
    for provider, model, display in RUNS:
        safe_model = _safe_model(model)
        paths = [
            (
                "Extracted events",
                OUTPUT_DIR / f"{provider}__{safe_model}__extracted_events_metrics.json",
            ),
            (
                "Gold events",
                GOLD_DIR / f"{provider}__{safe_model}__event_grounded_metrics.json",
            ),
        ]
        for source, path in paths:
            metrics = _load_metrics(path)
            before = metrics["before_repair"]
            after = metrics["after_repair"]
            cost = metrics["cost"]
            rows.append(
                {
                    "model": display,
                    "event_source": source,
                    "faithfulness": before["claim_faithfulness"],
                    "unsupported": before["unsupported_claim_rate"],
                    "partial": before["partial_support_rate"],
                    "claims_per_explanation": before["claims_per_explanation"],
                    "faithfulness_after": after["claim_faithfulness"],
                    "retention": metrics["claim_retention_rate"],
                    "semantic_preservation": metrics["supported_claim_preservation"],
                    "cost_per_explanation": cost["cost_per_explanation_usd"],
                }
            )
    return rows


def write_csv(rows: list[dict]) -> None:
    path = OUTPUT_DIR / "event_source_ablation_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[dict]) -> str:
    body = "\n".join(
        (
            f"{row['model']} & {row['event_source']} & {row['faithfulness']:.3f} "
            f"& {row['unsupported']:.3f} & {row['partial']:.3f} "
            f"& {row['claims_per_explanation']:.2f} & {row['faithfulness_after']:.3f} "
            f"& {row['retention']:.3f} & {row['semantic_preservation']:.3f} "
            f"& \\${row['cost_per_explanation']:.5f} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrrrrrr}
\toprule
Model & Event source & Faith. & Unsup. & Partial & Claims/Expl. & Faith.+Repair & Retention & Sem. Pres. & Cost/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Gold-event versus extracted-event ablation on the 400-example synthetic test split. Extracted-event rows reuse the original event-grounded generations and recompute verification with the same parser used for the gold-event rows. Gold-event rows rerun event-grounded generation with synthetic gold events as the event list.}
\label{tab:event-source-ablation}
\end{table*}
""" % body


def main() -> None:
    rows = collect_rows()
    write_csv(rows)
    table = latex_table(rows)
    write_text(ROOT / "outputs" / "tables" / "event_source_ablation.tex", table)
    print("Wrote event_source_ablation.tex")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
