#!/usr/bin/env python3
"""Create same-generation gold-event robustness tables."""

from __future__ import annotations

import csv
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import write_text


EXTRACTED_SUMMARY = ROOT / "outputs" / "predictions" / "full_synthetic_400" / "summary.csv"
GOLD_SUMMARY = ROOT / "outputs" / "predictions" / "gold_eval_from_full_generations_400" / "summary.csv"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"

PROMPT_ORDER = ["direct", "cot", "structured", "event_grounded"]
PROMPT_DISPLAY = {
    "direct": "Direct",
    "cot": "CoT",
    "structured": "Structured",
    "event_grounded": "Event-grounded",
}
MODEL_DISPLAY = {
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gemini-3-flash-preview": "Gemini-3-Flash",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
}
MODEL_ORDER = [
    "gpt-4.1-mini",
    "gpt-5.4-mini",
    "deepseek-v4-flash",
    "gemini-3-flash-preview",
    "claude-sonnet-4-5-20250929",
]


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    rows: dict[tuple[str, str], dict[str, float | str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["prompt"])
            rows[key] = {
                "provider": row["provider"],
                "model": row["model"],
                "prompt": row["prompt"],
                "faithfulness": float(row["faithfulness_before"]),
                "unsupported": float(row["unsupported_before"]),
                "partial": float(row["partial_before"]),
                "claims_per_explanation": float(row["claims_per_explanation_before"]),
            }
    return rows


def prompt_rows(extracted: dict, gold: dict) -> list[dict]:
    rows = []
    for prompt in PROMPT_ORDER:
        extracted_rows = [row for (model, p), row in extracted.items() if p == prompt]
        gold_rows = [row for (model, p), row in gold.items() if p == prompt]
        extracted_faith = stats.mean(float(row["faithfulness"]) for row in extracted_rows)
        gold_faith = stats.mean(float(row["faithfulness"]) for row in gold_rows)
        extracted_unsup = stats.mean(float(row["unsupported"]) for row in extracted_rows)
        gold_unsup = stats.mean(float(row["unsupported"]) for row in gold_rows)
        rows.append(
            {
                "prompt": prompt,
                "extracted_faith": extracted_faith,
                "gold_faith": gold_faith,
                "delta_faith": gold_faith - extracted_faith,
                "extracted_unsup": extracted_unsup,
                "gold_unsup": gold_unsup,
                "delta_unsup": gold_unsup - extracted_unsup,
                "claims_per_explanation": stats.mean(
                    float(row["claims_per_explanation"]) for row in extracted_rows
                ),
            }
        )
    return rows


def model_prompt_rows(extracted: dict, gold: dict) -> list[dict]:
    rows = []
    for model in MODEL_ORDER:
        for prompt in PROMPT_ORDER:
            key = (model, prompt)
            if key not in extracted or key not in gold:
                continue
            extracted_row = extracted[key]
            gold_row = gold[key]
            rows.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "extracted_faith": float(extracted_row["faithfulness"]),
                    "gold_faith": float(gold_row["faithfulness"]),
                    "delta_faith": float(gold_row["faithfulness"]) - float(extracted_row["faithfulness"]),
                    "extracted_unsup": float(extracted_row["unsupported"]),
                    "gold_unsup": float(gold_row["unsupported"]),
                    "delta_unsup": float(gold_row["unsupported"]) - float(extracted_row["unsupported"]),
                    "claims_per_explanation": float(extracted_row["claims_per_explanation"]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def prompt_table(rows: list[dict]) -> str:
    body = "\n".join(
        (
            f"{PROMPT_DISPLAY[row['prompt']]} & {row['extracted_faith']:.3f} "
            f"& {row['gold_faith']:.3f} & {fmt_delta(row['delta_faith'])} "
            f"& {row['extracted_unsup']:.3f} & {row['gold_unsup']:.3f} "
            f"& {fmt_delta(row['delta_unsup'])} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{lrrrrrr}
\toprule
Prompt & Ext. Faith. & Gold Faith. & $\Delta$Faith. & Ext. Unsup. & Gold Unsup. & $\Delta$Unsup. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Evaluation-side gold-event robustness on the same 400-example synthetic generations. Ext. rows use the default extracted-event verifier; Gold rows recompute verification against synthetic gold-event graphs without any new LLM calls. Because synthetic gold events encode designed reference events rather than exhaustive local patterns, this table is a strict evidence-source diagnostic rather than the main benchmark score.}
\label{tab:gold-eval-robustness}
\end{table}
""" % body


def model_prompt_table(rows: list[dict]) -> str:
    body = "\n".join(
        (
            f"{MODEL_DISPLAY[row['model']]} & {PROMPT_DISPLAY[row['prompt']]} "
            f"& {row['extracted_faith']:.3f} & {row['gold_faith']:.3f} "
            f"& {fmt_delta(row['delta_faith'])} & {row['extracted_unsup']:.3f} "
            f"& {row['gold_unsup']:.3f} & {fmt_delta(row['delta_unsup'])} "
            f"& {row['claims_per_explanation']:.2f} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrrrrr}
\toprule
Model & Prompt & Ext. Faith. & Gold Faith. & $\Delta$Faith. & Ext. Unsup. & Gold Unsup. & $\Delta$Unsup. & Claims/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Full model-by-prompt evaluation-side gold-event robustness. The same generated explanations are scored twice, once with extracted-event evidence and once with synthetic gold-event evidence. Claims/Expl. is unchanged by construction because no explanations are regenerated.}
\label{tab:gold-eval-robustness-full}
\end{table*}
""" % body


def main() -> None:
    extracted = read_summary(EXTRACTED_SUMMARY)
    gold = read_summary(GOLD_SUMMARY)
    prompt_level = prompt_rows(extracted, gold)
    model_level = model_prompt_rows(extracted, gold)

    write_csv(ANALYSIS_DIR / "gold_eval_robustness_by_prompt.csv", prompt_level)
    write_csv(ANALYSIS_DIR / "gold_eval_robustness_by_model_prompt.csv", model_level)
    write_text(ROOT / "outputs" / "tables" / "gold_eval_robustness.tex", prompt_table(prompt_level))
    write_text(
        ROOT / "outputs" / "tables" / "gold_eval_robustness_full.tex",
        model_prompt_table(model_level),
    )

    print("Prompt-level robustness:")
    for row in prompt_level:
        print(
            row["prompt"],
            f"faith {row['extracted_faith']:.3f}->{row['gold_faith']:.3f}",
            f"unsup {row['extracted_unsup']:.3f}->{row['gold_unsup']:.3f}",
        )


if __name__ == "__main__":
    main()
