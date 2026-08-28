"""Create the mismatched event-list ablation table."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


MODEL_DISPLAY = {
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gpt-4.1-mini": "GPT-4.1-mini",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_row(row: dict[str, str], condition: str) -> dict[str, float | str]:
    if "faithfulness_before" in row:
        return {
            "provider": row["provider"],
            "model": row["model"],
            "condition": condition,
            "faith": float(row["faithfulness_before"]),
            "unsupported": float(row["unsupported_before"]),
            "partial": float(row["partial_before"]),
            "claims": float(row["claims_per_explanation_before"]),
            "retention": float(row["claim_retention_rate"]),
            "cost": float(row["cost_per_explanation_usd"]),
        }
    return {
        "provider": row["provider"],
        "model": row["model"],
        "condition": condition,
        "faith": float(row["faith_before"]),
        "unsupported": float(row["unsup_before"]),
        "partial": float(row["partial_before"]),
        "claims": float(row["claims_before"]),
        "retention": float(row["retention"]),
        "cost": float(row["cost_per_expl"]),
    }


def collect() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    base_sources = [
        ROOT / "outputs/predictions/requested_80/summary.csv",
        ROOT / "outputs/predictions/multimodel_80/summary.csv",
    ]
    wanted = {"deepseek-v4-flash", "gpt-4.1-mini"}
    seen = set()
    for path in base_sources:
        for row in read_rows(path):
            if row["prompt"] != "event_grounded" or row["model"] not in wanted or row["model"] in seen:
                continue
            rows.append(normalize_row(row, "Matched events"))
            seen.add(row["model"])
    for row in read_rows(ROOT / "outputs/predictions/mismatched_events_80/summary.csv"):
        rows.append(normalize_row(row, "Mismatched events"))
    return sorted(rows, key=lambda row: (str(row["model"]), str(row["condition"])))


def make_table(rows: list[dict[str, float | str]]) -> str:
    def fmt(value: float) -> str:
        return f"{value:.3f}"

    lines = []
    matched_by_model = {row["model"]: row for row in rows if row["condition"] == "Matched events"}
    for row in rows:
        model = str(row["model"])
        matched = matched_by_model.get(model)
        delta = float(row["faith"]) - float(matched["faith"]) if matched else 0.0
        lines.append(
            f"{MODEL_DISPLAY.get(model, model)} & {row['condition']} & {fmt(float(row['faith']))} & "
            f"{delta:+.3f} & {fmt(float(row['unsupported']))} & {fmt(float(row['partial']))} & "
            f"{float(row['claims']):.2f} & {fmt(float(row['retention']))} & \\${float(row['cost']):.5f} \\\\"
        )

    return rf"""\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrr}}
\toprule
Model & Event evidence & Faith. $\uparrow$ & $\Delta$Faith. & Unsup. $\downarrow$ & Partial & Claims/Expl. & Retention & Cost/Expl. \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
}}
\caption{{Mismatched event-list ablation on the 80-example stratified synthetic subset. Matched events use the correct extracted event list in the event-grounded prompt. Mismatched events keep the same raw series but supply events from another series, then verification is performed against the correct event graph. The large faithfulness drop shows that event-grounded prompting is sensitive to the supplied evidence rather than simply benefiting from prompt formatting.}}
\label{{tab:mismatched-events}}
\end{{table*}}
"""


def main() -> None:
    rows = collect()
    analysis_dir = ROOT / "outputs/analysis"
    table_dir = ROOT / "outputs/tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "mismatched_event_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["provider", "model", "condition", "faith", "unsupported", "partial", "claims", "retention", "cost"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (table_dir / "mismatched_event_ablation.tex").write_text(make_table(rows), encoding="utf-8")
    print(f"Wrote {analysis_dir / 'mismatched_event_ablation.csv'}")
    print(f"Wrote {table_dir / 'mismatched_event_ablation.tex'}")


if __name__ == "__main__":
    main()
