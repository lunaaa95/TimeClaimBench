"""Coverage and informativeness diagnostics for generated explanations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from tsr.claims.claim_schema import Claim
from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph
from tsr.verification.metrics import interval_iou
from tsr.verification.verifier import candidate_event_types


FULL_DIR = Path("outputs/predictions/full_synthetic_400")
VERBALIZER_DIR = Path("outputs/predictions/event_verbalizer")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_graphs(path: Path) -> dict[str, EventGraph]:
    return {row["series_id"]: EventGraph.from_json(row) for row in read_jsonl(path)}


def load_gold_events(path: Path) -> dict[str, list[Event]]:
    return {row["id"]: [Event(**event) for event in row.get("events", [])] for row in read_jsonl(path)}


def top_events(events: Iterable[Event], k: int) -> list[Event]:
    return sorted(events, key=lambda event: (-float(event.score or 0.0), event.start, event.end, event.type))[:k]


def event_ids(results: list[dict], labels: set[str]) -> set[str]:
    return {
        event_id
        for result in results
        if result.get("label") in labels
        for event_id in result.get("support_events", [])
    }


def compatible_with_gold(claim: Claim, gold_event: Event) -> bool:
    return gold_event.type in set(candidate_event_types(claim))


def gold_event_covered(record: dict, gold_event: Event, min_iou: float, labels: set[str]) -> bool:
    claims = {claim["claim_id"]: Claim(**claim) for claim in record.get("claims", [])}
    for result in record.get("verification_results", []):
        if result.get("label") not in labels:
            continue
        claim = claims.get(result.get("claim_id"))
        if not claim or not compatible_with_gold(claim, gold_event):
            continue
        interval = result.get("support_interval") or claim.support_interval
        if interval is None:
            return True
        if interval_iou(tuple(interval), (gold_event.start, gold_event.end)) >= min_iou:
            return True
    return False


def record_metrics(record: dict, graph: EventGraph, gold_events: list[Event], top_k: int, gold_iou: float) -> dict:
    results = record.get("verification_results", [])
    labels = [result.get("label") for result in results]
    supported_events = event_ids(results, {"supported"})
    grounded_events = event_ids(results, {"supported", "partial"})
    top = top_events(graph.events, top_k)
    top_ids = {event.event_id for event in top}

    supported_top = len(top_ids & supported_events) / len(top_ids) if top_ids else 0.0
    grounded_top = len(top_ids & grounded_events) / len(top_ids) if top_ids else 0.0
    all_ids = {event.event_id for event in graph.events}
    grounded_all = len(all_ids & grounded_events) / len(all_ids) if all_ids else 0.0
    gold_supported = sum(1 for event in gold_events if gold_event_covered(record, event, gold_iou, {"supported"}))
    gold_grounded = sum(
        1 for event in gold_events if gold_event_covered(record, event, gold_iou, {"supported", "partial"})
    )
    gold_supported_coverage = gold_supported / len(gold_events) if gold_events else 0.0
    gold_grounded_coverage = gold_grounded / len(gold_events) if gold_events else 0.0
    supported = sum(1 for label in labels if label == "supported")
    partial = sum(1 for label in labels if label == "partial")
    unsupported = sum(1 for label in labels if label == "unsupported")

    return {
        "n_records": 1,
        "n_claims": len(labels),
        "supported_claims": supported,
        "partial_claims": partial,
        "unsupported_claims": unsupported,
        "top_supported_coverage": supported_top,
        "top_grounded_coverage": grounded_top,
        "all_grounded_coverage": grounded_all,
        "gold_supported_coverage": gold_supported_coverage,
        "gold_grounded_coverage": gold_grounded_coverage,
        "gold_events": len(gold_events),
        "gold_events_supported": gold_supported,
        "gold_events_grounded": gold_grounded,
    }


def parse_run_name(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix("_verification.jsonl")
    parts = stem.split("__")
    if len(parts) != 3:
        raise ValueError(f"Unexpected verification filename: {path.name}")
    return parts[0], parts[1], parts[2]


def collect_records(root: Path) -> list[tuple[str, str, str, Path]]:
    rows = []
    for path in sorted((root / FULL_DIR).glob("*_verification.jsonl")):
        rows.append((*parse_run_name(path), path))
    verbalizer = root / VERBALIZER_DIR / "deterministic__event-verbalizer__event_grounded_verification.jsonl"
    if verbalizer.exists():
        rows.append(("deterministic", "event-verbalizer", "event_verbalizer", verbalizer))
    return rows


def display_prompt(prompt: str) -> str:
    return {
        "direct": "Direct",
        "cot": "CoT",
        "structured": "Structured",
        "event_grounded": "Event-grounded",
        "event_verbalizer": "Event verbalizer",
    }.get(prompt, prompt)


def aggregate(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    grouped: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        acc = grouped[key]
        for col in group_cols:
            acc[col] = row[col]
        for metric in [
            "n_records",
            "n_claims",
            "supported_claims",
            "partial_claims",
            "unsupported_claims",
            "gold_events",
            "gold_events_supported",
            "gold_events_grounded",
        ]:
            acc[metric] += row[metric]
        for metric in [
            "top_supported_coverage",
            "top_grounded_coverage",
            "all_grounded_coverage",
            "gold_supported_coverage",
            "gold_grounded_coverage",
        ]:
            acc[f"{metric}_sum"] += row[metric]

    out = []
    for acc in grouped.values():
        n_records = acc["n_records"]
        n_claims = acc["n_claims"]
        row = {col: acc[col] for col in group_cols}
        row.update(
            {
                "n_records": int(n_records),
                "n_claims": int(n_claims),
                "claim_faithfulness": acc["supported_claims"] / n_claims if n_claims else 0.0,
                "unsupported_rate": acc["unsupported_claims"] / n_claims if n_claims else 0.0,
                "partial_rate": acc["partial_claims"] / n_claims if n_claims else 0.0,
                "claims_per_explanation": n_claims / n_records if n_records else 0.0,
                "top_supported_coverage": acc["top_supported_coverage_sum"] / n_records if n_records else 0.0,
                "top_grounded_coverage": acc["top_grounded_coverage_sum"] / n_records if n_records else 0.0,
                "all_grounded_coverage": acc["all_grounded_coverage_sum"] / n_records if n_records else 0.0,
                "gold_supported_coverage": acc["gold_supported_coverage_sum"] / n_records if n_records else 0.0,
                "gold_grounded_coverage": acc["gold_grounded_coverage_sum"] / n_records if n_records else 0.0,
                "micro_gold_supported_coverage": acc["gold_events_supported"] / acc["gold_events"]
                if acc["gold_events"]
                else 0.0,
                "micro_gold_grounded_coverage": acc["gold_events_grounded"] / acc["gold_events"]
                if acc["gold_events"]
                else 0.0,
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def make_prompt_table(prompt_df: pd.DataFrame) -> str:
    order = ["Direct", "CoT", "Structured", "Event-grounded", "Event verbalizer"]
    prompt_df = prompt_df.copy()
    prompt_df["order"] = prompt_df["prompt_display"].map({name: index for index, name in enumerate(order)})
    prompt_df = prompt_df.sort_values("order")

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    rows = []
    for _, row in prompt_df.iterrows():
        rows.append(
            f"{row['prompt_display']} & {int(row['n_records'])} & {fmt(row['claim_faithfulness'])} & "
            f"{fmt(row['unsupported_rate'])} & {row['claims_per_explanation']:.2f} & "
            f"{fmt(row['top_supported_coverage'])} & {fmt(row['top_grounded_coverage'])} & "
            f"{fmt(row['gold_supported_coverage'])} & {fmt(row['gold_grounded_coverage'])} \\\\"
        )

    return rf"""\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Prompt & Runs & Faith. $\uparrow$ & Unsup. $\downarrow$ & Claims/Expl. & Top-5 Sup. Cov. $\uparrow$ & Top-5 Grounded Cov. $\uparrow$ & Gold Sup. Cov. $\uparrow$ & Gold Grounded Cov. $\uparrow$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
}}
\caption{{Coverage and informativeness diagnostics on the 400-example synthetic test set. Top-5 coverage is the mean fraction of the five highest-scoring extracted events that are mentioned by supported claims, or by supported-or-partial claims for grounded coverage. Gold coverage is the mean fraction of synthetic gold events covered by a compatible claim with interval IoU at least 0.3. The deterministic event verbalizer is included as a no-API upper reference for event coverage.}}
\label{{tab:coverage-informativeness}}
\end{{table*}}
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    graphs = load_graphs(root / "data/processed/test_graphs.jsonl")
    gold_events = load_gold_events(root / "data/synthetic/test.jsonl")
    output_dir = root / "outputs/analysis"
    table_dir = root / "outputs/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for provider, model, prompt, path in collect_records(root):
        for record in read_jsonl(path):
            metrics = record_metrics(record, graphs[record["id"]], gold_events[record["id"]], top_k=5, gold_iou=0.3)
            metrics.update(
                {
                    "provider": provider,
                    "model": model,
                    "prompt": prompt,
                    "prompt_display": display_prompt(prompt),
                    "series_id": record["id"],
                }
            )
            rows.append(metrics)

    record_df = pd.DataFrame(rows)
    by_prompt = aggregate(rows, ["prompt", "prompt_display"])
    by_model_prompt = aggregate(rows, ["provider", "model", "prompt", "prompt_display"])

    record_df.to_csv(output_dir / "coverage_informativeness_records.csv", index=False)
    by_prompt.to_csv(output_dir / "coverage_informativeness_by_prompt.csv", index=False)
    by_model_prompt.to_csv(output_dir / "coverage_informativeness_by_model_prompt.csv", index=False)
    (table_dir / "coverage_informativeness.tex").write_text(make_prompt_table(by_prompt), encoding="utf-8")

    print(by_prompt.sort_values("prompt").to_string(index=False))
    print(f"Wrote {output_dir / 'coverage_informativeness_by_prompt.csv'}")
    print(f"Wrote {table_dir / 'coverage_informativeness.tex'}")


if __name__ == "__main__":
    main()
