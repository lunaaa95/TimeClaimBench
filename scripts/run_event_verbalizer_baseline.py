#!/usr/bin/env python3
"""Run a deterministic event-to-text verbalizer baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.claims.claim_extractor import extract_claims
from tsr.claims.claim_schema import Claim, VerificationResult
from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph
from tsr.utils.io import read_jsonl, write_json, write_jsonl
from tsr.verification.metrics import claims_per_explanation, summarize_verification
from tsr.verification.verifier import VerifierConfig, verify_claims


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_graphs(path: Path) -> dict[str, EventGraph]:
    return {item["series_id"]: EventGraph.from_json(item) for item in read_jsonl(path)}


def _event_priority(event: Event) -> tuple[int, float, int, int]:
    priority = {
        "anomaly": 0,
        "change_point": 1,
        "sharp_rise": 2,
        "sharp_drop": 2,
        "volatility_shift": 3,
        "upward_trend": 4,
        "downward_trend": 4,
        "seasonality": 5,
        "local_peak": 6,
        "local_trough": 6,
    }
    return (priority.get(event.type, 99), -float(event.score), int(event.start), int(event.end))


def _select_events(events: Iterable[Event], max_events: int) -> list[Event]:
    selected = sorted(events, key=_event_priority)[:max_events]
    return sorted(selected, key=lambda event: (event.start, event.end, event.type))


def _event_sentence(event: Event) -> str:
    attrs = event.attributes
    interval = f"from index {event.start} to {event.end}" if event.start != event.end else f"at index {event.start}"
    strength = str(attrs.get("strength", "")).strip()
    strength_prefix = f"{strength} " if strength else ""
    citation = f"[{event.event_id}]"
    if event.type == "upward_trend":
        return f"- {citation} The series shows a {strength_prefix}upward trend {interval} with slope {_fmt(attrs.get('slope'))}."
    if event.type == "downward_trend":
        return f"- {citation} The series shows a {strength_prefix}downward trend {interval} with slope {_fmt(attrs.get('slope'))}."
    if event.type == "sharp_rise":
        return f"- {citation} The series has a {strength_prefix}sharp rise {interval} with magnitude {_fmt(attrs.get('magnitude'))}."
    if event.type == "sharp_drop":
        return f"- {citation} The series has a {strength_prefix}sharp drop {interval} with magnitude {_fmt(attrs.get('magnitude'))}."
    if event.type == "volatility_shift":
        direction = attrs.get("direction", "change")
        return (
            f"- {citation} The series has a {strength_prefix}volatility shift {interval}, "
            f"with volatility {direction} and ratio {_fmt(attrs.get('ratio'))}."
        )
    if event.type == "change_point":
        return (
            f"- {citation} The series has a {strength_prefix}change point {interval}, "
            f"with mean changing from {_fmt(attrs.get('before_mean'))} to {_fmt(attrs.get('after_mean'))}."
        )
    if event.type == "seasonality":
        return f"- {citation} The series shows {strength_prefix}seasonality {interval} with period {attrs.get('period')}."
    if event.type == "anomaly":
        direction = attrs.get("direction", "")
        direction_phrase = f" {direction}" if direction else ""
        return f"- {citation} The series contains a {strength_prefix}{direction_phrase} anomaly {interval} with z-score {_fmt(attrs.get('z_score'))}."
    if event.type == "local_peak":
        return f"- {citation} The series has a {strength_prefix}local peak {interval} with value {_fmt(attrs.get('value'))}."
    if event.type == "local_trough":
        return f"- {citation} The series has a {strength_prefix}local trough {interval} with value {_fmt(attrs.get('value'))}."
    return f"- {citation} The series has a {strength_prefix}{event.type.replace('_', ' ')} {interval}."


def verbalize(graph: EventGraph, max_events: int) -> str:
    selected = _select_events(graph.events, max_events=max_events)
    if not selected:
        return "Explanation:\n- No temporal event is detected by the event extractor."
    lines = ["Explanation:"]
    lines.extend(_event_sentence(event) for event in selected)
    return "\n".join(lines)


def _metrics(records: list[dict], input_path: Path) -> dict:
    before = summarize_verification(result for record in records for result in record.get("verification_results", []))
    before["claims_per_explanation"] = claims_per_explanation(int(before["n_claims"]), len(records))
    return {
        "input": str(input_path.relative_to(ROOT)),
        "n_records": len(records),
        "before_repair": before,
        "after_repair": before,
        "claim_retention_rate": 1.0,
        "supported_claim_preservation": 1.0,
        "cost": {
            "generation_cost_usd": 0.0,
            "cost_per_explanation_usd": 0.0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "tokens_per_explanation": 0.0,
        },
    }


def run_baseline(
    samples_path: Path,
    graphs_path: Path,
    output_prefix: Path,
    config: VerifierConfig,
    max_events: int,
) -> tuple[Path, Path, Path]:
    samples = read_jsonl(samples_path)
    graphs = _load_graphs(graphs_path)
    generation_records = []
    verification_records = []
    for sample in samples:
        graph = graphs[sample["id"]]
        explanation = verbalize(graph, max_events=max_events)
        claims = extract_claims(explanation, mode="rule")
        results = verify_claims(claims, graph, config)
        generation_record = {
            "id": sample["id"],
            "prompt": "event_grounded",
            "provider": "deterministic",
            "model": "event-verbalizer",
            "explanation": explanation,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_tokens": False,
            },
            "estimated_cost_usd": 0.0,
            "reference_claims": sample.get("reference_claims", []),
        }
        generation_records.append(generation_record)
        verification_records.append(
            {
                **generation_record,
                "claims": [claim.model_dump() for claim in claims],
                "verification_results": [result.model_dump() for result in results],
            }
        )

    generation_path = output_prefix.with_name(output_prefix.name + "_generation.jsonl")
    verification_path = output_prefix.with_name(output_prefix.name + "_verification.jsonl")
    metrics_path = output_prefix.with_name(output_prefix.name + "_metrics.json")
    write_jsonl(generation_path, generation_records)
    write_jsonl(verification_path, verification_records)
    write_json(metrics_path, _metrics(verification_records, generation_path))
    return generation_path, verification_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/synthetic/test.jsonl")
    parser.add_argument("--graphs", default="data/processed/test_graphs.jsonl")
    parser.add_argument("--output-prefix", default="outputs/predictions/event_verbalizer/deterministic__event-verbalizer__event_grounded")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--max-events", type=int, default=12)
    args = parser.parse_args()

    config = VerifierConfig.from_yaml(_resolve(args.config))
    generation_path, verification_path, metrics_path = run_baseline(
        samples_path=_resolve(args.samples),
        graphs_path=_resolve(args.graphs),
        output_prefix=_resolve(args.output_prefix),
        config=config,
        max_events=args.max_events,
    )
    print(f"Generation: {generation_path.relative_to(ROOT)}")
    print(f"Verification: {verification_path.relative_to(ROOT)}")
    print(f"Metrics: {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
