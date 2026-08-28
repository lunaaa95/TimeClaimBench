#!/usr/bin/env python3
"""Run explanation generation with cached LLM calls or dummy provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.events.event_graph import EventGraph, build_event_graph
from tsr.prompting.costs import load_model_prices
from tsr.prompting.llm_client import cached_llm_call
from tsr.prompting.prompts import PROMPTS
from tsr.utils.io import read_jsonl, read_yaml, write_jsonl


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_graphs(path: Path) -> dict:
    if not path.exists():
        return {}
    return {item["series_id"]: EventGraph.from_json(item) for item in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--provider", default="dummy")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="event_grounded", choices=list(PROMPTS.keys()))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = read_yaml(_resolve(args.config))
    input_path = _resolve(config.get("input_path", "data/synthetic/test.jsonl"))
    graph_path = _resolve(config.get("graph_path", "data/processed/test_graphs.jsonl"))
    output_dir = _resolve(config.get("output_dir", "outputs/predictions"))
    cache_dir = str(_resolve(config.get("cache_dir", "outputs/cache")))
    cost_config = _resolve(config.get("cost_config", "configs/model_costs.yaml"))
    prices = load_model_prices(cost_config) if cost_config.exists() else {}
    output_path = _resolve(args.output) if args.output else output_dir / "generation.jsonl"
    model = args.model or ("local-dummy" if args.provider == "dummy" else "gpt-4o-mini")
    max_retries = int(config.get("max_retries", 2))

    samples = read_jsonl(input_path)
    graphs = _load_graphs(graph_path)
    outputs = read_jsonl(output_path) if output_path.exists() else []
    done_ids = {
        record["id"]
        for record in outputs
        if record.get("prompt") == args.prompt
        and record.get("provider") == args.provider
        and record.get("model") == model
    }
    template = PROMPTS[args.prompt]
    for sample in samples:
        if sample["id"] in done_ids:
            continue
        graph = graphs.get(sample["id"])
        if graph is None:
            graph = build_event_graph(sample["id"], series=sample["series"])
        events_json = json.dumps([event.model_dump() for event in graph.events], ensure_ascii=False)
        prompt = template.format(
            series=json.dumps(sample["series"]),
            events=events_json,
            question=sample.get("question", "Describe the major temporal patterns in the series."),
        )
        payload = cached_llm_call(
            provider=args.provider,
            model=model,
            prompt=prompt,
            cache_dir=cache_dir,
            temperature=0.0,
            prices=prices,
            return_metadata=True,
            max_retries=max_retries,
        )
        explanation = payload["response"]
        outputs.append(
            {
                "id": sample["id"],
                "prompt": args.prompt,
                "provider": args.provider,
                "model": model,
                "explanation": explanation,
                "usage": payload.get("usage", {}),
                "estimated_cost_usd": payload.get("usage", {}).get("estimated_cost_usd", 0.0),
                "reference_claims": sample.get("reference_claims", []),
            }
        )
        write_jsonl(output_path, outputs)
    write_jsonl(output_path, outputs)
    total_cost = sum(float(record.get("estimated_cost_usd", 0.0)) for record in outputs)
    print(f"Wrote {len(outputs)} generations to {output_path.relative_to(ROOT)}")
    print(f"Estimated generation cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
