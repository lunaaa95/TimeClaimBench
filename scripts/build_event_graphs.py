#!/usr/bin/env python3
"""Build event graph JSONL files from time-series samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.data.dataset_schema import Event
from tsr.events.event_graph import build_event_graph
from tsr.utils.io import read_jsonl, write_jsonl


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prefer-gold-events", action="store_true")
    args = parser.parse_args()

    input_path = _resolve(args.input)
    output_path = _resolve(args.output) if args.output else ROOT / "data" / "processed" / f"{input_path.stem}_graphs.jsonl"
    records = read_jsonl(input_path)
    graphs = []
    for record in records:
        events = [Event(**event) for event in record.get("events", [])] if args.prefer_gold_events else None
        graph = build_event_graph(
            series_id=record["id"],
            series=record.get("series"),
            events=events,
            prefer_gold_events=args.prefer_gold_events,
        )
        graphs.append(graph.to_json())
    write_jsonl(output_path, graphs)
    print(f"Wrote {len(graphs)} event graphs to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
