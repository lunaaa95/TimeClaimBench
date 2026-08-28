"""Build event graphs whose event lists come from a different series.

The output keeps each target series_id unchanged, but replaces its events and
relations with those from another sample. This supports a diagnostic ablation:
LLMs receive plausible but wrong event evidence, and their explanations are then
verified against the correct event graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.events.event_graph import EventGraph
from tsr.utils.io import read_jsonl, write_jsonl


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/synthetic/test_stratified_80.jsonl")
    parser.add_argument("--graphs", default="data/processed/test_graphs.jsonl")
    parser.add_argument("--output", default="data/processed/test_stratified_80_mismatched_graphs.jsonl")
    parser.add_argument("--offset", type=int, default=17)
    args = parser.parse_args()

    samples = read_jsonl(_resolve(args.samples))
    sample_ids = [sample["id"] for sample in samples]
    graphs = {row["series_id"]: EventGraph.from_json(row) for row in read_jsonl(_resolve(args.graphs))}
    if not sample_ids:
        raise SystemExit("No samples found.")

    output = []
    for index, series_id in enumerate(sample_ids):
        donor_id = sample_ids[(index + args.offset) % len(sample_ids)]
        donor = graphs[donor_id].to_json()
        donor["series_id"] = series_id
        donor["mismatched_source_series_id"] = donor_id
        output.append(donor)

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output)
    print(
        json.dumps(
            {
                "samples": len(sample_ids),
                "offset": args.offset,
                "output": str(output_path.relative_to(ROOT)),
                "first_mapping": {"target": sample_ids[0], "donor": sample_ids[args.offset % len(sample_ids)]},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
