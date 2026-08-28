#!/usr/bin/env python3
"""Create a balanced synthetic subset by pattern type."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import read_jsonl, write_jsonl


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/test.jsonl")
    parser.add_argument("--output", default="data/synthetic/test_stratified_80.jsonl")
    parser.add_argument("--n-per-pattern", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_jsonl(_resolve(args.input))
    groups = defaultdict(list)
    for record in records:
        pattern = tuple(record.get("metadata", {}).get("pattern_types", ["unknown"]))
        groups[pattern].append(record)

    rng = random.Random(args.seed)
    subset = []
    for pattern in sorted(groups):
        items = list(groups[pattern])
        rng.shuffle(items)
        subset.extend(items[: args.n_per_pattern])
    subset.sort(key=lambda record: record["id"])
    output_path = _resolve(args.output)
    write_jsonl(output_path, subset)
    print(f"Wrote {len(subset)} records to {output_path.relative_to(ROOT)}")
    print({"/".join(pattern): min(args.n_per_pattern, len(items)) for pattern, items in groups.items()})


if __name__ == "__main__":
    main()
