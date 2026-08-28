#!/usr/bin/env python3
"""Verify generated explanations against event graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.claims.claim_extractor import extract_claims
from tsr.events.event_graph import EventGraph
from tsr.utils.io import read_jsonl, write_jsonl
from tsr.verification.verifier import VerifierConfig, verify_claims


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_graphs(path: Path) -> dict:
    return {item["series_id"]: EventGraph.from_json(item) for item in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--output", default="outputs/predictions/verification.jsonl")
    parser.add_argument("--claim-mode", default="rule", choices=["rule", "llm"])
    args = parser.parse_args()

    records = read_jsonl(_resolve(args.input))
    graphs = _load_graphs(_resolve(args.graphs))
    config = VerifierConfig.from_yaml(_resolve(args.config))
    outputs = []
    for record in records:
        graph = graphs[record["id"]]
        claims = extract_claims(record["explanation"], mode=args.claim_mode)
        results = verify_claims(claims, graph, config)
        outputs.append(
            {
                **record,
                "claims": [claim.model_dump() for claim in claims],
                "verification_results": [result.model_dump() for result in results],
            }
        )
    output_path = _resolve(args.output)
    write_jsonl(output_path, outputs)
    print(f"Wrote verification results for {len(outputs)} records to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
