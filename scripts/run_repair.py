#!/usr/bin/env python3
"""Repair ungrounded explanations using verifier feedback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.claims.claim_extractor import extract_claims
from tsr.claims.claim_schema import Claim, VerificationResult
from tsr.events.event_graph import EventGraph
from tsr.utils.io import read_jsonl, write_jsonl
from tsr.verification.repair import repair_explanation
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
    parser.add_argument("--output", default="outputs/predictions/repaired.jsonl")
    args = parser.parse_args()

    records = read_jsonl(_resolve(args.input))
    graphs = _load_graphs(_resolve(args.graphs))
    config = VerifierConfig.from_yaml(_resolve(args.config))
    outputs = []
    for record in records:
        graph = graphs[record["id"]]
        claims = [Claim(**claim) for claim in record.get("claims", [])]
        results = [VerificationResult(**result) for result in record.get("verification_results", [])]
        repaired = repair_explanation(record["explanation"], claims, results, graph, mode="rule")
        repaired_claims = extract_claims(repaired, mode="rule")
        repaired_results = verify_claims(repaired_claims, graph, config)
        outputs.append(
            {
                **record,
                "repaired_explanation": repaired,
                "repaired_claims": [claim.model_dump() for claim in repaired_claims],
                "repaired_verification_results": [result.model_dump() for result in repaired_results],
            }
        )
    output_path = _resolve(args.output)
    write_jsonl(output_path, outputs)
    print(f"Wrote repaired explanations for {len(outputs)} records to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
