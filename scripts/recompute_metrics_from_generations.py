#!/usr/bin/env python3
"""Recompute verification, repair, metrics, and summary CSV from saved generations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.claims.claim_extractor import extract_claims
from tsr.claims.claim_schema import Claim, VerificationResult
from tsr.events.event_graph import EventGraph
from tsr.utils.io import read_jsonl, write_json, write_jsonl
from tsr.verification.metrics import (
    claim_retention_rate,
    claims_per_explanation,
    summarize_generation_cost,
    summarize_verification,
    supported_claim_preservation,
)
from tsr.verification.repair import repair_explanation
from tsr.verification.verifier import VerifierConfig, verify_claims


SUMMARY_FIELDS = [
    "provider",
    "model",
    "prompt",
    "n_records",
    "faithfulness_before",
    "unsupported_before",
    "partial_before",
    "chronology_before",
    "claims_per_explanation_before",
    "faithfulness_after",
    "unsupported_after",
    "partial_after",
    "claims_per_explanation_after",
    "claim_retention_rate",
    "semantic_preservation",
    "generation_cost_usd",
    "cost_per_explanation_usd",
    "tokens_per_explanation",
]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_graphs(path: Path) -> dict[str, EventGraph]:
    return {item["series_id"]: EventGraph.from_json(item) for item in read_jsonl(path)}


def _flatten(records: list[dict], field: str) -> list[dict]:
    results = []
    for record in records:
        results.extend(record.get(field, []))
    return results


def _parse_run_name(path: Path) -> tuple[str, str, str]:
    stem = path.name[: -len("_generation.jsonl")]
    provider, model, prompt = stem.split("__")
    return provider, model.replace("_", "."), prompt


def _metrics(records: list[dict], input_path: Path) -> dict:
    before = summarize_verification(_flatten(records, "verification_results"))
    after = summarize_verification(_flatten(records, "repaired_verification_results"))
    before["claims_per_explanation"] = claims_per_explanation(int(before["n_claims"]), len(records))
    after["claims_per_explanation"] = claims_per_explanation(int(after["n_claims"]), len(records))
    return {
        "input": str(input_path.relative_to(ROOT)),
        "n_records": len(records),
        "before_repair": before,
        "after_repair": after,
        "claim_retention_rate": claim_retention_rate(int(before["n_claims"]), int(after["n_claims"])),
        "supported_claim_preservation": supported_claim_preservation(records),
        "cost": summarize_generation_cost(records),
    }


def _summary_row(path: Path, metrics: dict) -> dict:
    provider, model, prompt = _parse_run_name(path)
    before = metrics["before_repair"]
    after = metrics["after_repair"]
    cost = metrics["cost"]
    return {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "n_records": metrics["n_records"],
        "faithfulness_before": before["claim_faithfulness"],
        "unsupported_before": before["unsupported_claim_rate"],
        "partial_before": before["partial_support_rate"],
        "chronology_before": before["chronology_violation_rate"],
        "claims_per_explanation_before": before["claims_per_explanation"],
        "faithfulness_after": after["claim_faithfulness"],
        "unsupported_after": after["unsupported_claim_rate"],
        "partial_after": after["partial_support_rate"],
        "claims_per_explanation_after": after["claims_per_explanation"],
        "claim_retention_rate": metrics["claim_retention_rate"],
        "semantic_preservation": metrics["supported_claim_preservation"],
        "generation_cost_usd": cost["generation_cost_usd"],
        "cost_per_explanation_usd": cost["cost_per_explanation_usd"],
        "tokens_per_explanation": cost["tokens_per_explanation"],
    }


def recompute_run(
    generation_path: Path,
    graphs: dict[str, EventGraph],
    config: VerifierConfig,
    claim_mode: str,
) -> tuple[Path, Path, Path, dict]:
    records = read_jsonl(generation_path)
    verified_records = []
    for record in records:
        graph = graphs[record["id"]]
        claims = extract_claims(record["explanation"], mode=claim_mode)
        results = verify_claims(claims, graph, config)
        verified_records.append(
            {
                **record,
                "claims": [claim.model_dump() for claim in claims],
                "verification_results": [result.model_dump() for result in results],
            }
        )

    verified_path = generation_path.with_name(generation_path.name.replace("_generation.jsonl", "_verification.jsonl"))
    write_jsonl(verified_path, verified_records)

    repaired_records = []
    for record in verified_records:
        graph = graphs[record["id"]]
        claims = [Claim(**claim) for claim in record.get("claims", [])]
        results = [VerificationResult(**result) for result in record.get("verification_results", [])]
        repaired = repair_explanation(record["explanation"], claims, results, graph, mode="rule")
        repaired_claims = extract_claims(repaired, mode=claim_mode)
        repaired_results = verify_claims(repaired_claims, graph, config)
        repaired_records.append(
            {
                **record,
                "repaired_explanation": repaired,
                "repaired_claims": [claim.model_dump() for claim in repaired_claims],
                "repaired_verification_results": [result.model_dump() for result in repaired_results],
            }
        )

    repaired_path = generation_path.with_name(generation_path.name.replace("_generation.jsonl", "_repaired.jsonl"))
    write_jsonl(repaired_path, repaired_records)

    run_metrics = _metrics(repaired_records, repaired_path)
    metrics_path = generation_path.with_name(generation_path.name.replace("_generation.jsonl", "_metrics.json"))
    write_json(metrics_path, run_metrics)
    return verified_path, repaired_path, metrics_path, run_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/predictions/full_synthetic_400")
    parser.add_argument("--graphs", default="data/processed/test_graphs.jsonl")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--claim-mode", default="rule", choices=["rule", "llm"])
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    graphs = _load_graphs(_resolve(args.graphs))
    config = VerifierConfig.from_yaml(_resolve(args.config))
    generation_paths = sorted(output_dir.glob("*_generation.jsonl"))
    if not generation_paths:
        raise SystemExit(f"No generation files found under {output_dir}")

    rows = []
    for generation_path in generation_paths:
        verified_path, repaired_path, metrics_path, run_metrics = recompute_run(
            generation_path,
            graphs,
            config,
            args.claim_mode,
        )
        rows.append(_summary_row(generation_path, run_metrics))
        print(
            f"Recomputed {generation_path.name}: "
            f"{verified_path.name}, {repaired_path.name}, {metrics_path.name}"
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
