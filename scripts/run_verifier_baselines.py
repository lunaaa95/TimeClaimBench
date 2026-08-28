#!/usr/bin/env python3
"""Run deterministic verifier baselines on synthetic claim-level labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.claims.claim_schema import Claim, VerificationResult
from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph, build_event_graph
from tsr.utils.io import read_jsonl, write_json, write_jsonl, write_text
from tsr.verification.metrics import classification_report, interval_iou
from tsr.verification.verifier import (
    VerifierConfig,
    event_direction,
    retrieve_candidate_events,
    verify_claim,
)


LABELS = ["supported", "partial", "unsupported"]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_graphs(path: Path) -> dict[str, EventGraph]:
    return {item["series_id"]: EventGraph.from_json(item) for item in read_jsonl(path)}


def _wrong_interval(start: int, end: int, length: int) -> tuple[int, int]:
    width = max(2, end - start + 1)
    if end < length // 2:
        new_start = min(length - 1, max(end + 5, length - width))
    else:
        new_start = 0
    new_end = min(length - 1, new_start + width - 1)
    return (new_start, new_end)


def _opposite_type(event_type: str) -> str:
    opposites = {
        "upward_trend": "downward_trend",
        "downward_trend": "upward_trend",
        "sharp_rise": "sharp_drop",
        "sharp_drop": "sharp_rise",
        "volatility_shift": "seasonality",
        "change_point": "seasonality",
        "seasonality": "change_point",
        "anomaly": "seasonality",
    }
    return opposites.get(event_type, "anomaly")


def _direction_for_type(event_type: str, event: Event) -> str | None:
    if event_type in {"upward_trend", "sharp_rise"}:
        return "up"
    if event_type in {"downward_trend", "sharp_drop"}:
        return "down"
    return event_direction(event)


def _claim_text(event_type: str, start: int, end: int, strength: str | None = None) -> str:
    event_name = event_type.replace("_", " ")
    modifier = f"{strength} " if strength else ""
    return f"The series shows a {modifier}{event_name} from t={start} to t={end}."


def build_cases(samples: list[dict]) -> list[dict]:
    """Create supported, partial, and unsupported claims from gold events."""

    cases: list[dict] = []
    for sample in samples:
        length = int(sample.get("metadata", {}).get("length", len(sample.get("series", []))))
        events = [Event(**event) for event in sample.get("events", [])]
        if not events:
            continue
        event = events[0]
        strength = event.attributes.get("strength")
        direction = event_direction(event)

        supported = Claim(
            claim_id=f"{sample['id']}::supported",
            text=_claim_text(event.type, event.start, event.end, str(strength) if strength else None),
            claim_type=event.type,
            direction=direction,
            strength=str(strength) if strength else None,
            support_interval=(event.start, event.end),
        )
        cases.append({"id": sample["id"], "case_type": "gold_supported", "gold_label": "supported", "claim": supported.model_dump()})

        wrong_start, wrong_end = _wrong_interval(event.start, event.end, length)
        partial = Claim(
            claim_id=f"{sample['id']}::partial_interval",
            text=_claim_text(event.type, wrong_start, wrong_end, str(strength) if strength else None),
            claim_type=event.type,
            direction=direction,
            strength=str(strength) if strength else None,
            support_interval=(wrong_start, wrong_end),
            error_type="interval_mislocalization",
        )
        cases.append({"id": sample["id"], "case_type": "interval_mislocalization", "gold_label": "partial", "claim": partial.model_dump()})

        opposite_type = _opposite_type(event.type)
        unsupported = Claim(
            claim_id=f"{sample['id']}::unsupported_type",
            text=_claim_text(opposite_type, event.start, event.end),
            claim_type=opposite_type,
            direction=_direction_for_type(opposite_type, event),
            support_interval=(event.start, event.end),
            error_type="pattern_confusion",
        )
        cases.append({"id": sample["id"], "case_type": "pattern_confusion", "gold_label": "unsupported", "claim": unsupported.model_dump()})

        causal = Claim(
            claim_id=f"{sample['id']}::unsupported_causal",
            text="The pattern happens because of an external intervention not shown in the series.",
            claim_type="causal_claim",
            error_type="unsupported_causal_claim",
        )
        cases.append({"id": sample["id"], "case_type": "unsupported_causal", "gold_label": "unsupported", "claim": causal.model_dump()})
    return cases


def _result(claim: Claim, label: str, score: float = 0.0, event: Event | None = None) -> VerificationResult:
    return VerificationResult(
        claim_id=claim.claim_id,
        label=label,  # type: ignore[arg-type]
        score=score,
        support_events=[event.event_id] if event else [],
        support_interval=(event.start, event.end) if event else None,
    )


def majority_unsupported(claim: Claim, graph: EventGraph) -> VerificationResult:
    return _result(claim, "unsupported", 0.0)


def type_only(claim: Claim, graph: EventGraph) -> VerificationResult:
    candidates = retrieve_candidate_events(claim, graph)
    if not candidates:
        return _result(claim, "unsupported", 0.0)
    return _result(claim, "supported", 1.0, candidates[0])


def type_direction(claim: Claim, graph: EventGraph) -> VerificationResult:
    candidates = retrieve_candidate_events(claim, graph)
    if not candidates:
        return _result(claim, "unsupported", 0.0)
    for event in candidates:
        if not claim.direction or event_direction(event) in {None, claim.direction}:
            return _result(claim, "supported", 1.0, event)
    return _result(claim, "unsupported", 0.0, candidates[0])


def type_interval(claim: Claim, graph: EventGraph) -> VerificationResult:
    candidates = retrieve_candidate_events(claim, graph)
    if not candidates:
        return _result(claim, "unsupported", 0.0)
    if claim.support_interval is None:
        return _result(claim, "supported", 1.0, candidates[0])
    scored = [(interval_iou(tuple(claim.support_interval), (event.start, event.end)), event) for event in candidates]
    best_iou, best_event = max(scored, key=lambda item: item[0])
    if best_iou >= 0.5:
        return _result(claim, "supported", best_iou, best_event)
    if best_iou > 0.0 or candidates:
        return _result(claim, "partial", best_iou, best_event)
    return _result(claim, "unsupported", 0.0)


def make_no_interval_verifier(config: VerifierConfig) -> Callable[[Claim, EventGraph], VerificationResult]:
    weights = dict(config.weights)
    interval_weight = weights.pop("interval", 0.0)
    if interval_weight:
        total = sum(weights.values()) or 1.0
        weights = {key: value + interval_weight * (value / total) for key, value in weights.items()}
    no_interval_config = VerifierConfig(
        supported_threshold=config.supported_threshold,
        partial_threshold=config.partial_threshold,
        weights=weights,
        interval_iou_threshold=0.0,
    )

    def _verify(claim: Claim, graph: EventGraph) -> VerificationResult:
        return verify_claim(claim, graph, no_interval_config)

    return _verify


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_variant(
    name: str,
    cases: list[dict],
    graphs: dict[str, EventGraph],
    verifier: Callable[[Claim, EventGraph], VerificationResult],
) -> tuple[dict, list[dict]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    ious: list[float] = []
    outputs: list[dict] = []
    for case in cases:
        claim = Claim(**case["claim"])
        graph = graphs[case["id"]]
        result = verifier(claim, graph)
        y_true.append(case["gold_label"])
        y_pred.append(result.label)
        if case["gold_label"] in {"supported", "partial"} and result.support_interval is not None and claim.support_interval is not None:
            ious.append(interval_iou(tuple(claim.support_interval), tuple(result.support_interval)))
        outputs.append({**case, "prediction": result.model_dump()})
    report = classification_report(y_true, y_pred)
    report["name"] = name
    report["support_iou"] = _mean(ious)
    return report, outputs


def latex_table(reports: list[dict]) -> str:
    rows = []
    for report in reports:
        rows.append(
            f"{report['name']} & {report['macro_f1']:.3f} & {report['supported_f1']:.3f} "
            f"& {report['partial_f1']:.3f} & {report['unsupported_f1']:.3f} "
            f"& {report['support_iou']:.3f} \\\\"
        )
    body = "\n".join(rows)
    return r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{lrrrrr}
\toprule
Verifier & Macro-F1 & Sup. F1 & Part. F1 & Unsup. F1 & IoU \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Verifier baseline and ablation results on 1,600 synthetic claim-level cases derived from the 400-example test split. The benchmark contains one supported claim, one interval-mislocalized partial claim, and two unsupported claims per series. IoU is computed for supported and partial cases with predicted support intervals.}
\label{tab:verification}
\end{table}
""" % body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/test.jsonl")
    parser.add_argument("--graphs", default="data/processed/test_graphs.jsonl")
    parser.add_argument("--config", default="configs/verifier.yaml")
    parser.add_argument("--output-dir", default="outputs/verification_baselines/full_synthetic_400")
    args = parser.parse_args()

    samples = read_jsonl(_resolve(args.input))
    extracted_graphs = _load_graphs(_resolve(args.graphs))
    gold_graphs = {
        sample["id"]: build_event_graph(
            sample["id"],
            events=[Event(**event) for event in sample.get("events", [])],
            prefer_gold_events=True,
        )
        for sample in samples
    }
    config = VerifierConfig.from_yaml(_resolve(args.config))
    cases = build_cases(samples)

    variants: list[tuple[str, dict[str, EventGraph], Callable[[Claim, EventGraph], VerificationResult]]] = [
        ("Majority unsupported", extracted_graphs, majority_unsupported),
        ("Type-only", extracted_graphs, type_only),
        ("Type+direction", extracted_graphs, type_direction),
        ("Type+interval", extracted_graphs, type_interval),
        ("Event graph (no interval)", extracted_graphs, make_no_interval_verifier(config)),
        ("Event graph", extracted_graphs, lambda claim, graph: verify_claim(claim, graph, config)),
        ("Gold-event oracle", gold_graphs, lambda claim, graph: verify_claim(claim, graph, config)),
    ]

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, graphs, verifier in variants:
        report, predictions = evaluate_variant(name, cases, graphs, verifier)
        reports.append(report)
        safe_name = name.lower().replace("+", "_").replace(" ", "_").replace("(", "").replace(")", "")
        write_jsonl(output_dir / f"{safe_name}_predictions.jsonl", predictions)

    write_jsonl(output_dir / "verification_cases.jsonl", cases)
    write_json(output_dir / "verification_baselines.json", {"reports": reports})
    table = latex_table(reports)
    write_text(ROOT / "outputs" / "tables" / "verification_results.tex", table)
    print(f"Wrote {len(cases)} cases and {len(reports)} reports to {output_dir.relative_to(ROOT)}")
    for report in reports:
        print(
            f"{report['name']}: macro_f1={report['macro_f1']:.3f}, "
            f"sup_f1={report['supported_f1']:.3f}, partial_f1={report['partial_f1']:.3f}, "
            f"unsup_f1={report['unsupported_f1']:.3f}, iou={report['support_iou']:.3f}"
        )


if __name__ == "__main__":
    main()
