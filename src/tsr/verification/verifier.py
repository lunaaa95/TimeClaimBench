"""Event-grounded claim verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tsr.claims.claim_extractor import parse_interval_text
from tsr.claims.claim_schema import Claim, VerificationResult
from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph
from tsr.utils.io import read_yaml
from tsr.verification.metrics import interval_iou


CLAIM_TO_EVENT_TYPES = {
    "trend_up": ["upward_trend"],
    "upward_trend": ["upward_trend"],
    "trend_down": ["downward_trend"],
    "downward_trend": ["downward_trend"],
    "sharp_rise": ["sharp_rise"],
    "sharp_drop": ["sharp_drop"],
    "local_peak": ["local_peak"],
    "local_trough": ["local_trough"],
    "volatility_shift": ["volatility_shift"],
    "volatility_increase": ["volatility_shift"],
    "volatility_decrease": ["volatility_shift"],
    "change_point": ["change_point"],
    "seasonality": ["seasonality"],
    "anomaly": ["anomaly"],
    "recovery": ["upward_trend", "sharp_rise"],
}

OPPOSITE_TYPES = {
    "upward_trend": "downward_trend",
    "downward_trend": "upward_trend",
    "sharp_rise": "sharp_drop",
    "sharp_drop": "sharp_rise",
}


@dataclass
class VerifierConfig:
    supported_threshold: float = 0.75
    partial_threshold: float = 0.45
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "type": 0.35,
            "direction": 0.20,
            "interval": 0.25,
            "strength": 0.10,
            "numeric": 0.10,
        }
    )
    interval_iou_threshold: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VerifierConfig":
        data = read_yaml(path)
        return cls(
            supported_threshold=float(data.get("supported_threshold", 0.75)),
            partial_threshold=float(data.get("partial_threshold", 0.45)),
            weights=dict(data.get("weights", {})) or cls().weights,
            interval_iou_threshold=float(data.get("interval_iou_threshold", 0.5)),
        )


@dataclass
class MatchScore:
    event: Event
    score: float
    components: Dict[str, float]


def graph_length(graph: EventGraph) -> Optional[int]:
    if not graph.events:
        return None
    return max(event.end for event in graph.events) + 1


def candidate_event_types(claim: Claim) -> List[str]:
    return CLAIM_TO_EVENT_TYPES.get(claim.claim_type, [])


def retrieve_candidate_events(claim: Claim, graph: EventGraph) -> List[Event]:
    event_types = candidate_event_types(claim)
    if not event_types:
        return []
    return [event for event in graph.events if event.type in event_types]


def event_direction(event: Event) -> Optional[str]:
    if "direction" in event.attributes:
        return str(event.attributes["direction"])
    if event.type in {"upward_trend", "sharp_rise"}:
        return "up"
    if event.type in {"downward_trend", "sharp_drop"}:
        return "down"
    return None


def direction_score(claim: Claim, event: Event) -> float:
    if not claim.direction:
        return 1.0
    direction = event_direction(event)
    if direction is None:
        return 0.8
    return 1.0 if claim.direction == direction else 0.0


def strength_score(claim: Claim, event: Event) -> float:
    if not claim.strength:
        return 1.0
    event_strength = event.attributes.get("strength")
    if not event_strength:
        return 0.8
    if claim.strength == event_strength:
        return 1.0
    order = {"weak": 0, "moderate": 1, "strong": 2}
    if claim.strength in order and event_strength in order:
        return max(0.0, 1.0 - 0.5 * abs(order[claim.strength] - order[event_strength]))
    return 0.0


def numeric_score(claim: Claim, event: Event) -> float:
    if claim.numeric_value is None:
        return 1.0
    numeric_attrs = [
        event.attributes.get(key)
        for key in ["period", "magnitude", "slope", "z_score", "before_mean", "after_mean"]
        if key in event.attributes
    ]
    if not numeric_attrs:
        return 0.8
    best = 0.0
    for value in numeric_attrs:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        denom = max(abs(value), abs(claim.numeric_value), 1.0)
        best = max(best, 1.0 - min(1.0, abs(value - claim.numeric_value) / denom))
    return best


def claim_interval(claim: Claim, graph: EventGraph) -> Optional[Tuple[int, int]]:
    if claim.support_interval is not None:
        return tuple(claim.support_interval)
    return parse_interval_text(claim.text, graph_length(graph))


def score_claim_event_match(claim: Claim, event: Event, graph: EventGraph, config: VerifierConfig) -> MatchScore:
    interval = claim_interval(claim, graph)
    if interval is None:
        interval_component = 1.0
    else:
        interval_component = interval_iou(interval, (event.start, event.end))
    components = {
        "type": 1.0,
        "direction": direction_score(claim, event),
        "interval": interval_component,
        "strength": strength_score(claim, event),
        "numeric": numeric_score(claim, event),
    }
    score = sum(config.weights.get(name, 0.0) * value for name, value in components.items())
    return MatchScore(event=event, score=float(score), components=components)


def infer_partial_error(claim: Claim, match: MatchScore, config: VerifierConfig) -> str:
    components = match.components
    if components.get("direction", 1.0) < 0.5:
        return "direction_error"
    if components.get("interval", 1.0) < config.interval_iou_threshold:
        return "interval_mislocalization"
    if components.get("strength", 1.0) < 0.75:
        return "magnitude_overclaim"
    if components.get("numeric", 1.0) < 0.75:
        return "numeric_error"
    return "partial_support"


def infer_error_type(claim: Claim, graph: EventGraph) -> str:
    if claim.claim_type == "causal_claim":
        return "unsupported_causal_claim"
    if claim.claim_type == "chronology_claim":
        return "chronology_violation"

    event_types = {event.type for event in graph.events}
    claim_types = candidate_event_types(claim)
    if claim_types:
        for event_type in claim_types:
            opposite = OPPOSITE_TYPES.get(event_type)
            if opposite in event_types:
                return "direction_error"
        if any(event_type in event_types for event_type in claim_types):
            return "interval_mislocalization"
        if event_types:
            return "pattern_confusion"
    return "pattern_hallucination"


def verify_claim(claim: Claim, graph: EventGraph, config: Optional[VerifierConfig] = None) -> VerificationResult:
    config = config or VerifierConfig()
    candidates = retrieve_candidate_events(claim, graph)
    if not candidates:
        return VerificationResult(
            claim_id=claim.claim_id,
            label="unsupported",
            score=0.0,
            error_type=infer_error_type(claim, graph),
            rationale="No candidate event of the requested type was found in the event graph.",
        )

    scored = [score_claim_event_match(claim, event, graph, config) for event in candidates]
    best = max(scored, key=lambda item: item.score)
    support_interval = (best.event.start, best.event.end)
    support_events = [best.event.event_id]
    explicit_interval = claim_interval(claim, graph) is not None
    interval_ok = (not explicit_interval) or best.components.get("interval", 1.0) >= config.interval_iou_threshold
    direction_ok = best.components.get("direction", 1.0) >= 0.5
    if best.score >= config.supported_threshold and interval_ok and direction_ok:
        return VerificationResult(
            claim_id=claim.claim_id,
            label="supported",
            support_events=support_events,
            support_interval=support_interval,
            score=best.score,
            error_type=None,
            rationale="Claim matches a grounded temporal event.",
        )
    if not direction_ok:
        return VerificationResult(
            claim_id=claim.claim_id,
            label="unsupported",
            support_events=support_events,
            support_interval=support_interval,
            score=best.score,
            error_type="direction_error",
            rationale="Claim direction is incompatible with the best matching event.",
        )
    if best.score >= config.partial_threshold:
        return VerificationResult(
            claim_id=claim.claim_id,
            label="partial",
            support_events=support_events,
            support_interval=support_interval,
            score=best.score,
            error_type=infer_partial_error(claim, best, config),
            rationale="Claim has a candidate event but one or more grounding components are weak.",
        )
    return VerificationResult(
        claim_id=claim.claim_id,
        label="unsupported",
        support_events=support_events,
        support_interval=support_interval,
        score=best.score,
        error_type=infer_error_type(claim, graph),
        rationale="Best candidate event score is below the partial-support threshold.",
    )


def verify_claims(
    claims: List[Claim],
    graph: EventGraph,
    config: Optional[VerifierConfig] = None,
) -> List[VerificationResult]:
    config = config or VerifierConfig()
    return [verify_claim(claim, graph, config) for claim in claims]
