"""Verifier-guided rationale repair."""

from __future__ import annotations

from typing import Dict, List

from tsr.claims.claim_schema import Claim, VerificationResult
from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph


def event_to_sentence(event: Event) -> str:
    strength = event.attributes.get("strength")
    strength_prefix = f"{strength} " if strength else ""
    if event.type == "upward_trend":
        return f"The series shows a {strength_prefix}upward trend from t={event.start} to t={event.end}."
    if event.type == "downward_trend":
        return f"The series shows a {strength_prefix}downward trend from t={event.start} to t={event.end}."
    if event.type == "sharp_rise":
        return f"There is a {strength_prefix}sharp rise from t={event.start} to t={event.end}."
    if event.type == "sharp_drop":
        return f"There is a {strength_prefix}sharp drop from t={event.start} to t={event.end}."
    if event.type == "volatility_shift":
        direction = event.attributes.get("direction", "change")
        return f"Volatility shows a {strength_prefix}{direction} after t={event.start}."
    if event.type == "change_point":
        return f"The mean level changes around t={event.start}."
    if event.type == "seasonality":
        period = event.attributes.get("period", "unknown")
        return f"The series shows {strength_prefix}seasonality with period about {period}."
    if event.type == "anomaly":
        direction = event.attributes.get("direction", "")
        return f"There is a {strength_prefix}{direction} anomaly at t={event.start}."
    return f"The event {event.event_id} supports a {event.type} pattern from t={event.start} to t={event.end}."


def repair_explanation(
    explanation: str,
    claims: List[Claim],
    verification_results: List[VerificationResult],
    graph: EventGraph,
    mode: str = "rule",
) -> str:
    """Return a repaired explanation with unsupported claims removed or corrected."""

    del explanation, mode
    result_by_id: Dict[str, VerificationResult] = {result.claim_id: result for result in verification_results}
    event_by_id: Dict[str, Event] = {event.event_id: event for event in graph.events}
    repaired: List[str] = []
    seen = set()

    for claim in claims:
        result = result_by_id.get(claim.claim_id)
        if result is None:
            continue
        if result.label == "supported":
            sentence = claim.text.strip()
        elif result.label == "partial" and result.support_events:
            event = event_by_id.get(result.support_events[0])
            if event is None:
                continue
            sentence = event_to_sentence(event)
        else:
            continue
        normalized = sentence.lower()
        if normalized not in seen:
            repaired.append(sentence)
            seen.add(normalized)

    if not repaired:
        supported_events = [event for event in graph.events if event.score >= 0.5]
        if supported_events:
            repaired.append(event_to_sentence(supported_events[0]))
        else:
            repaired.append("No verifiable temporal pattern is supported by the current event graph.")
    return " ".join(repaired)
