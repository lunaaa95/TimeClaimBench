"""Event extraction pipeline over univariate time series."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from tsr.data.dataset_schema import Event
from tsr.events import operators
from tsr.verification.metrics import interval_iou


def deduplicate_events(events: Iterable[Event], iou_threshold: float = 0.8) -> List[Event]:
    kept: List[Event] = []
    for event in sorted(events, key=lambda e: (-e.score, e.start, e.end)):
        duplicate = False
        for kept_event in kept:
            if event.type == kept_event.type and interval_iou((event.start, event.end), (kept_event.start, kept_event.end)) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(event)
    kept.sort(key=lambda e: (e.start, e.end, e.type))
    for i, event in enumerate(kept, start=1):
        event.event_id = f"E{i}"
    return kept


def extract_events(series: Sequence[float]) -> List[Event]:
    """Extract candidate temporal events with executable operators."""

    events: List[Event] = []
    events.extend(operators.detect_trend(series))
    events.extend(operators.detect_sharp_changes(series))
    events.extend(operators.detect_volatility_shift(series))
    events.extend(operators.detect_change_points(series))
    events.extend(operators.detect_seasonality(series))
    events.extend(operators.detect_anomalies(series))
    events.extend(operators.detect_local_peaks(series))
    events.extend(operators.detect_local_troughs(series))
    return deduplicate_events(events)
