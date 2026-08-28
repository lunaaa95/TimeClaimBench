"""Time-Series Event Graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from tsr.data.dataset_schema import Event
from tsr.events.event_extractor import deduplicate_events, extract_events
from tsr.verification.metrics import interval_iou


@dataclass
class EventRelation:
    source: str
    target: str
    relation_type: str

    def to_json(self) -> Dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
        }

    @classmethod
    def from_json(cls, data: Dict[str, str]) -> "EventRelation":
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=data["relation_type"],
        )


@dataclass
class EventGraph:
    series_id: str
    events: List[Event]
    relations: List[EventRelation]

    def to_json(self) -> Dict:
        return {
            "series_id": self.series_id,
            "events": [event.model_dump() for event in self.events],
            "relations": [relation.to_json() for relation in self.relations],
        }

    @classmethod
    def from_json(cls, data: Dict) -> "EventGraph":
        return cls(
            series_id=data["series_id"],
            events=[Event(**event) for event in data.get("events", [])],
            relations=[EventRelation.from_json(rel) for rel in data.get("relations", [])],
        )

    def find_events_by_type(self, event_type: str) -> List[Event]:
        return [event for event in self.events if event.type == event_type]

    def find_events_overlapping_interval(self, start: int, end: int) -> List[Event]:
        query = (start, end)
        return [event for event in self.events if interval_iou((event.start, event.end), query) > 0.0]

    def temporal_relation(self, a: Event, b: Event) -> str:
        if a.end < b.start:
            return "before"
        if b.end < a.start:
            return "after"
        if a.start <= b.start and a.end >= b.end:
            return "contains"
        if b.start <= a.start and b.end >= a.end:
            return "within"
        return "overlaps"


def build_relations(events: List[Event]) -> List[EventRelation]:
    relations: List[EventRelation] = []
    for i, source in enumerate(events):
        for target in events[i + 1 :]:
            relation = EventGraph("", [], []).temporal_relation(source, target)
            relations.append(EventRelation(source=source.event_id, target=target.event_id, relation_type=relation))
            if relation == "before":
                inverse = "after"
            elif relation == "after":
                inverse = "before"
            elif relation == "contains":
                inverse = "within"
            elif relation == "within":
                inverse = "contains"
            else:
                inverse = relation
            relations.append(EventRelation(source=target.event_id, target=source.event_id, relation_type=inverse))
    return relations


def build_event_graph(
    series_id: str,
    series: Optional[Sequence[float]] = None,
    events: Optional[List[Event]] = None,
    prefer_gold_events: bool = False,
) -> EventGraph:
    """Build an event graph from extracted events or supplied event annotations."""

    if events is None:
        if series is None:
            raise ValueError("Either series or events must be provided.")
        events = extract_events(series)
    elif not prefer_gold_events:
        events = deduplicate_events(events)
    else:
        for i, event in enumerate(sorted(events, key=lambda e: (e.start, e.end, e.type)), start=1):
            event.event_id = event.event_id or f"E{i}"

    events = sorted(events, key=lambda e: (e.start, e.end, e.type))
    return EventGraph(series_id=series_id, events=events, relations=build_relations(events))
