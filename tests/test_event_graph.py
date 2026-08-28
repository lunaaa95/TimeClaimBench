import unittest

from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph, build_relations


class EventGraphTests(unittest.TestCase):
    def test_temporal_relations(self):
        a = Event(event_id="E1", type="upward_trend", start=0, end=10, score=1.0)
        b = Event(event_id="E2", type="sharp_drop", start=12, end=14, score=1.0)
        c = Event(event_id="E3", type="anomaly", start=5, end=5, score=1.0)
        d = Event(event_id="E4", type="volatility_shift", start=8, end=20, score=1.0)
        graph = EventGraph(series_id="s", events=[a, b, c, d], relations=[])
        self.assertEqual(graph.temporal_relation(a, b), "before")
        self.assertEqual(graph.temporal_relation(b, a), "after")
        self.assertEqual(graph.temporal_relation(a, c), "contains")
        self.assertEqual(graph.temporal_relation(a, d), "overlaps")
        relations = build_relations([a, b])
        self.assertEqual(relations[0].relation_type, "before")
        self.assertEqual(relations[1].relation_type, "after")


if __name__ == "__main__":
    unittest.main()
