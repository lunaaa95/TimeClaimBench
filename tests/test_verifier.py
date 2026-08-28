import unittest

from tsr.claims.claim_schema import Claim
from tsr.data.dataset_schema import Event
from tsr.events.event_graph import EventGraph
from tsr.verification.verifier import VerifierConfig, verify_claim


class VerifierTests(unittest.TestCase):
    def setUp(self):
        event = Event(
            event_id="E1",
            type="upward_trend",
            start=5,
            end=35,
            score=1.0,
            attributes={"direction": "up", "strength": "moderate", "slope": 0.1},
        )
        self.graph = EventGraph(series_id="s", events=[event], relations=[])
        self.config = VerifierConfig()

    def test_supported_claim(self):
        claim = Claim(
            claim_id="C1",
            text="The series shows a moderate upward trend from t=5 to t=35.",
            claim_type="upward_trend",
            direction="up",
            strength="moderate",
            support_interval=(5, 35),
        )
        result = verify_claim(claim, self.graph, self.config)
        self.assertEqual(result.label, "supported")

    def test_partial_claim(self):
        claim = Claim(
            claim_id="C1",
            text="The series shows a moderate upward trend from t=50 to t=70.",
            claim_type="upward_trend",
            direction="up",
            strength="moderate",
            support_interval=(50, 70),
        )
        result = verify_claim(claim, self.graph, self.config)
        self.assertEqual(result.label, "partial")
        self.assertEqual(result.error_type, "interval_mislocalization")

    def test_direction_mismatch_is_unsupported(self):
        claim = Claim(
            claim_id="C1",
            text="The series shows a moderate downward trend from t=5 to t=35.",
            claim_type="upward_trend",
            direction="down",
            strength="moderate",
            support_interval=(5, 35),
        )
        result = verify_claim(claim, self.graph, self.config)
        self.assertEqual(result.label, "unsupported")
        self.assertEqual(result.error_type, "direction_error")

    def test_unsupported_claim(self):
        claim = Claim(claim_id="C1", text="The series is seasonal.", claim_type="seasonality")
        result = verify_claim(claim, self.graph, self.config)
        self.assertEqual(result.label, "unsupported")


if __name__ == "__main__":
    unittest.main()
