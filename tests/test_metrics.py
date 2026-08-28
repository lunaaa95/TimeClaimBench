import unittest

from tsr.verification.metrics import (
    claim_retention_rate,
    claims_per_explanation,
    claim_faithfulness,
    interval_iou,
    macro_f1,
    summarize_generation_cost,
    supported_claim_preservation,
    unsupported_claim_rate,
)


class MetricsTests(unittest.TestCase):
    def test_rates_and_iou(self):
        labels = ["supported", "partial", "unsupported", "supported"]
        self.assertEqual(claim_faithfulness(labels), 0.5)
        self.assertEqual(unsupported_claim_rate(labels), 0.25)
        self.assertAlmostEqual(interval_iou((0, 9), (5, 14)), 5 / 15)

    def test_macro_f1(self):
        y_true = ["supported", "unsupported", "partial"]
        y_pred = ["supported", "unsupported", "supported"]
        self.assertGreater(macro_f1(y_true, y_pred), 0.4)

    def test_repair_preservation_metrics(self):
        self.assertEqual(claims_per_explanation(10, 2), 5.0)
        self.assertEqual(claim_retention_rate(10, 7), 0.7)
        records = [
            {
                "verification_results": [
                    {"label": "supported", "support_events": ["E1"]},
                    {"label": "unsupported", "support_events": []},
                    {"label": "supported", "support_events": ["E2"]},
                ],
                "repaired_verification_results": [
                    {"label": "supported", "support_events": ["E1"]},
                ],
            }
        ]
        self.assertEqual(supported_claim_preservation(records), 0.5)

    def test_cost_summary(self):
        records = [
            {"estimated_cost_usd": 0.1, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            {"estimated_cost_usd": 0.2, "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}},
        ]
        summary = summarize_generation_cost(records)
        self.assertAlmostEqual(summary["generation_cost_usd"], 0.3)
        self.assertAlmostEqual(summary["cost_per_explanation_usd"], 0.15)
        self.assertEqual(summary["total_tokens"], 40.0)


if __name__ == "__main__":
    unittest.main()
