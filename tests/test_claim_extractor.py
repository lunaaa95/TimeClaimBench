import unittest

from tsr.claims.claim_extractor import parse_interval_text, rule_extract_claims


class ClaimExtractorTests(unittest.TestCase):
    def test_keyword_claim_extraction(self):
        explanation = "The series shows a moderate upward trend from t=5 to t=35. There is a sharp drop near t=60."
        claims = rule_extract_claims(explanation)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].claim_type, "upward_trend")
        self.assertEqual(claims[0].support_interval, (5, 35))
        self.assertEqual(claims[1].claim_type, "sharp_drop")

    def test_relative_interval_parsing(self):
        self.assertEqual(parse_interval_text("in the first half", series_length=80), (0, 39))
        self.assertEqual(parse_interval_text("near the end", series_length=90), (60, 89))


if __name__ == "__main__":
    unittest.main()
