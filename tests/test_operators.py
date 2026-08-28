import unittest

import numpy as np

from tsr.events.operators import (
    compute_slope,
    compute_volatility,
    compute_zscores,
    detect_anomalies,
    detect_change_points,
    detect_trend,
)


class OperatorTests(unittest.TestCase):
    def test_slope_and_volatility(self):
        x = [1, 2, 3, 4, 5]
        self.assertGreater(compute_slope(x, 0, 4), 0.9)
        self.assertAlmostEqual(compute_volatility(x, 0, 4), 0.0)

    def test_zscore_and_anomaly_detection(self):
        x = [0.0] * 20 + [10.0] + [0.0] * 20
        z = compute_zscores(x)
        self.assertGreater(z[20], 3.0)
        self.assertTrue(any(event.type == "anomaly" for event in detect_anomalies(x, z_threshold=3.0)))

    def test_trend_and_change_point_detection(self):
        trend = np.linspace(0, 10, 80)
        self.assertTrue(any(event.type == "upward_trend" for event in detect_trend(trend)))
        cp = [0.0] * 40 + [5.0] * 40
        self.assertTrue(any(event.type == "change_point" for event in detect_change_points(cp)))


if __name__ == "__main__":
    unittest.main()
