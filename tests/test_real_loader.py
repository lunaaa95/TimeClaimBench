import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from tsr.data.real_loader import load_nab_windows


class RealLoaderTests(unittest.TestCase):
    def test_nab_windows_are_converted_to_project_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "realKnownCause"
            label_dir = root / "labels"
            data_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            csv_path = data_dir / "example.csv"
            start = datetime(2020, 1, 1, 0, 0, 0)
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("timestamp,value\n")
                for i in range(120):
                    value = 10.0 if i != 50 else 30.0
                    f.write(f"{start + timedelta(minutes=i)},{value}\n")

            labels = {
                "realKnownCause/example.csv": [
                    ["2020-01-01 00:49:00", "2020-01-01 00:51:00"],
                ]
            }
            (label_dir / "combined_windows.json").write_text(json.dumps(labels), encoding="utf-8")

            samples = load_nab_windows(
                root,
                subsets=["realKnownCause"],
                length=20,
                max_anomaly_windows=1,
                max_normal_windows=1,
                seed=7,
            )

            self.assertEqual(len(samples), 2)
            anomaly = next(sample for sample in samples if sample.metadata.pattern_types == ["nab_anomaly"])
            normal = next(sample for sample in samples if sample.metadata.pattern_types == ["nab_normal"])
            self.assertEqual(len(anomaly.series), 20)
            self.assertEqual(len(anomaly.metadata.timestamps), 20)
            self.assertEqual(anomaly.events[0].type, "anomaly")
            self.assertEqual(anomaly.reference_claims[0].claim_type, "anomaly")
            self.assertEqual(normal.events, [])


if __name__ == "__main__":
    unittest.main()
