"""Compare scale-invariant shape features for synthetic and Real-NAB windows.

This is a coverage diagnostic after per-window z-normalization, not a test of
distributional equivalence or domain representativeness.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_PATH = ROOT / "data/synthetic/test.jsonl"
REAL_PATH = ROOT / "data/real/nab_windows.jsonl"
OUT_DIR = ROOT / "outputs/analysis/camera_ready"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def standardize(values: list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / std


def spectral_entropy(z: np.ndarray) -> float:
    power = np.abs(np.fft.rfft(z))[1:] ** 2
    total = float(np.sum(power))
    if total < 1e-12 or len(power) <= 1:
        return 0.0
    probabilities = power / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / float(np.log(len(probabilities)))


def shape_features(values: list[float]) -> dict[str, float]:
    z = standardize(values)
    n = len(z)
    time = np.linspace(-1.0, 1.0, n)
    slope = float(np.polyfit(time, z, 1)[0]) if n > 1 else 0.0
    if n > 2 and float(np.std(z[:-1])) > 1e-12 and float(np.std(z[1:])) > 1e-12:
        lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    else:
        lag1 = 0.0
    differences = np.diff(z)
    first_std = float(np.std(z[: n // 2]))
    second_std = float(np.std(z[n // 2 :]))
    volatility_ratio = max(first_std, second_std) / max(min(first_std, second_std), 1e-6)
    return {
        "abs_trend_slope": abs(slope),
        "lag1_autocorrelation": lag1,
        "difference_std": float(np.std(differences)),
        "max_abs_step": float(np.max(np.abs(differences))),
        "spectral_entropy": spectral_entropy(z),
        "peak_abs_z": float(np.max(np.abs(z))),
        "standardized_range": float(np.ptp(z)),
        "log_volatility_ratio": float(np.log1p(volatility_ratio)),
    }


def is_anomaly_window(record: dict) -> bool:
    return any(event.get("type") == "anomaly" for event in record.get("events", []))


def summarize_feature(
    feature: str,
    synthetic_rows: list[dict[str, float]],
    real_rows: list[dict[str, float]],
    anomaly_mask: np.ndarray,
) -> dict[str, float | str]:
    synthetic = np.asarray([row[feature] for row in synthetic_rows], dtype=float)
    real = np.asarray([row[feature] for row in real_rows], dtype=float)
    q05, q50, q95 = np.quantile(synthetic, [0.05, 0.50, 0.95])

    def coverage(values: np.ndarray) -> float:
        return float(np.mean((values >= q05) & (values <= q95)))

    return {
        "feature": feature,
        "synthetic_q05": float(q05),
        "synthetic_median": float(q50),
        "synthetic_q95": float(q95),
        "real_median": float(np.median(real)),
        "real_coverage": coverage(real),
        "anomaly_coverage": coverage(real[anomaly_mask]),
        "normal_coverage": coverage(real[~anomaly_mask]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_records = read_jsonl(SYNTHETIC_PATH)
    real_records = read_jsonl(REAL_PATH)
    synthetic_rows = [shape_features(record["series"]) for record in synthetic_records]
    real_rows = [shape_features(record["series"]) for record in real_records]
    anomaly_mask = np.asarray([is_anomaly_window(record) for record in real_records], dtype=bool)

    feature_names = list(synthetic_rows[0])
    summaries = [
        summarize_feature(feature, synthetic_rows, real_rows, anomaly_mask)
        for feature in feature_names
    ]

    with (OUT_DIR / "synthetic_real_shape_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    synthetic_bounds = {
        row["feature"]: (float(row["synthetic_q05"]), float(row["synthetic_q95"]))
        for row in summaries
    }
    per_window_coverage = []
    for row in real_rows:
        covered = sum(
            synthetic_bounds[feature][0] <= row[feature] <= synthetic_bounds[feature][1]
            for feature in feature_names
        )
        per_window_coverage.append(covered / len(feature_names))

    result = {
        "source": {
            "synthetic": str(SYNTHETIC_PATH.relative_to(ROOT)),
            "real": str(REAL_PATH.relative_to(ROOT)),
        },
        "n_synthetic": len(synthetic_rows),
        "n_real": len(real_rows),
        "n_real_anomaly": int(np.sum(anomaly_mask)),
        "n_real_normal": int(np.sum(~anomaly_mask)),
        "feature_definition": "All features are computed after per-window z-normalization.",
        "coverage_definition": "Real value lies within the synthetic 5th-95th percentile interval.",
        "mean_feature_coverage": float(np.mean([row["real_coverage"] for row in summaries])),
        "median_feature_coverage": float(np.median([row["real_coverage"] for row in summaries])),
        "mean_real_window_feature_fraction_covered": float(np.mean(per_window_coverage)),
        "real_windows_with_at_least_6_of_8_features_covered": float(
            np.mean(np.asarray(per_window_coverage) >= 0.75)
        ),
        "features": summaries,
        "interpretation_constraint": (
            "This is a scale-invariant shape-coverage diagnostic, not evidence that "
            "synthetic windows reproduce the full real-data distribution."
        ),
    }
    (OUT_DIR / "synthetic_real_shape_coverage.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
