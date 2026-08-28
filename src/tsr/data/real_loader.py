"""Load license-checked real-world time-series datasets."""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from tsr.data.dataset_schema import Event, ReferenceClaim, TimeSeriesMetadata, TimeSeriesSample


NAB_REAL_SUBSETS = (
    "realKnownCause",
    "realTraffic",
    "realTweets",
    "realAWSCloudwatch",
    "realAdExchange",
)

NAB_QUESTION = (
    "Describe the major temporal patterns in this real-world time-series window, "
    "including any anomaly or change-like behavior."
)


def _parse_timestamp(value: str) -> datetime:
    cleaned = value.strip().replace("T", " ").replace("Z", "")
    return datetime.fromisoformat(cleaned)


def _timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ")


def _read_nab_csv(path: Path) -> tuple[list[datetime], list[float]]:
    timestamps: list[datetime] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return timestamps, values
        timestamp_col = "timestamp" if "timestamp" in reader.fieldnames else reader.fieldnames[0]
        value_col = "value" if "value" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            raw_value = row.get(value_col)
            raw_timestamp = row.get(timestamp_col)
            if raw_value is None or raw_timestamp is None:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            timestamps.append(_parse_timestamp(raw_timestamp))
            values.append(value)
    return timestamps, values


def _load_windows(nab_root: Path) -> dict[str, list[tuple[datetime, datetime]]]:
    path = nab_root / "labels" / "combined_windows.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing NAB anomaly-window labels: {path}. "
            "Expected a checkout of https://github.com/numenta/NAB under data/raw/NAB."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: [(_parse_timestamp(start), _parse_timestamp(end)) for start, end in windows]
        for key, windows in raw.items()
    }


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _window_indices_for_label(
    timestamps: Sequence[datetime],
    label_start: datetime,
    label_end: datetime,
) -> tuple[int, int] | None:
    matches = [i for i, timestamp in enumerate(timestamps) if label_start <= timestamp <= label_end]
    if not matches:
        return None
    return matches[0], matches[-1]


def _clip_window_start(center: int, length: int, n: int) -> int:
    return max(0, min(max(0, n - length), center - length // 2))


def _normalize(values: Sequence[float], offset: float = 100.0, scale: float = 3.0) -> tuple[list[float], float, float]:
    mean = sum(values) / max(1, len(values))
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
    std = math.sqrt(variance)
    denom = std if std > 1e-8 else 1.0
    normalized = [round(offset + scale * ((value - mean) / denom), 6) for value in values]
    return normalized, float(mean), float(std)


def _make_sample(
    *,
    sample_id: str,
    source_file: str,
    timestamps: Sequence[datetime],
    values: Sequence[float],
    start: int,
    length: int,
    split: str,
    seed: int,
    label_interval: tuple[int, int] | None,
    label_timestamp_interval: tuple[datetime, datetime] | None,
) -> TimeSeriesSample:
    end = start + length - 1
    segment_values = list(values[start : end + 1])
    segment_timestamps = list(timestamps[start : end + 1])
    series, original_mean, original_std = _normalize(segment_values)

    events: list[Event] = []
    claims: list[ReferenceClaim] = []
    pattern_types = ["nab_normal"]
    decision_time = None
    if label_interval is not None and label_timestamp_interval is not None:
        label_start, label_end = label_interval
        local_start = max(0, label_start - start)
        local_end = min(length - 1, label_end - start)
        decision_time = local_start
        pattern_types = ["nab_anomaly"]
        event = Event(
            event_id="E1",
            type="anomaly",
            start=local_start,
            end=local_end,
            score=1.0,
            attributes={
                "label_source": "NAB",
                "timestamp_start": _timestamp(label_timestamp_interval[0]),
                "timestamp_end": _timestamp(label_timestamp_interval[1]),
                "source_file": source_file,
            },
        )
        events.append(event)
        claims.append(
            ReferenceClaim(
                claim_id="C1",
                text=f"The NAB annotation marks an anomalous interval from t={local_start} to t={local_end}.",
                claim_type="anomaly",
                label="supported",
                support_events=["E1"],
                support_interval=(local_start, local_end),
            )
        )

    return TimeSeriesSample(
        id=sample_id,
        series=series,
        metadata=TimeSeriesMetadata(
            domain="real_nab",
            length=length,
            pattern_types=pattern_types,
            decision_time=decision_time,
            seed=seed,
            split=split,
            source="Numenta Anomaly Benchmark",
            source_file=source_file,
            timestamp_start=_timestamp(segment_timestamps[0]),
            timestamp_end=_timestamp(segment_timestamps[-1]),
            timestamps=[_timestamp(ts) for ts in segment_timestamps],
            original_values=[round(float(value), 6) for value in segment_values],
            original_mean=round(original_mean, 6),
            original_std=round(original_std, 6),
            label_source="NAB combined_windows.json",
        ),
        events=events,
        question=NAB_QUESTION,
        reference_claims=claims,
    )


def _source_files(nab_root: Path, subsets: Iterable[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for subset in subsets:
        subset_dir = nab_root / "data" / subset
        if not subset_dir.exists():
            continue
        for path in sorted(subset_dir.glob("*.csv")):
            files.append((f"{subset}/{path.name}", path))
    return files


def load_nab_windows(
    nab_root: str | Path,
    *,
    subsets: Iterable[str] = NAB_REAL_SUBSETS,
    length: int = 80,
    max_anomaly_windows: int = 100,
    max_normal_windows: int = 100,
    split: str = "test",
    seed: int = 20260518,
) -> List[TimeSeriesSample]:
    """Convert NAB anomaly windows into fixed-length project-schema samples.

    The emitted `series` is z-normalized per window and shifted near 100 so the
    existing event operators remain comparable with the synthetic benchmark.
    Original values and timestamps are preserved in metadata.
    """

    root = Path(nab_root)
    windows_by_file = _load_windows(root)
    rng = random.Random(seed)
    anomaly_candidates: list[tuple[str, Path, int, int, tuple[datetime, datetime]]] = []
    normal_candidates: list[tuple[str, Path, int]] = []

    for source_file, path in _source_files(root, subsets):
        timestamps, values = _read_nab_csv(path)
        if len(values) < length:
            continue
        label_windows = windows_by_file.get(source_file, [])
        label_indices = []
        for label_start, label_end in label_windows:
            label_idx = _window_indices_for_label(timestamps, label_start, label_end)
            if label_idx is None:
                continue
            label_indices.append((label_idx[0], label_idx[1], (label_start, label_end)))
            center = (label_idx[0] + label_idx[1]) // 2
            anomaly_candidates.append(
                (source_file, path, _clip_window_start(center, length, len(values)), label_idx[0], label_idx[1], (label_start, label_end))
            )

        occupied = [(start, end) for start, end, _ in label_indices]
        for start in range(0, len(values) - length + 1, length):
            end = start + length - 1
            if any(_overlaps(start, end, label_start, label_end) for label_start, label_end in occupied):
                continue
            normal_candidates.append((source_file, path, start))

    rng.shuffle(anomaly_candidates)
    rng.shuffle(normal_candidates)
    anomaly_candidates = anomaly_candidates[:max_anomaly_windows]
    normal_candidates = normal_candidates[:max_normal_windows]

    csv_cache: dict[Path, tuple[list[datetime], list[float]]] = {}

    def read_cached(path: Path) -> tuple[list[datetime], list[float]]:
        if path not in csv_cache:
            csv_cache[path] = _read_nab_csv(path)
        return csv_cache[path]

    samples: list[TimeSeriesSample] = []
    idx = 0
    for source_file, path, start, label_start, label_end, label_ts in anomaly_candidates:
        timestamps, values = read_cached(path)
        samples.append(
            _make_sample(
                sample_id=f"real_nab_{split}_{idx:06d}",
                source_file=source_file,
                timestamps=timestamps,
                values=values,
                start=start,
                length=length,
                split=split,
                seed=seed + idx,
                label_interval=(label_start, label_end),
                label_timestamp_interval=label_ts,
            )
        )
        idx += 1

    for source_file, path, start in normal_candidates:
        timestamps, values = read_cached(path)
        samples.append(
            _make_sample(
                sample_id=f"real_nab_{split}_{idx:06d}",
                source_file=source_file,
                timestamps=timestamps,
                values=values,
                start=start,
                length=length,
                split=split,
                seed=seed + idx,
                label_interval=None,
                label_timestamp_interval=None,
            )
        )
        idx += 1

    samples.sort(key=lambda sample: (sample.metadata.pattern_types[0], sample.metadata.source_file or "", sample.id))
    for i, sample in enumerate(samples):
        sample.id = f"real_nab_{split}_{i:06d}"
        if sample.metadata.seed is not None:
            sample.metadata.seed = seed + i
    return samples


def load_real_dataset(path: str | Path, domain: str) -> List[TimeSeriesSample]:
    if domain.lower() in {"nab", "real_nab"}:
        return load_nab_windows(path)
    raise NotImplementedError(
        f"Real loader for domain '{domain}' is not implemented. "
        "Add license-checked public data and convert it to TimeSeriesSample JSONL."
    )
