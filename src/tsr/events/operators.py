"""Executable time-series operators used by the event-grounded verifier."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from tsr.data.dataset_schema import Event


def _as_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Only univariate time series are supported in the first release.")
    return arr


def _clip_interval(n: int, start: int, end: int) -> Tuple[int, int]:
    start = max(0, min(int(start), n - 1))
    end = max(start, min(int(end), n - 1))
    return start, end


def compute_slope(x: Sequence[float], start: int, end: int) -> float:
    arr = _as_array(x)
    start, end = _clip_interval(len(arr), start, end)
    y = arr[start : end + 1]
    if len(y) < 2:
        return 0.0
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t, y, deg=1)[0])


def compute_volatility(x: Sequence[float], start: int, end: int) -> float:
    arr = _as_array(x)
    start, end = _clip_interval(len(arr), start, end)
    y = arr[start : end + 1]
    if len(y) < 2:
        return 0.0
    return float(np.std(np.diff(y)))


def compute_pct_change(x: Sequence[float], start: int, end: int) -> float:
    arr = _as_array(x)
    start, end = _clip_interval(len(arr), start, end)
    denom = abs(float(arr[start])) + 1e-8
    return float((arr[end] - arr[start]) / denom)


def compute_zscores(x: Sequence[float]) -> np.ndarray:
    arr = _as_array(x)
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)


def strength_from_slope(slope: float) -> str:
    value = abs(float(slope))
    if value < 0.08:
        return "weak"
    if value < 0.16:
        return "moderate"
    return "strong"


def strength_from_score(score: float) -> str:
    score = abs(float(score))
    if score < 0.45:
        return "weak"
    if score < 0.75:
        return "moderate"
    return "strong"


def _mk_event(event_type: str, start: int, end: int, score: float, **attrs) -> Event:
    return Event(
        event_id="",
        type=event_type,
        start=int(start),
        end=int(end),
        score=float(max(0.0, min(1.0, score))),
        attributes=attrs,
    )


def detect_trend(x: Sequence[float], min_len: int = 8, slope_threshold: float = 0.03) -> List[Event]:
    arr = _as_array(x)
    n = len(arr)
    if n < min_len:
        return []
    candidates: List[Tuple[int, int]] = [(0, n - 1)]
    if n >= 2 * min_len:
        candidates.extend([(0, n // 2), (n // 2, n - 1), (n // 4, 3 * n // 4)])

    events: List[Event] = []
    for start, end in candidates:
        slope = compute_slope(arr, start, end)
        if abs(slope) < slope_threshold:
            continue
        event_type = "upward_trend" if slope > 0 else "downward_trend"
        events.append(
            _mk_event(
                event_type,
                start,
                end,
                min(1.0, abs(slope) / max(slope_threshold * 5.0, 1e-8)),
                slope=slope,
                direction="up" if slope > 0 else "down",
                strength=strength_from_slope(slope),
            )
        )
    return events


def detect_local_peaks(x: Sequence[float], prominence: float = 1.0) -> List[Event]:
    arr = _as_array(x)
    events: List[Event] = []
    if len(arr) < 3:
        return events
    for i in range(1, len(arr) - 1):
        left = arr[i] - arr[i - 1]
        right = arr[i] - arr[i + 1]
        prom = min(left, right)
        if prom >= prominence:
            events.append(
                _mk_event(
                    "local_peak",
                    i,
                    i,
                    min(1.0, prom / max(prominence * 3.0, 1e-8)),
                    value=float(arr[i]),
                    prominence=float(prom),
                    strength=strength_from_score(prom / max(prominence * 3.0, 1e-8)),
                )
            )
    return events


def detect_local_troughs(x: Sequence[float], prominence: float = 1.0) -> List[Event]:
    arr = _as_array(x)
    events: List[Event] = []
    if len(arr) < 3:
        return events
    for i in range(1, len(arr) - 1):
        left = arr[i - 1] - arr[i]
        right = arr[i + 1] - arr[i]
        prom = min(left, right)
        if prom >= prominence:
            events.append(
                _mk_event(
                    "local_trough",
                    i,
                    i,
                    min(1.0, prom / max(prominence * 3.0, 1e-8)),
                    value=float(arr[i]),
                    prominence=float(prom),
                    strength=strength_from_score(prom / max(prominence * 3.0, 1e-8)),
                )
            )
    return events


def detect_sharp_changes(x: Sequence[float], window: int = 5, threshold: float = 2.0) -> List[Event]:
    arr = _as_array(x)
    if len(arr) <= window:
        return []
    diffs = arr[window:] - arr[:-window]
    scale = np.std(np.diff(arr)) + 1e-8
    events: List[Event] = []
    for i, delta in enumerate(diffs):
        z = abs(delta) / scale
        if z < threshold:
            continue
        start = i
        end = min(len(arr) - 1, i + window)
        event_type = "sharp_rise" if delta > 0 else "sharp_drop"
        events.append(
            _mk_event(
                event_type,
                start,
                end,
                min(1.0, z / (threshold * 2.0)),
                magnitude=float(abs(delta)),
                direction="up" if delta > 0 else "down",
                strength=strength_from_score(z / (threshold * 2.0)),
            )
        )
    return events


def detect_volatility_shift(
    x: Sequence[float],
    window: int = 10,
    ratio_threshold: float = 1.8,
) -> List[Event]:
    arr = _as_array(x)
    n = len(arr)
    if n < window * 2:
        return []
    best = None
    for split in range(window, n - window):
        before = compute_volatility(arr, max(0, split - window), split - 1)
        after = compute_volatility(arr, split, min(n - 1, split + window - 1))
        small = max(min(before, after), 1e-8)
        ratio = max(before, after) / small
        if best is None or ratio > best[0]:
            best = (ratio, split, before, after)
    if best is None or best[0] < ratio_threshold:
        return []
    ratio, split, before, after = best
    direction = "increase" if after > before else "decrease"
    return [
        _mk_event(
            "volatility_shift",
            split,
            n - 1,
            min(1.0, ratio / (ratio_threshold * 2.0)),
            direction=direction,
            ratio=float(ratio),
            before_volatility=float(before),
            after_volatility=float(after),
            strength=strength_from_score(ratio / (ratio_threshold * 2.0)),
        )
    ]


def detect_change_points(x: Sequence[float], penalty: float = 5.0) -> List[Event]:
    arr = _as_array(x)
    n = len(arr)
    if n < 12:
        return []
    global_std = float(np.std(arr) + 1e-8)
    best = None
    for split in range(4, n - 4):
        before = float(np.mean(arr[:split]))
        after = float(np.mean(arr[split:]))
        score = abs(after - before) / global_std
        if best is None or score > best[0]:
            best = (score, split, before, after)
    if best is None or best[0] < 1.0:
        return []
    score, split, before, after = best
    return [
        _mk_event(
            "change_point",
            split,
            split,
            min(1.0, score / max(penalty / 2.0, 1e-8)),
            before_mean=before,
            after_mean=after,
            magnitude=float(abs(after - before)),
            strength=strength_from_score(score / max(penalty / 2.0, 1e-8)),
        )
    ]


def detect_seasonality(
    x: Sequence[float],
    min_period: int = 4,
    max_period: int = 24,
) -> List[Event]:
    arr = _as_array(x)
    n = len(arr)
    centered = arr - np.mean(arr)
    if n < max(min_period * 3, 12) or np.std(centered) < 1e-8:
        return []
    best = None
    max_period = min(max_period, n // 2)
    for period in range(min_period, max_period + 1):
        a = centered[:-period]
        b = centered[period:]
        if len(a) < 3:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(corr):
            continue
        if best is None or corr > best[0]:
            best = (corr, period)
    if best is None or best[0] < 0.45:
        return []
    corr, period = best
    return [
        _mk_event(
            "seasonality",
            0,
            n - 1,
            min(1.0, corr),
            period=int(period),
            strength=strength_from_score(corr),
        )
    ]


def detect_anomalies(x: Sequence[float], z_threshold: float = 3.0) -> List[Event]:
    arr = _as_array(x)
    zscores = compute_zscores(arr)
    events: List[Event] = []
    for i, z in enumerate(zscores):
        if abs(float(z)) >= z_threshold:
            events.append(
                _mk_event(
                    "anomaly",
                    i,
                    i,
                    min(1.0, abs(float(z)) / (z_threshold * 2.0)),
                    z_score=float(z),
                    direction="up" if z > 0 else "down",
                    strength=strength_from_score(abs(float(z)) / (z_threshold * 2.0)),
                )
            )
    return events
