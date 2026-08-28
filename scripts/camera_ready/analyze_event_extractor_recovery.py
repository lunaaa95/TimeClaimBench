#!/usr/bin/env python3
"""Evaluate recovery of sparse designed events by the frozen event extractor.

This diagnostic measures whether the extracted graph contains an event with the
same type as the designed primary event and interval IoU >= 0.5.  It is a recall
diagnostic: the sparse synthetic annotations do not exhaustively label every
valid local event, so they cannot identify false-positive extractions.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data/synthetic/test.jsonl"
GRAPH_PATH = ROOT / "data/processed/test_graphs.jsonl"
OUT_DIR = ROOT / "outputs/analysis/camera_ready"
IOU_THRESHOLD = 0.5


def read_jsonl(path: Path, key: str) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row[key]): row for row in rows}


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return intersection / union if union else 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_rows = read_jsonl(GOLD_PATH, "id")
    graph_rows = read_jsonl(GRAPH_PATH, "series_id")
    if set(gold_rows) != set(graph_rows):
        raise ValueError(
            "Synthetic test and extracted-graph IDs differ: "
            f"missing_graphs={sorted(set(gold_rows) - set(graph_rows))[:5]}, "
            f"extra_graphs={sorted(set(graph_rows) - set(gold_rows))[:5]}"
        )
    by_type: dict[str, list[float]] = defaultdict(list)

    for series_id, gold_row in gold_rows.items():
        gold_events = gold_row["events"]
        if len(gold_events) != 1:
            raise ValueError(f"Expected one designed event for {series_id}, found {len(gold_events)}")
        gold = gold_events[0]
        same_type = [event for event in graph_rows[series_id]["events"] if event["type"] == gold["type"]]
        best_iou = max(
            (
                interval_iou(
                    (int(gold["start"]), int(gold["end"])),
                    (int(event["start"]), int(event["end"])),
                )
                for event in same_type
            ),
            default=0.0,
        )
        by_type[str(gold["type"])].append(best_iou)

    rows = []
    all_ious = []
    for event_type, ious in sorted(by_type.items()):
        all_ious.extend(ious)
        rows.append(
            {
                "event_type": event_type,
                "n_gold": len(ious),
                "recovered": sum(iou >= IOU_THRESHOLD for iou in ious),
                "recall_at_iou_0_5": sum(iou >= IOU_THRESHOLD for iou in ious) / len(ious),
                "mean_best_iou": sum(ious) / len(ious),
            }
        )

    summary = {
        "definition": "Exact event type and interval IoU >= 0.5.",
        "n_series": len(all_ious),
        "n_recovered": sum(iou >= IOU_THRESHOLD for iou in all_ious),
        "micro_recall": sum(iou >= IOU_THRESHOLD for iou in all_ious) / len(all_ious),
        "macro_recall": sum(row["recall_at_iou_0_5"] for row in rows) / len(rows),
        "macro_mean_best_iou": sum(row["mean_best_iou"] for row in rows) / len(rows),
        "per_event_type": rows,
        "precision_constraint": (
            "Each synthetic series labels one designed primary event, while the extractor may detect "
            "additional valid local events; unmatched extracted events are therefore not identifiable false positives."
        ),
    }

    with (OUT_DIR / "event_extractor_recovery.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "event_extractor_recovery.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
