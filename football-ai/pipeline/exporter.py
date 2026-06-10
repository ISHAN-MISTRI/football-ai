"""CSV tracking export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import supervision as sv

from utils.dataset import get_class_name

CSV_COLUMNS = [
    "frame", "timestamp", "track_id", "class", "team", "jersey_number",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence",
]


def detections_to_rows(
    detections: sv.Detections,
    frame_id: int,
    timestamp: float,
    class_names: list[str],
    track_teams: dict[int, int],
) -> list[dict]:
    rows = []
    if detections.tracker_id is None:
        return rows

    for i in range(len(detections)):
        tid = int(detections.tracker_id[i])
        cls_id = int(detections.class_id[i])
        cls_name = get_class_name(cls_id, class_names)
        x1, y1, x2, y2 = detections.xyxy[i]
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        team = track_teams.get(tid, -1)
        team_label = f"Team {'A' if team == 0 else 'B'}" if team >= 0 else ""

        rows.append({
            "frame": frame_id,
            "timestamp": round(timestamp, 4),
            "track_id": tid,
            "class": cls_name,
            "team": team_label,
            "jersey_number": "",
            "bbox_x1": round(float(x1), 2),
            "bbox_y1": round(float(y1), 2),
            "bbox_x2": round(float(x2), 2),
            "bbox_y2": round(float(y2), 2),
            "confidence": round(conf, 4),
        })
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
