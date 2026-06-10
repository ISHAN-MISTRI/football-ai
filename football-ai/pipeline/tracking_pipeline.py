"""ByteTrack + OSNet ReID tracking (BoT-SORT-style identity recovery)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from models.tracker.reid_manager import ReIDIdentityManager


@dataclass
class TrackHistory:
    """Per-track centroid history for analytics export."""

    positions: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)

    def update(self, detections: sv.Detections, frame_id: int) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return
        feet = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for tid, (x, y) in zip(detections.tracker_id, feet):
            tid = int(tid)
            self.positions.setdefault(tid, []).append((frame_id, float(x), float(y)))

    def get(self, track_id: int) -> list[tuple[int, float, float]]:
        return self.positions.get(track_id, [])


class TrackingPipeline:
    """
    Two-stage tracking:
        1. ByteTrack — short-term bbox association
        2. OSNet ReID — long-term identity recovery after occlusion / exit
    """

    def __init__(
        self,
        device: str = "cuda",
        minimum_consecutive_frames: int = 5,
        lost_track_buffer: int = 90,
        reid_enabled: bool = True,
        reid_threshold: float = 0.65,
        reid_model: str = "osnet_x0_25",
        person_class_ids: tuple[int, ...] = (1, 2, 3),
    ):
        self.person_class_ids = person_class_ids
        self.byte_tracker = sv.ByteTrack(minimum_consecutive_frames=minimum_consecutive_frames)
        self.history = TrackHistory()
        self.reid = (
            ReIDIdentityManager(
                device=device if device == "cuda" else "cpu",
                model_name=reid_model,
                similarity_threshold=reid_threshold,
                lost_buffer_frames=lost_track_buffer,
                person_class_ids=person_class_ids,
            )
            if reid_enabled
            else None
        )

    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        frame_id: int,
        team_ids: dict[int, int] | None = None,
    ) -> sv.Detections:
        detections = self.byte_tracker.update_with_detections(detections)

        persons = detections[np.isin(detections.class_id, self.person_class_ids)]
        others = detections[~np.isin(detections.class_id, self.person_class_ids)]

        if self.reid and len(persons) > 0:
            persons = self.reid.update(frame, persons, team_ids=team_ids or {})

        if len(persons) and len(others):
            detections = sv.Detections.merge([persons, others])
        elif len(persons):
            detections = persons
        else:
            detections = others

        self.history.update(detections, frame_id)
        return detections

    def get_stats(self) -> dict:
        if self.reid:
            return self.reid.get_stats_dict()
        return {}
