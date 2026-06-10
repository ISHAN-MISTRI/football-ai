"""Ball tracking utilities adapted from sports-main."""

from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    def __init__(self, radius: int = 6, buffer_size: int = 10, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib("jet", buffer_size)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return frame
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, positions in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            r = self.radius if len(self.buffer) == 1 else int(1 + i * (self.radius - 1) / (len(self.buffer) - 1))
            for center in positions:
                frame = cv2.circle(frame, tuple(center), r, color.as_bgr(), self.thickness)
        return frame


class BallTracker:
    def __init__(self, buffer_size: int = 20):
        self.buffer: deque = deque(maxlen=buffer_size)
        self.history: list[tuple[int, float, float]] = []

    def update(self, detections: sv.Detections, frame_id: int = 0) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        all_points = np.concatenate(self.buffer) if self.buffer else xy
        centroid = np.mean(all_points, axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = int(np.argmin(distances))
        best = detections[[index]]
        center = best.get_anchors_coordinates(sv.Position.CENTER)[0]
        self.history.append((frame_id, float(center[0]), float(center[1])))
        return best

    def get_history(self) -> list[tuple[int, float, float]]:
        return list(self.history)
