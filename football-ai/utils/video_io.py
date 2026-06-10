"""Video I/O helpers (avoids VideoSink shutdown crash on some Windows builds)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv


class VideoWriter:
    def __init__(self, path: Path | str, video_info: sv.VideoInfo):
        self.path = str(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.path,
            fourcc,
            video_info.fps,
            (video_info.width, video_info.height),
        )

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
