"""Jersey number recognition with EasyOCR and temporal smoothing."""

from collections import Counter, defaultdict, deque
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from loguru import logger


class JerseyNumberReader:
    def __init__(
        self,
        languages: list[str] | None = None,
        min_confidence: float = 0.4,
        smoothing_window: int = 15,
        gpu: bool = True,
    ):
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))
        self._results: dict[int, str] = {}
        self._reader = None
        self._languages = languages or ["en"]
        self._gpu = gpu

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
            logger.info("EasyOCR reader initialized")

    def _crop_torso(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        torso_y1 = y1 + int(h * 0.15)
        torso_y2 = y1 + int(h * 0.55)
        crop = frame[torso_y1:torso_y2, x1:x2]
        if crop.size == 0:
            return frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _parse_number(self, text: str) -> Optional[str]:
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            return None
        num = int(digits)
        if 1 <= num <= 99:
            return str(num)
        return None

    def read_crop(self, crop: np.ndarray) -> tuple[str, float]:
        self._ensure_reader()
        results = self._reader.readtext(crop, allowlist="0123456789")
        best_num, best_conf = "UNKNOWN", 0.0
        for _bbox, text, conf in results:
            parsed = self._parse_number(text)
            if parsed and conf > best_conf:
                best_num, best_conf = parsed, conf
        if best_conf < self.min_confidence:
            return "UNKNOWN", best_conf
        return best_num, best_conf

    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        person_class_ids: tuple[int, ...] = (1, 2),
    ) -> dict[int, str]:
        if detections.tracker_id is None:
            return self._results

        for idx in range(len(detections)):
            if detections.class_id[idx] not in person_class_ids:
                continue
            track_id = int(detections.tracker_id[idx])
            crop = self._crop_torso(frame, detections.xyxy[idx])
            number, conf = self.read_crop(crop)
            if number != "UNKNOWN":
                self._history[track_id].append(number)

            if self._history[track_id]:
                voted = Counter(self._history[track_id]).most_common(1)[0][0]
                self._results[track_id] = voted
            elif track_id not in self._results:
                self._results[track_id] = "UNKNOWN"

        return dict(self._results)

    def get_jersey(self, track_id: int) -> str:
        return self._results.get(track_id, "UNKNOWN")
