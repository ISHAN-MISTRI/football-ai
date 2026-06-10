"""YOLO11 football object detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
from loguru import logger
from ultralytics import YOLO

from utils.config import load_config, resolve_path


class FootballDetector:
    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        half: bool = True,
        class_names: list[str] | None = None,
    ):
        config = load_config()
        path = Path(model_path) if model_path else resolve_path(config, "detector_model")
        if not path.exists():
            path = Path(config["_project_root"]) / "models/detector/best26x.pt"
        if not path.exists():
            raise FileNotFoundError(f"Detector weights not found: {path}")

        self.class_names = class_names or config.get("detection", {}).get("class_names") or ["ball", "goalkeeper", "player", "referee"]
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.half = half and device == "cuda"

        logger.info(f"Loading detector: {path}")
        self.model = YOLO(str(path))
        self.model.to(device)

    def predict(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            half=self.half,
            device=self.device,
        )[0]
        return sv.Detections.from_ultralytics(result)
