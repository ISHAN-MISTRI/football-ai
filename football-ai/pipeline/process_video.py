"""End-to-end video pipeline: detect → track → team → annotate → export."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv
from loguru import logger
from tqdm import tqdm

from pipeline.detector import FootballDetector
from pipeline.exporter import detections_to_rows, save_csv
from pipeline.team_assigner import TeamAssigner
from pipeline.tracking_pipeline import TrackingPipeline
from pipeline.visualizer import FrameVisualizer
from utils.config import load_config, resolve_path
from utils.dataset import get_class_id_map
from utils.device import clear_cuda_cache, resolve_device
from utils.video_io import VideoWriter


@dataclass
class ProcessingResult:
    output_video: Path
    csv_path: Path
    session_id: str
    stats: dict = field(default_factory=dict)


class VideoProcessor:
    def __init__(self, config: dict | None = None, device: str = "auto"):
        self.config = config or load_config()
        self.device = resolve_device(device)

        det_cfg = self.config["detection"]
        self.detector = FootballDetector(
            device=self.device,
            imgsz=det_cfg.get("imgsz", 640),
            conf=det_cfg.get("conf_threshold", 0.25),
            half=det_cfg.get("half_precision", True),
        )
        self.class_names = self.detector.class_names
        self.class_id_map = get_class_id_map(self.class_names)

        trk_cfg = self.config["tracking"]
        reid_cfg = self.config["reid"]
        self.tracker = TrackingPipeline(
            device=self.device,
            minimum_consecutive_frames=trk_cfg.get("minimum_consecutive_frames", 5),
            lost_track_buffer=trk_cfg.get("lost_track_buffer", 90),
            reid_enabled=reid_cfg.get("enabled", True),
            reid_threshold=reid_cfg.get("similarity_threshold", 0.65),
            reid_model=reid_cfg.get("model_name", "osnet_x0_25"),
            person_class_ids=tuple(
                self.class_id_map[name]
                for name in ("goalkeeper", "player", "referee")
                if name in self.class_id_map
            ),
        )

        tc_cfg = self.config["team_classification"]
        self.team_assigner = TeamAssigner(
            detector=self.detector,
            device=self.device,
            stride=tc_cfg.get("stride", 60),
            batch_size=tc_cfg.get("batch_size", 16),
        )

        self.visualizer = FrameVisualizer(
            colors=self.config["annotation"]["colors"],
            class_names=self.class_names,
            class_id_map=self.class_id_map,
        )

    def process(
        self,
        source_video: str | Path,
        output_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> ProcessingResult:
        source = Path(source_video)
        out_root = Path(output_dir) if output_dir else resolve_path(self.config, "outputs_dir")
        out_root.mkdir(parents=True, exist_ok=True)

        session_id = str(uuid.uuid4())[:8]
        output_video = out_root / f"tracked_{session_id}.mp4"
        csv_path = out_root / self.config["export"]["csv_filename"]

        logger.info(f"Fitting team classifier on {source.name}")
        self.team_assigner.fit(str(source))

        video_info = sv.VideoInfo.from_video_path(str(source))
        total_frames = video_info.total_frames
        if max_frames:
            total_frames = min(total_frames, max_frames)

        ball_id = self.class_id_map.get("ball", -1)
        player_id = self.class_id_map.get("player", -1)
        gk_id = self.class_id_map.get("goalkeeper", -1)
        ref_id = self.class_id_map.get("referee", -1)

        csv_rows: list[dict] = []
        stats = {
            "frames_processed": 0,
            "ball_detections": 0,
            "unique_track_ids": set(),
        }

        logger.info(f"Processing {total_frames} frames → {output_video}")

        with VideoWriter(output_video, video_info) as writer:
            for frame_idx, frame in enumerate(
                tqdm(sv.get_video_frames_generator(str(source)), total=total_frames, desc="Tracking")
            ):
                if max_frames and frame_idx >= max_frames:
                    break

                timestamp = frame_idx / video_info.fps
                detections = self.detector.predict(frame)
                detections = self.tracker.update(
                    frame, detections, frame_idx, self.team_assigner.track_teams,
                )

                players = detections[detections.class_id == player_id]
                goalkeepers = detections[detections.class_id == gk_id]
                referees = detections[detections.class_id == ref_id]
                ball = detections[detections.class_id == ball_id]

                team_ids = self.team_assigner.assign(frame, players, goalkeepers)

                annotated = self.visualizer.annotate(
                    frame, players, goalkeepers, referees, ball, team_ids, frame_idx,
                )
                writer.write(annotated)

                csv_rows.extend(
                    detections_to_rows(
                        detections, frame_idx, timestamp,
                        self.class_names, self.team_assigner.track_teams,
                    )
                )
                if len(ball):
                    stats["ball_detections"] += len(ball)
                if detections.tracker_id is not None:
                    stats["unique_track_ids"].update(int(t) for t in detections.tracker_id)
                stats["frames_processed"] = frame_idx + 1

        stats["reid"] = self.tracker.get_stats()
        stats["unique_track_ids"] = len(stats["unique_track_ids"])
        stats["track_history_count"] = len(self.tracker.history.positions)

        logger.info(f"Done — video: {output_video}, csv: {csv_path}")
        save_csv(csv_rows, csv_path)
        return ProcessingResult(
            output_video=output_video,
            csv_path=csv_path,
            session_id=session_id,
            stats=stats,
        )
