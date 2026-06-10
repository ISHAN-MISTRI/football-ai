"""SigLIP + UMAP + KMeans team classification."""

from __future__ import annotations

from typing import Optional

import numpy as np
import supervision as sv
from loguru import logger
from tqdm import tqdm

from analytics.team_classifier import (
    TeamClassifier,
    get_crops,
    resolve_goalkeepers_team_id,
)
from pipeline.detector import FootballDetector
from utils.device import clear_cuda_cache


class TeamAssigner:
    def __init__(
        self,
        detector: FootballDetector,
        device: str = "cuda",
        stride: int = 60,
        batch_size: int = 16,
        siglip_model: str = "google/siglip-base-patch16-224",
        umap_components: int = 3,
        n_clusters: int = 2,
    ):
        self.detector = detector
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        self.siglip_model = siglip_model
        self.umap_components = umap_components
        self.n_clusters = n_clusters
        self.classifier: Optional[TeamClassifier] = None
        self.track_teams: dict[int, int] = {}

    def fit(self, video_path: str) -> None:
        player_id = self.detector.class_names.index("player")
        frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=self.stride)
        video_info = sv.VideoInfo.from_video_path(video_path)
        total = max(1, video_info.total_frames // self.stride)

        crops = []
        for frame in tqdm(frame_gen, total=total, desc="Fitting team classifier"):
            detections = self.detector.predict(frame)
            players = detections[detections.class_id == player_id]
            crops.extend(get_crops(frame, players))

        if len(crops) < 10:
            logger.warning("Few player crops — team colors may be unreliable")

        self.classifier = TeamClassifier(
            device=self.device,
            batch_size=self.batch_size,
            umap_components=self.umap_components,
            n_clusters=self.n_clusters,
            siglip_model=self.siglip_model,
        )
        self.classifier.fit(crops)
        clear_cuda_cache()
        logger.info(f"Team classifier fitted on {len(crops)} player crops")

    def assign(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        goalkeepers: sv.Detections,
    ) -> np.ndarray:
        """Return per-detection team id array aligned with merged [players, goalkeepers]."""
        player_teams = np.array([], dtype=int)
        if len(players) > 0 and self.classifier:
            crops = get_crops(frame, players)
            player_teams = self.classifier.predict(crops)
            for tid, team in zip(players.tracker_id, player_teams):
                self.track_teams[int(tid)] = int(team)

        gk_teams = resolve_goalkeepers_team_id(players, player_teams, goalkeepers)
        for tid, team in zip(goalkeepers.tracker_id, gk_teams):
            self.track_teams[int(tid)] = int(team)

        if len(players) == 0 and len(goalkeepers) == 0:
            return np.array([])
        parts = [a for a in (player_teams, gk_teams) if len(a)]
        return np.concatenate(parts) if parts else np.array([])
