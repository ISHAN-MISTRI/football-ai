"""Team classification via SigLIP + UMAP + KMeans (adapted from sports-main)."""

from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")
SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch: list[V] = []
    for element in sequence:
        current_batch.append(element)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


class TeamClassifier:
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 16,
        umap_components: int = 3,
        n_clusters: int = 2,
        siglip_model: str = "google/siglip-base-patch16-224",
    ):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(siglip_model).to(device)
        self.processor = AutoProcessor.from_pretrained(siglip_model)
        self.reducer = umap.UMAP(n_components=umap_components)
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self._fitted = False

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 768))
        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        data: list[np.ndarray] = []
        with torch.no_grad():
            for batch in create_batches(crops_pil, self.batch_size):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        if len(data) < 10:
            from loguru import logger
            logger.warning("Need at least 10 player crops to fit team classifier. Bypassing fit and defaulting to team 0.")
            self._fitted = True
            return
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])
        if not self._fitted:
            raise RuntimeError("TeamClassifier must be fitted before predict()")
        # If KMeans was not fitted due to insufficient crops
        if not hasattr(self.cluster_model, "cluster_centers_"):
            return np.zeros(len(crops), dtype=int)
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([])
    if len(players) == 0 or len(players_team_id) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0 = players_xy[players_team_id == 0]
    team_1 = players_xy[players_team_id == 1]
    team_0_centroid = team_0.mean(axis=0) if len(team_0) else players_xy.mean(axis=0)
    team_1_centroid = team_1.mean(axis=0) if len(team_1) else players_xy.mean(axis=0)

    result = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        result.append(0 if dist_0 < dist_1 else 1)
    return np.array(result)


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
