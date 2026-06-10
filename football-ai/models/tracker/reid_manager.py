"""ReID identity manager: OSNet embeddings + identity recovery for lost tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch
from loguru import logger


@dataclass
class IdentityRecord:
    track_id: int
    embedding: np.ndarray
    team_id: int = -1
    jersey_number: str = "UNKNOWN"
    last_frame: int = 0
    class_id: int = 2
    active: bool = True


@dataclass
class ReIDStats:
    id_switches: int = 0
    id_recoveries: int = 0
    total_tracks: int = 0


class ReIDIdentityManager:
    """
    Layer on top of ByteTrack for persistent player identities.

    Pipeline:
        ByteTrack ID -> OSNet embedding -> compare with lost identity bank -> restore or assign
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "osnet_x0_25",
        similarity_threshold: float = 0.65,
        lost_buffer_frames: int = 90,
        max_lost_identities: int = 50,
        ema_alpha: float = 0.85,
        update_interval: int = 3,
        person_class_ids: tuple[int, ...] = (1, 2),
    ):
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.lost_buffer_frames = lost_buffer_frames
        self.max_lost_identities = max_lost_identities
        self.ema_alpha = ema_alpha
        self.update_interval = update_interval
        self.person_class_ids = person_class_ids
        self.stats = ReIDStats()
        self._next_id = 1
        self._active: dict[int, IdentityRecord] = {}
        self._lost: dict[int, IdentityRecord] = {}
        self._byte_to_persistent: dict[int, int] = {}
        self._frame_id = 0
        self._extractor = self._build_extractor(model_name)

    def _build_extractor(self, model_name: str):
        try:
            import torchreid

            extractor = torchreid.utils.FeatureExtractor(
                model_name=model_name,
                model_path="",
                device=self.device,
            )
            logger.info(f"ReID: loaded {model_name} on {self.device}")
            return extractor
        except Exception as exc:
            logger.warning(f"torchreid unavailable ({exc}), using SigLIP fallback via torchvision")
            return None

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self._extractor is not None:
            feat = self._extractor([crop])
            if isinstance(feat, torch.Tensor):
                emb = feat.cpu().numpy().flatten()
            else:
                emb = np.asarray(feat).flatten()
        else:
            from torchvision import models, transforms

            if not hasattr(self, "_fallback_model"):
                self._fallback_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                self._fallback_model.classifier = torch.nn.Identity()
                self._fallback_model = self._fallback_model.to(self.device).eval()
                self._fallback_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._fallback_transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._fallback_model(tensor).cpu().numpy().flatten()

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else emb

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _allocate_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        self.stats.total_tracks += 1
        return new_id

    def _update_ema(self, record: IdentityRecord, new_emb: np.ndarray) -> None:
        record.embedding = self.ema_alpha * record.embedding + (1 - self.ema_alpha) * new_emb
        norm = np.linalg.norm(record.embedding)
        if norm > 1e-8:
            record.embedding = record.embedding / norm

    def _mark_lost(self, persistent_id: int, frame_id: int) -> None:
        if persistent_id not in self._active:
            return
        record = self._active.pop(persistent_id)
        record.active = False
        record.last_frame = frame_id
        self._lost[persistent_id] = record
        if len(self._lost) > self.max_lost_identities:
            oldest = min(self._lost.values(), key=lambda r: r.last_frame)
            self._lost.pop(oldest.track_id, None)

    def _try_recover(self, embedding: np.ndarray, class_id: int, jersey: str) -> Optional[int]:
        best_id: Optional[int] = None
        best_sim = self.similarity_threshold

        for lost_id, record in self._lost.items():
            if record.class_id != class_id:
                continue
            if self._frame_id - record.last_frame > self.lost_buffer_frames:
                continue
            sim = self._cosine_similarity(embedding, record.embedding)
            if jersey != "UNKNOWN" and record.jersey_number == jersey:
                sim += 0.1
            if sim > best_sim:
                best_sim = sim
                best_id = lost_id

        if best_id is not None:
            record = self._lost.pop(best_id)
            record.active = True
            record.last_frame = self._frame_id
            self._active[best_id] = record
            self.stats.id_recoveries += 1
            logger.debug(f"ReID recovered identity {best_id} (sim={best_sim:.3f})")
        return best_id

    def update(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        jersey_numbers: Optional[dict[int, str]] = None,
        team_ids: Optional[dict[int, int]] = None,
    ) -> sv.Detections:
        """Remap ByteTrack IDs to persistent ReID IDs for person detections."""
        self._frame_id += 1
        jersey_numbers = jersey_numbers or {}
        team_ids = team_ids or {}

        if detections.tracker_id is None or len(detections) == 0:
            return detections

        person_mask = np.isin(detections.class_id, self.person_class_ids)
        persistent_ids = np.array(detections.tracker_id, dtype=int).copy()
        seen_persistent: set[int] = set()
        byte_ids_this_frame: set[int] = set()

        for idx in range(len(detections)):
            byte_id = int(detections.tracker_id[idx])
            byte_ids_this_frame.add(byte_id)
            class_id = int(detections.class_id[idx])

            if not person_mask[idx]:
                persistent_ids[idx] = byte_id
                continue

            crop = sv.crop_image(frame, detections.xyxy[idx])
            should_update = (
                byte_id not in self._byte_to_persistent
                or self._frame_id % self.update_interval == 0
            )
            embedding = self._extract_embedding(crop) if should_update else None
            jersey = jersey_numbers.get(byte_id, "UNKNOWN")
            team = team_ids.get(byte_id, -1)

            if byte_id in self._byte_to_persistent:
                pid = self._byte_to_persistent[byte_id]
                if pid in self._active and embedding is not None:
                    self._update_ema(self._active[pid], embedding)
                    self._active[pid].last_frame = self._frame_id
                    if jersey != "UNKNOWN":
                        self._active[pid].jersey_number = jersey
                    if team >= 0:
                        self._active[pid].team_id = team
                persistent_ids[idx] = pid
                seen_persistent.add(pid)
                continue

            recovered = None
            if embedding is not None:
                recovered = self._try_recover(embedding, class_id, jersey)

            if recovered is not None:
                pid = recovered
                self._byte_to_persistent[byte_id] = pid
                if embedding is not None:
                    self._update_ema(self._active[pid], embedding)
            else:
                pid = self._allocate_id()
                self._byte_to_persistent[byte_id] = pid
                self._active[pid] = IdentityRecord(
                    track_id=pid,
                    embedding=embedding if embedding is not None else np.zeros(512, dtype=np.float32),
                    team_id=team,
                    jersey_number=jersey,
                    last_frame=self._frame_id,
                    class_id=class_id,
                )

            persistent_ids[idx] = pid
            seen_persistent.add(pid)

        stale_byte = [b for b in self._byte_to_persistent if b not in byte_ids_this_frame]
        for byte_id in stale_byte:
            pid = self._byte_to_persistent.pop(byte_id)
            if pid in seen_persistent:
                self._byte_to_persistent[byte_id] = pid
            elif pid in self._active:
                self._mark_lost(pid, self._frame_id)

        expired = [
            lid for lid, rec in self._lost.items()
            if self._frame_id - rec.last_frame > self.lost_buffer_frames
        ]
        for lid in expired:
            self._lost.pop(lid, None)

        detections.tracker_id = persistent_ids
        return detections

    def get_stats_dict(self) -> dict:
        return {
            "id_switches": self.stats.id_switches,
            "id_recoveries": self.stats.id_recoveries,
            "total_tracks": self.stats.total_tracks,
            "active_identities": len(self._active),
            "lost_identities": len(self._lost),
        }
