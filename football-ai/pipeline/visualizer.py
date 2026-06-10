"""Professional football analytics frame visualization."""

from __future__ import annotations

import numpy as np
import supervision as sv

from analytics.ball_tracker import BallAnnotator, BallTracker
from utils.annotators import build_color_lookup


class FrameVisualizer:
    """Ellipse markers at feet, track IDs above players, triangle for ball."""

    def __init__(self, colors: dict[str, str], class_names: list[str], class_id_map: dict[str, int]):
        self.class_names = class_names
        self.class_id_map = class_id_map
        palette = sv.ColorPalette.from_hex([
            colors["team_a"],
            colors["team_b"],
            colors["goalkeeper"],
            colors["referee"],
        ])
        self.ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
        self.label = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=5,
            text_thickness=1,
            text_scale=0.55,
            text_position=sv.Position.TOP_CENTER,
        )
        self.triangle = sv.TriangleAnnotator(
            color=sv.Color.from_hex(colors["ball"]),
            base=20,
            height=16,
        )
        self.ball_label = sv.LabelAnnotator(
            color=sv.Color.from_hex(colors["ball"]),
            text_color=sv.Color.from_hex("#000000"),
            text_padding=4,
            text_thickness=1,
            text_scale=0.45,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.ball_tracker = BallTracker(buffer_size=20)
        self.ball_trail = BallAnnotator(radius=7, buffer_size=12, thickness=2)

    def _labels(self, track_ids: np.ndarray, class_ids: np.ndarray) -> list[str]:
        labels = []
        for tid, cid in zip(track_ids, class_ids):
            name = self.class_names[int(cid)].capitalize() if int(cid) < len(self.class_names) else "Object"
            labels.append(f"#{int(tid)}\n{name}")
        return labels

    def annotate(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        goalkeepers: sv.Detections,
        referees: sv.Detections,
        ball: sv.Detections,
        team_ids: np.ndarray,
        frame_id: int = 0,
    ) -> np.ndarray:
        out = frame.copy()

        person_parts = [d for d in (players, goalkeepers, referees) if len(d) > 0]
        if person_parts:
            persons = person_parts[0] if len(person_parts) == 1 else sv.Detections.merge(person_parts)
            n_players = len(players)
            n_gks = len(goalkeepers)
            person_team = np.concatenate([
                team_ids[:n_players] if n_players else np.array([]),
                team_ids[n_players:n_players + n_gks] if n_gks else np.array([]),
                np.full(len(referees), -1),
            ]) if len(team_ids) else np.full(len(persons), -1)

            color_lookup = build_color_lookup(
                persons.class_id, person_team, self.class_names, self.class_id_map,
            )
            labels = self._labels(persons.tracker_id, persons.class_id)
            out = self.ellipse.annotate(out, persons, custom_color_lookup=color_lookup)
            out = self.label.annotate(out, persons, labels=labels, custom_color_lookup=color_lookup)

        if len(ball) > 0:
            ball = self.ball_tracker.update(ball, frame_id)
            out = self.triangle.annotate(out, ball)
            out = self.ball_label.annotate(out, ball, labels=["Ball"] * len(ball))
            out = self.ball_trail.annotate(out, ball)

        return out
