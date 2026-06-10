"""Professional-style annotation helpers adapted from sports-main."""

import numpy as np
import supervision as sv


def build_annotators(colors: dict[str, str]) -> dict:
    palette = [
        colors["team_a"],
        colors["team_b"],
        colors["goalkeeper"],
        colors["referee"],
    ]
    color_palette = sv.ColorPalette.from_hex(palette)

    return {
        "ellipse": sv.EllipseAnnotator(color=color_palette, thickness=2),
        "ellipse_label": sv.LabelAnnotator(
            color=color_palette,
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=6,
            text_thickness=1,
            text_scale=0.6,
            text_position=sv.Position.BOTTOM_CENTER,
        ),
        "triangle": sv.TriangleAnnotator(
            color=sv.Color.from_hex(colors["ball"]),
            base=18,
            height=14,
        ),
        "ball_label": sv.LabelAnnotator(
            color=sv.Color.from_hex(colors["ball"]),
            text_color=sv.Color.from_hex("#000000"),
            text_padding=4,
            text_thickness=1,
            text_scale=0.5,
            text_position=sv.Position.TOP_CENTER,
        ),
    }


def build_color_lookup(
    class_ids: np.ndarray,
    team_ids: np.ndarray,
    class_names: list[str],
    class_id_map: dict[str, int],
) -> np.ndarray:
    """Map detections to color indices: 0=team_a, 1=team_b, 2=gk, 3=referee."""
    lookup = np.zeros(len(class_ids), dtype=int)
    player_id = class_id_map.get("player", -1)
    gk_id = class_id_map.get("goalkeeper", -1)
    ref_id = class_id_map.get("referee", -1)

    for i, (cls_id, team_id) in enumerate(zip(class_ids, team_ids)):
        if cls_id == player_id:
            lookup[i] = int(team_id) if team_id >= 0 else 0
        elif cls_id == gk_id:
            lookup[i] = 2
        elif cls_id == ref_id:
            lookup[i] = 3
        else:
            lookup[i] = 0
    return lookup


def format_labels(
    track_ids: np.ndarray,
    class_ids: np.ndarray,
    jersey_numbers: dict[int, str],
    class_names: list[str],
) -> list[str]:
    labels = []
    for track_id, class_id in zip(track_ids, class_ids):
        cls_name = class_names[class_id] if class_id < len(class_names) else "object"
        cls_name = cls_name.capitalize()
        jersey = jersey_numbers.get(int(track_id), "")
        if jersey and jersey != "UNKNOWN":
            labels.append(f"#{jersey}\n{cls_name}")
        else:
            labels.append(f"#{int(track_id)}\n{cls_name}")
    return labels
