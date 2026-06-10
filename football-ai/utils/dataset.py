"""Dataset validation and data.yaml path resolution."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from utils.paths import DATASET_DIR


def load_data_yaml(dataset_dir: Path | None = None) -> dict[str, Any]:
    root = Path(dataset_dir) if dataset_dir else DATASET_DIR
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path, encoding="utf-8") as file:
        data = yaml.safe_load(file)

    data["_yaml_path"] = str(yaml_path)
    data["_dataset_root"] = str(root)
    return data


def fix_data_yaml_paths(data: dict[str, Any]) -> Path:
    """Resolve train/val/test paths and write a corrected yaml for Ultralytics."""
    root = Path(data["_dataset_root"])
    corrected = dict(data)

    for split in ("train", "val", "test"):
        if split not in corrected:
            continue
        raw = Path(corrected[split])
        if not raw.is_absolute():
            candidates = [
                root / corrected[split],
                root / corrected[split].replace("../", ""),
                (root.parent / corrected[split].lstrip("../")),
            ]
            resolved = next((c for c in candidates if c.exists()), None)
            if resolved is None:
                resolved = root / split / "images"
            corrected[split] = str(resolved.resolve())

    out_path = root / "data_resolved.yaml"
    export = {k: v for k, v in corrected.items() if not k.startswith("_")}
    with open(out_path, "w", encoding="utf-8") as file:
        yaml.dump(export, file, default_flow_style=False)
    return out_path


def validate_dataset(dataset_dir: Path | None = None) -> dict[str, Any]:
    data = load_data_yaml(dataset_dir)
    root = Path(data["_dataset_root"])

    class_names = data.get("names", [])
    num_classes = data.get("nc", len(class_names))
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    logger.info(f"Dataset: {root.name}")
    logger.info(f"Classes ({num_classes}): {class_names}")

    stats: dict[str, Any] = {
        "dataset_root": str(root),
        "num_classes": num_classes,
        "class_names": class_names,
        "splits": {},
        "valid": True,
        "warnings": [],
    }

    split_dirs = {"train": "train", "val": "valid", "test": "test"}
    for split, folder in split_dirs.items():
        images_dir = root / folder / "images"
        labels_dir = root / folder / "labels"
        image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
        stats["splits"][split] = {"images": image_count, "labels": label_count}
        if label_count > 0 and image_count == 0:
            stats["warnings"].append(
                f"{split}: {label_count} labels but 0 images — re-download from Roboflow"
            )
            stats["valid"] = False

    if stats["warnings"]:
        for warning in stats["warnings"]:
            logger.warning(warning)

    return stats


def get_class_id_map(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def get_class_name(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return "unknown"
