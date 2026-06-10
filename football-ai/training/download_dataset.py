#!/usr/bin/env python3
"""Download football-players-detection dataset from Roboflow."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.secrets import get_roboflow_api_key
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_OUT = WORKSPACE_ROOT / "football-players-detection.v1i.yolov11"


def download_via_roboflow(
    output_dir: Path,
    workspace: str = "roboflow-jvuqo",
    project: str = "football-players-detection-3zvbc",
    version: int = 1,
    fmt: str = "yolov11",
) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=get_roboflow_api_key())
    ds = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(model_format=fmt, location=str(output_dir), overwrite=True)
    )
    location = Path(ds.location)
    logger.info(f"Downloaded dataset to {location}")
    return location


def validate_images(dataset_dir: Path) -> int:
    count = 0
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        if images_dir.exists():
            count += len(list(images_dir.glob("*.*")))
    return count


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    try:
        location = download_via_roboflow(out)
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Roboflow download failed: {exc}")
        sys.exit(1)

    n_images = validate_images(location)
    if n_images == 0:
        logger.error("Download completed but no images found")
        sys.exit(1)
    logger.info(f"Dataset ready: {n_images} images")


if __name__ == "__main__":
    main()
