#!/usr/bin/env python3
"""
Train YOLO11 football detector on Colab GPU.

Usage (Colab):
    !git clone <repo> && cd football-ai
    !pip install -r requirements.txt
    !python training/train_detector.py --dataset /content/football-players-detection.v1i.yolov11

Local (inference machine — training will abort without GPU):
    python training/train_detector.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from loguru import logger
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import fix_data_yaml_paths, load_data_yaml, validate_dataset
from utils.secrets import load_secrets
from utils.paths import DATASET_DIR, MODELS_DIR


def require_gpu() -> str:
    if not torch.cuda.is_available():
        logger.error("GPU required for training. Use Google Colab with GPU runtime.")
        sys.exit(1)
    device = "0"
    logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    return device


def train(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = "yolo11m.pt",
    epochs: int = 50,
    batch: int = 8,
    imgsz: int = 1280,
    workers: int = 4,
    amp: bool = True,
    cache: bool | str = False,
) -> Path:
    stats = validate_dataset(dataset_dir)
    if not stats["valid"]:
        logger.error("Dataset validation failed. Ensure images are downloaded from Roboflow.")
        for warning in stats["warnings"]:
            logger.error(f"  - {warning}")
        sys.exit(1)

    data = load_data_yaml(dataset_dir)
    resolved_yaml = fix_data_yaml_paths(data)
    logger.info(f"Resolved data.yaml: {resolved_yaml}")
    logger.info(f"Training {stats['num_classes']} classes: {stats['class_names']}")

    device = require_gpu()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_model)
    results = model.train(
        data=str(resolved_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        amp=amp,
        cache=cache,
        workers=workers,
        project=str(output_dir / "runs"),
        name="detector",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = output_dir / "best.pt"
    shutil.copy2(best_src, best_dst)
    logger.info(f"Best model saved to {best_dst}")
    return best_dst


def main():
    load_secrets()
    parser = argparse.ArgumentParser(description="Train YOLO11 football detector")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "detector"))
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cache", choices=["false", "disk", "ram"], default="false",
                        help="Image cache mode (use 'false' on 4GB GPU / 16GB RAM)")
    args = parser.parse_args()

    cache_val: bool | str = False if args.cache == "false" else args.cache

    train(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        amp=not args.no_amp,
        cache=cache_val,
    )


if __name__ == "__main__":
    main()
