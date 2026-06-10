#!/usr/bin/env python3
"""Standalone detector inference on image or video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, resolve_path
from utils.device import resolve_device


def run_image(model_path: Path, source: Path, output: Path, device: str, imgsz: int) -> None:
    model = YOLO(str(model_path))
    frame = cv2.imread(str(source))
    results = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    print(f"Saved {output} ({len(detections)} detections)")


def run_video(model_path: Path, source: Path, output: Path, device: str, imgsz: int) -> None:
    model = YOLO(str(model_path))
    video_info = sv.VideoInfo.from_video_path(str(source))
    annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    with sv.VideoSink(str(output), video_info) as sink:
        for frame in sv.get_video_frames_generator(str(source)):
            results = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            annotated = annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections)
            sink.write_frame(annotated)
    print(f"Saved {output}")


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="YOLO11 detector inference")
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--model", default=str(resolve_path(config, "detector_model")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=config["detection"]["imgsz"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    source, output = Path(args.source), Path(args.output)
    suffix = source.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        run_video(Path(args.model), source, output, device, args.imgsz)
    else:
        run_image(Path(args.model), source, output, device, args.imgsz)


if __name__ == "__main__":
    main()
