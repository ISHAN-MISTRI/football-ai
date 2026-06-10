#!/usr/bin/env python3
"""CLI entry point for tracking pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.process_video import VideoProcessor
from utils.paths import ensure_dirs


def main():
    parser = argparse.ArgumentParser(description="Football AI tracking pipeline")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames (for quick tests)")
    args = parser.parse_args()

    ensure_dirs()
    processor = VideoProcessor(device=args.device)
    result = processor.process(
        source_video=args.source,
        output_dir=args.output,
        max_frames=args.max_frames,
    )
    print(f"Video: {result.output_video}")
    print(f"CSV:   {result.csv_path}")
    print(f"Stats: {result.stats}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
