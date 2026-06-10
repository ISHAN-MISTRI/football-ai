# Football AI Video Analytics MVP

Professional-style football video analytics: player/goalkeeper/referee/ball detection, multi-object tracking with ReID identity recovery, team classification, jersey OCR, annotated video export, and CSV tracking data.

Built on patterns from [Roboflow sports-main](../sports-main).

## Features

| Feature | Implementation |
|---------|----------------|
| Detection | YOLO11 (4 classes: ball, goalkeeper, player, referee) |
| Tracking | ByteTrack + OSNet ReID identity recovery |
| Team classification | SigLIP → UMAP → KMeans |
| Jersey numbers | EasyOCR + temporal majority voting |
| Annotation | Ellipse markers, triangle ball marker, ID/team/jersey labels |
| Export | Annotated MP4 + `tracking_results.csv` + SQLite |

## Hardware

- **Training:** Google Colab GPU (required — script aborts on CPU)
- **Inference:** NVIDIA GTX 1650 (4GB) with FP16, CPU fallback supported
- **RAM:** 16GB recommended for 25-minute matches

## Roboflow API Key

1. Open **`football-ai/.env`** in Cursor
2. Paste your key on this line (no quotes):

```
ROBOFLOW_API_KEY=paste_your_key_here
```

3. Save the file. **Do not paste the key in chat** — `.env` is gitignored.

Get a free key at: https://app.roboflow.com/settings/api

## Quick Start

Use the **Python 3.12 GPU virtual environment** (required for CUDA on GTX 1650):

```bash
cd football-ai

# First-time setup (already done if .venv exists)
py -3.12 -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt

# Download dataset (requires ROBOFLOW_API_KEY in .env)
.\.venv\Scripts\python training/download_dataset.py

# Train detector on GPU
.\.venv\Scripts\python training/train_detector.py --batch 4

# Run Streamlit app
.\.venv\Scripts\streamlit run app/main.py
```

## Training (Google Colab)

1. Upload `football-players-detection.v1i.yolov11` with images to Colab
2. Clone this repo and install dependencies
3. Run:

```bash
python training/train_detector.py \
  --dataset /content/football-players-detection.v1i.yolov11 \
  --epochs 50 --batch 8 --imgsz 1280
```

Output: `models/detector/best.pt`

The script reads `data.yaml` automatically — class names and paths are never hardcoded.

## CLI Processing

```python
from pipeline.process_video import VideoProcessor

processor = VideoProcessor(device="cuda")
result = processor.process("path/to/match.mp4")
print(result.csv_path, result.output_video)
```

## Project Structure

```
football-ai/
├── app/                 # Streamlit UI
├── analytics/           # Team classifier, ball tracker (from sports-main)
├── configs/             # YAML configuration
├── database/            # SQLite tracking storage
├── models/
│   ├── detector/        # YOLO weights (best.pt)
│   └── tracker/         # ReID identity manager
├── ocr/                 # Jersey number reader
├── pipeline/            # End-to-end video processor
├── training/            # Colab training script
├── outputs/             # Annotated videos & CSV
└── uploads/             # Uploaded source videos
```

## Pipeline Architecture

```
Video (frame-by-frame)
    ↓
YOLO11 Detection (1280px, FP16 on CUDA)
    ↓
ByteTrack (short-term association)
    ↓
OSNet ReID (identity recovery for lost tracks)
    ↓
SigLIP Team Classification (fit on stride=60, predict per frame)
    ↓
EasyOCR Jersey Recognition (every N frames, majority vote)
    ↓
Supervision Annotators (ellipse + labels + ball triangle)
    ↓
MP4 + CSV + SQLite export
```

## ReID Identity Recovery

When a player leaves the frame or is occluded, their OSNet embedding is stored in a lost-identity bank. When a new track appears, cosine similarity matching (plus jersey number boost) attempts to restore the original track ID.

Configure in `configs/default.yaml`:

```yaml
reid:
  similarity_threshold: 0.65
  lost_track_buffer: 90
  embedding_ema_alpha: 0.85
```

## CSV Output Columns

`frame`, `timestamp`, `track_id`, `class`, `team`, `jersey_number`, `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `confidence`

## Dataset Note

The local `football-players-detection.v1i.yolov11` folder may contain labels only. Re-download the full dataset (with images) from [Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) before training.

## V1 Scope

Included: detection, tracking, ReID, team colors, jersey OCR, professional overlays, CSV/video export.

Excluded: pitch homography, heatmaps, speed/distance metrics, advanced tactical analytics.
