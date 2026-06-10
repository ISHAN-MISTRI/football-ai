# ⚽ Football AI Video Analytics MVP

A professional-grade football match video analytics application built with **YOLO11, ByteTrack, OSNet ReID, and Streamlit**. It automates player and ball detection, multi-object tracking, team jersey color classification, and generates beautifully annotated videos and tracking statistic CSV sheets.

---

## 🌟 Key Features

* **Advanced Object Detection**: Custom YOLO11 model detecting **players** and **balls** optimized for high resolution (1280px).
* **Robust Multi-Object Tracking**: Integrates `ByteTrack` for short-term bounding box association.
* **Persistent ReID Identity Recovery**: Utilizes `OSNet` deep feature embeddings to recover player tracking IDs after occlusion or camera transitions.
* **Team Classification**: Clusters players automatically into Team A and Team B using UMAP dimensionality reduction and KMeans clustering.
* **Professional Annotations**: Ring/ellipse overlays at players' feet, tracking ID labels above players' heads, and ball marker indicators.
* **Export Utilities**: Outputs annotated video files (`.mp4`), tracking datasets (`.csv`), and stores entries in an SQLite database.
* **User Interface**: Streamlit UI for simple match uploads, live processing bars, and stats previews.

---

## 📂 Project Directory Structure

```
football-ai/
├── app/                 # Streamlit UI
│   └── main.py          # Streamlit entry point
├── analytics/           # Team classification and ball tracking
├── configs/             # Configuration files
│   └── default.yaml     # Application parameters
├── database/            # SQLite tracking storage
├── models/
│   ├── detector/        # YOLO weights (best26x.pt)
│   └── tracker/         # ReID identity manager
├── ocr/                 # Jersey recognition modules (optional)
├── pipeline/            # End-to-end processing pipeline
├── training/            # Custom model training scripts
├── outputs/             # Processed videos & CSV exports (gitignored)
├── uploads/             # Uploaded source videos (gitignored)
└── utils/               # Path, config, and video I/O helpers
```

---

## 💻 System Requirements

* **OS**: Windows (tested on Windows 10/11)
* **Python**: `3.11` or `3.12`
* **GPU**: NVIDIA GTX 1650 (4GB VRAM) or better recommended for CUDA acceleration. CPU fallback is supported but slower.
* **RAM**: 16GB recommended for full-length match clips.

---

## 🚀 Setup Instructions

### 1. Initialize Virtual Environment

Open your terminal in the repository root directory and run:

```bash
# Navigate to project subfolder (if desired)
cd football-ai

# Create virtual environment
py -3.12 -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install PyTorch with CUDA acceleration (strongly recommended for real-time tracking speed) followed by general requirements:

```bash
# Install PyTorch + CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt
```

### 3. Add YOLO Weights Model

> [!WARNING]
> The trained YOLO weight file `best26x.pt` is **gitignored** because it exceeds GitHub's 100MB file push limit.

Before running the application, place your custom trained weights file `best26x.pt` in the directory:
📂 `football-ai/models/detector/best26x.pt`

---

## 🛠️ Running the Application

### Option A: Run via CLI (CommandLine)

Run the video analytics pipeline on a sample video from your console:

```bash
# Process a video with CUDA acceleration
.venv\Scripts\python pipeline/run.py --source short_clip.mp4 --device cuda

# Run a quick check on the first 100 frames only
.venv\Scripts\python pipeline/run.py --source short_clip.mp4 --device cuda --max-frames 100
```

### Option B: Run Streamlit Web Application

To launch the web dashboard:

```bash
.venv\Scripts\streamlit run app/main.py
```
Open the local link printed in the terminal (usually `http://localhost:8501`) in your browser to upload and analyze videos interactively.
