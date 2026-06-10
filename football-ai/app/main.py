"""Streamlit UI for Football AI Video Analytics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.process_video import VideoProcessor
from utils.dataset import validate_dataset
from utils.device import resolve_device
from utils.paths import OUTPUTS_DIR, UPLOADS_DIR, ensure_dirs

st.set_page_config(
    page_title="Football AI Analytics",
    page_icon="⚽",
    layout="wide",
)

ensure_dirs()

if "result" not in st.session_state:
    st.session_state.result = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def save_upload(uploaded_file) -> Path:
    dest = UPLOADS_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


st.title("⚽ Football AI Video Analytics")
st.caption("Professional player tracking, team classification, and jersey recognition")

tab_upload, tab_process, tab_results = st.tabs(["1. Upload Video", "2. Processing", "3. Results"])

with tab_upload:
    st.header("Upload Match Video")
    st.info("Supported: MP4, AVI, MOV. Optimized for broadcast football footage (~25 min matches).")
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    device = st.selectbox("Inference device", ["auto", "cuda", "cpu"], index=0)

    if uploaded:
        video_path = save_upload(uploaded)
        st.session_state.video_path = str(video_path)
        st.video(str(video_path))
        st.success(f"Saved to `{video_path.name}`")

    with st.expander("Dataset & Model Status"):
        try:
            ds_stats = validate_dataset()
            st.json(ds_stats)
        except Exception as exc:
            st.warning(f"Dataset check: {exc}")
        model_path = PROJECT_ROOT / "models" / "detector" / "best.pt"
        if model_path.exists():
            st.success(f"Detector model found: {model_path.name}")
        else:
            st.warning("No trained detector at models/detector/best.pt — train on Colab or use sports-main fallback.")

with tab_process:
    st.header("Process Video")
    if "video_path" not in st.session_state:
        st.warning("Upload a video in the Upload tab first.")
    else:
        st.write(f"**Source:** `{Path(st.session_state.video_path).name}`")
        disable_team_classifier = st.checkbox(
            "Disable player team classification (clustering)",
            value=False,
            help="If selected, all players will be detected and tracked with unique IDs but not divided into teams."
        )
        progress_bar = st.progress(0.0)
        status = st.empty()

        def on_progress(stage: str, value: float):
            if stage == "team_fit":
                status.text("Fitting team classifier (SigLIP + UMAP + KMeans)...")
                progress_bar.progress(min(0.15, value))
            elif stage == "processing":
                progress_bar.progress(0.15 + value * 0.85)
                status.text(f"Processing frames... {value * 100:.1f}%")
            elif stage == "done":
                progress_bar.progress(1.0)
                status.text("Complete!")

        if st.button("Start Processing", disabled=st.session_state.processing, type="primary"):
            st.session_state.processing = True
            try:
                with st.spinner("Running analytics pipeline..."):
                    processor = VideoProcessor(device=device)
                    result = processor.process(
                        st.session_state.video_path,
                        output_dir=OUTPUTS_DIR,
                        disable_team_classifier=disable_team_classifier,
                        progress_callback=on_progress,
                    )
                    st.session_state.result = result
                st.success("Processing complete!")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.exception(exc)
            finally:
                st.session_state.processing = False

with tab_results:
    st.header("Results")
    result = st.session_state.result
    if result is None:
        st.info("Process a video to see results here.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        stats = result.stats
        col1.metric("Frames", stats.get("frames_processed", 0))
        col2.metric("Unique Tracks", stats.get("unique_track_ids", 0))
        col3.metric("Ball Detections", stats.get("ball_detections", 0))
        col4.metric("ID Recoveries", stats.get("reid", {}).get("id_recoveries", 0))

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Players (detections)", stats.get("players_detected", 0))
            st.metric("Goalkeepers (detections)", stats.get("goalkeepers_detected", 0))
        with col_b:
            st.metric("Referees (detections)", stats.get("referees_detected", 0))
            if "reid" in stats:
                st.json(stats["reid"])

        st.subheader("Annotated Video")
        st.video(str(result.output_video))

        st.subheader("Downloads")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            with open(result.output_video, "rb") as vf:
                st.download_button("Download Annotated Video", vf, file_name=result.output_video.name)
        with col_dl2:
            with open(result.csv_path, "rb") as cf:
                st.download_button("Download Tracking CSV", cf, file_name=result.csv_path.name)

        st.subheader("Tracking Preview")
        import pandas as pd
        df = pd.read_csv(result.csv_path)
        st.dataframe(df.head(500), use_container_width=True)
