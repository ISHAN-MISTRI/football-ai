@echo off
cd /d "%~dp0"
.\.venv\Scripts\python training/train_detector.py --batch 4 --epochs 50 --imgsz 1280
