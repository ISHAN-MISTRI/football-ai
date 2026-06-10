from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATABASE_DIR = PROJECT_ROOT / "database"
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_DIR = WORKSPACE_ROOT / "football-players-detection.v1i.yolov11"


def ensure_dirs() -> None:
    for directory in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR / "detector", DATABASE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
