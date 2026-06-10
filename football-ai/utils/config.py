from pathlib import Path
from typing import Any

import yaml

from utils.paths import CONFIGS_DIR, PROJECT_ROOT


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else CONFIGS_DIR / "default.yaml"
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    return PROJECT_ROOT / config["paths"][key]
