"""Load API keys from .env (never commit .env to git)."""

import os
from pathlib import Path

from dotenv import load_dotenv

from utils.paths import PROJECT_ROOT

_ENV_LOADED = False


def load_secrets() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    _ENV_LOADED = True


def get_roboflow_api_key() -> str:
    load_secrets()
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key or key == "your_api_key_here":
        raise EnvironmentError(
            f"ROBOFLOW_API_KEY missing. Paste your key in:\n  {PROJECT_ROOT / '.env'}\n"
            "Get a free key at: https://app.roboflow.com/settings/api"
        )
    return key
