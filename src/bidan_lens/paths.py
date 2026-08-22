from __future__ import annotations

from pathlib import Path


def user_data_dir() -> Path:
    try:
        from platformdirs import user_data_path

        return user_data_path("BiDan Lens", "BiDan Lens", roaming=False, ensure_exists=True)
    except ImportError:
        return Path.home() / ".bidan-lens"


def config_path() -> Path:
    return user_data_dir() / "config.json"


def assets_path() -> Path:
    return user_data_dir() / "assets"
