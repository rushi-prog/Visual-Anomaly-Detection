"""Configuration loading and management utilities.

Loads YAML config files and provides a clean interface for accessing
model, data, and training parameters across the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Project root directory (tier2/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_config(config_name: str) -> dict[str, Any]:
    """Load a YAML config file from the configs/ directory.

    Args:
        config_name: Name of the config file (with or without .yaml extension).
            Examples: "patchcore", "patchcore.yaml", "autoencoder"

    Returns:
        Dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: {list_configs()}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def list_configs() -> list[str]:
    """List all available YAML config files.

    Returns:
        List of config file names (without directory prefix).
    """
    if not CONFIGS_DIR.exists():
        return []
    return [f.name for f in CONFIGS_DIR.glob("*.yaml")]


def get_results_dir(model_name: str, category: str) -> Path:
    """Get the results directory for a specific model and category.

    Creates the directory if it doesn't exist.

    Args:
        model_name: Name of the model (e.g., "patchcore").
        category: MVTec AD category (e.g., "bottle").

    Returns:
        Path to the results directory.
    """
    results_path = RESULTS_DIR / model_name / category
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same path, guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
