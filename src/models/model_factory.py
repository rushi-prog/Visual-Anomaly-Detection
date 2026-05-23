"""Model factory for instantiating Anomalib anomaly detection models.

Provides a unified interface to create any supported Anomalib model
by name, with optional config overrides. Supports:
  - PatchCore: Memory-bank coreset approach (highest accuracy)
  - EfficientAD: Student-teacher + autoencoder (fastest inference)
  - STFPM: Student-teacher feature pyramid matching (balanced)
  - PaDiM: Gaussian modeling of patch features (fast training)
"""

from __future__ import annotations

from typing import Any

from anomalib.models import (
    EfficientAd,
    Padim,
    Patchcore,
    Stfpm,
)

from src.utils.config import load_config

# Registry of supported Anomalib models
_MODEL_REGISTRY: dict[str, type] = {
    "patchcore": Patchcore,
    "efficientad": EfficientAd,
    "stfpm": Stfpm,
    "padim": Padim,
}


def create_model(
    model_name: str,
    config_override: dict[str, Any] | None = None,
) -> Patchcore | EfficientAd | Stfpm | Padim:
    """Create an Anomalib model by name.

    Loads default config from configs/ directory if available, then applies
    any overrides.

    Args:
        model_name: Model identifier. One of: "patchcore", "efficientad",
            "stfpm", "padim".
        config_override: Optional dictionary of model-specific parameters
            to override defaults. Keys depend on the model.

    Returns:
        Instantiated Anomalib model ready for training.

    Raises:
        ValueError: If model_name is not supported.

    Examples:
        >>> model = create_model("patchcore")
        >>> model = create_model("efficientad", {"model_size": "medium"})
    """
    model_name = model_name.lower().strip()

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {list(_MODEL_REGISTRY.keys())}"
        )

    # Load config if available
    model_params = {}
    try:
        config = load_config(model_name)
        model_params = config.get("model", {})
        # Remove the 'name' key — it's metadata, not a constructor arg
        model_params.pop("name", None)
    except FileNotFoundError:
        pass  # No config file, use model defaults

    # Apply overrides
    if config_override:
        model_params.update(config_override)

    model_class = _MODEL_REGISTRY[model_name]
    model = model_class(**model_params)

    return model


def list_models() -> list[str]:
    """List all supported Anomalib model names.

    Returns:
        List of model identifiers.
    """
    return list(_MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> dict[str, str]:
    """Get descriptive information about a model.

    Args:
        model_name: Model identifier.

    Returns:
        Dictionary with model description and key characteristics.
    """
    info = {
        "patchcore": {
            "name": "PatchCore",
            "type": "Memory Bank (Coreset)",
            "description": (
                "Stores representative patch features from a pretrained CNN. "
                "Detects anomalies via nearest-neighbor distance in feature space."
            ),
            "strengths": "Highest accuracy, strong on small datasets",
            "weaknesses": "Memory-intensive, slower inference",
            "training": "Single-pass (no gradient descent)",
        },
        "efficientad": {
            "name": "EfficientAD",
            "type": "Student-Teacher + Autoencoder",
            "description": (
                "Uses knowledge distillation for structural anomalies and an "
                "autoencoder for global/logical anomalies. Designed for speed."
            ),
            "strengths": "Fastest inference, handles logical anomalies",
            "weaknesses": "Needs more training epochs, tuning sensitive",
            "training": "Gradient-based (70-200 epochs)",
        },
        "stfpm": {
            "name": "STFPM",
            "type": "Student-Teacher Feature Pyramid",
            "description": (
                "Multi-scale feature pyramid matching between pretrained teacher "
                "and trained student network."
            ),
            "strengths": "Fast inference, balanced performance",
            "weaknesses": "Lower accuracy than PatchCore on complex defects",
            "training": "Gradient-based (100 epochs)",
        },
        "padim": {
            "name": "PaDiM",
            "type": "Gaussian Modeling",
            "description": (
                "Models patch-level feature distributions as multivariate "
                "Gaussians. Anomalies are patches with low probability."
            ),
            "strengths": "Very fast training and inference",
            "weaknesses": "Lower pixel-level accuracy",
            "training": "Single-pass (no gradient descent)",
        },
    }

    model_name = model_name.lower().strip()
    if model_name not in info:
        raise ValueError(f"Unknown model '{model_name}'.")
    return info[model_name]
