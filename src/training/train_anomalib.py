"""Anomalib model training wrapper.

Provides a clean interface around Anomalib's Engine for training
PatchCore, EfficientAD, STFPM, and PaDiM on MVTec AD or custom datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anomalib.engine import Engine

from src.data.mvtec_explorer import MVTEC_CATEGORIES, get_mvtec_datamodule
from src.models.model_factory import create_model
from src.utils.config import get_results_dir, load_config


def train_anomalib_model(
    model_name: str = "patchcore",
    category: str = "bottle",
    config_override: dict[str, Any] | None = None,
    max_epochs: int | None = None,
    image_size: tuple[int, int] = (256, 256),
    accelerator: str = "auto",
    devices: int = 1,
) -> dict[str, Any]:
    """Train an Anomalib model on a MVTec AD category.

    This is the main entry point for training. It:
    1. Creates the model from the factory
    2. Sets up the MVTec datamodule (auto-downloads if needed)
    3. Trains using Anomalib's Engine
    4. Returns training results and checkpoint path

    Args:
        model_name: Model to train ("patchcore", "efficientad", "stfpm", "padim").
        category: MVTec AD category ("bottle", "carpet", etc.).
        config_override: Optional model parameter overrides.
        max_epochs: Override max training epochs from config.
        image_size: Input image resolution.
        accelerator: PyTorch accelerator ("auto", "gpu", "cpu").
        devices: Number of devices.

    Returns:
        Dictionary with:
            - model: Trained model instance
            - results_dir: Path to results directory
            - checkpoint_path: Path to best checkpoint
            - metrics: Test metrics (if available)

    Example:
        >>> result = train_anomalib_model("patchcore", "bottle")
        >>> print(f"Checkpoint: {result['checkpoint_path']}")
    """
    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()} on MVTec AD / {category}")
    print(f"{'='*60}\n")

    # Load config for training parameters
    try:
        config = load_config(model_name)
        trainer_config = config.get("trainer", {})
    except FileNotFoundError:
        trainer_config = {}

    # Resolve max_epochs
    if max_epochs is None:
        max_epochs = trainer_config.get("max_epochs", 1)

    # Create model
    model = create_model(model_name, config_override)

    # Create datamodule
    data_config = {}
    try:
        full_config = load_config(model_name)
        data_config = full_config.get("data", {})
    except FileNotFoundError:
        pass

    datamodule = get_mvtec_datamodule(
        category=category,
        image_size=image_size,
        train_batch_size=data_config.get("train_batch_size", 32),
        eval_batch_size=data_config.get("eval_batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
    )

    # Setup results directory
    results_dir = get_results_dir(model_name, category)

    # Create engine and train
    engine = Engine(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=str(results_dir),
    )

    engine.fit(model=model, datamodule=datamodule)

    # Run test to get metrics
    test_results = engine.test(model=model, datamodule=datamodule)

    # Find checkpoint path
    ckpt_path = _find_latest_checkpoint(results_dir)

    result = {
        "model": model,
        "model_name": model_name,
        "category": category,
        "results_dir": results_dir,
        "checkpoint_path": ckpt_path,
        "test_results": test_results,
    }

    print(f"\n✓ Training complete!")
    print(f"  Results: {results_dir}")
    if ckpt_path:
        print(f"  Checkpoint: {ckpt_path}")
    if test_results:
        print(f"  Test metrics: {test_results}")

    return result


def train_multiple_categories(
    model_name: str = "patchcore",
    categories: list[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Train a model across multiple MVTec AD categories.

    Args:
        model_name: Model to train.
        categories: List of categories. Defaults to a representative subset.
        **kwargs: Additional arguments passed to train_anomalib_model.

    Returns:
        List of result dictionaries, one per category.
    """
    if categories is None:
        # Default: representative subset (3 objects + 2 textures)
        categories = ["bottle", "hazelnut", "screw", "carpet", "tile"]

    results = []
    for cat in categories:
        if cat not in MVTEC_CATEGORIES:
            print(f"⚠ Skipping unknown category: {cat}")
            continue

        result = train_anomalib_model(model_name, cat, **kwargs)
        results.append(result)

    return results


def _find_latest_checkpoint(results_dir: Path) -> Path | None:
    """Find the latest checkpoint file in a results directory.

    Searches recursively for .ckpt files and returns the most recent.

    Args:
        results_dir: Root directory to search.

    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    ckpt_files = list(Path(results_dir).rglob("*.ckpt"))
    if not ckpt_files:
        return None
    return max(ckpt_files, key=lambda p: p.stat().st_mtime)
