"""MVTec AD dataset explorer and visualization utilities.

Provides functions to load, inspect, and visualize the MVTec Anomaly Detection
dataset. Supports both Anomalib's built-in MVTec datamodule and manual loading
for custom analysis.

MVTec AD contains 15 categories:
  - Objects: bottle, cable, capsule, hazelnut, metal_nut, pill, screw,
             toothbrush, transistor, zipper
  - Textures: carpet, grid, leather, tile, wood
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from anomalib.data import MVTecAD
from torchvision.transforms.v2 import Resize

from src.utils.config import DATASETS_DIR, ensure_dir

# All 15 MVTec AD categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]


def get_mvtec_datamodule(
    category: str = "bottle",
    image_size: tuple[int, int] = (256, 256),
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 4,
) -> MVTecAD:
    """Create an Anomalib MVTec datamodule for a given category.

    Auto-downloads the dataset on first use (~350MB per category).

    Args:
        category: MVTec AD category name (e.g., "bottle", "carpet").
        image_size: Target image size as (height, width).
        train_batch_size: Batch size for training dataloader.
        eval_batch_size: Batch size for evaluation dataloaders.
        num_workers: Number of data loading workers.

    Returns:
        Configured MVTecAD datamodule (call .setup() before accessing data).

    Raises:
        ValueError: If category is not a valid MVTec AD category.
    """
    if category not in MVTEC_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Must be one of: {MVTEC_CATEGORIES}"
        )

    # In Anomalib v2.4+, image resizing is handled via augmentations
    resize_transform = Resize(image_size)

    datamodule = MVTecAD(
        root=str(DATASETS_DIR / "MVTec"),
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        augmentations=resize_transform,
    )

    return datamodule


def get_dataset_stats(category: str) -> dict:
    """Get dataset statistics for a given MVTec category.

    Sets up the datamodule and counts samples in each split.

    Args:
        category: MVTec AD category name.

    Returns:
        Dictionary with counts for train, val, and test splits,
        plus anomaly type breakdown.
    """
    datamodule = get_mvtec_datamodule(category, num_workers=0)
    datamodule.setup()

    stats = {
        "category": category,
        "train_normal": len(datamodule.train_data),
        "test_total": len(datamodule.test_data),
    }

    # Count normal vs anomalous in test set
    test_labels = []
    for sample in datamodule.test_data:
        label = sample["label"] if isinstance(sample["label"], int) else sample["label"].item()
        test_labels.append(label)

    stats["test_normal"] = sum(1 for l in test_labels if l == 0)
    stats["test_anomalous"] = sum(1 for l in test_labels if l == 1)

    return stats


def visualize_samples(
    category: str = "bottle",
    num_samples: int = 4,
    split: Literal["train", "test"] = "test",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Visualize sample images from a MVTec category.

    Shows images side-by-side with labels. For test samples, also shows
    ground truth masks if available.

    Args:
        category: MVTec AD category.
        num_samples: Number of samples to display.
        split: Which split to visualize ("train" or "test").
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    datamodule = get_mvtec_datamodule(category, num_workers=0)
    datamodule.setup()

    dataset = datamodule.train_data if split == "train" else datamodule.test_data
    num_samples = min(num_samples, len(dataset))

    # Randomly sample indices
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    has_masks = split == "test"
    nrows = 2 if has_masks else 1

    fig, axes = plt.subplots(nrows, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = np.array(axes).reshape(nrows, 1)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"MVTec AD — {category.title()} ({split})",
        fontsize=16, fontweight="bold",
    )

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]

        # Convert from tensor (C, H, W) to numpy (H, W, C)
        if hasattr(image, "numpy"):
            img_np = image.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = image

        label = sample.get("label", 0)
        if hasattr(label, "item"):
            label = label.item()
        label_str = "Anomalous" if label == 1 else "Normal"
        color = "red" if label == 1 else "green"

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(label_str, color=color, fontweight="bold")
        axes[0, i].axis("off")

        # Show mask for test split
        if has_masks and "mask" in sample:
            mask = sample["mask"]
            if hasattr(mask, "numpy"):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = np.squeeze(mask)

            axes[1, i].imshow(mask_np, cmap="hot")
            axes[1, i].set_title("Ground Truth Mask", fontsize=9)
            axes[1, i].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    return fig


def visualize_normal_vs_anomalous(
    category: str = "bottle",
    num_each: int = 4,
    save_path: Path | None = None,
) -> plt.Figure:
    """Show normal vs anomalous samples side by side with masks.

    Useful for understanding what anomalies look like in a category.

    Args:
        category: MVTec AD category.
        num_each: Number of normal and anomalous samples each.
        save_path: Optional save path.

    Returns:
        Matplotlib figure with normal (top row) vs anomalous (bottom two rows).
    """
    datamodule = get_mvtec_datamodule(category, num_workers=0)
    datamodule.setup()

    test_data = datamodule.test_data
    normal_indices = []
    anomalous_indices = []

    for i in range(len(test_data)):
        sample = test_data[i]
        label = sample["label"]
        if hasattr(label, "item"):
            label = label.item()
        if label == 0:
            normal_indices.append(i)
        else:
            anomalous_indices.append(i)

    rng = np.random.default_rng(42)
    n_normal = min(num_each, len(normal_indices))
    n_anomalous = min(num_each, len(anomalous_indices))

    normal_pick = rng.choice(normal_indices, size=n_normal, replace=False)
    anomalous_pick = rng.choice(anomalous_indices, size=n_anomalous, replace=False)

    fig, axes = plt.subplots(3, max(n_normal, n_anomalous), figsize=(4 * max(n_normal, n_anomalous), 12))
    fig.suptitle(f"MVTec AD — {category.title()}: Normal vs Anomalous", fontsize=16, fontweight="bold")

    # Row 0: Normal samples
    for i, idx in enumerate(normal_pick):
        sample = test_data[idx]
        img = sample["image"].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title("Normal ✓", color="green", fontweight="bold")
        axes[0, i].axis("off")

    # Row 1 & 2: Anomalous samples + masks
    for i, idx in enumerate(anomalous_pick):
        sample = test_data[idx]
        img = sample["image"].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[1, i].imshow(img)
        axes[1, i].set_title("Anomalous ✗", color="red", fontweight="bold")
        axes[1, i].axis("off")

        if "mask" in sample:
            mask = sample["mask"].squeeze().numpy()
            # Overlay mask on image
            overlay = img.copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            axes[2, i].imshow(overlay)
            axes[2, i].set_title("Defect Overlay", fontsize=9)
            axes[2, i].axis("off")

    # Hide unused axes
    for row in range(3):
        for col in range(max(n_normal, n_anomalous)):
            if row == 0 and col >= n_normal:
                axes[row, col].axis("off")
            elif row > 0 and col >= n_anomalous:
                axes[row, col].axis("off")

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
