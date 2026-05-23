"""Visualization utilities for anomaly detection results.

Provides functions to create:
- Anomaly heatmap overlays on original images
- Side-by-side comparisons (input → anomaly map → prediction → ground truth)
- Multi-model comparison grids
- ROC curves and metric bar charts
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from src.utils.config import ensure_dir


def overlay_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay an anomaly heatmap on the original image.

    Args:
        image: Original image in RGB, shape (H, W, 3), values in [0, 1] or [0, 255].
        anomaly_map: Anomaly scores per pixel, shape (H, W), any range.
        alpha: Transparency of the heatmap overlay (0=transparent, 1=opaque).
        colormap: OpenCV colormap to use.

    Returns:
        Overlaid image in RGB, shape (H, W, 3), values in [0, 255] as uint8.
    """
    # Normalize image to uint8
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    # Normalize anomaly map to [0, 255]
    if anomaly_map.max() > anomaly_map.min():
        normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        normalized = np.zeros_like(anomaly_map)
    heatmap_uint8 = (normalized * 255).astype(np.uint8)

    # Resize heatmap to match image if needed
    if heatmap_uint8.shape[:2] != image_uint8.shape[:2]:
        heatmap_uint8 = cv2.resize(heatmap_uint8, (image_uint8.shape[1], image_uint8.shape[0]))

    # Apply colormap (OpenCV uses BGR)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay


def visualize_predictions(
    images: np.ndarray,
    anomaly_maps: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray | None = None,
    threshold: float | None = None,
    num_samples: int = 8,
    save_path: Path | None = None,
    title: str = "Anomaly Detection Results",
) -> plt.Figure:
    """Create a visualization grid of anomaly detection predictions.

    For each sample, shows:
    Row 1: Original image
    Row 2: Anomaly heatmap overlay
    Row 3: Ground truth mask (if available)

    Args:
        images: Array of images, shape (N, H, W, 3) in [0, 1].
        anomaly_maps: Anomaly maps, shape (N, H, W).
        labels: Ground truth labels, shape (N,).
        scores: Anomaly scores, shape (N,).
        masks: Optional ground truth masks, shape (N, H, W).
        threshold: Classification threshold for pass/fail annotation.
        num_samples: Number of samples to display.
        save_path: Optional path to save the figure.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    num_samples = min(num_samples, len(images))

    has_masks = masks is not None
    nrows = 3 if has_masks else 2

    fig, axes = plt.subplots(nrows, num_samples, figsize=(3 * num_samples, 3 * nrows))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    if num_samples == 1:
        axes = axes.reshape(nrows, 1)

    for i in range(num_samples):
        img = images[i]
        amap = anomaly_maps[i]
        label = int(labels[i])
        score = float(scores[i])

        # Original image
        axes[0, i].imshow(img)
        label_str = "Anomalous" if label == 1 else "Normal"
        color = "#FF5252" if label == 1 else "#4CAF50"

        if threshold is not None:
            pred = "FAIL" if score >= threshold else "PASS"
            pred_color = "#FF5252" if score >= threshold else "#4CAF50"
            axes[0, i].set_title(
                f"{label_str}\nScore: {score:.3f} → {pred}",
                fontsize=9, color=color, fontweight="bold",
            )
        else:
            axes[0, i].set_title(
                f"{label_str} (Score: {score:.3f})",
                fontsize=9, color=color, fontweight="bold",
            )
        axes[0, i].axis("off")

        # Anomaly heatmap overlay
        overlay = overlay_anomaly_map(img, amap)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Anomaly Heatmap", fontsize=9)
        axes[1, i].axis("off")

        # Ground truth mask
        if has_masks:
            axes[2, i].imshow(masks[i], cmap="hot", vmin=0, vmax=1)
            axes[2, i].set_title("Ground Truth", fontsize=9)
            axes[2, i].axis("off")

    # Add row labels
    row_labels = ["Input", "Heatmap"]
    if has_masks:
        row_labels.append("GT Mask")
    for ax, label in zip(axes[:, 0], row_labels):
        ax.set_ylabel(label, fontsize=12, fontweight="bold", rotation=90, labelpad=15)

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "image_auroc",
    save_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Create a bar chart comparing multiple models on a metric.

    Args:
        results: Dict of {model_name: {metric_name: value, ...}}.
        metric: Metric to compare.
        save_path: Optional save path.
        title: Optional title override.

    Returns:
        Matplotlib figure.
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    # Color palette
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, values, color=colors[:len(models)], edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13)
    ax.set_title(
        title or f"Model Comparison — {metric.replace('_', ' ').title()}",
        fontsize=15, fontweight="bold",
    )
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curves(
    results: dict[str, dict],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot ROC curves for multiple models on the same axes.

    Args:
        results: Dict of {model_name: {roc_fpr: [...], roc_tpr: [...], image_auroc: float}}.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, (model_name, data) in enumerate(results.items()):
        if "roc_fpr" not in data or "roc_tpr" not in data:
            continue
        fpr = data["roc_fpr"]
        tpr = data["roc_tpr"]
        auroc = data.get("image_auroc", 0)

        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{model_name} (AUROC={auroc:.3f})",
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_category_comparison(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str = "image_auroc",
    save_path: Path | None = None,
) -> plt.Figure:
    """Create a grouped bar chart comparing models across categories.

    Args:
        results: Nested dict {model_name: {category: {metric: value}}}.
        metric: Metric to compare.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    models = list(results.keys())
    categories = sorted(set(
        cat for model_data in results.values() for cat in model_data
    ))

    x = np.arange(len(categories))
    width = 0.8 / len(models)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, model in enumerate(models):
        values = [
            results[model].get(cat, {}).get(metric, 0)
            for cat in categories
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])

    ax.set_xlabel("Category", fontsize=13)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13)
    ax.set_title(
        f"Model Comparison by Category — {metric.replace('_', ' ').title()}",
        fontsize=15, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
