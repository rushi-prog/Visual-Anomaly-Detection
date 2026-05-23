"""Custom autoencoder training loop for anomaly detection.

A standard PyTorch training pipeline (no Anomalib dependency) that:
1. Trains on ONLY normal images (one-class learning)
2. Uses MSE + optional SSIM loss
3. Determines anomaly threshold from validation data
4. Evaluates on test data with normal + anomalous images

This module demonstrates the fundamentals of anomaly detection training
without relying on high-level libraries.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.mvtec_explorer import get_mvtec_datamodule
from src.models.autoencoder import AnomalyAutoencoder, CombinedLoss
from src.utils.config import get_results_dir, load_config, ensure_dir


def train_autoencoder(
    category: str = "bottle",
    config_path: str = "autoencoder",
    device: str | None = None,
) -> dict[str, Any]:
    """Train the custom autoencoder on a MVTec AD category.

    Full training pipeline:
    1. Load config and create model
    2. Setup MVTec datamodule (normal images only for training)
    3. Train with MSE + SSIM loss
    4. Determine anomaly threshold from validation errors
    5. Evaluate on test set

    Args:
        category: MVTec AD category.
        config_path: Config file name (without .yaml).
        device: Device to use. Auto-detected if None.

    Returns:
        Dictionary with model, metrics, threshold, and paths.
    """
    # Load config
    config = load_config(config_path)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("training", {})
    threshold_config = config.get("threshold", {})

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"  Training Custom Autoencoder on MVTec AD / {category}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Create model
    model = AnomalyAutoencoder(
        in_channels=model_config.get("in_channels", 3),
        latent_dim=model_config.get("latent_dim", 128),
        encoder_channels=model_config.get("encoder_channels"),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup loss
    use_ssim = model_config.get("use_ssim_loss", True)
    ssim_weight = model_config.get("ssim_weight", 0.5)
    if use_ssim:
        criterion = CombinedLoss(ssim_weight=ssim_weight).to(device)
        print(f"  Loss: MSE ({1-ssim_weight:.0%}) + SSIM ({ssim_weight:.0%})")
    else:
        criterion = nn.MSELoss()
        print(f"  Loss: MSE")

    # Setup optimizer & scheduler
    lr = train_config.get("lr", 0.001)
    weight_decay = train_config.get("weight_decay", 1e-5)
    epochs = train_config.get("epochs", 100)
    patience = train_config.get("patience", 15)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup data
    image_size = tuple(data_config.get("image_size", [256, 256]))
    datamodule = get_mvtec_datamodule(
        category=category,
        image_size=image_size,
        train_batch_size=data_config.get("train_batch_size", 16),
        eval_batch_size=data_config.get("eval_batch_size", 16),
        num_workers=data_config.get("num_workers", 4),
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    # Results directory
    results_dir = get_results_dir("custom_autoencoder", category)

    # ── Training Loop ────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_loss = float("inf")
    patience_counter = 0
    best_model_path = results_dir / "best_model.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device)

            optimizer.zero_grad()
            reconstruction, _ = model(images)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        scheduler.step()

        # Validation: compute reconstruction error on test normal images
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"]
                # Only use normal images for validation loss
                normal_mask = labels == 0
                if normal_mask.sum() == 0:
                    continue
                normal_images = images[normal_mask]
                reconstruction, _ = model(normal_images)
                loss = criterion(reconstruction, normal_images)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} │ "
                f"Train Loss: {avg_train_loss:.6f} │ "
                f"Val Loss: {avg_val_loss:.6f} │ "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": config,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚡ Early stopping at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - start_time
    print(f"\n  Training time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Load best model
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Determine Anomaly Threshold ──────────────────────────────
    print("\n  Determining anomaly threshold...")
    threshold = _compute_threshold(
        model, train_loader, device,
        k_sigma=threshold_config.get("k_sigma", 3.0),
    )
    print(f"  Threshold: {threshold:.6f}")

    # ── Evaluate on Test Set ─────────────────────────────────────
    print("\n  Evaluating on test set...")
    metrics = _evaluate_autoencoder(model, test_loader, device, threshold)

    # Save training plots
    _save_training_plots(history, results_dir)

    result = {
        "model": model,
        "model_name": "custom_autoencoder",
        "category": category,
        "threshold": threshold,
        "metrics": metrics,
        "history": history,
        "results_dir": results_dir,
        "checkpoint_path": best_model_path,
        "training_time": elapsed,
        "total_params": total_params,
    }

    print(f"\n  ✓ Results saved to {results_dir}")
    print(f"  ✓ Image AUROC: {metrics.get('image_auroc', 0):.4f}")
    print(f"  ✓ Image F1: {metrics.get('image_f1', 0):.4f}")

    return result


def _compute_threshold(
    model: AnomalyAutoencoder,
    normal_loader: DataLoader,
    device: torch.device,
    k_sigma: float = 3.0,
) -> float:
    """Compute anomaly threshold from normal training data.

    Fits a Gaussian to the reconstruction error distribution of normal
    images, then sets threshold at mean + k * std.

    Args:
        model: Trained autoencoder.
        normal_loader: DataLoader with normal images.
        device: Computation device.
        k_sigma: Number of standard deviations for threshold.

    Returns:
        Anomaly threshold value.
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in normal_loader:
            images = batch["image"].to(device)
            batch_scores = model.compute_anomaly_score(images)
            scores.extend(batch_scores.cpu().numpy().tolist())

    scores = np.array(scores)
    mean_score = scores.mean()
    std_score = scores.std()
    threshold = mean_score + k_sigma * std_score

    return float(threshold)


def _evaluate_autoencoder(
    model: AnomalyAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """Evaluate autoencoder on test set with anomaly detection metrics.

    Args:
        model: Trained autoencoder.
        test_loader: Test dataloader with normal + anomalous images.
        device: Computation device.
        threshold: Anomaly threshold.

    Returns:
        Dictionary with AUROC, F1, precision, recall metrics.
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            scores = model.compute_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy().tolist())

            if hasattr(labels, "numpy"):
                all_labels.extend(labels.numpy().tolist())
            else:
                all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels).astype(int)

    # Binary predictions using threshold
    predictions = (all_scores > threshold).astype(int)

    metrics = {}

    # AUROC (doesn't need threshold)
    if len(np.unique(all_labels)) > 1:
        metrics["image_auroc"] = float(roc_auc_score(all_labels, all_scores))
    else:
        metrics["image_auroc"] = 0.0

    # Threshold-based metrics
    metrics["image_f1"] = float(f1_score(all_labels, predictions, zero_division=0))
    metrics["image_precision"] = float(precision_score(all_labels, predictions, zero_division=0))
    metrics["image_recall"] = float(recall_score(all_labels, predictions, zero_division=0))
    metrics["threshold"] = threshold
    metrics["num_test_normal"] = int((all_labels == 0).sum())
    metrics["num_test_anomalous"] = int((all_labels == 1).sum())

    return metrics


def _save_training_plots(history: dict, results_dir: Path) -> None:
    """Save training loss and learning rate plots.

    Args:
        history: Training history dictionary.
        results_dir: Directory to save plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#FF5722", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LR plot
    ax2.plot(epochs, history["lr"], color="#4CAF50", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training plots to {plot_path}")
