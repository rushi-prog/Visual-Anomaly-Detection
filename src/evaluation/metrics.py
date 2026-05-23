"""Anomaly detection evaluation metrics.

Computes image-level and pixel-level metrics for anomaly detection:
- Image AUROC: How well the model separates normal vs anomalous images
- Pixel AUROC: How well the model localizes anomalies at pixel level
- F1 Score: Threshold-dependent classification performance
- PRO (Per-Region Overlap): Measures overlap quality of detected regions
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_image_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute image-level anomaly detection metrics.

    Args:
        labels: Ground truth binary labels (0=normal, 1=anomalous).
        scores: Predicted anomaly scores (higher = more anomalous).
        threshold: Classification threshold. If None, uses optimal F1 threshold.

    Returns:
        Dictionary with AUROC, F1, precision, recall, and optimal threshold.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    metrics = {}

    # AUROC
    if len(np.unique(labels)) > 1:
        metrics["image_auroc"] = float(roc_auc_score(labels, scores))

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        metrics["roc_fpr"] = fpr.tolist()
        metrics["roc_tpr"] = tpr.tolist()
    else:
        metrics["image_auroc"] = 0.0

    # Find optimal threshold (maximizes F1)
    if threshold is None:
        threshold = _find_optimal_threshold(labels, scores)
    metrics["threshold"] = float(threshold)

    # Threshold-based metrics
    predictions = (scores >= threshold).astype(int)
    metrics["image_f1"] = float(f1_score(labels, predictions, zero_division=0))
    metrics["image_precision"] = float(precision_score(labels, predictions, zero_division=0))
    metrics["image_recall"] = float(recall_score(labels, predictions, zero_division=0))

    # AUPRC (Area Under Precision-Recall Curve)
    if len(np.unique(labels)) > 1:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
        metrics["image_auprc"] = float(auc(recall_vals, precision_vals))

    return metrics


def compute_pixel_metrics(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> dict[str, float]:
    """Compute pixel-level anomaly localization metrics.

    Args:
        masks: Ground truth binary masks of shape (N, H, W).
            1 = anomalous pixel, 0 = normal pixel.
        anomaly_maps: Predicted anomaly maps of shape (N, H, W).
            Higher values = more anomalous.

    Returns:
        Dictionary with pixel AUROC and per-region overlap (PRO).
    """
    masks = np.asarray(masks).astype(int).flatten()
    anomaly_maps = np.asarray(anomaly_maps).astype(float).flatten()

    metrics = {}

    if len(np.unique(masks)) > 1:
        metrics["pixel_auroc"] = float(roc_auc_score(masks, anomaly_maps))
    else:
        metrics["pixel_auroc"] = 0.0

    return metrics


def compute_pro_score(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    num_thresholds: int = 200,
    integration_limit: float = 0.3,
) -> float:
    """Compute Per-Region Overlap (PRO) score.

    PRO measures how well anomaly predictions overlap with each individual
    connected component in the ground truth. It's less biased towards
    large anomaly regions than pixel AUROC.

    Args:
        masks: Binary ground truth masks (N, H, W).
        anomaly_maps: Predicted anomaly maps (N, H, W).
        num_thresholds: Number of thresholds to sweep.
        integration_limit: Upper limit for FPR integration.

    Returns:
        PRO score (higher is better, max = 1.0).
    """
    from scipy.ndimage import label as scipy_label

    masks = np.asarray(masks).astype(int)
    anomaly_maps = np.asarray(anomaly_maps).astype(float)

    # Normalize anomaly maps to [0, 1]
    if anomaly_maps.max() > anomaly_maps.min():
        anomaly_maps = (anomaly_maps - anomaly_maps.min()) / (anomaly_maps.max() - anomaly_maps.min())

    thresholds = np.linspace(0, 1, num_thresholds)
    pro_values = []
    fpr_values = []

    for thresh in thresholds:
        binary_pred = (anomaly_maps >= thresh).astype(int)

        # FPR
        normal_mask = masks == 0
        if normal_mask.sum() > 0:
            fpr = binary_pred[normal_mask].sum() / normal_mask.sum()
        else:
            fpr = 0.0

        # Per-region overlap
        region_overlaps = []
        for i in range(len(masks)):
            if masks[i].max() == 0:
                continue  # Skip images with no anomaly

            labeled, num_regions = scipy_label(masks[i])
            for region_id in range(1, num_regions + 1):
                region_mask = labeled == region_id
                region_size = region_mask.sum()
                if region_size == 0:
                    continue
                overlap = (binary_pred[i] * region_mask).sum() / region_size
                region_overlaps.append(overlap)

        if region_overlaps:
            avg_overlap = np.mean(region_overlaps)
        else:
            avg_overlap = 0.0

        pro_values.append(avg_overlap)
        fpr_values.append(fpr)

    # Sort by FPR and compute area under curve up to integration_limit
    sorted_indices = np.argsort(fpr_values)
    fpr_sorted = np.array(fpr_values)[sorted_indices]
    pro_sorted = np.array(pro_values)[sorted_indices]

    # Clip to integration limit
    valid = fpr_sorted <= integration_limit
    if valid.sum() < 2:
        return 0.0

    pro_score = float(np.trapz(pro_sorted[valid], fpr_sorted[valid]) / integration_limit)
    return pro_score


def _find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Find the threshold that maximizes F1 score.

    Args:
        labels: Ground truth binary labels.
        scores: Predicted anomaly scores.

    Returns:
        Optimal threshold value.
    """
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_f1 = 0.0
    best_threshold = float(scores.mean())

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)

    return best_threshold
