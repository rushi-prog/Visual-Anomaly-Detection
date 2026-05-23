"""Phase 3 tests — Benchmarking and visualization utilities.

Tests the evaluation, visualization, and benchmarking components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestMetricsAdvanced:
    """Advanced metric computation tests."""

    def test_pixel_metrics_perfect(self):
        from src.evaluation.metrics import compute_pixel_metrics
        masks = np.array([[[0, 0], [1, 1]]])
        anomaly_maps = np.array([[[0.1, 0.2], [0.9, 0.8]]])
        metrics = compute_pixel_metrics(masks, anomaly_maps)
        assert metrics["pixel_auroc"] == 1.0

    def test_pixel_metrics_random(self):
        from src.evaluation.metrics import compute_pixel_metrics
        rng = np.random.default_rng(42)
        masks = rng.integers(0, 2, size=(10, 64, 64))
        anomaly_maps = rng.random((10, 64, 64))
        metrics = compute_pixel_metrics(masks, anomaly_maps)
        # Random should be ~0.5
        assert 0.3 < metrics["pixel_auroc"] < 0.7

    def test_optimal_threshold_search(self):
        from src.evaluation.metrics import _find_optimal_threshold
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = _find_optimal_threshold(labels, scores)
        # Threshold should be between normal and anomalous scores
        assert 0.3 < threshold < 0.7


class TestVisualization:
    """Test visualization utilities."""

    def test_overlay_anomaly_map_shape(self):
        from src.evaluation.visualize import overlay_anomaly_map
        image = np.random.rand(256, 256, 3)
        anomaly_map = np.random.rand(256, 256)
        overlay = overlay_anomaly_map(image, anomaly_map)
        assert overlay.shape == (256, 256, 3)
        assert overlay.dtype == np.uint8

    def test_overlay_anomaly_map_uint8_input(self):
        from src.evaluation.visualize import overlay_anomaly_map
        image = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        anomaly_map = np.random.rand(128, 128)
        overlay = overlay_anomaly_map(image, anomaly_map)
        assert overlay.shape == (128, 128, 3)

    def test_overlay_different_sizes(self):
        """Test that overlay handles size mismatch between image and anomaly map."""
        from src.evaluation.visualize import overlay_anomaly_map
        image = np.random.rand(256, 256, 3)
        anomaly_map = np.random.rand(64, 64)  # Smaller anomaly map
        overlay = overlay_anomaly_map(image, anomaly_map)
        assert overlay.shape == (256, 256, 3)

    def test_plot_model_comparison(self):
        """Test model comparison bar chart creation."""
        import matplotlib
        matplotlib.use("Agg")
        from src.evaluation.visualize import plot_model_comparison
        results = {
            "PatchCore": {"image_auroc": 0.99},
            "STFPM": {"image_auroc": 0.97},
            "Autoencoder": {"image_auroc": 0.88},
        }
        fig = plot_model_comparison(results, metric="image_auroc")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_roc_curves(self):
        """Test ROC curve plotting."""
        import matplotlib
        matplotlib.use("Agg")
        from src.evaluation.visualize import plot_roc_curves
        results = {
            "PatchCore": {
                "roc_fpr": [0.0, 0.1, 0.5, 1.0],
                "roc_tpr": [0.0, 0.8, 0.95, 1.0],
                "image_auroc": 0.95,
            },
        }
        fig = plot_roc_curves(results)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestBenchmark:
    """Test benchmarking utilities (lightweight, no training)."""

    def test_load_benchmark_raises_when_missing(self):
        from src.evaluation.benchmark import load_benchmark_results
        with pytest.raises(FileNotFoundError):
            load_benchmark_results(Path("/nonexistent/path.csv"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
