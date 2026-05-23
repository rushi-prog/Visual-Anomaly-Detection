"""Phase 2 tests — Custom autoencoder architecture and components.

Tests the custom convolutional autoencoder:
- Forward pass shapes
- Encoder/decoder independently
- Anomaly map generation
- Anomaly score computation
- SSIM loss
- Combined loss
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestAnomalyAutoencoder:
    """Test the custom autoencoder architecture."""

    @pytest.fixture
    def model(self):
        from src.models.autoencoder import AnomalyAutoencoder
        return AnomalyAutoencoder(
            in_channels=3,
            latent_dim=128,
            encoder_channels=[32, 64, 128, 256],
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a small batch of fake images."""
        return torch.randn(2, 3, 256, 256)

    def test_forward_pass_shapes(self, model, sample_batch):
        """Verify output shapes match input shapes."""
        reconstruction, latent = model(sample_batch)
        assert reconstruction.shape == sample_batch.shape, \
            f"Expected {sample_batch.shape}, got {reconstruction.shape}"
        assert latent.shape == (2, 128), \
            f"Expected (2, 128), got {latent.shape}"

    def test_encoder_output_shape(self, model, sample_batch):
        """Test encoder produces correct latent vector size."""
        latent = model.encode(sample_batch)
        assert latent.shape == (2, 128)

    def test_decoder_output_shape(self, model):
        """Test decoder produces correct image shape."""
        latent = torch.randn(2, 128)
        decoded = model.decode(latent)
        # Decoder output should be spatial
        assert decoded.dim() == 4
        assert decoded.shape[1] == 3  # RGB channels

    def test_reconstruction_in_valid_range(self, model, sample_batch):
        """Verify sigmoid output is in [0, 1]."""
        reconstruction, _ = model(sample_batch)
        assert reconstruction.min() >= 0.0, "Reconstruction has negative values"
        assert reconstruction.max() <= 1.0, "Reconstruction exceeds 1.0"

    def test_anomaly_map_shape(self, model, sample_batch):
        """Test anomaly map generation."""
        anomaly_map = model.compute_anomaly_map(sample_batch)
        assert anomaly_map.shape[0] == 2  # Batch size
        assert anomaly_map.shape[1] == 1  # Single channel
        assert anomaly_map.shape[2] == 256  # Height
        assert anomaly_map.shape[3] == 256  # Width

    def test_anomaly_map_nonnegative(self, model, sample_batch):
        """Anomaly map (MSE) should be non-negative."""
        anomaly_map = model.compute_anomaly_map(sample_batch, method="mse")
        assert (anomaly_map >= 0).all()

    def test_anomaly_score_shape(self, model, sample_batch):
        """Test image-level anomaly score."""
        scores = model.compute_anomaly_score(sample_batch)
        assert scores.shape == (2,)

    def test_different_latent_dims(self):
        """Test model with different latent dimensions."""
        from src.models.autoencoder import AnomalyAutoencoder

        for latent_dim in [64, 128, 256]:
            model = AnomalyAutoencoder(latent_dim=latent_dim)
            x = torch.randn(1, 3, 256, 256)
            recon, latent = model(x)
            assert latent.shape[1] == latent_dim

    def test_single_image_input(self, model):
        """Test with batch size 1."""
        x = torch.randn(1, 3, 256, 256)
        recon, latent = model(x)
        assert recon.shape == x.shape

    def test_model_parameter_count(self, model):
        """Verify model has a reasonable number of parameters."""
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
        # Should be in the range of 1M-20M for this architecture
        assert total < 50_000_000, f"Model has too many params: {total}"


class TestSSIMLoss:
    """Test SSIM loss computation."""

    def test_ssim_identical_images(self):
        """SSIM loss should be ~0 for identical images."""
        from src.models.autoencoder import SSIMLoss
        ssim = SSIMLoss()
        x = torch.rand(1, 3, 64, 64)
        loss = ssim(x, x)
        assert loss.item() < 0.01, f"SSIM loss for identical images should be ~0, got {loss.item()}"

    def test_ssim_different_images(self):
        """SSIM loss should be > 0 for different images."""
        from src.models.autoencoder import SSIMLoss
        ssim = SSIMLoss()
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)
        loss = ssim(x, y)
        assert loss.item() > 0.0

    def test_ssim_output_is_scalar(self):
        from src.models.autoencoder import SSIMLoss
        ssim = SSIMLoss()
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss = ssim(x, y)
        assert loss.dim() == 0  # Scalar


class TestCombinedLoss:
    """Test combined MSE + SSIM loss."""

    def test_combined_loss_output(self):
        from src.models.autoencoder import CombinedLoss
        loss_fn = CombinedLoss(ssim_weight=0.5)
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        loss = loss_fn(x, y)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_combined_loss_gradient_flows(self):
        """Verify gradients flow through the combined loss."""
        from src.models.autoencoder import CombinedLoss
        loss_fn = CombinedLoss(ssim_weight=0.5)
        x = torch.rand(1, 3, 64, 64, requires_grad=True)
        y = torch.rand(1, 3, 64, 64)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


class TestMetrics:
    """Test evaluation metrics."""

    def test_image_metrics_perfect(self):
        """Perfect predictions should give AUROC=1."""
        import numpy as np
        from src.evaluation.metrics import compute_image_metrics
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        metrics = compute_image_metrics(labels, scores)
        assert metrics["image_auroc"] == 1.0

    def test_image_metrics_keys(self):
        import numpy as np
        from src.evaluation.metrics import compute_image_metrics
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = compute_image_metrics(labels, scores)
        assert "image_auroc" in metrics
        assert "image_f1" in metrics
        assert "threshold" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
