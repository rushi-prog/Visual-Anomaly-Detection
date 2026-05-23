"""Custom Convolutional Autoencoder for anomaly detection.

One-class learning approach: the autoencoder is trained to reconstruct
ONLY normal images. At inference time, anomalous regions produce high
reconstruction error, generating an anomaly heatmap.

Architecture:
    Encoder: Conv2d → BatchNorm → ReLU → MaxPool (4 blocks)
    Bottleneck: Compressed latent representation
    Decoder: ConvTranspose2d → BatchNorm → ReLU (4 blocks)

Anomaly scoring:
    - Pixel-wise MSE between input and reconstruction
    - Optional SSIM-based scoring for structural awareness
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Deconvolutional block: ConvTranspose2d → BatchNorm → ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2,
            stride=2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.deconv(x)))


class AnomalyAutoencoder(nn.Module):
    """Convolutional Autoencoder for visual anomaly detection.

    Trained on normal images only. Anomalies are detected by measuring
    reconstruction error — regions the model can't reconstruct well
    are likely anomalous.

    Args:
        in_channels: Number of input image channels (3 for RGB).
        latent_dim: Size of the bottleneck feature dimension.
        encoder_channels: Channel sizes for each encoder block.
            Default: [32, 64, 128, 256] (4 downsampling stages).

    Example:
        >>> model = AnomalyAutoencoder(in_channels=3, latent_dim=128)
        >>> img = torch.randn(1, 3, 256, 256)
        >>> reconstruction, latent = model(img)
        >>> anomaly_map = model.compute_anomaly_map(img)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        encoder_channels: list[int] | None = None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels
        decoder_channels = list(reversed(encoder_channels))

        # ── Encoder ──────────────────────────────────────────────
        encoder_layers = []
        prev_ch = in_channels
        for ch in encoder_channels:
            encoder_layers.append(ConvBlock(prev_ch, ch))
            encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = ch
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Bottleneck ───────────────────────────────────────────
        # After 4 pooling stages: 256x256 → 16x16
        # Feature map size: encoder_channels[-1] x 16 x 16
        self.bottleneck_down = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                    # → (B, C, 1, 1)
            nn.Flatten(),                                # → (B, C)
            nn.Linear(encoder_channels[-1], latent_dim), # → (B, latent_dim)
            nn.ReLU(inplace=True),
        )
        self.bottleneck_up = nn.Sequential(
            nn.Linear(latent_dim, encoder_channels[-1] * 16 * 16),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────────
        decoder_layers = []
        prev_ch = decoder_channels[0]
        for ch in decoder_channels[1:]:
            decoder_layers.append(DeconvBlock(prev_ch, ch))
            prev_ch = ch
        # Final upsampling + conv to reconstruct original channels
        decoder_layers.append(DeconvBlock(prev_ch, prev_ch))
        decoder_layers.append(
            nn.Conv2d(prev_ch, in_channels, kernel_size=3, padding=1)
        )
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent representation.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Latent vector of shape (B, latent_dim).
        """
        features = self.encoder(x)
        latent = self.bottleneck_down(features)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstructed image.

        Args:
            z: Latent tensor of shape (B, latent_dim).

        Returns:
            Reconstructed image of shape (B, C, H, W).
        """
        x = self.bottleneck_up(z)
        x = x.view(-1, self.encoder_channels[-1], 16, 16)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple of (reconstruction, latent_vector).
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)

        # Resize reconstruction to match input if sizes differ
        if reconstruction.shape != x.shape:
            reconstruction = F.interpolate(
                reconstruction, size=x.shape[2:], mode="bilinear", align_corners=False,
            )

        return reconstruction, latent

    def compute_anomaly_map(
        self,
        x: torch.Tensor,
        method: str = "mse",
    ) -> torch.Tensor:
        """Compute pixel-level anomaly map from reconstruction error.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            method: Error method — "mse" for mean squared error,
                "l1" for absolute error.

        Returns:
            Anomaly map of shape (B, 1, H, W) with higher values = more anomalous.
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self(x)

            if method == "mse":
                error = (x - reconstruction) ** 2
            elif method == "l1":
                error = torch.abs(x - reconstruction)
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'mse' or 'l1'.")

            # Average across channels to get single-channel anomaly map
            anomaly_map = error.mean(dim=1, keepdim=True)

        return anomaly_map

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image-level anomaly score.

        Takes the maximum value of the anomaly map as the image-level score.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Anomaly scores of shape (B,).
        """
        anomaly_map = self.compute_anomaly_map(x)
        # Use max of anomaly map as image-level score
        scores = anomaly_map.view(x.size(0), -1).max(dim=1)[0]
        return scores


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss for perceptual quality.

    SSIM captures structural information that MSE misses, making it
    better at detecting subtle texture anomalies.

    Args:
        window_size: Size of the sliding window. Must be odd.
        channels: Number of image channels.
    """

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels

        # Create Gaussian window
        _1d_window = self._gaussian(window_size, 1.5).unsqueeze(1)  # (N, 1)
        _2d_window = _1d_window.mm(_1d_window.t())  # (N, N)
        window = _2d_window.expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        gauss = torch.tensor([
            -(x - window_size // 2) ** 2 / (2 * sigma ** 2)
            for x in range(window_size)
        ]).exp()
        return gauss / gauss.sum()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM).

        Args:
            x: Predicted image.
            y: Target image.

        Returns:
            SSIM loss (lower = more similar).
        """
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=self.channels)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=self.channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, self.window, padding=self.window_size // 2, groups=self.channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, self.window, padding=self.window_size // 2, groups=self.channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size // 2, groups=self.channels) - mu_xy

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )

        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Combined MSE + SSIM loss for autoencoder training.

    MSE provides pixel-level accuracy while SSIM captures structural
    and perceptual quality.

    Args:
        ssim_weight: Weight for SSIM component (MSE weight = 1 - ssim_weight).
    """

    def __init__(self, ssim_weight: float = 0.5):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            x: Predicted (reconstructed) image.
            y: Target (original) image.

        Returns:
            Combined loss value.
        """
        mse_loss = self.mse(x, y)
        ssim_loss = self.ssim(x, y)
        return (1 - self.ssim_weight) * mse_loss + self.ssim_weight * ssim_loss
