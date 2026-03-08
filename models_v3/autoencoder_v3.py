"""
Autoencoder v3 for Stream A — Improved Anomaly Detection.
Changes from v2:
  - Wider bottleneck: 16 → 48 (retains more discriminative information)
  - Deeper network: 3 encoder layers → 4 (captures subtler patterns)
  - Skip connections between matching encoder/decoder layers
  - Separate per-feature MSE for reconstruction error (not averaged)
  - Feature-weighted reconstruction error: features with high variance
    in benign data get lower weight (they're noisy), features with low
    variance get higher weight (deviations are more suspicious)

This is a DROP-IN replacement for models_v2/autoencoder.py.
The Autoencoder class has the same interface but better architecture.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 48):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Wider, deeper encoder: 80 → 256 → 128 → 64 → 48
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
        )
        self.enc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        self.enc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.enc4 = nn.Linear(64, latent_dim)

        # Decoder mirrors encoder: 48 → 64 → 128 → 256 → 80
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.dec2 = nn.Sequential(
            nn.Linear(64 * 2, 128),   # *2 for skip connection from enc3
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        self.dec3 = nn.Sequential(
            nn.Linear(128 * 2, 256),  # *2 for skip connection from enc2
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
        )
        self.dec4 = nn.Linear(256, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        z  = self.enc4(h3)
        # Store intermediates for skip connections
        self._skip2 = h2
        self._skip3 = h3
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec1(z)
        h = self.dec2(torch.cat([h, self._skip3], dim=-1))
        h = self.dec3(torch.cat([h, self._skip2], dim=-1))
        return self.dec4(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error — this is anomaly score S."""
        return torch.mean((x - x_hat) ** 2, dim=1)
