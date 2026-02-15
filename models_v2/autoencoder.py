"""
Autoencoder for Stream A — Anomaly Detection (Unsupervised).
Algorithm Step 8: Pass V to Autoencoder AM; compute reconstruction error → S.

Improvements over v1:
  - Batch normalisation for stable training
  - Dropout for regularisation / better generalisation
  - Separate encode() method so latent embedding can be reused
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error — this is anomaly score S."""
        return torch.mean((x - x_hat) ** 2, dim=1)
