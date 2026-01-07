import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Simple Feedforward Autoencoder for Network Flow Anomaly Detection (AGILE NIDS)
    - Encoder compresses input feature vectors into a latent embedding
    - Decoder reconstructs the original features
    """

    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()
        
        # Encoder: compresses the feature vector
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        # Decoder: reconstructs original vector
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode → decode → reconstruct."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
