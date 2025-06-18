import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqtorch.nn import VectorQuant

class FeatureNormJet(nn.Module):
    """Custom feature-wise normalization using external global stats for jets"""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)  # expected shape: [1, 4]
        self.register_buffer("std", std)

    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def denorm(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean

class EncoderJet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, z_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):  # x: [B, 4]
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        z = self.fc3(x)
        return z

class DecoderJet(nn.Module):
    def __init__(self, z_dim=32, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):  # z: [B, z_dim]
        x = F.gelu(self.fc1(z))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

class VQVAEJet(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_dim=128,
        z_dim=32,
        num_embeddings=256,
        commitment_cost=0.25,
        mean=None,
        std=None,
    ):
        super().__init__()
        assert mean is not None and std is not None, "Must provide global mean and std as [1, 4] tensors"

        self.norm = FeatureNormJet(mean, std)
        self.encoder = EncoderJet(input_dim, hidden_dim, z_dim)
        self.vq = VectorQuant(
            feature_size=z_dim,
            num_codes=num_embeddings,
            beta=commitment_cost,
            sync_nu=0.1,
            affine_lr=0.1,
            affine_groups=1,
            replace_freq=10,
        )
        self.decoder = DecoderJet(z_dim, hidden_dim, input_dim)

    def forward(self, x):  # x: [B, 4]
        x_norm = self.norm(x)
        z = self.encoder(x_norm)
        quantized, vq_loss = self.vq(z)
        x_recon_norm = self.decoder(quantized)
        x_recon = self.norm.denorm(x_recon_norm)
        return x_recon, vq_loss  # [B, 4]



