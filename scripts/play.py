
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import models.NormFormer as vqvae
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.dataloader import load_jetclass_label_as_tensor
import vector

# === Setup ===
batch_size = 512
num_epochs = 10
lr = 2e-4
start = 10
end = 12
checkpoint_dir = "checkpoints/checkpoints_vqvae_normformer_new"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader = load_jetclass_label_as_tensor(label="HToBB", start=start, end=end, batch_size=batch_size)

# === Model ===
model = vqvae.VQVAENormFormer(
    input_dim=3,
    latent_dim=128,
    hidden_dim=256,
    num_heads=8,
    num_blocks=3,
    vq_kwargs={"num_codes": 2048, "beta": 0.25,"kmeans_init": "true",  "affine_lr": 0.0, "sync_nu": 2,
    "replace_freq": 20,
    "dim": -1},).to(device)
for x_particles, _, _ in dataloader:
    x_particles = x_particles.to(device)
    print(x_particles.shape)
    x_particles = x_particles.transpose(1, 2)    # [B, 128, 4]
    model(x_particles)
    
