import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import models.NormFormer as vqvae
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.dataloader import load_jetclass_label_as_tensor

# === Setup ===
batch_size = 512
num_epochs = 10
lr = 2e-4
start = 10
end = 11
checkpoint_dir = "checkpoints/checkpoints_vqvae_normformer_new"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader = load_jetclass_label_as_tensor(label="HToBB", start=start, end=end, batch_size=batch_size)

model = vqvae.VQVAENormFormer(
    input_dim=3,
    latent_dim=128,
    hidden_dim=256,
    num_heads=8,
    num_blocks=3,
    vq_kwargs={"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0, "sync_nu": 2,
    "replace_freq": 20,},
).to(device)

# === Compute global mean and std with log1p on pt only ===
all_particles = []
for x_particles, _, _ in dataloader:
    all_particles.append(x_particles)
all_particles = torch.cat(all_particles, dim=0)
all_particles = all_particles.transpose(1, 2)
flat_particles = all_particles.reshape(-1, 3)

global_mean = flat_particles.mean(dim=0).to(device)
global_std = flat_particles.std(dim=0).to(device) + 1e-6


checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    print(f"🔁 Resuming from: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    start_epoch = checkpoint["epoch"]

model.eval()
all_orig_jets, all_recon_jets = [], []
dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=15, end=16, batch_size=batch_size)

with torch.no_grad():
    for i, (x_particles, _, _) in enumerate(dataloader_eval):
        if i >= 300:
            break

        x_particles = x_particles.to(device)  # [B, 3, 128]
        x_particles = x_particles.transpose(1, 2)  # [B, 128, 3]

        # Normalize
        x_particles_normed = (x_particles - global_mean) / global_std

        # Model forward
        x_recon_normed, _ = model(x_particles_normed)

        # De-normalize
        x_particles_denorm = x_particles_normed * global_std + global_mean
        x_recon_denorm = x_recon_normed * global_std + global_mean

        # Reconstruct jet-level features
        orig_jet = reconstruct_jet_features_from_particles(x_particles_denorm)
        recon_jet = reconstruct_jet_features_from_particles(x_recon_denorm)

        all_orig_jets.append(orig_jet.to(device))
        all_recon_jets.append(recon_jet.to(device))

all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)

plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename="jet_recon_overlay_normformer_particles_new.png"
)



