import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import models.NormFormer_Flash as vqvae
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.masked_dataloader import load_jetclass_label_as_tensor
import vector
import wandb

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

wandb.init(project="vqvae-normformer-flash_masked", config={
    "batch_size": 512,
    "num_epochs": 10,
    "learning_rate": 2e-4,
    "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0, "sync_nu": 2, "replace_freq": 20, "dim": -1},
})
config = wandb.config

# === Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints/checkpoints_vqvae_normformer_flash_masked"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load data with masking
dataloader = load_jetclass_label_as_tensor(label="HToBB", start=70, end=80, batch_size=config.batch_size)

# === Model ===
model = vqvae.VQVAENormFormer(
    input_dim=3,
    latent_dim=128,
    hidden_dim=256,
    num_heads=8,
    num_blocks=3,
    vq_kwargs=config.vq_kwargs,
).to(device)

# === Compute Global Normalization (log-pt) ===
all_parts = []
all_masks = []
for x_particles, _, _, mask in dataloader:
    all_parts.append(x_particles)
    all_masks.append(mask)
all_parts = torch.cat(all_parts, dim=0).transpose(1, 2)
all_masks = torch.cat(all_masks, dim=0)
all_parts[:, :, 0] = torch.log(all_parts[:, :, 0] + 1e-6)
flat_parts = all_parts.reshape(-1, 3)
flat_masks = all_masks.reshape(-1)
valid_parts = flat_parts[flat_masks.bool()]

global_mean = valid_parts.mean(dim=0).to(device)
global_std = valid_parts.std(dim=0).to(device) + 1e-6

# === Optimizer, Loss, Scaler ===
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95))
recon_loss_fn = nn.MSELoss()
scaler = GradScaler()

# === Training Loop ===
for epoch in range(config.num_epochs):
    model.train()
    epoch_loss, total_recon_loss, total_vq_loss = [], [], []

    for x_particles, _, _, mask in dataloader:
        x_particles = x_particles.to(device).transpose(1, 2)
        mask = mask.to(device)
        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
        x_particles_normed = (x_particles - global_mean) / global_std

        optimizer.zero_grad()
        with autocast():
            x_recon, loss_dict = model(x_particles_normed, mask=mask)
            diff = (x_recon - x_particles_normed) ** 2
            recon_loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum()
            vq_loss = loss_dict.get("loss", 0.0) if isinstance(loss_dict, dict) else loss_dict
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(loss.detach())
        total_recon_loss.append(recon_loss.detach())
        total_vq_loss.append(vq_loss.detach())

    loss_mean = torch.stack(epoch_loss).mean()
    recon_mean = torch.stack(total_recon_loss).mean()
    vq_mean = torch.stack(total_vq_loss).mean()

    wandb.log({
        "epoch": epoch + 1,
        "loss": loss_mean.item(),
        "recon_loss": recon_mean.item(),
        "vq_loss": vq_mean.item(),
        "unique_codes": loss_dict["q"].unique().numel() if isinstance(loss_dict, dict) else 0,
    })

    print(f"📈 Epoch [{epoch+1}/{config.num_epochs}] - Total: {loss_mean:.4f} | Recon: {recon_mean:.4f} | VQ: {vq_mean:.4f}")

    if epoch + 1 == config.num_epochs:
        save_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

# === Post-Training Analysis ===
model.eval()
dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=15, end=18, batch_size=config.batch_size)
all_orig_jets, all_recon_jets = [], []

with torch.no_grad():
    for i, (x_particles, _, _, mask) in enumerate(dataloader_eval):
        if i >= 300:
            break
        x_particles = x_particles.to(device).transpose(1, 2)
        mask = mask.to(device)
        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
        x_norm = (x_particles - global_mean) / global_std
        x_recon, _ = model(x_norm, mask=mask)
        x_recon_denorm = x_recon * global_std + global_mean
        x_recon_denorm[:, :, 0] = torch.exp(x_recon_denorm[:, :, 0]) - 1e-6

        x_particles[:, :, 0] = torch.exp(x_particles[:, :, 0]) - 1e-6
        orig_jet = reconstruct_jet_features_from_particles(x_particles * mask.unsqueeze(-1))
        recon_jet = reconstruct_jet_features_from_particles(x_recon_denorm * mask.unsqueeze(-1))

        all_orig_jets.append(orig_jet)
        all_recon_jets.append(recon_jet)

all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)

wandb.log({
    "eval/orig_mean": all_orig_jets.mean(dim=0).tolist(),
    "eval/recon_mean": all_recon_jets.mean(dim=0).tolist(),
    "eval/orig_std": all_orig_jets.std(dim=0).tolist(),
    "eval/recon_std": all_recon_jets.std(dim=0).tolist(),
})

print("📊 Final Evaluation Stats")
print("Original mean:", all_orig_jets.mean(dim=0))
print("Reconstructed mean:", all_recon_jets.mean(dim=0))
print("Original std:", all_orig_jets.std(dim=0))
print("Reconstructed std:", all_recon_jets.std(dim=0))

plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename=os.path.join(PLOT_DIR, "jet_recon_overlay_normformer_particles_masked.png"),
)

# Plot model structure
x_particles, _, _, mask = next(iter(dataloader_eval))
x_particles = x_particles.to(device).transpose(1, 2)
mask = mask.to(device)
x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
vqvae.plot_model(
    model,
    x_particles,
    masks=mask,
    saveas=os.path.join(PLOT_DIR, "vqvae_model_normformer_particles_masked.png"),
)
