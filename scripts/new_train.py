import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import models.NormFormer_Flash as vqvae
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.dataloader import load_jetclass_label_as_tensor
import vector
import wandb

# Directory to store plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# === W&B Init ===
wandb.init(project="vqvae-normformer-flash_new", config={
    "batch_size": 512,
    "num_epochs": 1,
    "learning_rate": 2e-4,
    "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0, "sync_nu": 2, "replace_freq": 20, "dim": -1},
})
config = wandb.config

# === Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints/checkpoints_vqvae_normformer_flash"
os.makedirs(checkpoint_dir, exist_ok=True)

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

# === Global Normalization ===
all_particles = [x for x, _, _ in dataloader]
all_particles = torch.cat(all_particles, dim=0).transpose(1, 2)  # [N, 128, 3]
flat_particles = all_particles.reshape(-1, 3)
global_mean = flat_particles.mean(dim=0).to(device)
global_std = flat_particles.std(dim=0).to(device) + 1e-6

# === Optimizer, Loss, Scaler ===
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95))
recon_loss_fn = nn.MSELoss()
scaler = GradScaler()

# === Resume Checkpoint ===
start_epoch = 0
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    print(f"🔁 Resuming from: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]

# === Training Loop ===
for epoch in range(start_epoch, start_epoch + config.num_epochs):
    model.train()
    epoch_loss, total_recon_loss, total_vq_loss = [], [], []

    for x_particles, _, _ in dataloader:
        x_particles = x_particles.to(device).transpose(1, 2)  # [B, 128, 4]
        x_particles_normed = (x_particles - global_mean) / global_std

        optimizer.zero_grad()
        with autocast():
            x_recon, loss_dict = model(x_particles_normed)
            recon_loss = recon_loss_fn(x_recon, x_particles_normed)
            vq_loss = loss_dict.get("loss", 0.0) if isinstance(loss_dict, dict) else loss_dict
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(loss.detach())
        total_recon_loss.append(recon_loss.detach())
        total_vq_loss.append(vq_loss.detach())

    # === Logging ===
    loss_mean = torch.stack(epoch_loss).mean()
    recon_mean = torch.stack(total_recon_loss).mean()
    vq_mean = torch.stack(total_vq_loss).mean()

    wandb.log({
        "epoch": epoch + 1,
        "loss": loss_mean.item(),
        "recon_loss": recon_mean.item(),
        "vq_loss": vq_mean.item(),
        "unique_codes": loss_dict["q"].unique().numel() if isinstance(loss_dict, dict) else 0
    })

    print(f"📈 Epoch [{epoch+1}/{start_epoch + config.num_epochs}] - Total: {loss_mean:.4f} | Recon: {recon_mean:.4f} | VQ: {vq_mean:.4f}")

    # === Save Checkpoint ===
    if epoch + 1 == start_epoch + config.num_epochs:
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
    for i, (x_particles, _, _) in enumerate(dataloader_eval):
        if i >= 300:
            break

        x_particles = x_particles.to(device).transpose(1, 2)
        x_particles_normed = (x_particles - global_mean) / global_std
        x_recon, _ = model(x_particles_normed)
        x_recon_denorm = x_recon * global_std + global_mean

        orig_jet = reconstruct_jet_features_from_particles(x_particles)
        recon_jet = reconstruct_jet_features_from_particles(x_recon_denorm)

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
    filename=os.path.join(PLOT_DIR, "jet_recon_overlay_normformer_particles.png")
)

# Plot model structure
x_particles, _, _ = next(iter(dataloader_eval))
x_particles = x_particles.to(device).transpose(1, 2)
vqvae.plot_model(
    model,
    x_particles,
    saveas=os.path.join(PLOT_DIR, "vqvae_model_normformer_particles.png")
)

