
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
# Use the flash attention version of the NormFormer VQVAE
import models.NormFormer_Flash as vqvae
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.masked_dataloader import load_jetclass_label_as_tensor
import vector

# Directory to store plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# === Setup ===
batch_size = 512
num_epochs = 10
lr = 2e-4
start = 10
end = 15
checkpoint_dir = "checkpoints/checkpoints_vqvae_normformer_mask"
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
    vq_kwargs={"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0, "sync_nu": 2,
    "replace_freq": 20,
    "dim": -1},).to(device)
# --- Compute global mean and std with log-pt transformation ---
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


optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
recon_loss_fn = nn.MSELoss()
scaler = GradScaler()

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

# === Training Loop with Global Normalization ===
for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    epoch_loss, total_recon_loss, total_vq_loss = [], [], []

    for x_particles, _, _, mask in dataloader:
        x_particles = x_particles.to(device)  # [B, 4, 128]
        mask = mask.to(device)
        x_particles = x_particles.transpose(1, 2)  # -> [B, 128, 4]
        # Apply log transform on pt
        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)

        # 🧼 Apply global normalization before feeding into the model
        x_particles_normed = (x_particles - global_mean) / global_std

        optimizer.zero_grad()

        with autocast():
            x_recon, loss_dict = model(x_particles_normed,mask=mask)  # model expects normalized input
            mse = (x_recon - x_particles_normed) ** 2
            recon_loss = (mse * mask.unsqueeze(-1)).sum() / mask.sum()

            vq_loss = loss_dict.get("loss", 0.0) if isinstance(loss_dict, dict) else loss_dict
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(loss.detach())
        total_recon_loss.append(recon_loss.detach())
        total_vq_loss.append(vq_loss.detach())

    # === Logging and Checkpointing ===
    if epoch % 1 == 0 or epoch + 1 == num_epochs:
        print(f"📈 Epoch [{epoch+1}/{start_epoch + num_epochs}] - Total: {torch.stack(epoch_loss).mean():.4f} | Recon: {torch.stack(total_recon_loss).mean():.4f} | VQ: {torch.stack(total_vq_loss).mean():.4f}")
        if isinstance(loss_dict, dict):
            unique_codes = loss_dict["q"].unique().numel()
            print(f"🧩 Unique codes used: {unique_codes}")

    if epoch + 1 == start_epoch + num_epochs:
        save_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

# === Post-Training Analysis ===
model.eval()
all_orig_jets, all_recon_jets = [], []
dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=15, end=18, batch_size=batch_size)

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

        all_orig_jets.append(orig_jet.to(device))
        all_recon_jets.append(recon_jet.to(device))

all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)

plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename=os.path.join(PLOT_DIR, "jet_recon_overlay_normformer_particles.png"),
)
x_particles, _, _, mask = next(iter(dataloader_eval))
x_particles = x_particles.to(device).transpose(1, 2)
mask = mask.to(device)
x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
vqvae.plot_model(
    model,
    x_particles,
    masks=mask,
    saveas=os.path.join(PLOT_DIR, "vqvae_model_normformer_particles.png"),
)

