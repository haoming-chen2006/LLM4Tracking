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
num_epochs = 1
lr = 2e-4
start = 10
end = 11
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
    vq_kwargs={"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0, "sync_nu": 2,
    "replace_freq": 20,
    "dim": -1},).to(device)
all_particles = []
for x_particles, _, _ in dataloader:
    all_particles.append(x_particles)
all_particles = torch.cat(all_particles, dim=0)  
all_particles = all_particles.transpose(1, 2)       
flat_particles = all_particles.reshape(-1, 3) 

global_mean = flat_particles.mean(dim=0).to(device)
global_std = flat_particles.std(dim=0).to(device) + 1e-6


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

    for x_particles, _, _ in dataloader:
        x_particles = x_particles.to(device)  # [B, 4, 128]
        x_particles = x_particles.transpose(1, 2)  # -> [B, 128, 4]

        # 🧼 Apply global normalization before feeding into the model
        x_particles_normed = (x_particles - global_mean) / global_std

        optimizer.zero_grad()

        with autocast():
            x_recon, loss_dict = model(x_particles_normed)  # model expects normalized input
            recon_loss = recon_loss_fn(x_recon, x_particles_normed)  # compare in normalized space

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
    for i, (x_particles, _, _) in enumerate(dataloader_eval):
        if i >= 300:
            break

        x_particles = x_particles.to(device).transpose(1, 2)  # [B, 128, 4]
        x_particles_normed = (x_particles - global_mean) / global_std
        x_recon, _ = model(x_particles_normed)
        x_recon_denorm = x_recon * global_std + global_mean

        orig_jet = reconstruct_jet_features_from_particles(x_particles)
        recon_jet = reconstruct_jet_features_from_particles(x_recon_denorm)

        all_orig_jets.append(orig_jet.to(device))
        all_recon_jets.append(recon_jet.to(device))

all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)

print("all_orig_jets shape:", all_orig_jets.shape)
print("all_recon_jets shape:", all_recon_jets.shape)
print("Original mean:", all_orig_jets.mean(dim=0))
print("Reconstructed mean:", all_recon_jets.mean(dim=0))
print("Original std:", all_orig_jets.std(dim=0))
print("Reconstructed std:", all_recon_jets.std(dim=0))
print("plotting jet features...")
plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename="jet_recon_overlay_normformer_particles.png"
)
x_particles,x_jets,_ = next(iter(dataloader_eval))
x_particles = x_particles.to(device)
x_particles = x_particles.transpose(1, 2)  # [B, 4, 128]
vqvae.plot_model(model, x_particles,saveas="vqvae_model_normformer_particles.png")
