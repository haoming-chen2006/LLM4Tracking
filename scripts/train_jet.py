import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import models.vqvaeMLP_jet as vqvae
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.dataloader import load_jetclass_label_as_tensor
import vector

batch_size = 128
num_epochs = 50
lr = 2e-4
start = 10
end = 14
checkpoint_dir = "checkpoints/checkpoints_vqvae_jet"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\U0001F4E6 Loading full dataset for global normalization...")
dataloader = load_jetclass_label_as_tensor(label="HToBB", start=start, end=end, batch_size=batch_size)
all_jets = []
for _, jets, _ in dataloader:
    all_jets.append(jets)
all_jets = torch.cat(all_jets, dim=0)  
global_mean = all_jets.mean(dim=0).to(device)  # [1, 1, 4]
global_std = all_jets.std(dim = 0).to(device) + 1e-6
print(f"✅ Computed global mean: {global_mean.flatten()}")
print(f"✅ Computed global std: {global_std.flatten()}")


model = vqvae.VQVAEJet(
    input_dim=4,
    hidden_dim=256,
    z_dim=128,
    num_embeddings=1024,
    commitment_cost=0.25,
    mean=global_mean,  # [1, 1, 4]
    std=global_std     # [1, 1, 4]
)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
recon_loss_fn = nn.MSELoss()
scaler = GradScaler()

start_epoch = 0
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    print(f"\U0001F501 Resuming from: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    epoch_loss = []
    total_recon_loss = []
    total_vq_loss = []

    for _, x_jet, _ in dataloader:
        x_jets = x_jet.to(device)       # (B,  4)

        optimizer.zero_grad()
        with autocast():
            x_recon, loss_dict = model(x_jets)
            recon_loss = recon_loss_fn(x_recon, x_jets)
            vq_loss = loss_dict.get("loss", 0.0) if isinstance(loss_dict, dict) else loss_dict
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(loss.detach())
        total_recon_loss.append(recon_loss.detach())
        total_vq_loss.append(vq_loss.detach())

    if epoch % 10 == 0:
        print(f"\U0001F4C8 Epoch [{epoch+1}/{num_epochs}] - Total: {torch.stack(epoch_loss).mean().item():.4f} | Recon: {torch.stack(total_recon_loss).mean().item():.4f} | VQ: {torch.stack(total_vq_loss).mean().item():.4f}")
        nique_codes = loss_dict['q'].unique().numel()
        print(f"\U0001F4C8 Number of unique codes: {nique_codes}")

    if epoch + 1 == num_epochs:
        save_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, save_path)
        print(f"\U0001F4BE Saved checkpoint: {save_path}")

# === Post-Training Recon Analysis ===
model.eval()

all_orig_jets = []
all_recon_jets = []

with torch.no_grad():
    dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=12, end=18, batch_size=batch_size)
    for i, (_, x_jets, _) in enumerate(dataloader_eval):
        if i >= 300:
            break

        x_jets = x_jets.to(device)
        x_recon, _ = model(x_jets)  # model handles normalization internally

        all_orig_jets.append(x_jets.to(device))
        all_recon_jets.append(x_recon.to(device))

all_orig_jets = torch.cat(all_orig_jets, dim=0)   # [30*B, 4]
all_recon_jets = torch.cat(all_recon_jets, dim=0) # [30*B, 4]
print("all_orig_jets shape: ", all_orig_jets.shape)
print("all_recon_jets shape: ", all_recon_jets.shape)
print("all_orig_jets mean: ", all_orig_jets.mean(dim=0))
print("all_recon_jets mean: ", all_recon_jets.mean(dim=0))
print("all_orig_jets std: ", all_orig_jets.std(dim=0))
print("all_recon_jets std: ", all_recon_jets.std(dim=0))

plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename="jet_recon_overlay_30batch.png"
)  # [B, 4]