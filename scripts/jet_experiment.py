import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import models.vqvaeMLP_jet as vqvae
from plot.plot import plot_tensor_jet_features
from dataloader.dataloader import load_jetclass_label_as_tensor
import vector

# Directory to store plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

batch_size = 128
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
global_mean = all_jets.mean(dim=0).to(device)
global_std = all_jets.std(dim=0).to(device) + 1e-6
print(f"✅ Computed global mean: {global_mean.flatten()}")
print(f"✅ Computed global std: {global_std.flatten()}")

model = vqvae.VQVAEJet(
    input_dim=4,
    hidden_dim=256,
    z_dim=128,
    num_embeddings=1024,
    commitment_cost=0.25,
    mean=global_mean,
    std=global_std
)
model = model.to(device)
model.eval()

# Load checkpoint if available
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    print(f"\U0001F501 Resuming from: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])

# === Begin Diagnostic Forward Pass ===
all_orig_jets = []
all_recon_jets = []
print("\n🚑 Running diagnostics on model outputs...")

with torch.no_grad():
    dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=12, end=13, batch_size=batch_size)
    for i, (_, x_jets, _) in enumerate(dataloader_eval):
        if i >= 1: break  # Just one batch for inspection

        x_jets = x_jets.to(device)
        print(f"\n🔍 Batch {i+1} - Input shape: {x_jets.shape}")

        # Step-by-step internal inspection
        x_norm = model.norm(x_jets)
        z = model.encoder(x_norm)
        z_q, vq_out = model.vq(z)
        x_recon_norm = model.decoder(z_q)
        x_recon = model.norm.denorm(x_recon_norm)

        # Log internal stats
        print(f"\n📦 Input jets (mean): {x_jets.mean(dim=0)}")
        print(f"📦 Input jets (std):  {x_jets.std(dim=0)}")
        print(f"\n🧠 Normed input (std):  {x_norm.std(dim=0)}")
        print(f"🎯 Encoder output z (std): {z.std(dim=0).mean():.5f}")
        print(f"📚 Quantized z_q (std):     {z_q.std(dim=0).mean():.5f}")
        print(f"🔁 Decoder output (std):   {x_recon_norm.std(dim=0).mean():.5f}")
        print(f"🎯 Final x_recon (std):     {x_recon.std(dim=0)}")
        print(f"🧮 Unique codes used:       {torch.unique(vq_out['q']).numel()}")

        all_orig_jets.append(x_jets)
        all_recon_jets.append(x_recon)

# Stack and plot
all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)
plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename=os.path.join(PLOT_DIR, "jet_recon_diagnostic.png")
)
