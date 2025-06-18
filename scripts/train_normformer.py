import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import models.NormFormer as vqvae
from plot.plot import plot_tensor_jet_features
from dataloader.dataloader import load_jetclass_label_as_tensor

# Directory to store plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# === Setup ===
batch_size = 512
num_epochs = 50
lr = 2e-4
start = 10
end = 14
checkpoint_dir = "checkpoints/checkpoints_vqvae_normformer"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader = load_jetclass_label_as_tensor(label="HToBB", start=start, end=end, batch_size=batch_size)

# === Build Model ===
model = vqvae.VQVAENormFormer(
    input_dim=4,
    latent_dim=128,
    hidden_dim=256,
    num_heads=1,
    num_blocks=2,
    vq_kwargs={"num_codes": 128, "beta": 0.1},
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
recon_loss_fn = nn.MSELoss()
scaler = GradScaler()

# === Resume Checkpoint ===
start_epoch = 0
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, latest)
    print(f"🔄 Resuming from: {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]

# === Training Loop ===
for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    epoch_loss, recon_losses, vq_losses = [], [], []

    for _, x_jets, _ in dataloader:
        x_jets = x_jets.to(device)

        optimizer.zero_grad()
        with autocast():
            x_recon, loss_dict = model(x_jets)
            recon_loss = recon_loss_fn(x_recon, x_jets)
            vq_loss = loss_dict.get("loss", 0.0)
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(loss.detach())
        recon_losses.append(recon_loss.detach())
        vq_losses.append(vq_loss.detach())

    if epoch % 10 == 0 or epoch == start_epoch + num_epochs - 1:
        print(f"📈 Epoch [{epoch+1}/{start_epoch+num_epochs}] | Total: {torch.stack(epoch_loss).mean():.4f} | Recon: {torch.stack(recon_losses).mean():.4f} | VQ: {torch.stack(vq_losses).mean():.4f}")
        unique_codes = loss_dict["q"].unique().numel()
        print(f"🧩 Unique codes used: {unique_codes}")

    if epoch + 1 == num_epochs:
        save_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

# === Post-Training Evaluation ===
model.eval()
all_orig_jets, all_recon_jets = [], []
dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=12, end=18, batch_size=batch_size)

with torch.no_grad():
    for i, (_, x_jets, _) in enumerate(dataloader_eval):
        if i >= 300:
            break
        x_jets = x_jets.to(device)
        x_recon, _ = model(x_jets)
        all_orig_jets.append(x_jets)
        all_recon_jets.append(x_recon)

all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)

print("✅ Evaluation Stats:")
print("Original mean:", all_orig_jets.mean(dim=0))
print("Reconstructed mean:", all_recon_jets.mean(dim=0))
print("Original std:", all_orig_jets.std(dim=0))
print("Reconstructed std:", all_recon_jets.std(dim=0))

plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", "Reconstructed"),
    filename=os.path.join(PLOT_DIR, "jet_recon_overlay_normformer.png")
)



