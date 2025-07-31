import os
import sys
import torch
import matplotlib.pyplot as plt

# Allow importing modules from the repository root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import (
    load_jetclass_label_as_dataset,
    load_jetclass_label_as_tensor,
)
from models.MOE import VQVAENormFormer
from scripts.moe import MOE_CONFIGS, compute_global_stats


def load_latest_checkpoint(model: VQVAENormFormer, ckpt_dir: str, device: torch.device) -> None:
    """Load the most recent checkpoint from ``ckpt_dir`` into ``model``."""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("moe_epoch_") and f.endswith(".pth")]
    if not ckpts:
        print(f"No checkpoint found in {ckpt_dir}")
        return
    latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    path = os.path.join(ckpt_dir, latest)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    print(f"Loaded checkpoint: {path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MOE_CONFIGS["MOE_med"]

    # Compute global normalisation statistics
    dataset = load_jetclass_label_as_dataset(
        label="HToBB", start=config["start"], end=config["start"] + 1
    )
    mean, std = compute_global_stats(dataset, config["batch_size"], False, False)
    mean = mean.to(device)
    std = std.to(device)

    # Build and load model
    model = VQVAENormFormer(
        input_dim=3,
        latent_dim=16,
        hidden_dim=128,
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)
    load_latest_checkpoint(model, config["checkpoint_dir"], device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Grab a single batch for testing
    loader = load_jetclass_label_as_tensor(
        label="HToBB",
        start=config["start"],
        end=config["start"] + 1,
        batch_size=config["batch_size"],
    )
    x_particles, _, _ = next(iter(loader))
    x_particles = x_particles.to(device).transpose(1, 2)
    x_norm = (x_particles - mean) / std

    with torch.no_grad():
        embed = model.encode(x_norm)
        recon = model.decode(embed)

    print("Embedding shape:", embed.shape)
    print("Reconstruction shape:", recon.shape)

    # Simple visualisation
    e_cpu = embed[..., :2].detach().cpu().view(-1, 2)
    r_cpu = recon[..., :2].detach().cpu().view(-1, 2)
    x_cpu = x_particles[..., :2].detach().cpu().view(-1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(e_cpu[:, 0], e_cpu[:, 1], s=4, alpha=0.5)
    axes[0].set_title("Latent space (first 2 dims)")

    axes[1].scatter(x_cpu[:, 0], x_cpu[:, 1], label="orig", s=4, alpha=0.5)
    axes[1].scatter(r_cpu[:, 0], r_cpu[:, 1], label="recon", s=4, alpha=0.5)
    axes[1].legend()
    axes[1].set_title("Original vs Reconstruction")

    fig.tight_layout()
    plt.savefig("test_encode_decode.png")
    print("Saved plot to test_encode_decode.png")


if __name__ == "__main__":
    main()
