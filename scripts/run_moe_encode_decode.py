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
    print("original shape", x_norm.shape)
    print("Embedding shape:", embed.shape)
    print("Reconstruction shape:", recon.shape)

    # === Plot original vs reconstruction (jet features) ===
    from plot.plot import reconstruct_jet_features_from_particles, plot_tensor_jet_features
    orig_jets = reconstruct_jet_features_from_particles(x_particles)
    recon_jets = reconstruct_jet_features_from_particles(recon)
    plot_tensor_jet_features([orig_jets, recon_jets], labels=["Original", "Reconstruction"], filename="moe_encode_decode_jet_overlay.png")

    # === Embedding space visualization (PCA) ===
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    embed_np = embed.detach().cpu().numpy().reshape(embed.shape[0], -1)
    pca = PCA(n_components=2)
    embed_2d = pca.fit_transform(embed_np)
    plt.figure(figsize=(6, 5))
    plt.scatter(embed_2d[:, 0], embed_2d[:, 1], alpha=0.5, s=10)
    plt.title("Embedding Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("moe_encode_decode_embedding_pca.png", dpi=200)
    print("✅ Saved PCA embedding plot to moe_encode_decode_embedding_pca.png")


if __name__ == "__main__":
    main()
