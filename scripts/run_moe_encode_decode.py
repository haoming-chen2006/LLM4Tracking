import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt

# Allow importing modules from the repository root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import (
    load_jetclass_label_as_dataset,
    load_jetclass_label_as_tensor,
)
from models.MOE import VQVAENormFormer
from scripts.moe import compute_global_stats


CONFIGS = {
    "new": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash",
        "vq_kwargs": {
            "num_codes": 2048,
            "beta": 0.25,
            "affine_lr": 0.0,
            "sync_nu": 2,
            "replace_freq": 20,
            "dim": -1,
        },
    },
    "MOE_med": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_med",
        "vq_kwargs": {
            "num_codes": 4096,
            "beta": 0.8,
            "affine_lr": 1.0,
            "sync_nu": 2,
            "replace_freq": 3,
            "dim": -1,
        },
    },
    "MOE_large": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_large",
        "vq_kwargs": {
            "num_codes": 8192,
            "beta": 0.9,
            "affine_lr": 0.0,
            "sync_nu": 5,
            "replace_freq": 2,
            "dim": -1,
        },
    },
    "masked": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash_masked",
        "vq_kwargs": {
            "num_codes": 2048,
            "beta": 0.25,
            "affine_lr": 0.0,
            "sync_nu": 2,
            "replace_freq": 20,
            "dim": -1,
        },
    },
    "particle": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_new",
        "vq_kwargs": {
            "num_codes": 2048,
            "beta": 0.25,
            "affine_lr": 0.0,
            "sync_nu": 2,
            "replace_freq": 20,
            "dim": -1,
        },
    },
}

LABELS = [
    "HToBB",
    "HToCC",
    "HToGG",
    "HToWW4Q",
    "HToWW2Q1L",
    "ZToQQ",
    "WToQQ",
    "TTBar",
    "TTBarLep",
    "ZJetsToNuNu",
]


def load_checkpoint(
    model: VQVAENormFormer,
    ckpt_dir: str,
    device: torch.device,
    epoch: int | None = None,
) -> None:
    """Load checkpoint ``epoch`` from ``ckpt_dir`` into ``model``.

    If ``epoch`` is ``None`` the latest checkpoint is used."""
    ckpt_path = None
    if epoch is not None:
        for prefix in ("moe_epoch_", "vqvae_epoch_"):
            candidate = os.path.join(ckpt_dir, f"{prefix}{epoch}.pth")
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
    if ckpt_path is None:
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if not ckpts:
            print(f"No checkpoint found in {ckpt_dir}")
            return
        ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode/decode with a trained model")
    parser.add_argument(
        "--model",
        choices=CONFIGS.keys(),
        default="MOE_med",
        help="Model configuration to use",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch number of the checkpoint to load (default: latest)",
    )
    parser.add_argument(
        "--label",
        choices=LABELS,
        default="HToBB",
        help="JetClass label to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CONFIGS[args.model]

    dataset = load_jetclass_label_as_dataset(label=args.label, start=10, end=11)
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
    load_checkpoint(model, config["checkpoint_dir"], device, args.epoch)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    loader = load_jetclass_label_as_tensor(
        label=args.label,
        start=10,
        end=11,
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
