import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
    plot_difference,
)
# -----------------------------------------------------------------------------
# Configuration: choose which training script to mimic
# Options: "new", "masked", "particle"
# Default to the masked configuration which applies log-pt transformation
TRAIN_TYPE = "new_medium"
WORLD_SIZE = 4

CONFIGS = {
    "new": {
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 2e-4,
        "start": 10,
        "end": 40,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/checkpoints_vqvae_normformer_flash",
    },
    "new_large": {
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 2e-4,
        "start": 10,
        "end": 40,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "moe_kwargs": {"num_experts": 8, "expert_capacity": 64},
        "checkpoint_dir": "checkpoints/checkpoints_vqvae_moe_large",
    },
    "new_medium": {
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 2e-4,
        "start": 30,
        "end": 40,
        "vq_kwargs": {"num_codes": 4096, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "moe_kwargs": {"num_experts": 4, "expert_capacity": 32},
        "checkpoint_dir": "checkpoints/checkpoints_vqvae_moe_medium",
    },
    "masked": {
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 2e-4,
        "start": 70,
        "end": 80,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/checkpoints_vqvae_normformer_flash_masked",
    },
    "particle": {
        "batch_size": 512,
        "num_epochs": 1,
        "learning_rate": 2e-4,
        "start": 10,
        "end": 11,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/checkpoints_vqvae_normformer_new",
    },
}
# -----------------------------------------------------------------------------

def setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup() -> None:
    dist.destroy_process_group()


def compute_global_stats(dataset, batch_size, log_pt=False, use_mask=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_parts = []
    all_masks = [] if use_mask else None
    for batch in loader:
        if use_mask:
            x_part, _, _, mask = batch
            all_masks.append(mask)
        else:
            x_part, _, _ = batch
        all_parts.append(x_part)
    particles = torch.cat(all_parts, dim=0).transpose(1, 2)
    if use_mask:
        masks = torch.cat(all_masks, dim=0)
        particles[:, :, 0] = torch.log(particles[:, :, 0] + 1e-6)
        flat = particles.reshape(-1, particles.shape[-1])
        valid = masks.reshape(-1).bool()
        flat = flat[valid]
    else:
        flat = particles.reshape(-1, particles.shape[-1])
        if log_pt:
            flat[:, 0] = torch.log(flat[:, 0] + 1e-6)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0) + 1e-6
    return mean, std


def plot_token_usage(token_counts, num_codes, model_name, save_path):
    """Plot histogram of token usage distribution"""
    plt.figure(figsize=(12, 6))
    
    # Create histogram of token usage
    usage_counts = np.bincount(token_counts.cpu().numpy(), minlength=num_codes)
    used_tokens = np.sum(usage_counts > 0)
    
    plt.subplot(1, 2, 1)
    plt.hist(usage_counts[usage_counts > 0], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of times used')
    plt.ylabel('Number of tokens')
    plt.title(f'Token Usage Distribution\n{model_name}')
    plt.grid(True, alpha=0.3)
    
    # Plot token utilization
    plt.subplot(1, 2, 2)
    token_ids = np.arange(num_codes)
    plt.bar(token_ids[usage_counts > 0], usage_counts[usage_counts > 0], 
            alpha=0.7, width=max(1, num_codes//1000))
    plt.xlabel('Token ID')
    plt.ylabel('Usage count')
    plt.title(f'Token Utilization\n{used_tokens}/{num_codes} tokens used ({used_tokens/num_codes*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return used_tokens, usage_counts

def ddp_train(rank: int, world_size: int, config: dict) -> None:
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Determine model type and configuration
    if config["type"] == "new":
        from dataloader.dataloader import load_jetclass_label_as_dataset
        dataset = load_jetclass_label_as_dataset(
            label="HToBB", start=config["start"], end=config["end"])
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] in ["new_large", "new_medium"]:
        from dataloader.dataloader import load_jetclass_label_as_dataset
        dataset = load_jetclass_label_as_dataset(
            label="HToBB", start=config["start"], end=config["end"])
        use_mask = False
        log_pt = False
        model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])
    elif config["type"] == "masked":
        from dataloader.masked_dataloader import load_jetclass_label_as_dataset
        dataset = load_jetclass_label_as_dataset(
            label="HToBB", start=config["start"], end=config["end"])
        use_mask = True
        log_pt = True
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    else:
        from dataloader.dataloader import load_jetclass_label_as_dataset
        dataset = load_jetclass_label_as_dataset(
            label="HToBB", start=config["start"], end=config["end"])
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer", fromlist=["VQVAENormFormer"])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)

    if rank == 0:
        mean, std = compute_global_stats(dataset, config["batch_size"], log_pt, use_mask)
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean = torch.zeros(3, device=device)
        std = torch.ones(3, device=device)
    dist.broadcast(mean, 0)
    dist.broadcast(std, 0)
    mean = mean.to(device)
    std = std.to(device)

    # Create model with MOE-specific parameters for medium/large
    if config["type"] in ["new_large", "new_medium"]:
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_blocks=3,
            vq_kwargs=config["vq_kwargs"],
            moe_kwargs=config["moe_kwargs"],
        ).to(device)
    else:
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_blocks=3,
            vq_kwargs=config["vq_kwargs"],
        ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))
    recon_loss_fn = nn.MSELoss(reduction="none")
    scaler = GradScaler()

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Track token usage for analysis
    all_token_counts = []

    for epoch in range(config["num_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = torch.zeros(1, device=device)
        recon_loss = torch.zeros(1, device=device)
        vq_loss = torch.zeros(1, device=device)
        epoch_tokens = []

        for batch in dataloader:
            if use_mask:
                x_particles, _, _, mask = [b.to(device) for b in batch]
            else:
                x_particles, _, _ = [b.to(device) for b in batch]
                mask = None
            x_particles = x_particles.transpose(1, 2)
            if log_pt:
                x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
            x_norm = (x_particles - mean) / std

            optimizer.zero_grad()
            with autocast():
                if mask is not None:
                    out, loss_dict = model(x_norm, mask=mask)
                    diff = (out - x_norm) ** 2
                    r_loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum()
                else:
                    out, loss_dict = model(x_norm)
                    r_loss = recon_loss_fn(out, x_norm).mean()
                v_loss = loss_dict.get("loss", loss_dict if isinstance(loss_dict, torch.Tensor) else 0.0)
                loss = r_loss + v_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach()
            recon_loss += r_loss.detach()
            vq_loss += v_loss.detach()

            # Collect token usage
            if rank == 0:  # Only collect on rank 0 to avoid duplicates
                epoch_tokens.append(loss_dict["q"].detach())

        epoch_loss /= len(dataloader)
        recon_loss /= len(dataloader)
        vq_loss /= len(dataloader)
        for t in (epoch_loss, recon_loss, vq_loss):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size

        if rank == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']} - Total: {epoch_loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f}")
            
            # Analyze token usage for this epoch
            if epoch_tokens:
                epoch_token_tensor = torch.cat(epoch_tokens, dim=0)
                all_token_counts.append(epoch_token_tensor)
                unique_codes = epoch_token_tensor.unique().numel()
                print(f"🧩 Unique codes used: {unique_codes}/{config['vq_kwargs']['num_codes']}")
            
            if epoch + 1 == config["num_epochs"]:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    os.path.join(config["checkpoint_dir"], f"vqvae_epoch_{epoch+1}.pth"),
                )
                
                # Create token usage plot
                if all_token_counts:
                    all_tokens = torch.cat(all_token_counts, dim=0)
                    plot_path = os.path.join(PLOT_DIR, f"token_usage_{config['type']}.png")
                    used_tokens, usage_dist = plot_token_usage(
                        all_tokens, 
                        config['vq_kwargs']['num_codes'], 
                        f"{config['type']} ({config['vq_kwargs']['num_codes']} tokens)",
                        plot_path
                    )
                    print(f"📊 Token usage plot saved to {plot_path}")
                    print(f"📈 Final utilization: {used_tokens}/{config['vq_kwargs']['num_codes']} tokens ({used_tokens/config['vq_kwargs']['num_codes']*100:.1f}%)")

    cleanup()


def ddp_eval(config: dict) -> None:
    """Run evaluation and create comparison plots for different token sizes"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config["type"] == "masked":
        from dataloader.masked_dataloader import (
            load_jetclass_label_as_dataset,
            load_jetclass_label_as_tensor,
        )
        use_mask = True
        log_pt = True
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] == "new":
        from dataloader.dataloader import (
            load_jetclass_label_as_dataset,
            load_jetclass_label_as_tensor,
        )
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] in ["new_large", "new_medium"]:
        from dataloader.dataloader import (
            load_jetclass_label_as_dataset,
            load_jetclass_label_as_tensor,
        )
        use_mask = False
        log_pt = False
        model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])
    else:
        from dataloader.dataloader import (
            load_jetclass_label_as_dataset,
            load_jetclass_label_as_tensor,
        )
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer", fromlist=["VQVAENormFormer"])

    dataset = load_jetclass_label_as_dataset(label="HToBB", start=config["start"], end=config["end"])
    mean, std = compute_global_stats(dataset, config["batch_size"], log_pt, use_mask)
    mean = mean.to(device)
    std = std.to(device)

    # Create model for evaluation
    if config["type"] in ["new_large", "new_medium"]:
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_blocks=3,
            vq_kwargs=config["vq_kwargs"],
            moe_kwargs=config["moe_kwargs"],
        ).to(device)
    else:
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=128,
            hidden_dim=256,
            num_heads=8,
            num_blocks=3,
            vq_kwargs=config["vq_kwargs"],
        ).to(device)

    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint = torch.load(os.path.join(config["checkpoint_dir"], latest), map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    model.eval()
    dataloader_eval = load_jetclass_label_as_tensor(label="HToBB", start=15, end=18, batch_size=config["batch_size"])
    all_orig_jets, all_recon_jets = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader_eval):
            if i >= 300:
                break

            if use_mask:
                x_particles, _, _, mask = [b.to(device) for b in batch]
            else:
                x_particles, _, _ = [b.to(device) for b in batch]
                mask = None

            x_particles = x_particles.transpose(1, 2)
            if log_pt:
                x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
            x_norm = (x_particles - mean) / std

            if mask is not None:
                out, _ = model(x_norm, mask=mask)
            else:
                out, _ = model(x_norm)

            out_denorm = out * std + mean
            if log_pt:
                out_denorm[:, :, 0] = torch.exp(out_denorm[:, :, 0]) - 1e-6
                x_particles[:, :, 0] = torch.exp(x_particles[:, :, 0]) - 1e-6

            if mask is not None:
                orig_jet = reconstruct_jet_features_from_particles(x_particles * mask.unsqueeze(-1))
                recon_jet = reconstruct_jet_features_from_particles(out_denorm * mask.unsqueeze(-1))
            else:
                orig_jet = reconstruct_jet_features_from_particles(x_particles)
                recon_jet = reconstruct_jet_features_from_particles(out_denorm)

            all_orig_jets.append(orig_jet)
            all_recon_jets.append(recon_jet)

    all_orig_jets = torch.cat(all_orig_jets, dim=0)
    all_recon_jets = torch.cat(all_recon_jets, dim=0)

    # Add model name to plot filenames for distinction
    model_suffix = f"_{config['type']}"
    
    plot_tensor_jet_features(
        [all_orig_jets, all_recon_jets],
        labels=("Original", f"Reconstructed ({config['vq_kwargs']['num_codes']} tokens)"),
        filename=os.path.join(PLOT_DIR, f"jet_recon_overlay_ddp{model_suffix}.png"),
    )
    plot_difference(
        all_orig_jets,
        all_recon_jets,
        filename=os.path.join(PLOT_DIR, f"jet_feature_difference_ddp{model_suffix}.png"),
    )


def main() -> None:
    # Train all three models
    models_to_train = ["new", "new_medium", "new_large"]
    
    for model_type in models_to_train:
        print(f"\n🚀 Training {model_type} model...")
        config = CONFIGS[model_type].copy()
        config["type"] = model_type
        
        mp.spawn(ddp_train, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
        print(f"✅ Completed training {model_type}")
        
        # Run evaluation
        print(f"🔍 Evaluating {model_type} model...")
        ddp_eval(config)
        print(f"✅ Completed evaluation {model_type}")
    
    # Create comparison plot of token utilizations
    create_token_comparison_plot()

def create_token_comparison_plot():
    """Create a comparison plot of token utilization across different model sizes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_configs = [
        ("new", "2048 tokens"), 
        ("new_medium", "4 experts, 32 capacity"), 
        ("new_large", "8 experts, 64 capacity")
    ]
    
    for i, (model_name, description) in enumerate(model_configs):
        plot_path = os.path.join(PLOT_DIR, f"token_usage_{model_name}.png")
        if os.path.exists(plot_path):
            axes[i].text(0.5, 0.5, f'{model_name}\n{description}\nSee individual plot', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
        else:
            axes[i].text(0.5, 0.5, f'{model_name}\n{description}\nNot trained yet', 
                        ha='center', va='center', transform=axes[i].transAxes, fontsize=12)
        axes[i].set_title(f'{model_name.replace("_", " ").title()}\n{description}')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Model Architecture Comparison: Standard vs MOE', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Model comparison overview saved to {os.path.join(PLOT_DIR, 'model_comparison_overview.png')}")
