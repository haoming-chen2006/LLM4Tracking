import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
)

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "checkpoint_comparison")
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_TYPE = "MOE_large"  # Change this as needed
CHECKPOINT_EPOCH = 1  # Single epoch for simplified comparison

CONFIGS = {
    "MOE_med": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_med_1",
        "vq_kwargs": {"num_codes": 4096, "beta": 0.8, "affine_lr": 1.0,
                      "sync_nu": 2, "replace_freq": 3, "dim": -1},
    },
    "MOE_large": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_large_1",
        "vq_kwargs": {"num_codes": 8192, "beta": 0.9, "affine_lr": 1.0,
                      "sync_nu": 5, "replace_freq": 2, "dim": -1},
    },
}

LABELS = [
    "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L",
    "ZToQQ", "WToQQ", "TTBar", "TTBarLep", "ZJetsToNuNu",
]

def load_all_labels_dataset(start: int, end: int, use_mask: bool):
    if use_mask:
        from dataloader.masked_dataloader import load_jetclass_label_as_dataset
    else:
        from dataloader.dataloader import load_jetclass_label_as_dataset

    datasets = []
    for lbl in LABELS:
        try:
            ds = load_jetclass_label_as_dataset(label=lbl, start=start, end=end)
            datasets.append(ds)
            print(f"✅ Loaded {lbl}: {len(ds)} samples")
        except Exception as e:
            print(f"❌ Failed to load {lbl}: {e}")
            continue

    if not datasets:
        raise RuntimeError("No valid datasets loaded for any label")

    from torch.utils.data import TensorDataset
    x_parts = torch.cat([d.tensors[0] for d in datasets], dim=0)
    x_jets = torch.cat([d.tensors[1] for d in datasets], dim=0)
    y = torch.cat([d.tensors[2] for d in datasets], dim=0)
    
    print(f"📊 Combined dataset: {len(x_parts)} total samples")
    
    if use_mask:
        masks = torch.cat([d.tensors[3] for d in datasets], dim=0)
        return TensorDataset(x_parts, x_jets, y, masks)
    return TensorDataset(x_parts, x_jets, y)

def compute_global_stats(dataset, batch_size, log_pt=False, use_mask=False):
    """Compute global mean and std statistics - matches MOE training script exactly"""
    print(f"🔢 Computing global statistics with log_pt={log_pt}, use_mask={use_mask}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_parts = []
    all_masks = [] if use_mask else None
    
    for batch_idx, batch in enumerate(loader):
        if use_mask:
            x_part, _, _, mask = batch
            all_masks.append(mask)
        else:
            x_part, _, _ = batch
        all_parts.append(x_part)
        
        if batch_idx >= 100:  # Limit for large datasets
            break
    
    particles = torch.cat(all_parts, dim=0)  # [B, 3, T] 
    particles = particles.transpose(1, 2)    # [B, T, 3] for easier processing
    
    if use_mask:
        masks = torch.cat(all_masks, dim=0)  # [B, T]
        
        if log_pt:
            particles[:, :, 0] = torch.log(particles[:, :, 0] + 1e-6)
        
        flat_particles = particles.reshape(-1, particles.shape[-1])  # [B*T, 3]
        flat_mask = masks.reshape(-1).bool()  # [B*T]
        valid_particles = flat_particles[flat_mask]  # [N_valid, 3]
        
    else:
        flat_particles = particles.reshape(-1, particles.shape[-1])  # [B*T, 3]
        if log_pt:
            flat_particles[:, 0] = torch.log(flat_particles[:, 0] + 1e-6)
        valid_particles = flat_particles
    
    mean = valid_particles.mean(dim=0)
    std = valid_particles.std(dim=0) + 1e-6
    
    print(f"📈 Global statistics: Mean={mean.tolist()}, Std={std.tolist()}")
    return mean, std

def load_model_and_checkpoint(config, checkpoint_path, device):
    use_mask = False
    log_pt = False
    model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])

    model = model_module.VQVAENormFormer(
        input_dim=3,
        latent_dim=16,
        hidden_dim=128,
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    
    print("✅ Model loaded successfully")
    return model, use_mask, log_pt

def evaluate_model_with_tokens(model, mean, std, use_mask, log_pt, device, 
                              start=10, end=11, batch_size=512):
    """Evaluate model and return original jets, reconstructed jets, and all tokens"""
    model.eval()
    
    eval_dataset = load_all_labels_dataset(start, end, use_mask)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    orig_particles_list = []
    recon_particles_list = []
    all_tokens = []
    
    with torch.no_grad():
        for batch in eval_loader:
            if use_mask:
                x_part, _, _, mask = batch
                mask = mask.to(device)
            else:
                x_part, _, _ = batch
                mask = None
            
            x_part = x_part.to(device)
            orig_particles_list.append(x_part.cpu())
            
            # Normalize input (match training preprocessing)
            x_norm = x_part.transpose(1, 2)  # [B, T, 3]
            
            if log_pt:
                x_norm[:, :, 0] = torch.log(x_norm[:, :, 0] + 1e-6)
            
            x_norm = (x_norm - mean) / std
            x_norm = x_norm.transpose(1, 2)  # Back to [B, 3, T]
            
            # Forward pass
            if mask is not None:
                x_recon, tokens, _ = model(x_norm, mask=mask)
            else:
                x_recon, tokens, _ = model(x_norm)
            
            # Denormalize reconstruction
            x_recon = x_recon.transpose(1, 2)  # [B, T, 3]
            x_recon = x_recon * std + mean
            
            if log_pt:
                x_recon[:, :, 0] = torch.exp(x_recon[:, :, 0]) - 1e-6
            
            x_recon = x_recon.transpose(1, 2)  # Back to [B, 3, T]
            
            recon_particles_list.append(x_recon.cpu())
            all_tokens.append(tokens.cpu())
    
    # Concatenate all batches
    orig_particles = torch.cat(orig_particles_list, dim=0)
    recon_particles = torch.cat(recon_particles_list, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)
    
    print(f"💫 Processed {orig_particles.shape[0]} samples")
    
    # Convert to physical coordinates and reconstruct jet features
    orig_particles_phys = orig_particles.transpose(1, 2)  # [B, T, 3]
    recon_particles_phys = recon_particles.transpose(1, 2)  # [B, T, 3]
    
    # Reconstruct jet features using physical coordinates
    orig_jets = reconstruct_jet_features_from_particles(orig_particles_phys)
    recon_jets = reconstruct_jet_features_from_particles(recon_particles_phys)
    
    return orig_jets, recon_jets, all_tokens.flatten()

def plot_token_usage_histogram(token_counts, num_codes, model_name, save_path):
    """Plot histogram of unique token usage"""
    plt.figure(figsize=(12, 8))
    
    # Get unique tokens and their counts
    unique_tokens, counts = torch.unique(token_counts, return_counts=True)
    unique_tokens = unique_tokens.cpu().numpy()
    counts = counts.cpu().numpy()
    
    # Create histogram of token usage frequency
    plt.subplot(2, 2, 1)
    plt.hist(counts, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Usage frequency')
    plt.ylabel('Number of tokens')
    plt.title(f'Token Usage Frequency Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot token utilization across vocabulary
    plt.subplot(2, 2, 2)
    all_token_usage = np.zeros(num_codes)
    all_token_usage[unique_tokens] = counts
    used_mask = all_token_usage > 0
    
    plt.bar(np.arange(num_codes)[used_mask], all_token_usage[used_mask], 
            alpha=0.7, width=max(1, num_codes//1000), color='lightcoral')
    plt.xlabel('Token ID')
    plt.ylabel('Usage count')
    plt.title(f'Token Utilization Across Vocabulary')
    plt.grid(True, alpha=0.3)
    
    # Cumulative usage plot
    plt.subplot(2, 2, 3)
    sorted_counts = np.sort(counts)[::-1]  # Sort in descending order
    cumulative_usage = np.cumsum(sorted_counts) / np.sum(sorted_counts)
    plt.plot(range(len(sorted_counts)), cumulative_usage, linewidth=2, color='green')
    plt.xlabel('Token rank (by usage)')
    plt.ylabel('Cumulative usage fraction')
    plt.title('Cumulative Token Usage')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    total_tokens = len(unique_tokens)
    utilization_rate = total_tokens / num_codes * 100
    
    stats_text = f"""
    Model: {model_name}
    Total vocabulary: {num_codes:,}
    Unique tokens used: {total_tokens:,}
    Utilization rate: {utilization_rate:.1f}%
    
    Most used token: {counts.max():,} times
    Least used token: {counts.min():,} times
    Average usage: {counts.mean():.1f} times
    """
    
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    plt.axis('off')
    
    plt.suptitle(f'Token Analysis: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Token histogram saved: {save_path}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = CONFIGS[TRAIN_TYPE].copy()
    config["type"] = TRAIN_TYPE
    
    print(f"🔍 Evaluating {TRAIN_TYPE} model checkpoint")
    
    # Find checkpoint
    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) 
             if f.startswith("moe_epoch_") and f.endswith(".pth")]
    
    if not ckpts:
        print("❌ No checkpoints found!")
        return
    
    target_file = f"moe_epoch_{CHECKPOINT_EPOCH}.pth"
    if target_file not in ckpts:
        print(f"❌ Checkpoint for epoch {CHECKPOINT_EPOCH} not found!")
        return
    
    checkpoint_path = os.path.join(config["checkpoint_dir"], target_file)
    print(f"📊 Evaluating checkpoint: {target_file}")
    
    # Load model
    model, use_mask, log_pt = load_model_and_checkpoint(config, checkpoint_path, device)
    
    # Compute normalization stats from training data
    train_dataset = load_all_labels_dataset(20, 21, use_mask)
    mean, std = compute_global_stats(train_dataset, config["batch_size"], log_pt, use_mask)
    mean, std = mean.to(device), std.to(device)
    
    # Evaluate model
    orig_jets, recon_jets, all_tokens = evaluate_model_with_tokens(
        model, mean, std, use_mask, log_pt, device, start=10, end=11, 
        batch_size=config["batch_size"]
    )
    
    # Create the two required plots
    print("📈 Creating plots...")
    
    model_description = f"{TRAIN_TYPE} (Epoch {CHECKPOINT_EPOCH})"
    
    # 1. Original vs reconstructed jet features plot using imported function
    plot_tensor_jet_features(
        [orig_jets, recon_jets],
        labels=["Original", f"Reconstructed {model_description}"],
        filename=os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{CHECKPOINT_EPOCH}_comparison.png"),
    )
    
    # 2. Token usage histogram
    token_hist_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{CHECKPOINT_EPOCH}_token_usage.png")
    plot_token_usage_histogram(
        all_tokens, 
        config["vq_kwargs"]["num_codes"], 
        model_description,
        token_hist_path
    )
    
    print(f"✅ Plots saved to {PLOT_DIR}")
    print(f"📊 Evaluated checkpoint on {len(orig_jets)} samples from ALL LABELS")
    print("🎯 Generated two plots:")
    print(f"   1. Token usage histogram: {token_hist_path}")
    print(f"   2. Original vs reconstructed comparison: {os.path.join(PLOT_DIR, f'{TRAIN_TYPE}_epoch_{CHECKPOINT_EPOCH}_comparison.png')}")

if __name__ == "__main__":
    main()
