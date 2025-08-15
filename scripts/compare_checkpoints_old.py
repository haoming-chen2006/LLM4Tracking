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
CHECKPOINT_EPOCH = [1,2] # Change to list of epochs or single epoch or "latest"

CONFIGS = {
    "new": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash",
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
    },
    "MOE_med": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_med_1",
        "vq_kwargs": {"num_codes": 4096, "beta": 0.8, "affine_lr": 1.0,  # Changed from 0.4 to 0.8, 0.0 to 1.0
                      "sync_nu": 2, "replace_freq": 3, "dim": -1},  # Changed from 10 to 3
    },
    "MOE_large": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_large_1",
        "vq_kwargs": {"num_codes": 8192, "beta": 0.9, "affine_lr": 1.0,  # Changed from 0.4 to 0.9
                      "sync_nu": 5, "replace_freq": 2, "dim": -1},  # Changed from 10 to 2
    },
    "masked": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash_masked",
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
    },
    "particle": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_new",
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
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
    """Compute global mean and std statistics - MUST match MOE training script exactly"""
    print(f"🔢 Computing global statistics with log_pt={log_pt}, use_mask={use_mask}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_parts = []
    all_masks = [] if use_mask else None
    
    # Collect all data first (SAME as MOE training)
    for batch_idx, batch in enumerate(loader):
        if use_mask:
            x_part, _, _, mask = batch
            all_masks.append(mask)
        else:
            x_part, _, _ = batch
        all_parts.append(x_part)
        
        # Early break for very large datasets (SAME as MOE training)
        if batch_idx >= 100:
            print(f"⚠️ Limited global stats computation to first {batch_idx + 1} batches")
            break
    
    particles = torch.cat(all_parts, dim=0)  # [B, 3, T] 
    particles = particles.transpose(1, 2)    # [B, T, 3] for easier processing
    
    if use_mask:
        masks = torch.cat(all_masks, dim=0)  # [B, T]
        
        # Apply log transformation BEFORE masking and flattening (SAME as MOE training)
        if log_pt:
            particles[:, :, 0] = torch.log(particles[:, :, 0] + 1e-6)
        
        # Flatten and apply mask (SAME as MOE training)
        flat_particles = particles.reshape(-1, particles.shape[-1])  # [B*T, 3]
        flat_mask = masks.reshape(-1).bool()  # [B*T]
        valid_particles = flat_particles[flat_mask]  # [N_valid, 3]
        
        print(f"📊 Mask statistics for global stats:")
        print(f"  Total tokens: {flat_mask.shape[0]:,}")
        print(f"  Valid tokens: {flat_mask.sum():,}")
        print(f"  Valid ratio: {flat_mask.float().mean()*100:.2f}%")
        
    else:
        # No masking case (SAME as MOE training)
        flat_particles = particles.reshape(-1, particles.shape[-1])  # [B*T, 3]
        if log_pt:
            flat_particles[:, 0] = torch.log(flat_particles[:, 0] + 1e-6)
        valid_particles = flat_particles
    
    # Compute statistics on valid particles only (SAME as MOE training)
    mean = valid_particles.mean(dim=0)
    std = valid_particles.std(dim=0) + 1e-6  # Add small epsilon for numerical stability
    
    print(f"📈 Global statistics computed:")
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std: {std.tolist()}")
    print(f"  Valid samples used: {valid_particles.shape[0]:,}")
    
    return mean, std

def load_model_and_checkpoint(config, checkpoint_path, device):
    if config["type"] == "masked":
        use_mask = True
        log_pt = True
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] == "new":
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] in ["MOE_med", "MOE_large"]:
        use_mask = False
        log_pt = False
        model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])
    else:
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer", fromlist=["VQVAENormFormer"])

    # Create model - align parameters with MOE training script
    model = model_module.VQVAENormFormer(
        input_dim=3,
        latent_dim=16,  # Match MOE training script
        hidden_dim=128, # Match MOE training script
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use strict=False to handle potential model architecture changes
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state"], strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint: {unexpected_keys}")
            print("📝 This is likely due to VQ layer architecture changes - continuing with available parameters")
            
        print("✅ Model loaded successfully with available parameters")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🔄 Attempting to load with key filtering...")
        
        # Filter out problematic keys
        state_dict = checkpoint["model_state"]
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            # Skip affine transform keys if they cause issues
            if "affine_transform" in key:
                print(f"Skipping key: {key}")
                continue
            filtered_state_dict[key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ Model loaded with filtered state dict")
        if missing_keys:
            print(f"⚠️  Missing keys after filtering: {missing_keys}")
    
    model.eval()
    
    return model, use_mask, log_pt

def evaluate_model(model, dataloader, mean, std, use_mask, log_pt, device):
    """Evaluate single model - MUST match MOE evaluation logic exactly"""
    all_orig_jets, all_recon_jets = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 200:  # Limit for faster evaluation
                break

            if use_mask:
                x_particles, x_jets, y, mask = [b.to(device) for b in batch]
            else:
                x_particles, x_jets, y = [b.to(device) for b in batch]
                mask = None

            # Store original particles in PHYSICAL units for jet reconstruction
            x_particles_physical = x_particles.clone()
            if x_particles_physical.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                x_particles_physical = x_particles_physical.transpose(1, 2)

            # Apply SAME preprocessing as MOE training using data module logic
            # Ensure proper tensor format [B, T, 3]
            if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                x_particles = x_particles.transpose(1, 2)
            
            # Apply log transformation if configured (SAME as MOE training)
            if log_pt:
                x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
            
            # Apply normalization (SAME as MOE training)
            x_norm = (x_particles - mean) / std
            
            # Apply masking after normalization (SAME as MOE training)
            if use_mask:
                x_norm = x_norm * mask.unsqueeze(-1)

            if mask is not None:
                out, _ = model(x_norm, mask=mask)
            else:
                out, _ = model(x_norm)

            # Denormalize outputs to get PHYSICAL particles (SAME inverse as MOE training preprocessing)
            out_denorm = out * std + mean
            
            # Apply inverse log transformation if configured to get PHYSICAL pt values
            if log_pt:
                out_denorm[:, :, 0] = torch.exp(out_denorm[:, :, 0]) - 1e-6
                out_denorm[:, :, 0] = torch.clamp(out_denorm[:, :, 0], min=1e-6)

            # Reconstruct jet features from PHYSICAL particles (no additional normalization needed)
            if mask is not None:
                # Apply mask to PHYSICAL particles
                orig_particles_masked = x_particles_physical * mask.unsqueeze(-1)
                recon_particles_masked = out_denorm * mask.unsqueeze(-1)
                
                orig_jet = reconstruct_jet_features_from_particles(orig_particles_masked)
                recon_jet = reconstruct_jet_features_from_particles(recon_particles_masked)
            else:
                orig_jet = reconstruct_jet_features_from_particles(x_particles_physical)
                recon_jet = reconstruct_jet_features_from_particles(out_denorm)

            all_orig_jets.append(orig_jet)
            all_recon_jets.append(recon_jet)

    return torch.cat(all_orig_jets, dim=0), torch.cat(all_recon_jets, dim=0)

def evaluate_model_all_labels(model, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=512):
    """Evaluate model on all labels - MUST match MOE evaluation logic exactly"""
    all_orig_jets = []
    all_recon_jets = []
    
    for label in LABELS:
        try:
            if use_mask:
                from dataloader.masked_dataloader import load_jetclass_label_as_tensor
            else:
                from dataloader.dataloader import load_jetclass_label_as_tensor
            
            print(f"🔄 Processing {label}...")
            dataloader = load_jetclass_label_as_tensor(label=label, start=start, end=end, batch_size=batch_size)
            
            label_orig_jets = []
            label_recon_jets = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 50:  # Limit batches per label for faster evaluation
                        break

                    if use_mask:
                        x_particles, x_jets, y, mask = [b.to(device) for b in batch]
                    else:
                        x_particles, x_jets, y = [b.to(device) for b in batch]
                        mask = None
                    
                    # Store original particles in PHYSICAL units for jet reconstruction
                    x_particles_physical = x_particles.clone()
                    if x_particles_physical.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                        x_particles_physical = x_particles_physical.transpose(1, 2)

                    # Store original particles in PHYSICAL units for jet reconstruction
                    x_particles_physical = x_particles.clone()
                    if x_particles_physical.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                        x_particles_physical = x_particles_physical.transpose(1, 2)

                    # Apply SAME preprocessing as MOE training using data module logic
                    # Ensure proper tensor format [B, T, 3]
                    if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                        x_particles = x_particles.transpose(1, 2)
                    
                    # Apply log transformation if configured (SAME as MOE training)
                    if log_pt:
                        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
                    
                    # Apply normalization (SAME as MOE training)
                    x_norm = (x_particles - mean) / std
                    
                    # Apply masking after normalization (SAME as MOE training)
                    if use_mask:
                        x_norm = x_norm * mask.unsqueeze(-1)

                    # Get model reconstruction
                    if mask is not None:
                        out, _ = model(x_norm, mask=mask)
                    else:
                        out, _ = model(x_norm)

                    # Denormalize outputs to get PHYSICAL particles (SAME inverse as MOE training preprocessing)
                    out_denorm = out * std + mean
                    
                    # Apply inverse log transformation if configured to get PHYSICAL pt values
                    if log_pt:
                        out_denorm[:, :, 0] = torch.exp(out_denorm[:, :, 0]) - 1e-6
                        out_denorm[:, :, 0] = torch.clamp(out_denorm[:, :, 0], min=1e-6)

                    # Reconstruct jet features from PHYSICAL particles (no additional normalization needed)
                    if mask is not None:
                        # Apply mask to PHYSICAL particles
                        orig_particles_masked = x_particles_physical * mask.unsqueeze(-1)
                        recon_particles_masked = out_denorm * mask.unsqueeze(-1)
                        
                        orig_jet = reconstruct_jet_features_from_particles(orig_particles_masked)
                        recon_jet = reconstruct_jet_features_from_particles(recon_particles_masked)
                    else:
                        orig_jet = reconstruct_jet_features_from_particles(x_particles_physical)
                        recon_jet = reconstruct_jet_features_from_particles(out_denorm)

                    label_orig_jets.append(orig_jet)
                    label_recon_jets.append(recon_jet)
            
            if label_orig_jets:
                all_orig_jets.append(torch.cat(label_orig_jets, dim=0))
                all_recon_jets.append(torch.cat(label_recon_jets, dim=0))
                print(f"✅ {label}: {len(torch.cat(label_orig_jets, dim=0))} jets processed")
            
        except Exception as e:
            print(f"❌ Failed to process {label}: {e}")
            continue
    
    if not all_orig_jets:
        raise RuntimeError("No data processed for any label")
    
    # Concatenate all labels
    return torch.cat(all_orig_jets, dim=0), torch.cat(all_recon_jets, dim=0)

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
    print(f"🎯 Utilization: {total_tokens}/{num_codes} tokens ({utilization_rate:.1f}%)")
    return total_tokens, utilization_rate

def evaluate_model_with_tokens(model, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=512):
    """Evaluate model and collect both reconstructions and token usage"""
    all_orig_jets = []
    all_recon_jets = []
    all_tokens = []
    
    for label in LABELS:
        try:
            if use_mask:
                from dataloader.masked_dataloader import load_jetclass_label_as_tensor
            else:
                from dataloader.dataloader import load_jetclass_label_as_tensor
            
            print(f"🔄 Processing {label}...")
            dataloader = load_jetclass_label_as_tensor(label=label, start=start, end=end, batch_size=batch_size)
            
            label_orig_jets = []
            label_recon_jets = []
            label_tokens = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 50:  # Limit batches per label
                        break

                    if use_mask:
                        x_particles, x_jets, y, mask = [b.to(device) for b in batch]
                    else:
                        x_particles, x_jets, y = [b.to(device) for b in batch]
                        mask = None
                    
                    # Apply SAME preprocessing as MOE training using data module logic
                    # Ensure proper tensor format [B, T, 3]
                    if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                        x_particles = x_particles.transpose(1, 2)
                    
                    # Apply log transformation if configured (SAME as MOE training)
                    if log_pt:
                        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
                    
                    # Apply normalization (SAME as MOE training)
                    x_norm = (x_particles - mean) / std
                    
                    # Apply masking after normalization (SAME as MOE training)
                    if use_mask:
                        x_norm = x_norm * mask.unsqueeze(-1)

                    # Get model reconstruction and tokens
                    if mask is not None:
                        out, loss_dict = model(x_norm, mask=mask)
                    else:
                        out, loss_dict = model(x_norm)
                    
                    # Collect tokens (SAME as MOE training)
                    if isinstance(loss_dict, dict) and "q" in loss_dict:
                        label_tokens.append(loss_dict["q"].detach())

                    # Denormalize outputs (SAME inverse as MOE training preprocessing)
                    out_denorm = out * std + mean
                    
                    # Apply inverse log transformation if configured
                    if log_pt:
                        out_denorm[:, :, 0] = torch.exp(out_denorm[:, :, 0]) - 1e-6
                        out_denorm[:, :, 0] = torch.clamp(out_denorm[:, :, 0], min=1e-6)
                        
                        # Also inverse log transform the original for consistency
                        x_particles_denorm = x_particles * std + mean
                        x_particles_denorm[:, :, 0] = torch.exp(x_particles_denorm[:, :, 0]) - 1e-6
                        x_particles_denorm[:, :, 0] = torch.clamp(x_particles_denorm[:, :, 0], min=1e-6)
                    else:
                        x_particles_denorm = x_particles * std + mean

                    # Reconstruct jet features from denormalized particles (SAME as MOE training)
                    if mask is not None:
                        # Apply mask to denormalized particles
                        orig_particles_masked = x_particles_denorm * mask.unsqueeze(-1)
                        recon_particles_masked = out_denorm * mask.unsqueeze(-1)
                        
                        orig_jet = reconstruct_jet_features_from_particles(orig_particles_masked)
                        recon_jet = reconstruct_jet_features_from_particles(recon_particles_masked)
                    else:
                        orig_jet = reconstruct_jet_features_from_particles(x_particles_denorm)
                        recon_jet = reconstruct_jet_features_from_particles(out_denorm)

                    label_orig_jets.append(orig_jet)
                    label_recon_jets.append(recon_jet)
            
            if label_orig_jets:
                all_orig_jets.append(torch.cat(label_orig_jets, dim=0))
                all_recon_jets.append(torch.cat(label_recon_jets, dim=0))
                all_tokens.append(torch.cat(label_tokens, dim=0))
                print(f"✅ {label}: {len(torch.cat(label_orig_jets, dim=0))} jets processed")
            
        except Exception as e:
            print(f"❌ Failed to process {label}: {e}")
            continue
    
    if not all_orig_jets:
        raise RuntimeError("No data processed for any label")
    
    # Concatenate all results
    return (torch.cat(all_orig_jets, dim=0), 
            torch.cat(all_recon_jets, dim=0), 
            torch.cat(all_tokens, dim=0))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = CONFIGS[TRAIN_TYPE].copy()
    config["type"] = TRAIN_TYPE
    
    print(f"🔍 Evaluating {TRAIN_TYPE} model checkpoint(s)")
    
    # Find checkpoints
    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) 
             if f.startswith("moe_epoch_") and f.endswith(".pth")]
    
    if not ckpts:
        print("❌ No checkpoints found!")
        return
    
    # Sort checkpoints by epoch number
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    # Handle different input types for CHECKPOINT_EPOCH
    if isinstance(CHECKPOINT_EPOCH, list):
        # Multiple epochs for comparison
        selected_epochs = []
        selected_ckpts = []
        
        for epoch in CHECKPOINT_EPOCH:
            target_file = f"moe_epoch_{epoch}.pth"
            if target_file in ckpts:
                selected_epochs.append(epoch)
                selected_ckpts.append(target_file)
            else:
                print(f"⚠️ Checkpoint for epoch {epoch} not found, skipping")
        
        if not selected_ckpts:
            print("❌ No valid checkpoints found from the list!")
            return
            
        print(f"📊 Comparing checkpoints: {selected_epochs}")
        
        # Evaluate multiple checkpoints
        all_recon_results = []
        orig_jets = None
        
        for i, (epoch, ckpt) in enumerate(zip(selected_epochs, selected_ckpts)):
            print(f"🔄 Loading checkpoint {i+1}/{len(selected_ckpts)}: {ckpt} (epoch {epoch})")
            checkpoint_path = os.path.join(config["checkpoint_dir"], ckpt)
            model, use_mask, log_pt = load_model_and_checkpoint(config, checkpoint_path, device)
            
            # Load evaluation dataset and compute stats (only once)
            if orig_jets is None:
                eval_dataset = load_all_labels_dataset(10, 11, use_mask)
                
                # Compute normalization stats
                if config["type"] == "masked":
                    train_dataset = load_all_labels_dataset(20, 21, True)
                    log_pt = True
                elif config["type"] in ["MOE_med", "MOE_large"]:
                    train_dataset = load_all_labels_dataset(20, 21, False)
                    log_pt = False
                else:
                    train_dataset = load_all_labels_dataset(10, 11, False)
                    log_pt = False
                
                mean, std = compute_global_stats(train_dataset, config["batch_size"], log_pt, use_mask)
                mean, std = mean.to(device), std.to(device)
            
            # Evaluate model and collect tokens
            orig_jets_current, recon_jets, all_tokens = evaluate_model_with_tokens(
                model, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=config["batch_size"]
            )
            
            # Store original jets from first checkpoint
            if orig_jets is None:
                orig_jets = orig_jets_current
            
            all_recon_results.append(recon_jets)
            
            # Create individual token usage plot
            token_hist_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{epoch}_token_usage.png")
            plot_token_usage_histogram(
                all_tokens, 
                config["vq_kwargs"]["num_codes"], 
                f"{TRAIN_TYPE} (Epoch {epoch})",
                token_hist_path
            )
        
        # Create comparison plots
        print("📈 Creating comparison plots...")
        
        # Prepare data for overlay plot
        jet_data = [orig_jets] + all_recon_results
        labels = ["Original"] + [f"Epoch {epoch}" for epoch in selected_epochs]
        
        # Plot overlay comparison
        plot_tensor_jet_features(
            jet_data,
            labels=labels,
            filename=os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epochs_{'_'.join(map(str, selected_epochs))}_comparison.png"),
        )
        
        # Plot differences from original for each epoch
        for i, (epoch, recon_jets) in enumerate(zip(selected_epochs, all_recon_results)):
            plot_difference(
                orig_jets,
                recon_jets,
                filename=os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{epoch}_difference.png"),
            )
        
        print(f"✅ Comparison plots saved to {PLOT_DIR}")
        print(f"📊 Evaluated {len(selected_epochs)} checkpoints on {len(orig_jets)} samples from ALL LABELS")
        
    else:
        # Single checkpoint evaluation (existing logic)
        if CHECKPOINT_EPOCH == "latest":
            selected_ckpt = ckpts[-1]
        else:
            target_file = f"moe_epoch_{CHECKPOINT_EPOCH}.pth"
            if target_file in ckpts:
                selected_ckpt = target_file
            else:
                print(f"❌ Checkpoint for epoch {CHECKPOINT_EPOCH} not found!")
                print(f"Available checkpoints: {[int(f.split('_')[-1].split('.')[0]) for f in ckpts]}")
                return
        
        selected_epoch = int(selected_ckpt.split("_")[-1].split(".")[0])
        print(f"📊 Evaluating checkpoint: {selected_ckpt} (epoch {selected_epoch})")
        
        # Load evaluation dataset
        use_mask = config["type"] == "masked"
        eval_dataset = load_all_labels_dataset(10, 11, use_mask)
        
        # Compute normalization stats
        if config["type"] == "masked":
            train_dataset = load_all_labels_dataset(20, 21, True)
            log_pt = True
        elif config["type"] in ["MOE_med", "MOE_large"]:
            train_dataset = load_all_labels_dataset(20, 21, False)
            log_pt = False
        else:
            train_dataset = load_all_labels_dataset(10, 11, False)
            log_pt = False
        
        mean, std = compute_global_stats(train_dataset, config["batch_size"], log_pt, use_mask)
        mean, std = mean.to(device), std.to(device)
        
        # Load and evaluate selected checkpoint
        print(f"🔄 Loading checkpoint: {selected_ckpt}")
        checkpoint_path = os.path.join(config["checkpoint_dir"], selected_ckpt)
        model, _, _ = load_model_and_checkpoint(config, checkpoint_path, device)
        
        # Evaluate model and collect tokens
        orig_jets, recon_jets, all_tokens = evaluate_model_with_tokens(
            model, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=config["batch_size"]
        )
        
        # Create plots
        print("📈 Creating plots...")
        
        model_description = f"{TRAIN_TYPE}"
        
        # Plot original vs reconstruction
        plot_tensor_jet_features(
            [orig_jets, recon_jets],
            labels=("Original", f"Reconstructed {model_description} (Epoch {selected_epoch})"),
            filename=os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_reconstruction.png"),
        )
        
        # Plot reconstruction difference
        plot_difference(
            orig_jets,
            recon_jets,
            filename=os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_difference.png"),
        )
        
        # Plot token usage histogram
        token_hist_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_token_usage.png")
        plot_token_usage_histogram(
            all_tokens, 
            config["vq_kwargs"]["num_codes"], 
            f"{model_description} (Epoch {selected_epoch})",
            token_hist_path
        )
        
        # Detailed feature comparison plots
        detailed_comp_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_detailed_comparison.png")
        plot_jet_feature_comparison_detailed(orig_jets, recon_jets, f"{model_description} (Epoch {selected_epoch})", detailed_comp_path)
        
        # Difference-only plots
        diff_only_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_differences_only.png")
        create_difference_only_plots(orig_jets, recon_jets, f"{model_description} (Epoch {selected_epoch})", diff_only_path)
        
        # Physical range validation plots
        validation_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_physical_validation.png")
        create_physical_range_validation_plots(orig_jets, recon_jets, f"{model_description} (Epoch {selected_epoch})", validation_path)
        diff_only_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_difference_only.png")
        create_difference_only_plots(orig_jets, recon_jets, model_description, diff_only_path)
        
        # Physical range validation plots
        phys_range_val_path = os.path.join(PLOT_DIR, f"{TRAIN_TYPE}_epoch_{selected_epoch}_physical_range_validation.png")
        create_physical_range_validation_plots(orig_jets, recon_jets, model_description, phys_range_val_path)
        
        print(f"✅ All plots saved to {PLOT_DIR}")
        print(f"📊 Evaluated on {len(orig_jets)} samples from ALL LABELS (parts 10-11)")

if __name__ == "__main__":
    main()

def plot_jet_feature_comparison_detailed(orig_jets, recon_jets, model_name, save_path):
    """Create detailed plots showing original vs reconstructed vs difference for jet features"""
    orig_np = orig_jets.detach().cpu().numpy()
    recon_np = recon_jets.detach().cpu().numpy()
    diff_np = recon_np - orig_np
    
    # Feature names with proper physics notation
    feature_names = ['pt [GeV]', 'eta', 'phi', 'mass [GeV]']
    
    # Create a 3x4 subplot (original, reconstructed, difference for each feature)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, feature in enumerate(feature_names):
        if i >= orig_np.shape[1]:
            # Hide unused subplots
            for row in range(3):
                axes[row, i].set_visible(False)
            continue
            
        orig_vals = orig_np[:, i]
        recon_vals = recon_np[:, i]
        diff_vals = diff_np[:, i]
        
        # Determine reasonable bin ranges for each feature
        if i == 0:  # pt
            bins_range = (400, 1200)
            diff_range = (-50, 50)
        elif i == 1:  # eta  
            bins_range = (-2.5, 2.5)
            diff_range = (-0.2, 0.2)
        elif i == 2:  # phi
            bins_range = (-np.pi, np.pi)
            diff_range = (-0.2, 0.2)
        else:  # mass
            bins_range = (0, 300)
            diff_range = (-30, 30)
        
        # Plot original distribution
        axes[0, i].hist(orig_vals, bins=50, alpha=0.7, label='Original', 
                       color='royalblue', density=True, range=bins_range)
        axes[0, i].set_xlabel(f'Jet {feature}')
        axes[0, i].set_ylabel('Density')
        axes[0, i].set_title(f'Original Jet {feature}')
        axes[0, i].legend()
        axes[0, i].grid(alpha=0.3)
        
        # Plot reconstructed distribution
        axes[1, i].hist(recon_vals, bins=50, alpha=0.7, label='Reconstructed', 
                       color='forestgreen', density=True, range=bins_range)
        axes[1, i].set_xlabel(f'Jet {feature}')
        axes[1, i].set_ylabel('Density')
        axes[1, i].set_title(f'Reconstructed Jet {feature}')
        axes[1, i].legend()
        axes[1, i].grid(alpha=0.3)
        
        # Plot difference distribution
        axes[2, i].hist(diff_vals, bins=50, alpha=0.7, label='Difference', 
                       color='darkred', density=True, range=diff_range)
        axes[2, i].set_xlabel(f'$\\Delta$ Jet {feature}')
        axes[2, i].set_ylabel('Density')
        axes[2, i].set_title(f'Difference (Recon - Orig) Jet {feature}')
        axes[2, i].legend()
        axes[2, i].grid(alpha=0.3)
        
        # Add statistics to difference plot
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        axes[2, i].axvline(mean_diff, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_diff:.3f}')
        axes[2, i].text(0.02, 0.95, f'μ: {mean_diff:.3f}\nσ: {std_diff:.3f}', 
                       transform=axes[2, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle(f'Detailed Jet Feature Analysis: {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Detailed jet feature comparison saved: {save_path}")
    
    # Print summary statistics
    print(f"📊 Jet Feature Reconstruction Statistics for {model_name}:")
    for i, feature in enumerate(feature_names[:orig_np.shape[1]]):
        diff_vals = diff_np[:, i]
        print(f"  {feature}: Mean Δ = {diff_vals.mean():.4f}, Std Δ = {diff_vals.std():.4f}")


def create_difference_only_plots(orig_jets, recon_jets, model_name, save_path):
    """Create plots showing only the differences between original and reconstructed jets"""
    orig_np = orig_jets.detach().cpu().numpy()
    recon_np = recon_jets.detach().cpu().numpy()
    diff_np = recon_np - orig_np
    
    feature_names = ['pt [GeV]', 'eta', 'phi', 'mass [GeV]']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        if i >= orig_np.shape[1]:
            ax.set_visible(False)
            continue
            
        diff_vals = diff_np[:, i]
        
        # Create histogram of differences
        n, bins, patches = ax.hist(diff_vals, bins=50, alpha=0.8, 
                                  color='darkred', density=True, edgecolor='black')
        
        # Add statistics
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        median_diff = np.median(diff_vals)
        
        # Add vertical lines for statistics
        ax.axvline(mean_diff, color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_diff:.4f}')
        ax.axvline(median_diff, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_diff:.4f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel(f'$\\Delta$ Jet {feature} (Reconstructed - Original)')
        ax.set_ylabel('Density')
        ax.set_title(f'Reconstruction Error: {feature}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add text box with detailed statistics
        stats_text = f'μ: {mean_diff:.4f}\nσ: {std_diff:.4f}\nmedian: {median_diff:.4f}'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Jet Reconstruction Errors: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Difference-only plots saved: {save_path}")


def create_physical_range_validation_plots(orig_jets, recon_jets, model_name, save_path):
    """Create plots to validate that eta and phi are in correct physical ranges"""
    orig_np = orig_jets.detach().cpu().numpy()
    recon_np = recon_jets.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Check eta range (should be roughly -2.5 to 2.5)
    axes[0, 0].hist(orig_np[:, 1], bins=50, alpha=0.6, label='Original eta', 
                   color='royalblue', density=True)
    axes[0, 0].axvline(-2.5, color='red', linestyle='--', alpha=0.7, label='Expected range')
    axes[0, 0].axvline(2.5, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('eta')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Original eta Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[1, 0].hist(recon_np[:, 1], bins=50, alpha=0.6, label='Reconstructed eta', 
                   color='forestgreen', density=True)
    axes[1, 0].axvline(-2.5, color='red', linestyle='--', alpha=0.7, label='Expected range')
    axes[1, 0].axvline(2.5, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('eta')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Reconstructed eta Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Check phi range (should be roughly -π to π)
    axes[0, 1].hist(orig_np[:, 2], bins=50, alpha=0.6, label='Original phi', 
                   color='royalblue', density=True)
    axes[0, 1].axvline(-np.pi, color='red', linestyle='--', alpha=0.7, label='Expected range')
    axes[0, 1].axvline(np.pi, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('phi')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Original phi Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 1].hist(recon_np[:, 2], bins=50, alpha=0.6, label='Reconstructed phi', 
                   color='forestgreen', density=True)
    axes[1, 1].axvline(-np.pi, color='red', linestyle='--', alpha=0.7, label='Expected range')
    axes[1, 1].axvline(np.pi, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('phi')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Reconstructed phi Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Check mass range (should be non-negative, typical range 0-300 GeV)
    axes[0, 2].hist(orig_np[:, 3], bins=50, alpha=0.6, label='Original mass', 
                   color='royalblue', density=True)
    axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.7, label='Physical boundary')
    axes[0, 2].set_xlabel('mass [GeV]')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Original mass Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    axes[1, 2].hist(recon_np[:, 3], bins=50, alpha=0.6, label='Reconstructed mass', 
                   color='forestgreen', density=True)
    axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.7, label='Physical boundary')
    axes[1, 2].set_xlabel('mass [GeV]')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Reconstructed mass Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    # Add range validation text
    eta_orig_range = (orig_np[:, 1].min(), orig_np[:, 1].max())
    eta_recon_range = (recon_np[:, 1].min(), recon_np[:, 1].max())
    phi_orig_range = (orig_np[:, 2].min(), orig_np[:, 2].max())
    phi_recon_range = (recon_np[:, 2].min(), recon_np[:, 2].max())
    mass_orig_range = (orig_np[:, 3].min(), orig_np[:, 3].max())
    mass_recon_range = (recon_np[:, 3].min(), recon_np[:, 3].max())
    
    validation_text = f"""
Physical Range Validation for {model_name}:

eta ranges:
  Original: [{eta_orig_range[0]:.3f}, {eta_orig_range[1]:.3f}]
  Reconstructed: [{eta_recon_range[0]:.3f}, {eta_recon_range[1]:.3f}]
  Expected: [-2.5, 2.5] approximately

phi ranges:
  Original: [{phi_orig_range[0]:.3f}, {phi_orig_range[1]:.3f}]
  Reconstructed: [{phi_recon_range[0]:.3f}, {phi_recon_range[1]:.3f}]
  Expected: [-π, π] = [-3.142, 3.142]

mass ranges:
  Original: [{mass_orig_range[0]:.1f}, {mass_orig_range[1]:.1f}] GeV
  Reconstructed: [{mass_recon_range[0]:.1f}, {mass_recon_range[1]:.1f}] GeV
  Expected: [0, ~300] GeV, non-negative
    """
    
    plt.suptitle(f'Physical Range Validation: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save validation text
    validation_text_path = save_path.replace('.png', '_validation.txt')
    with open(validation_text_path, 'w') as f:
        f.write(validation_text)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Physical range validation plots saved: {save_path}")
    print(f"📝 Validation text saved: {validation_text_path}")
    print(validation_text)
