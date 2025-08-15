import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Allow importing modules from the repository root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
    plot_difference,
)

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "encode_decode_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and Torch for reproducible dataloaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id: int) -> None:
    """Seed individual dataloader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

TRAIN_TYPE = "MOE_large"  # Change this as needed (e.g. MOE_large, or others)
CHECKPOINT_EPOCH = "latest"
MAX_PARTICLES_PER_JET = 5000  # Number of particles to load per jet (default: 5000, 1 batch = 128 particles)

CONFIGS = {
    "new": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash",
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
    },
    "MOE_med": {
        "batch_size": 512,
        "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "moe_checkpoints_vqvae_moe_med"),
        "vq_kwargs": {"num_codes": 4096, "beta": 0.8, "affine_lr": 1.0,
                      "sync_nu": 2, "replace_freq": 3, "dim": -1},
    },
    "MOE_large": {
        "batch_size": 512,
        "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "moe_checkpoints_vqvae_moe_large_1"),
        "vq_kwargs": {"num_codes": 8192, "beta": 0.9, "affine_lr": 0.0,
                      "sync_nu": 5, "replace_freq": 2, "dim": -1},
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

    x_parts = torch.cat([d.tensors[0] for d in datasets], dim=0)
    x_jets = torch.cat([d.tensors[1] for d in datasets], dim=0)
    y = torch.cat([d.tensors[2] for d in datasets], dim=0)
    
    print(f"📊 Combined dataset: {len(x_parts)} total samples")
    
    if use_mask:
        masks = torch.cat([d.tensors[3] for d in datasets], dim=0)
        return TensorDataset(x_parts, x_jets, y, masks)
    return TensorDataset(x_parts, x_jets, y)

def compute_global_stats(dataset, batch_size, log_pt=False, use_mask=False):
    """Compute global mean and std statistics for normalization with robust handling."""
    print(f"🔢 Computing global statistics with log_pt={log_pt}, use_mask={use_mask}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_parts = []
    all_masks = [] if use_mask else None
    
    # Collect all data first
    for batch_idx, batch in enumerate(loader):
        if use_mask:
            x_part, _, _, mask = batch
            all_masks.append(mask)
        else:
            x_part, _, _ = batch
        all_parts.append(x_part)
        
        # Early break for very large datasets to avoid memory issues
        if batch_idx >= 100:  # Limit to ~50k samples for stats
            print(f"⚠️ Limited global stats computation to first {batch_idx + 1} batches")
            break
    
    particles = torch.cat(all_parts, dim=0)  # [B, 3, T] 
    particles = particles.transpose(1, 2)    # [B, T, 3] for easier processing
    
    if use_mask:
        masks = torch.cat(all_masks, dim=0)  # [B, T]
        
        # Apply log transformation BEFORE masking and flattening
        if log_pt:
            particles[:, :, 0] = torch.log(particles[:, :, 0] + 1e-6)
        
        # Flatten and apply mask
        flat_particles = particles.reshape(-1, particles.shape[-1])  # [B*T, 3]
        flat_mask = masks.reshape(-1).bool()  # [B*T]
        valid_particles = flat_particles[flat_mask]  # [N_valid, 3]
        
        print(f"📊 Mask statistics for global stats:")
        print(f"  Total tokens: {flat_mask.shape[0]:,}")
        print(f"  Valid tokens: {flat_mask.sum():,}")
        print(f"  Valid ratio: {flat_mask.float().mean()*100:.2f}%")
        
    else:
        # No masking case
        flat_particles = particles.reshape(-1, particles.shape[-1])  # [B*T, 3]
        if log_pt:
            flat_particles[:, 0] = torch.log(flat_particles[:, 0] + 1e-6)
        valid_particles = flat_particles
    
    # Compute statistics on valid particles only
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

    # Create model with different configs for MOE vs non-MOE models
    if config["type"] in ["MOE_med", "MOE_large"]:
        # MOE model configuration
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=16,  # Match MOE training script
            hidden_dim=128, # Match MOE training script
            num_heads=8,
            num_blocks=3,
            vq_kwargs=config["vq_kwargs"],
        ).to(device)
    else:
        # Non-MOE model configuration (new, masked, particle)
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=128,
            hidden_dim=256,  # Fixed: changed from 128 to 256
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


def plot_embedding_pca(z_e, z_q, recon_particles, orig_particles, epoch, train_type, save_dir):
    """Create PCA plots showing original and reconstructed embeddings similar to MOE plot_model."""
    
    # Convert to numpy for PCA
    z_e_np = z_e.detach().cpu().numpy()
    z_q_np = z_q.detach().cpu().numpy()
    
    # Reshape if needed
    if len(z_e_np.shape) > 2:
        z_e_np = z_e_np.reshape(z_e_np.shape[0], -1)
    if len(z_q_np.shape) > 2:
        z_q_np = z_q_np.reshape(z_q_np.shape[0], -1)
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # Fit PCA on combined data for consistent scaling
    combined = np.vstack([z_e_np, z_q_np])
    pca.fit(combined)
    
    z_e_2d = pca.transform(z_e_np)
    z_q_2d = pca.transform(z_q_np)
    
    # Limit number of points for clarity
    n_plot = min(500, len(z_e_2d))
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Embedding space (original vs quantized)
    ax = axes[0]
    ax.scatter(z_e_2d[:n_plot, 0], z_e_2d[:n_plot, 1], 
              alpha=0.4, marker="o", color="darkorchid", 
              label="Original embeddings", s=20)
    ax.scatter(z_q_2d[:n_plot, 0], z_q_2d[:n_plot, 1], 
              alpha=0.6, marker="x", color="darkorange", 
              label="Quantized embeddings", s=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Embedding Space (PCA)\n(Original vs Quantized)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Particle space reconstruction
    orig_np = orig_particles.detach().cpu().numpy()
    recon_np = recon_particles.detach().cpu().numpy()
    
    ax = axes[1]
    ax.scatter(orig_np[:n_plot, 0], orig_np[:n_plot, 1], 
              alpha=0.4, marker="o", color="royalblue", 
              label="Original particles", s=20)
    ax.scatter(recon_np[:n_plot, 0], recon_np[:n_plot, 1], 
              alpha=0.6, marker="x", color="forestgreen", 
              label="Reconstructed particles", s=30)
    ax.set_xlabel("pt")
    ax.set_ylabel("eta")
    ax.set_title("Particle Space\n(Original vs Reconstructed)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Jet features comparison
    # Since we have flattened particles, we need to group them back into jets for reconstruction
    # For simplicity, let's just plot the particle-level pt vs eta directly
    ax = axes[2]
    
    # Take a subset of particles and plot their pt vs eta directly
    orig_subset = orig_np[:n_plot]
    recon_subset = recon_np[:n_plot]
    
    ax.scatter(orig_subset[:, 0], orig_subset[:, 2], 
              alpha=0.4, marker="o", color="royalblue", 
              label="Original particles", s=20)
    ax.scatter(recon_subset[:, 0], recon_subset[:, 2], 
              alpha=0.6, marker="x", color="forestgreen", 
              label="Reconstructed particles", s=30)
    ax.set_xlabel("pt")
    ax.set_ylabel("phi")
    ax.set_title("Particle Features\n(pt vs phi)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{train_type}_epoch_{epoch}_embedding_analysis.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved embedding analysis plot: {filepath}")
    
    return filepath


def create_particle_overlay_plots(orig_particles, recon_particles, epoch, train_type, save_dir):
    """Create overlay plots for particle-level features."""
    
    orig_np = orig_particles.detach().cpu().numpy()
    recon_np = recon_particles.detach().cpu().numpy()
    
    # Use all available particles (already limited by MAX_PARTICLES_PER_JET)
    n_plot = min(len(orig_np), MAX_PARTICLES_PER_JET)
    orig_subset = orig_np[:n_plot]
    recon_subset = recon_np[:n_plot]
    
    print(f"📊 Creating particle overlay plots with {n_plot} particles")
    
    # Create 2x2 subplot for particle features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: pt distribution
    ax = axes[0, 0]
    ax.hist(orig_subset[:, 0], bins=50, alpha=0.6, label='Original', color='royalblue', density=True)
    ax.hist(recon_subset[:, 0], bins=50, alpha=0.6, label='Reconstructed', color='forestgreen', density=True)
    ax.set_xlabel('pt')
    ax.set_ylabel('Density')
    ax.set_title('Particle pt Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: eta distribution
    ax = axes[0, 1]
    ax.hist(orig_subset[:, 1], bins=50, alpha=0.6, label='Original', color='royalblue', density=True)
    ax.hist(recon_subset[:, 1], bins=50, alpha=0.6, label='Reconstructed', color='forestgreen', density=True)
    ax.set_xlabel('eta')
    ax.set_ylabel('Density')
    ax.set_title('Particle eta Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: phi distribution
    ax = axes[1, 0]
    ax.hist(orig_subset[:, 2], bins=50, alpha=0.6, label='Original', color='royalblue', density=True)
    ax.hist(recon_subset[:, 2], bins=50, alpha=0.6, label='Reconstructed', color='forestgreen', density=True)
    ax.set_xlabel('phi')
    ax.set_ylabel('Density')
    ax.set_title('Particle phi Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: 2D scatter plot (pt vs eta)
    ax = axes[1, 1]
    n_scatter = min(5000, len(orig_subset))
    ax.scatter(orig_subset[:n_scatter, 0], orig_subset[:n_scatter, 1], 
              alpha=0.4, s=5, color='royalblue', label='Original')
    ax.scatter(recon_subset[:n_scatter, 0], recon_subset[:n_scatter, 1], 
              alpha=0.4, s=5, color='forestgreen', label='Reconstructed')
    ax.set_xlabel('pt')
    ax.set_ylabel('eta')
    ax.set_title('Particle pt vs eta')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{train_type}_epoch_{epoch}_particle_overlay.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved particle overlay plot: {filepath}")
    return filepath


def create_jet_level_plots(dataloader, model, mean, std, use_mask, log_pt, epoch, train_type, save_dir, device):
    """Create jet-level reconstruction plots by processing batches properly."""
    
    all_orig_jets = []
    all_recon_jets = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 20:  # Limit batches for speed
                break
            
            if use_mask:
                x_particles, _, _, mask = [b.to(device) for b in batch]
            else:
                x_particles, _, _ = [b.to(device) for b in batch]
                mask = None
            
            # Apply SAME preprocessing as MOE training using preprocess_batch logic
            # Keep original particles for jet reconstruction (before any transformation)
            x_particles_orig = x_particles.clone()
            
            # Ensure proper tensor format [B, T, 3] (SAME as MOE preprocess_batch)
            if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                x_particles = x_particles.transpose(1, 2)
            
            # Apply log transformation if configured (SAME as MOE preprocess_batch)
            if log_pt:
                x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
            
            # Apply normalization (SAME as MOE preprocess_batch)
            x_norm = (x_particles - mean) / std
            
            # Apply masking after normalization (SAME as MOE preprocess_batch)
            if use_mask:
                x_norm = x_norm * mask.unsqueeze(-1)
            
            # Model forward pass
            recon, vq_out = model(x_norm, mask=mask)
            
            # Denormalize outputs (SAME inverse as MOE training preprocessing)
            recon_denorm = recon * std + mean
            
            # Apply inverse log transformation if configured to get PHYSICAL values
            if log_pt:
                recon_denorm[:, :, 0] = torch.exp(recon_denorm[:, :, 0]) - 1e-6
                recon_denorm[:, :, 0] = torch.clamp(recon_denorm[:, :, 0], min=1e-6)
            
            # Reconstruct jet features from PHYSICAL particles (SAME as MOE evaluation)
            # Original jets from untransformed particles
            orig_jets = reconstruct_jet_features_from_particles(x_particles_orig.transpose(1, 2))
            # Reconstructed jets from denormalized particles  
            recon_jets = reconstruct_jet_features_from_particles(recon_denorm)
            
            all_orig_jets.append(orig_jets)
            all_recon_jets.append(recon_jets)
    
    if not all_orig_jets:
        raise RuntimeError("No jet data processed")
    
    # Concatenate results
    orig_jets_combined = torch.cat(all_orig_jets, dim=0)
    recon_jets_combined = torch.cat(all_recon_jets, dim=0)
    
    print(f"📊 Processed {len(orig_jets_combined)} jets for jet-level analysis")
    
    # Create jet overlay plot
    jet_filename = os.path.join(save_dir, f"{train_type}_epoch_{epoch}_jet_overlay.png")
    plot_tensor_jet_features(
        [orig_jets_combined, recon_jets_combined], 
        labels=["Original", "Reconstructed"],
        filename=jet_filename
    )
    
    # Create jet difference plot
    diff_filename = os.path.join(save_dir, f"{train_type}_epoch_{epoch}_jet_difference.png")
    plot_difference(orig_jets_combined, recon_jets_combined, filename=diff_filename)
    
    # Create detailed jet feature comparison plots
    create_detailed_jet_plots(orig_jets_combined, recon_jets_combined, epoch, train_type, save_dir)
    
    print(f"✅ Created jet-level plots")
    
    return jet_filename, diff_filename


def create_detailed_jet_plots(orig_jets, recon_jets, epoch, train_type, save_dir):
    """Create detailed jet feature comparison plots."""
    
    orig_np = orig_jets.detach().cpu().numpy()
    recon_np = recon_jets.detach().cpu().numpy()
    
    # Limit to reasonable number for plotting
    n_plot = min(50000, len(orig_np))
    orig_subset = orig_np[:n_plot]
    recon_subset = recon_np[:n_plot]
    
    # Create 2x2 subplot for jet features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    feature_names = ['pt', 'eta', 'phi', 'mass']
    
    for i, (ax, feature) in enumerate(zip(axes.flat, feature_names)):
        if i < orig_subset.shape[1]:  # Make sure we have this feature
            ax.hist(orig_subset[:, i], bins=50, alpha=0.6, label='Original', 
                   color='royalblue', density=True)
            ax.hist(recon_subset[:, i], bins=50, alpha=0.6, label='Reconstructed', 
                   color='forestgreen', density=True)
            ax.set_xlabel(f'Jet {feature}')
            ax.set_ylabel('Density')
            ax.set_title(f'Jet {feature} Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{train_type}_epoch_{epoch}_jet_features_detailed.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved detailed jet features plot: {filepath}")
    return filepath


def encode_decode_analysis(config, epochs_to_analyze, device):
    """Perform encode/decode analysis for specified epochs."""
    
    # Determine model configuration
    config["type"] = TRAIN_TYPE
    
    # Load dataset (similar to compare_checkpoints)
    use_mask = config["type"] == "masked"
    dataset = load_all_labels_dataset(start=10, end=11, use_mask=use_mask)
    
    # Compute global statistics using the SAME parameters as MOE training
    use_log_pt = config.get("log_pt", False)  # Match MOE logic exactly
    mean, std = compute_global_stats(dataset, config["batch_size"], use_log_pt, use_mask)
    mean = mean.to(device)
    std = std.to(device)
    
    # Create dataloader for evaluation
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    print(f"🔍 Analyzing {TRAIN_TYPE} model encoding/decoding...")
    print(f"📊 Dataset: {len(dataset)} samples")
    print(f"🔢 Max particles per jet: {MAX_PARTICLES_PER_JET}")
    
    # Process each epoch
    for epoch in epochs_to_analyze:
        print(f"\n🔄 Processing epoch {epoch}...")
        
        # Find checkpoint file
        checkpoint_path = None
        checkpoint_dir = config["checkpoint_dir"]
        
        print(f"🔍 Looking for checkpoints in: {checkpoint_dir}")
        print(f"🔍 Directory exists: {os.path.exists(checkpoint_dir)}")
        
        if not os.path.exists(checkpoint_dir):
            print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
            continue
        
        if isinstance(epoch, str) and epoch == "latest":
            # Find latest checkpoint
            ckpts = [f for f in os.listdir(checkpoint_dir) 
                    if f.endswith(".pth") and ("moe_epoch_" in f or "vqvae_epoch_" in f)]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint_path = os.path.join(checkpoint_dir, ckpts[-1])
                epoch_num = int(ckpts[-1].split("_")[-1].split(".")[0])
        else:
            # Find specific epoch
            for prefix in ("moe_epoch_", "vqvae_epoch_"):
                candidate = os.path.join(checkpoint_dir, f"{prefix}{epoch}.pth")
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    epoch_num = epoch
                    break
        
        if checkpoint_path is None:
            print(f"❌ Checkpoint not found for epoch {epoch}")
            continue
        
        print(f"📂 Loading checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Load model
        model, use_mask, log_pt = load_model_and_checkpoint(config, checkpoint_path, device)
        
        # Use log_pt from config, not from model loading function  
        use_log_pt = config.get("log_pt", False)  # Match MOE logic exactly
        
        # Perform encode/decode analysis
        all_orig_particles = []
        all_recon_particles = []
        all_z_e = []  # Original embeddings
        all_z_q = []  # Quantized embeddings
        
        particles_collected = 0
        target_particles = MAX_PARTICLES_PER_JET
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if particles_collected >= target_particles:
                    print(f"📊 Collected {particles_collected} particles, stopping...")
                    break
                
                if use_mask:
                    x_particles, _, _, mask = [b.to(device) for b in batch]
                else:
                    x_particles, _, _ = [b.to(device) for b in batch]
                    mask = None
                
                # Apply SAME preprocessing as MOE training using preprocess_batch logic
                # Keep original particles for later comparison (before any transformation)
                x_particles_orig = x_particles.clone()
                
                # Ensure proper tensor format [B, T, 3] (SAME as MOE preprocess_batch)
                if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                    x_particles = x_particles.transpose(1, 2)
                
                # Apply log transformation if configured (SAME as MOE preprocess_batch)
                if use_log_pt:
                    x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
                
                # Apply normalization (SAME as MOE preprocess_batch)
                x_norm = (x_particles - mean) / std
                
                # Apply masking after normalization (SAME as MOE preprocess_batch)
                if use_mask:
                    x_norm = x_norm * mask.unsqueeze(-1)
                
                # Model forward pass
                recon, vq_out = model(x_norm, mask=mask)
                
                # Get embeddings
                if isinstance(vq_out, dict):
                    z_e = vq_out.get("z", vq_out.get("z_e"))  # Original embeddings
                    z_q = vq_out.get("z_q")  # Quantized embeddings
                else:
                    # Fallback for simpler VQ outputs
                    z_e = model.encode(x_norm)
                    z_q = z_e  # If no separate quantized embeddings
                
                # Denormalize outputs (SAME inverse as MOE training preprocessing)
                recon_denorm = recon * std + mean
                
                # Apply inverse log transformation if configured to get PHYSICAL values
                if use_log_pt:
                    recon_denorm[:, :, 0] = torch.exp(recon_denorm[:, :, 0]) - 1e-6
                    recon_denorm[:, :, 0] = torch.clamp(recon_denorm[:, :, 0], min=1e-6)
                
                # Store results (flatten particles for analysis) using PHYSICAL values
                # Original particles: use untransformed x_particles_orig
                x_particles_physical = x_particles_orig.transpose(1, 2) if x_particles_orig.shape[1] == 3 else x_particles_orig
                
                if mask is not None:
                    # Only keep valid particles for both original and reconstructed
                    valid_mask = mask.bool()
                    orig_flat = x_particles_physical[valid_mask]
                    recon_flat = recon_denorm[valid_mask]
                    z_e_flat = z_e[valid_mask] if z_e.shape[:2] == mask.shape else z_e.reshape(-1, z_e.shape[-1])
                    z_q_flat = z_q[valid_mask] if z_q.shape[:2] == mask.shape else z_q.reshape(-1, z_q.shape[-1])
                else:
                    orig_flat = x_particles_physical.reshape(-1, x_particles_physical.shape[-1])
                    recon_flat = recon_denorm.reshape(-1, recon_denorm.shape[-1])
                    z_e_flat = z_e.reshape(-1, z_e.shape[-1])
                    z_q_flat = z_q.reshape(-1, z_q.shape[-1])
                
                # Apply particle limit per batch
                batch_particles = min(len(orig_flat), target_particles - particles_collected)
                if batch_particles > 0:
                    all_orig_particles.append(orig_flat[:batch_particles])
                    all_recon_particles.append(recon_flat[:batch_particles])
                    all_z_e.append(z_e_flat[:batch_particles])
                    all_z_q.append(z_q_flat[:batch_particles])
                    particles_collected += batch_particles
                
                if i % 10 == 0:
                    print(f"  Processed batch {i}, collected {particles_collected}/{target_particles} particles")
        
        if not all_orig_particles:
            print(f"⚠️ No data processed for epoch {epoch}")
            continue
        
        # Concatenate results
        orig_particles = torch.cat(all_orig_particles, dim=0)
        recon_particles = torch.cat(all_recon_particles, dim=0)
        z_e_combined = torch.cat(all_z_e, dim=0)
        z_q_combined = torch.cat(all_z_q, dim=0)
        
        print(f"📊 Final processed particles: {len(orig_particles)}")
        
        # Create embedding analysis plot
        plot_embedding_pca(z_e_combined, z_q_combined, recon_particles, 
                          orig_particles, epoch_num, TRAIN_TYPE, PLOT_DIR)
        
        # Create reconstruction overlay plots
        print(f"📊 Creating reconstruction overlay plots...")
        
        # 1. Particle-level overlay plots (pt, eta, phi distributions)
        create_particle_overlay_plots(orig_particles, recon_particles, epoch_num, TRAIN_TYPE, PLOT_DIR)
        
        # 2. Try to create jet-level plots if we can reconstruct jets properly
        try:
            create_jet_level_plots(dataloader, model, mean, std, use_mask, use_log_pt, epoch_num, TRAIN_TYPE, PLOT_DIR, device)
        except Exception as e:
            print(f"⚠️  Could not create jet-level plots: {e}")
        
        print(f"✅ Completed analysis for epoch {epoch_num}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CONFIGS[TRAIN_TYPE].copy()
    
    print(f"🚀 Starting encode/decode analysis for {TRAIN_TYPE}")
    print(f"📋 Analyzing epochs: {CHECKPOINT_EPOCH}")
    
    # Handle different epoch specifications
    if isinstance(CHECKPOINT_EPOCH, list):
        epochs_to_analyze = CHECKPOINT_EPOCH
    elif isinstance(CHECKPOINT_EPOCH, (int, str)):
        epochs_to_analyze = [CHECKPOINT_EPOCH]
    else:
        epochs_to_analyze = ["latest"]
    
    encode_decode_analysis(config, epochs_to_analyze, device)


if __name__ == "__main__":
    main()
