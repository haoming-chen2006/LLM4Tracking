import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import json
import time
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
from datetime import timedelta

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "moe_training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Git utilities for reproducibility
def get_git_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def get_git_status() -> Optional[str]:
    """Get current git status."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def get_last_commit_message() -> Optional[str]:
    """Get last git commit message."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def save_config(config: Dict[str, Any], checkpoint_dir: str) -> None:
    """Save configuration for reproducibility."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    # Add git information
    config_with_git = config.copy()
    config_with_git["git_info"] = {
        "hash": get_git_hash(),
        "status": get_git_status(),
        "last_commit": get_last_commit_message(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Add environment info
    config_with_git["environment"] = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    with open(config_path, "w") as f:
        json.dump(config_with_git, f, indent=2, default=str)
    
    print(f"💾 Saved config to {config_path}")

def find_best_checkpoint(checkpoint_dir: str, pattern: str = "moe_epoch_*.pth") -> Optional[str]:
    """Find the best (latest) checkpoint in directory with robust error handling."""
    if not os.path.exists(checkpoint_dir):
        print(f"⚠️ Checkpoint directory {checkpoint_dir} does not exist")
        return None
    
    ckpts = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not ckpts:
        print(f"⚠️ No checkpoints found with pattern {pattern} in {checkpoint_dir}")
        return None
    
    try:
        # Sort by epoch number in filename
        latest = max(ckpts, key=lambda x: int(Path(x).stem.split("_")[-1]))
        print(f"🔍 Found latest checkpoint: {latest}")
        return latest
    except Exception as e:
        print(f"❌ Error finding latest checkpoint: {e}")
        return None

def log_hyperparameters(config: Dict[str, Any], rank: int = 0) -> None:
    """Log hyperparameters for tracking."""
    if rank == 0:
        print("\n📋 Hyperparameters:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print()

def setup_environment() -> None:
    """Setup environment for robust training."""
    # Set CUDA_LAUNCH_BLOCKING for better debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Set PyTorch precision
    torch.set_float32_matmul_precision("medium")
    
    # NCCL settings for distributed training
    os.environ.setdefault("NCCL_TIMEOUT", "7200")  # 2 hours
    os.environ.setdefault("NCCL_DEBUG", "INFO")

def get_gpu_properties() -> Dict[str, Any]:
    """Get GPU properties for logging."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    gpu_props = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_props[f"gpu_{i}"] = {
            "name": props.name,
            "total_memory": props.total_memory,
            "major": props.major,
            "minor": props.minor,
        }
    
    return {
        "cuda_available": True,
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpu_props,
    }

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
    plot_difference,
)

TRAIN_TYPE = "MOE_large"  # Options: "MOE_med", "MOE_large"
WORLD_SIZE = 4

MOE_CONFIGS = {
    "MOE_med": {
        "batch_size": 512,
        "num_epochs": 40,
        "learning_rate": 1e-4,
        "start": 40,
        "end": 50,
        "vq_kwargs": {"num_codes": 4096, "beta": 0.45, "affine_lr": 1.0,
                      "sync_nu": 2, "replace_freq": 3, "dim": -1},
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_med",
    },
    "MOE_large": {
        "type": "MOE_large",
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "start": 10,
        "end": 11,
        "vq_kwargs": {"num_codes": 8192, "beta": 0.9, "affine_lr": 1.0,
                      "sync_nu": 5, "replace_freq": 2, "dim": -1},
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_large_1",
    },
}

LABELS = [
    "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L",
    "ZToQQ", "WToQQ", "TTBar", "TTBarLep", "ZJetsToNuNu",
]

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

def setup(rank: int, world_size: int) -> None:
    """Setup distributed training with robust error handling."""
    setup_environment()
    
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")  # Different port for MOE training
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=7200))
        torch.cuda.set_device(rank)
        
        if rank == 0:
            print(f"✅ Distributed training setup complete")
            print(f"  World size: {world_size}")
            print(f"  Backend: nccl")
            gpu_props = get_gpu_properties()
            print(f"  GPU info: {gpu_props}")
        
    except Exception as e:
        print(f"❌ Failed to setup distributed training: {e}")
        raise

def cleanup() -> None:
    dist.destroy_process_group()

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

def load_all_labels_dataset(start: int, end: int, use_mask: bool):
    """Load datasets for all labels with proper validation and error handling."""
    from dataloader.dataloader import load_jetclass_label_as_dataset
    from dataloader.masked_dataloader import load_jetclass_label_as_masked_dataset

    print(f"📥 Loading datasets from files {start} to {end-1} with masking={use_mask}")
    
    datasets = []
    global_mask_stats = {
        "total_valid_tokens": 0,
        "total_possible_tokens": 0,
        "total_tensors": 0,
        "masking_lengths": []
    }
    
    total_samples = 0
    successful_labels = []
    
    for lbl in LABELS:
        try:
            print(f"  Loading {lbl}...")
            if use_mask:
                ds = load_jetclass_label_as_masked_dataset(label=lbl, start=start, end=end)
            else:
                ds = load_jetclass_label_as_dataset(label=lbl, start=start, end=end)
            
            # Validate dataset
            if len(ds) == 0:
                print(f"    ⚠️ Empty dataset for {lbl}, skipping")
                continue
                
            # Check tensor shapes
            x_part = ds.tensors[0]  # [B, 3, T] or [B, T, 3]
            x_jet = ds.tensors[1]   # [B, jet_features]
            y = ds.tensors[2]       # [B, labels]
            
            print(f"    ✅ {lbl}: {len(ds):,} samples")
            print(f"       Particle shape: {x_part.shape}")
            print(f"       Jet shape: {x_jet.shape}")
            print(f"       Label shape: {y.shape}")
            
            # Validate particle features (should be non-negative pt, reasonable eta/phi)
            if x_part.shape[1] == 3:  # [B, 3, T] format
                pt_values = x_part[:, 0, :]
            else:  # [B, T, 3] format  
                pt_values = x_part[:, :, 0]
            
            # Check for invalid values
            invalid_pt = (pt_values < 0).any()
            if invalid_pt:
                print(f"    ⚠️ Warning: Found negative pt values in {lbl}")
            
            # Check for reasonable ranges
            max_pt = pt_values.max().item()
            min_pt = pt_values[pt_values > 0].min().item() if (pt_values > 0).any() else 0
            print(f"       Pt range: [{min_pt:.3f}, {max_pt:.3f}]")
            
            datasets.append(ds)
            successful_labels.append(lbl)
            total_samples += len(ds)
            
            # Collect mask statistics if using masking
            if use_mask and len(ds.tensors) > 3:
                mask = ds.tensors[3]  # [B, T]
                
                # Validate mask shape
                expected_mask_shape = (x_part.shape[0], x_part.shape[-1] if x_part.shape[1] == 3 else x_part.shape[1])
                if mask.shape != expected_mask_shape:
                    print(f"    ⚠️ Warning: Mask shape {mask.shape} doesn't match expected {expected_mask_shape}")
                
                # Collect mask statistics
                total_valid = mask.sum().item()
                total_possible = mask.numel()
                
                global_mask_stats["total_valid_tokens"] += total_valid
                global_mask_stats["total_possible_tokens"] += total_possible
                global_mask_stats["total_tensors"] += len(mask)
                
                # Calculate masking lengths
                for m in mask:
                    valid_length = m.sum().item()
                    global_mask_stats["masking_lengths"].append(valid_length)
                
                valid_ratio = total_valid / total_possible * 100
                print(f"       Mask stats: {total_valid:,}/{total_possible:,} ({valid_ratio:.1f}% valid)")
                
        except Exception as e:
            print(f"    ❌ Failed to load {lbl}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not datasets:
        raise RuntimeError("No valid datasets loaded for any label")
    
    print(f"\n📊 Dataset loading summary:")
    print(f"  Successful labels: {successful_labels}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Average samples per label: {total_samples/len(datasets):.0f}")
        
    if use_mask and global_mask_stats["total_tensors"] > 0:
        # Print comprehensive mask statistics
        print(f"\n📊 Global Mask Statistics:")
        print(f"  Total tensors: {global_mask_stats['total_tensors']:,}")
        print(f"  Total valid tokens: {global_mask_stats['total_valid_tokens']:,}")
        print(f"  Total possible tokens: {global_mask_stats['total_possible_tokens']:,}")
        
        valid_ratio = (global_mask_stats['total_valid_tokens'] / 
                      global_mask_stats['total_possible_tokens'] * 100)
        print(f"  Overall valid token ratio: {valid_ratio:.2f}%")
        
        # Calculate masking statistics
        lengths = torch.tensor(global_mask_stats['masking_lengths'])
        print(f"\n📏 Token Length Distribution:")
        print(f"  Mean: {lengths.float().mean():.1f}")
        print(f"  Median: {lengths.float().median():.1f}")
        print(f"  Std: {lengths.float().std():.1f}")
        print(f"  Min: {lengths.min().item()}")
        print(f"  Max: {lengths.max().item()}")
        
        # Print percentiles
        percentiles = [10, 25, 75, 90, 95, 99]
        for p in percentiles:
            val = torch.quantile(lengths.float(), p/100.0)
            print(f"  {p}th percentile: {val:.1f}")

    # Concatenate all datasets
    print(f"\n🔄 Concatenating {len(datasets)} datasets...")
    x_parts = torch.cat([d.tensors[0] for d in datasets], dim=0)
    x_jets = torch.cat([d.tensors[1] for d in datasets], dim=0)
    y = torch.cat([d.tensors[2] for d in datasets], dim=0)
    
    print(f"📦 Final dataset shapes:")
    print(f"  Particles: {x_parts.shape}")
    print(f"  Jets: {x_jets.shape}")
    print(f"  Labels: {y.shape}")
    
    if use_mask:
        # Include masks in the dataset when use_mask is True
        masks = torch.cat([d.tensors[3] for d in datasets], dim=0)
        print(f"  Masks: {masks.shape}")
        return TensorDataset(x_parts, x_jets, y, masks)
    
    return TensorDataset(x_parts, x_jets, y)

def ddp_train_moe(rank: int, world_size: int, config: dict) -> None:
    """Main MOE training function with robust pipeline components."""
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        # Log training start
        if rank == 0:
            print(f"\n🚀 Starting MOE training with config:")
            log_hyperparameters(config, rank)
            
            # Save config for reproducibility
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            save_config(config, config["checkpoint_dir"])

        # Seeding for reproducibility
        base_seed = config.get("seed", 42)
        seed = base_seed + rank
        generator = seed_everything(seed)

        # Remove wandb initialization - using offline logging only
        if rank == 0:
            print("📊 Training without online logging (offline mode)")

        # Setup data module for better data management
        data_module = MOEDataModule(config, rank, world_size)
        data_module.prepare_data()
        data_module.setup_normalization(device)
        data_module.setup_dataloader(generator, seed_worker)
        
        # Get dataloader and normalization stats
        dataloader = data_module.dataloader
        use_mask = data_module.use_mask
        log_pt = data_module.log_pt
        mean = data_module.mean
        std = data_module.std

        # Create MOE model
        model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])
        model = model_module.VQVAENormFormer(
            input_dim=3,
            latent_dim=16,
            hidden_dim=128,
            num_heads=8,
            num_blocks=3,
            vq_kwargs=config["vq_kwargs"],
        ).to(device)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))
        
        # Add cosine learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config["num_epochs"], 
            eta_min=config["learning_rate"] * 0.01  # Minimum LR is 1% of initial LR
        )
        
        recon_loss_fn = nn.MSELoss(reduction="none")
        scaler = GradScaler()

        # Load most recent checkpoint with robust checkpoint finding
        start_epoch = 0
        checkpoint = None
        if rank == 0:
            checkpoint_path = find_best_checkpoint(config["checkpoint_dir"])
            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    start_epoch = checkpoint["epoch"]
                    print(f"🔄 Loaded MOE checkpoint from {checkpoint_path} (epoch {start_epoch})")
                except Exception as e:
                    print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
                    start_epoch = 0
            else:
                print("🆕 No MOE checkpoint found, starting from scratch")

        # Broadcast checkpoint to all ranks
        obj_list = [checkpoint]
        dist.broadcast_object_list(obj_list, src=0)
        checkpoint = obj_list[0]

        if checkpoint is not None:
            missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint["model_state"], strict=False)
            if not missing_keys:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if "scheduler_state" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                if "scaler_state" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint.get("epoch", 0)
            if rank == 0:
                if missing_keys:
                    print(f"⚠️  Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Broadcast start_epoch to all processes
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, 0)
        start_epoch = start_epoch_tensor.item()

        if rank == 0:
            print(f"🚀 MOE Training from epoch {start_epoch + 1} to {config['num_epochs']}")
            print(f"📊 Dataset size: {len(data_module.dataset)}")
            print(f"🔢 Total batches per epoch: {len(dataloader)}")

        # Check if we need to train at all
        if start_epoch >= config["num_epochs"]:
            if rank == 0:
                print(f"⚠️  MOE Training already completed! start_epoch ({start_epoch}) >= num_epochs ({config['num_epochs']})")
            cleanup()
            return

        # Training loop with better error handling and metrics tracking
        training_metrics = []
        
        for epoch in range(start_epoch, config["num_epochs"]):
            epoch_start_time = time.time()
            
            # Initialize per-epoch statistics
            epoch_loss = torch.tensor(0.0, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            vq_loss = torch.tensor(0.0, device=device)
            aux_loss = torch.tensor(0.0, device=device)
            
            # Initialize code histogram for tracking token usage (like train_jet.py)                           
            code_hist = torch.zeros(config["vq_kwargs"]["num_codes"], device=device, dtype=torch.long)
            
            if use_mask:
                # Reset mask stats for each epoch
                epoch_mask_stats = {
                    "valid_tokens": 0,
                    "total_tokens": 0,
                    "sample_count": 0
                }
            
            batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                batch_count += 1
                
                # Use data module for consistent preprocessing
                x_norm, mask, x_particles, x_jets, y = data_module.preprocess_batch(batch, device)
                
                # Validate batch data
                if not data_module.validate_batch(x_particles, mask, batch_idx):
                    continue
                
                # Print mask statistics for first batch of first epoch
                if use_mask and epoch == start_epoch and batch_idx == 0 and rank == 0:
                    print("\n🔍 First batch mask statistics:")
                    valid_tokens = mask.sum(dim=1)  # [B]
                    total_tokens = mask.shape[1]    # T
                    for i in range(min(6, len(mask))):
                        print(f"  Sample {i}: {valid_tokens[i].item()}/{total_tokens} tokens "
                              f"({valid_tokens[i].item()/total_tokens*100:.1f}% valid)")
                    print(f"  Batch average: {valid_tokens.float().mean().item():.1f} valid tokens")
                    print(f"  Mask shape: {mask.shape}")
                    print(f"  Particle shape: {x_particles.shape}")
                    print(f"  Normalized range: [{x_norm.min():.3f}, {x_norm.max():.3f}]\n")
                
                # Accumulate mask stats for this batch only
                if use_mask:
                    epoch_mask_stats["valid_tokens"] += int(mask.sum().item())
                    epoch_mask_stats["total_tokens"] += int(mask.numel())
                    epoch_mask_stats["sample_count"] += int(len(mask))

                optimizer.zero_grad()
                with autocast():
                    # Pass mask to model for proper attention masking
                    out, loss_dict = model(x_norm, mask=mask)
                    
                    # Compute reconstruction loss with proper masking
                    if use_mask:
                        # Only compute loss on valid (unmasked) positions
                        reconstruction_error = (out - x_norm) ** 2  # [B, T, 3]
                        masked_error = reconstruction_error * mask.unsqueeze(-1)  # Zero out invalid positions
                        
                        # Average over valid positions only
                        total_valid_elements = mask.sum() * reconstruction_error.shape[-1]  # Total valid features
                        if total_valid_elements > 0:
                            r_loss = masked_error.sum() / total_valid_elements
                        else:
                            r_loss = torch.tensor(0.0, device=device, requires_grad=True)
                            print(f"⚠️ Warning: No valid tokens in batch {batch_idx}")
                    else:
                        r_loss = recon_loss_fn(out, x_norm).mean()

                    # Safely handle dict vs tensor loss_dict for MOE
                    if isinstance(loss_dict, dict):
                        vq_loss_val = loss_dict.get("vq_loss", loss_dict.get("loss", torch.tensor(0.0, device=device)))
                        aux_loss_val = loss_dict.get("aux_loss", torch.tensor(0.0, device=device))
                        total_latent_loss = loss_dict.get("total_loss", vq_loss_val + 0.01 * aux_loss_val)
                        
                        # Track code usage histogram (like train_jet.py)
                        codes = loss_dict.get("q")
                        if codes is not None:
                            try:
                                hist = torch.bincount(codes.view(-1), minlength=config["vq_kwargs"]["num_codes"])
                                code_hist += hist.to(device)
                            except Exception as e:
                                print(f"⚠️ Warning: Error computing code histogram: {e}")
                    else:
                        vq_loss_val = loss_dict
                        aux_loss_val = torch.tensor(0.0, device=device)
                        total_latent_loss = loss_dict

                    # Validate loss values
                    if torch.isnan(r_loss) or torch.isinf(r_loss):
                        print(f"❌ Invalid reconstruction loss detected: {r_loss}")
                        continue
                        
                    if torch.isnan(total_latent_loss) or torch.isinf(total_latent_loss):
                        print(f"❌ Invalid latent loss detected: {total_latent_loss}")
                        continue

                    loss = r_loss + total_latent_loss
                
                scaler.scale(loss).backward()
                
                # Add gradient clipping for stability (like train_jet.py)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.detach()
                recon_loss += r_loss.detach()
                vq_loss += vq_loss_val.detach()
                aux_loss += aux_loss_val.detach()

                # Log batch metrics every 25 batches (more frequent for MOE)
                if rank == 0 and batch_idx % 25 == 0:
                    print(f"  MOE Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

            if rank == 0:
                # Print epoch completion and masking statistics
                epoch_time = time.time() - epoch_start_time
                print(f"\n✅ MOE Epoch {epoch + 1} completed in {epoch_time:.1f}s:")
                print(f"  Processed {batch_count} batches")
                if use_mask:
                    # Only print local stats, skip distributed reduction
                    if epoch_mask_stats["total_tokens"] > 0 and epoch_mask_stats["sample_count"] > 0:
                        valid_ratio = epoch_mask_stats["valid_tokens"] / epoch_mask_stats["total_tokens"] * 100
                        avg_valid_tokens = epoch_mask_stats["valid_tokens"] / epoch_mask_stats["sample_count"]
                    else:
                        valid_ratio = 0.0
                        avg_valid_tokens = 0.0
                    print(f"\n📊 Epoch Mask Statistics (local process):")
                    print(f"  Total samples processed: {epoch_mask_stats['sample_count']:,}")
                    print(f"  Total valid tokens: {epoch_mask_stats['valid_tokens']:,}")
                    print(f"  Total possible tokens: {epoch_mask_stats['total_tokens']:,}")
                    print(f"  Valid token ratio: {valid_ratio:.1f}%")
                    print(f"  Average valid tokens per sample: {avg_valid_tokens:.1f}\n")

            epoch_loss /= len(dataloader)
            recon_loss /= len(dataloader)
            vq_loss /= len(dataloader)
            aux_loss /= len(dataloader)
            for t in (epoch_loss, recon_loss, vq_loss, aux_loss):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t /= world_size
            
            # Reduce code histogram across all processes (like train_jet.py)
            dist.all_reduce(code_hist)
            unique_codes = torch.count_nonzero(code_hist).item()

            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                # Store metrics for later analysis
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "epoch_loss": epoch_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "vq_loss": vq_loss.item(),
                    "aux_loss": aux_loss.item(),
                    "learning_rate": current_lr,
                    "unique_codes": unique_codes,
                    "epoch_time": epoch_time,
                }
                training_metrics.append(epoch_metrics)
                
                print(
                    f"MOE Epoch {epoch+1}/{config['num_epochs']} - Total: {epoch_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                    f"Codes: {unique_codes}/{config['vq_kwargs']['num_codes']} | LR: {current_lr:.6f}"
                )

                # Save checkpoint every epoch with better error handling
                try:
                    checkpoint_path = os.path.join(config["checkpoint_dir"], f"moe_epoch_{epoch+1}.pth")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": model.module.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "config": config,
                            "training_metrics": training_metrics,
                        },
                        checkpoint_path,
                    )
                    print(f"💾 Saved MOE checkpoint at {checkpoint_path}")
                except Exception as e:
                    print(f"❌ Failed to save checkpoint: {e}")
            
            # Step the scheduler at the end of each epoch
            scheduler.step()

        # Save final training metrics
        if rank == 0:
            try:
                metrics_path = os.path.join(config["checkpoint_dir"], "training_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(training_metrics, f, indent=2)
                print(f"📊 Saved training metrics to {metrics_path}")
            except Exception as e:
                print(f"⚠️ Failed to save training metrics: {e}")

        cleanup()
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        if rank == 0:
            import traceback
            traceback.print_exc()
        cleanup()
        raise

def ddp_eval_moe(config: dict) -> None:
    """Evaluation function with robust checkpoint handling."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = seed_everything(config.get("seed", 42))
    
    print(f"\n🔍 Starting MOE evaluation for {config['type']}")
    
    # Load training dataset for stats (use same parameters as training)
    print("📊 Loading training dataset to compute normalization statistics...")
    dataset = load_all_labels_dataset(config["start"], config["end"], False)  # Don't use mask for stats
    model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])

    # Compute global stats using the SAME parameters as training
    use_log_pt = config.get("log_pt", False)
    mean, std = compute_global_stats(dataset, config["batch_size"], use_log_pt, False)
    mean = mean.to(device)
    std = std.to(device)

    # Create MOE model for evaluation
    model = model_module.VQVAENormFormer(
        input_dim=3,
        latent_dim=16,
        hidden_dim=128,
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    # Load checkpoint with robust error handling
    checkpoint_path = None
    if "specific_checkpoint" in config:
        checkpoint_path = config["specific_checkpoint"]
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Specified checkpoint {checkpoint_path} does not exist")
            checkpoint_path = None
    
    if not checkpoint_path:
        checkpoint_path = find_best_checkpoint(config["checkpoint_dir"])
    
    if checkpoint_path:
        try:
            print(f"📊 Evaluating checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state"], strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys when loading checkpoint: {unexpected_keys}")
                
            # Load training metrics if available
            if "training_metrics" in checkpoint:
                print(f"📈 Loaded training history with {len(checkpoint['training_metrics'])} epochs")
                
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            print("⚠️ Using randomly initialized model for evaluation")
    else:
        print("⚠️ No checkpoint found, using randomly initialized model")

    model.eval()
    all_orig_jets = []
    all_recon_jets = []
    all_tokens = []
    
    # Process each label individually
    for label in LABELS:
        try:
            print(f"🔄 Evaluating label: {label}")
            from dataloader.dataloader import load_jetclass_label_as_tensor
            
            # Create dataloader for this label
            dataloader_eval = load_jetclass_label_as_tensor(
                label=label,
                start=11,
                end=12,
                batch_size=config["batch_size"],
                generator=generator,
                worker_init_fn=seed_worker,
            )
            
            if len(dataloader_eval) == 0:
                print(f"⚠️ No data found for label {label}")
                continue
                
            label_orig_jets = []
            label_recon_jets = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader_eval):
                    if i >= 50:  # Limit batches per label
                        break
                        
                    x_particles, x_jets, y = [b.to(device) for b in batch]
                    
                    # Apply SAME preprocessing as training using data module logic
                    # Ensure proper tensor format [B, T, 3]
                    if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
                        x_particles = x_particles.transpose(1, 2)
                    
                    # Apply log transformation if configured (SAME as training)
                    if use_log_pt:
                        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
                    
                    # Apply normalization (SAME as training)
                    x_norm = (x_particles - mean) / std

                    # Model forward pass (no mask in evaluation)
                    out, loss_dict = model(x_norm)
                    
                    # Collect token info
                    if isinstance(loss_dict, dict) and "q" in loss_dict:
                        all_tokens.append(loss_dict["q"].detach().cpu())
                        
                    # Denormalize outputs (SAME inverse as training preprocessing)
                    out_denorm = out * std + mean
                    
                    # Apply inverse log transformation if configured
                    if use_log_pt:
                        out_denorm[:, :, 0] = torch.exp(out_denorm[:, :, 0]) - 1e-6
                        out_denorm[:, :, 0] = torch.clamp(out_denorm[:, :, 0], min=1e-6)
                        
                        # Also inverse log transform the original for consistency
                        x_particles_denorm = x_particles * std + mean
                        x_particles_denorm[:, :, 0] = torch.exp(x_particles_denorm[:, :, 0]) - 1e-6
                        x_particles_denorm[:, :, 0] = torch.clamp(x_particles_denorm[:, :, 0], min=1e-6)
                    else:
                        x_particles_denorm = x_particles * std + mean

                    # Reconstruct jet features from denormalized particles
                    orig_jet = reconstruct_jet_features_from_particles(x_particles_denorm)
                    recon_jet = reconstruct_jet_features_from_particles(out_denorm)

                    label_orig_jets.append(orig_jet)
                    label_recon_jets.append(recon_jet)
                    
            # Add this label's jets to the overall collection
            if label_orig_jets:
                label_orig = torch.cat(label_orig_jets, dim=0)
                label_recon = torch.cat(label_recon_jets, dim=0)
                
                all_orig_jets.append(label_orig)
                all_recon_jets.append(label_recon)
                print(f"✅ {label}: Processed {len(label_orig)} jets")
            
        except Exception as e:
            print(f"❌ Error processing label {label}: {e}")
    
    # Concatenate results from all labels
    if not all_orig_jets:
        print("⚠️ No valid evaluation data for any label")
        return
        
    all_orig_jets = torch.cat(all_orig_jets, dim=0)
    all_recon_jets = torch.cat(all_recon_jets, dim=0)
    
    print(f"📊 Total: {len(all_orig_jets)} jets evaluated across all labels")
    
    # Create plots with all labels combined
    try:
        plot_tensor_jet_features(
            [all_orig_jets, all_recon_jets],
            labels=("Original (All Labels)", f"MOE Reconstructed (All Labels)"),
            filename=os.path.join(PLOT_DIR, f"moe_{config['type']}_all_labels.png"),
        )
        
        plot_difference(
            all_orig_jets,
            all_recon_jets,
            filename=os.path.join(PLOT_DIR, f"moe_{config['type']}_all_labels_diff.png"),
        )
        print(f"📈 Saved evaluation plots to {PLOT_DIR}")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
    
    # Plot token usage with better error handling
    if all_tokens:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Concatenate all token indices
            all_tokens_tensor = torch.cat(all_tokens)
            token_indices = all_tokens_tensor.flatten().numpy()
            
            # Count unique tokens
            unique_tokens = np.unique(token_indices)
            utilization_rate = len(unique_tokens) / config["vq_kwargs"]["num_codes"] * 100
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(token_indices, bins=min(100, config["vq_kwargs"]["num_codes"]), alpha=0.7)
            plt.title(f'Token Usage - {len(unique_tokens)}/{config["vq_kwargs"]["num_codes"]} tokens ({utilization_rate:.1f}%)')
            plt.xlabel('Token ID')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(PLOT_DIR, f"moe_{config['type']}_token_usage.png"))
            plt.close()
            
            print(f"🔖 Token utilization: {len(unique_tokens)}/{config['vq_kwargs']['num_codes']} tokens ({utilization_rate:.1f}%)")
        except Exception as e:
            print(f"⚠️ Error creating token histogram: {e}")
    
    print("✅ MOE evaluation completed")

class MOEDataModule:
    """Data module for MOE training following Lightning pattern for better data management."""
    
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_mask = config.get("use_mask", False)
        self.log_pt = config.get("log_pt", False)
        self.batch_size = config["batch_size"]
        
        # Data statistics
        self.mean = None
        self.std = None
        self.dataset = None
        self.dataloader = None
        
    def prepare_data(self):
        """Load and prepare the dataset."""
        if self.rank == 0:
            print("🔄 Preparing data...")
        
        # Load dataset
        self.dataset = load_all_labels_dataset(
            self.config["start"], 
            self.config["end"], 
            self.use_mask
        )
        
        if self.rank == 0:
            print(f"📊 Dataset prepared with {len(self.dataset)} samples")
    
    def setup_normalization(self, device):
        """Setup normalization statistics."""
        if self.rank == 0:
            self.mean, self.std = compute_global_stats(
                self.dataset, 
                self.batch_size, 
                self.log_pt, 
                self.use_mask
            )
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.zeros(3, device=device)
            self.std = torch.ones(3, device=device)
        
        # Broadcast to all ranks
        if dist.is_initialized():
            dist.broadcast(self.mean, 0)
            dist.broadcast(self.std, 0)
        
        if self.rank == 0:
            print(f"🔢 Normalization setup complete")
    
    def setup_dataloader(self, generator, worker_init_fn):
        """Setup the dataloader with proper distributed sampling."""
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.dataset, 
                num_replicas=self.world_size, 
                rank=self.rank, 
                shuffle=True
            )
        else:
            sampler = None
            
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        
        if self.rank == 0:
            print(f"🔄 Dataloader setup with {len(self.dataloader)} batches per epoch")
    
    def preprocess_batch(self, batch, device):
        """Preprocess a batch with consistent normalization and masking."""
        if self.use_mask:
            x_particles, x_jets, y, mask = [b.to(device) for b in batch]
        else:
            x_particles, x_jets, y = [b.to(device) for b in batch]
            mask = None
        
        # Ensure proper tensor format [B, T, 3]
        if x_particles.shape[1] == 3:  # [B, 3, T] -> [B, T, 3]
            x_particles = x_particles.transpose(1, 2)
        
        # Apply log transformation if configured
        if self.log_pt:
            x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
        
        # Apply normalization
        x_norm = (x_particles - self.mean) / self.std
        
        # Apply masking after normalization
        if self.use_mask:
            x_norm = x_norm * mask.unsqueeze(-1)
        
        return x_norm, mask, x_particles, x_jets, y
    
    def validate_batch(self, x_particles, mask=None, batch_idx=0):
        """Validate batch data for common issues."""
        # Check for NaN/Inf
        if torch.isnan(x_particles).any() or torch.isinf(x_particles).any():
            print(f"⚠️ Warning: NaN/Inf in particle data at batch {batch_idx}")
            return False
        
        # Check particle ranges
        pt_values = x_particles[:, :, 0]
        if (pt_values < 0).any():
            print(f"⚠️ Warning: Negative pt values at batch {batch_idx}")
        
        # Check mask if present
        if mask is not None:
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                print(f"⚠️ Warning: NaN/Inf in mask at batch {batch_idx}")
                return False
            
            if not torch.all((mask == 0) | (mask == 1)):
                print(f"⚠️ Warning: Invalid mask values at batch {batch_idx}")
                return False
        
        return True

def parse_args():
    """Parse command line arguments for better usability."""
    parser = argparse.ArgumentParser(description="MOE training and evaluation pipeline")
    parser.add_argument(
        "--model-type", 
        default=TRAIN_TYPE, 
        choices=list(MOE_CONFIGS.keys()), 
        help="Model configuration to use"
    )
    parser.add_argument(
        "--world-size", 
        type=int, 
        default=WORLD_SIZE, 
        help="Number of GPUs for distributed training"
    )
    parser.add_argument(
        "--train-only", 
        action="store_true", 
        help="Only run training, skip evaluation"
    )
    parser.add_argument(
        "--eval-only", 
        action="store_true", 
        help="Only run evaluation, skip training"
    )
    parser.add_argument(
        "--checkpoint-path", 
        type=str, 
        help="Specific checkpoint path for evaluation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def main() -> None:
    """Main function with robust configuration and error handling."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup global environment before training
        setup_environment()
        
        config = MOE_CONFIGS[args.model_type].copy()
        if "type" not in config:
            config["type"] = args.model_type
        
        # Override config with command line arguments
        config["seed"] = args.seed
        config.setdefault("use_mask", False)
        config.setdefault("log_pt", False)
        
        print(f"🚀 Starting MOE pipeline for {args.model_type}")
        print(f"🔧 Git Hash: {get_git_hash()}")
        print(f"🔧 Git Status: {'Clean' if not get_git_status() else 'Modified'}")
        
        # Log configuration
        log_hyperparameters(config)
        
        # Ensure absolute checkpoint path
        if not os.path.isabs(config["checkpoint_dir"]):
            config["checkpoint_dir"] = os.path.abspath(config["checkpoint_dir"])
        
        print(f"📁 Checkpoint directory: {config['checkpoint_dir']}")
        
        # Handle specific checkpoint path for evaluation
        if args.checkpoint_path:
            config["specific_checkpoint"] = args.checkpoint_path
        
        # Run training unless eval-only mode
        if not args.eval_only:
            print(f"🏋️ Starting distributed training with {args.world_size} processes")
            mp.spawn(ddp_train_moe, args=(args.world_size, config), nprocs=args.world_size, join=True)
        
        # Run evaluation unless train-only mode
        if not args.train_only:
            print(f"🔍 Starting evaluation")
            ddp_eval_moe(config)
        
        print("✅ MOE pipeline completed successfully")
        
    except Exception as e:
        print(f"❌ MOE pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()