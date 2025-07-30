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
import wandb

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "moe_training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

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
        "num_epochs": 30,
        "learning_rate": 1e-4,
        "start": 20,
        "end": 30,
        "vq_kwargs": {"num_codes": 4096, "beta": 0.45, "affine_lr": 1.0,
                      "sync_nu": 2, "replace_freq": 3, "dim": -1},
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_med",
    },
    "MOE_large": {
        "type": "MOE_large",
        "batch_size": 512,
        "num_epochs": 80,
        "learning_rate": 1e-4,
        "start": 80,
        "end": 90,
        "vq_kwargs": {"num_codes": 8192, "beta": 0.9, "affine_lr": 0.0,
                      "sync_nu": 5, "replace_freq": 2, "dim": -1},
        "checkpoint_dir": "checkpoints/moe_checkpoints_vqvae_moe_large",
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
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")  # Different port for MOE training
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

def load_all_labels_dataset(start: int, end: int, use_mask: bool):
    from dataloader.dataloader import load_jetclass_label_as_dataset
    from dataloader.masked_dataloader import load_jetclass_label_as_masked_dataset

    datasets = []
    global_mask_stats = {
        "total_valid_tokens": 0,
        "total_possible_tokens": 0,
        "total_tensors": 0,
        "masking_lengths": []
    }
    
    for lbl in LABELS:
        try:
            if use_mask:
                ds = load_jetclass_label_as_masked_dataset(label=lbl, start=start, end=end)
            else:
                ds = load_jetclass_label_as_dataset(label=lbl, start=start, end=end)
            datasets.append(ds)
            
            # Add diagnostic prints for first dataset of each type
            if len(datasets) == 1:
                print(f"\n🔍 Examining first dataset ({lbl}):")
                print(f"  Dataset size: {len(ds)}")
                x_part = ds.tensors[0]  # [B, T, 3]
                print(f"  Particle tensor shape: {x_part.shape}")
                
                if use_mask:
                    mask = ds.tensors[3]  # [B, T]
                    # Print mask statistics for first 6 samples
                    for i in range(min(6, len(mask))):
                        valid_tokens = mask[i].sum().item()
                        total_tokens = mask[i].shape[0]
                        print(f"  Sample {i}: {valid_tokens}/{total_tokens} tokens "
                              f"({valid_tokens/total_tokens*100:.1f}% valid)")
                    
                    # Global mask statistics for this dataset
                    total_valid = mask.sum().item()
                    total_possible = mask.numel()
                    
                    # Update global statistics
                    global_mask_stats["total_valid_tokens"] += total_valid
                    global_mask_stats["total_possible_tokens"] += total_possible
                    global_mask_stats["total_tensors"] += len(mask)
                    
                    # Calculate masking lengths for this dataset
                    for m in mask:
                        valid_length = m.sum().item()
                        global_mask_stats["masking_lengths"].append(valid_length)
                    
                    print(f"\n📊 Dataset mask statistics ({lbl}):")
                    print(f"  Total valid tokens: {total_valid:,}")
                    print(f"  Total possible tokens: {total_possible:,}")
                    print(f"  Valid token ratio: {total_valid/total_possible*100:.1f}%\n")
        except Exception as e:
            print(f"⚠️  Failed to load dataset for label {lbl}: {e}")
            continue

    if not datasets:
        raise RuntimeError("No valid datasets loaded for any label")
        
    if use_mask:
        # Print final global mask statistics
        print("\n📊 Final Global Mask Statistics:")
        print(f"  Total tensors processed: {global_mask_stats['total_tensors']:,}")
        print(f"  Total valid tokens: {global_mask_stats['total_valid_tokens']:,}")
        print(f"  Total possible tokens: {global_mask_stats['total_possible_tokens']:,}")
        valid_ratio = (global_mask_stats['total_valid_tokens'] / 
                      global_mask_stats['total_possible_tokens'] * 100)
        print(f"  Overall valid token ratio: {valid_ratio:.1f}%")
        
        # Calculate and print masking length statistics
        lengths = torch.tensor(global_mask_stats['masking_lengths'])
        print(f"\n📏 Masking Length Statistics:")
        print(f"  Mean valid tokens per tensor: {lengths.float().mean():.1f}")
        print(f"  Median valid tokens per tensor: {lengths.float().median():.1f}")
        print(f"  Min valid tokens: {lengths.min().item()}")
        print(f"  Max valid tokens: {lengths.max().item()}\n")

    x_parts = torch.cat([d.tensors[0] for d in datasets], dim=0)
    x_jets = torch.cat([d.tensors[1] for d in datasets], dim=0)
    y = torch.cat([d.tensors[2] for d in datasets], dim=0)
    
    if use_mask:
        # Include masks in the dataset when use_mask is True
        masks = torch.cat([d.tensors[3] for d in datasets], dim=0)
        return TensorDataset(x_parts, x_jets, y, masks)
    return TensorDataset(x_parts, x_jets, y)

def ddp_train_moe(rank: int, world_size: int, config: dict) -> None:
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Seeding for reproducibility
    base_seed = config.get("seed", 42)
    seed = base_seed + rank
    generator = seed_everything(seed)

    # Initialize wandb only on rank 0 with better error handling
    if rank == 0:
        try:
            # Set wandb mode to offline if login fails
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project="hep-models-moe",
                name=f"moe_{config['type']}_{config['start']}_{config['end']}",
                config={
                    "batch_size": config["batch_size"],
                    "num_epochs": config["num_epochs"],
                    "learning_rate": config["learning_rate"],
                    "world_size": world_size,
                    "train_type": config["type"],
                    "data_range": f"{config['start']}-{config['end']}",
                    **config["vq_kwargs"]
                }
            )
            print("✅ WandB initialized successfully")
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            print("📊 Continuing training without WandB logging...")
            # Set a flag to disable wandb logging
            os.environ["WANDB_DISABLED"] = "true"

    # Load MOE model with proper masking
    dataset = load_all_labels_dataset(config["start"], config["end"], config.get("use_mask", False))
    use_mask = config.get("use_mask", False)
    log_pt = config.get("log_pt", False)
    model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        generator=generator,
        worker_init_fn=seed_worker,
    )

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

    # Create MOE model
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

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Load most recent checkpoint with scheduler state
    start_epoch = 0
    checkpoint = None
    if rank == 0:
        ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("moe_epoch_") and f.endswith(".pth")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_path = os.path.join(config["checkpoint_dir"], latest)
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
        print(f"📊 Dataset size: {len(dataset)}")
        print(f"🔢 Total batches per epoch: {len(dataloader)}")

    # Check if we need to train at all
    if start_epoch >= config["num_epochs"]:
        if rank == 0:
            print(f"⚠️  MOE Training already completed! start_epoch ({start_epoch}) >= num_epochs ({config['num_epochs']})")
            wandb.finish()
        cleanup()
        return

    for epoch in range(start_epoch, config["num_epochs"]):
        # Initialize per-epoch statistics
        epoch_loss = torch.tensor(0.0, device=device)
        recon_loss = torch.tensor(0.0, device=device)
        vq_loss = torch.tensor(0.0, device=device)
        aux_loss = torch.tensor(0.0, device=device)
        
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
            if use_mask:
                x_particles, _, _, mask = [b.to(device) for b in batch]
                # Print mask statistics for first batch of first epoch
                if epoch == start_epoch and batch_idx == 0 and rank == 0:
                    print("\n🔍 First batch mask statistics:")
                    valid_tokens = mask.sum(dim=1)  # [B]
                    total_tokens = mask.shape[1]    # T
                    for i in range(min(6, len(mask))):
                        print(f"  Sample {i}: {valid_tokens[i].item()}/{total_tokens} tokens "
                              f"({valid_tokens[i].item()/total_tokens*100:.1f}% valid)")
                    print(f"  Batch average: {valid_tokens.float().mean().item():.1f} valid tokens")
                    print(f"  Mask shape: {mask.shape}\n")
                # Accumulate mask stats for this batch only
                epoch_mask_stats["valid_tokens"] += int(mask.sum().item())
                epoch_mask_stats["total_tokens"] += int(mask.numel())
                epoch_mask_stats["sample_count"] += int(len(mask))
            else:
                x_particles, _, _ = [b.to(device) for b in batch]
                mask = None
                
            x_particles = x_particles.transpose(1, 2)
            if log_pt:
                x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)

            x_norm = (x_particles - mean) / std if not use_mask else (
                (x_particles - mean) / std) * mask.unsqueeze(-1)

            optimizer.zero_grad()
            with autocast():
                # Pass mask to model
                out, loss_dict = model(x_norm, mask=mask)
                
                # Use masked loss computation
                if use_mask:
                    r_loss = (((out - x_norm) ** 2) * mask.unsqueeze(-1)).sum() / mask.sum()
                else:
                    r_loss = recon_loss_fn(out, x_norm).mean()

                # Safely handle dict vs tensor loss_dict for MOE
                if isinstance(loss_dict, dict):
                    vq_loss_val = loss_dict.get("vq_loss", loss_dict.get("loss", torch.tensor(0.0, device=device)))
                    aux_loss_val = loss_dict.get("aux_loss", torch.tensor(0.0, device=device))
                    total_latent_loss = loss_dict.get("total_loss", vq_loss_val + 0.01 * aux_loss_val)
                else:
                    vq_loss_val = loss_dict
                    aux_loss_val = torch.tensor(0.0, device=device)
                    total_latent_loss = loss_dict

                loss = r_loss + total_latent_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach()
            recon_loss += r_loss.detach()
            vq_loss += vq_loss_val.detach()
            aux_loss += aux_loss_val.detach()

            # Log batch metrics every 25 batches (more frequent for MOE)
            if rank == 0 and batch_idx % 25 == 0:
                print(f"  MOE Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
                try:
                    if os.environ.get("WANDB_DISABLED") != "true":
                        wandb.log({
                            "batch_loss": loss.item(),
                            "batch_recon_loss": r_loss.item(),
                            "batch_vq_loss": vq_loss_val.item(),
                            "batch_aux_loss": aux_loss_val.item(),
                            "batch_total_latent_loss": total_latent_loss.item(),
                            "epoch": epoch + 1,
                            "batch": batch_idx
                        })
                except Exception:
                    pass  # Continue without logging

        if rank == 0:
            # Print epoch completion and masking statistics
            print(f"\n✅ MOE Epoch {epoch + 1} completed:")
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

        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"MOE Epoch {epoch+1}/{config['num_epochs']} - Total: {epoch_loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                f"LR: {current_lr:.6f}"
            )
            unique_codes = loss_dict["q"].unique().numel() if isinstance(loss_dict, dict) and "q" in loss_dict else 0
            
            # Log epoch metrics to wandb with learning rate
            try:
                if rank == 0 and os.environ.get("WANDB_DISABLED") != "true":
                    wandb.log({
                        "epoch": epoch + 1,
                        "epoch_loss": epoch_loss.item(),
                        "epoch_recon_loss": recon_loss.item(),
                        "epoch_vq_loss": vq_loss.item(),
                        "epoch_aux_loss": aux_loss.item(),
                        "unique_codes": unique_codes,
                        "learning_rate": current_lr
                    })
            except Exception:
                pass  # Continue without logging

            # Save checkpoint every epoch
            checkpoint_path = os.path.join(config["checkpoint_dir"], f"moe_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                },
                checkpoint_path,
            )
            print(f"💾 Saved MOE checkpoint at {checkpoint_path}")
        
        # Step the scheduler at the end of each epoch
        scheduler.step()

    if rank == 0:
        try:
            if os.environ.get("WANDB_DISABLED") != "true":
                wandb.finish()
        except Exception:
            pass

    cleanup()

def ddp_eval_moe(config: dict) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = seed_everything(config.get("seed", 42))
    
    # Load training dataset for stats
    dataset = load_all_labels_dataset(config["start"], config["end"], False)
    model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])

    # Compute global stats from training data
    mean, std = compute_global_stats(dataset, config["batch_size"], False, False)
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

    # Load latest checkpoint
    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("moe_epoch_") and f.endswith(".pth")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(config["checkpoint_dir"], latest)
        print(f"📊 Evaluating checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
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
                        
                    x_particles, _, _ = [b.to(device) for b in batch]
                    x_particles = x_particles.transpose(1, 2)
                    x_norm = (x_particles - mean) / std

                    # Model forward pass
                    out, loss_dict = model(x_norm)
                    
                    # Collect token info
                    if "q" in loss_dict:
                        all_tokens.append(loss_dict["q"].detach().cpu())
                        
                    # Denormalize outputs
                    out_denorm = out * std + mean

                    # Reconstruct jet features
                    orig_jet = reconstruct_jet_features_from_particles(x_particles)
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
    
    # Plot token usage
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

def main() -> None:
    config = MOE_CONFIGS[TRAIN_TYPE].copy()
    if "type" not in config:
        config["type"] = TRAIN_TYPE
    
    print(f"🔍 Evaluating {TRAIN_TYPE} model on all labels")
    mp.spawn(ddp_train_moe, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
    ddp_eval_moe(config)

if __name__ == "__main__":
    main()