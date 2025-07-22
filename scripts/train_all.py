import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.cuda.amp import GradScaler, autocast

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
    plot_difference,
)

TRAIN_TYPE = "new"
WORLD_SIZE = 4

CONFIGS = {
    "new": {
        "batch_size": 512,
        "num_epochs": 40,
        "learning_rate": 2e-4,
        "start": 50,
        "end": 60,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_new_flash",
    },
    "MOE_med": {
        "batch_size": 512,
        "num_epochs": 20,
        "learning_rate": 2e-4,
        "start": 50,
        "end": 60,
        "vq_kwargs": {"num_codes": 4096, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_moe_med",
    },
    "MOE_large": {
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "start": 50,
        "end": 60,
        "vq_kwargs": {"num_codes": 8192, "beta": 0.8, "affine_lr": 0.0,
                      "sync_nu": 5, "replace_freq": 5, "dim": -1},
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_moe_large",
    },
    "masked": {
        "batch_size": 512,
        "num_epochs": 40,
        "learning_rate": 2e-4,
        "start": 20,
        "end": 30,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash_masked",
    },
    "particle": {
        "batch_size": 512,
        "num_epochs": 10,
        "learning_rate": 2e-4,
        "start": 10,
        "end": 20,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_new",
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
        except Exception:
            continue

    if not datasets:
        raise RuntimeError("No valid datasets loaded for any label")

    x_parts = torch.cat([d.tensors[0] for d in datasets], dim=0)
    x_jets = torch.cat([d.tensors[1] for d in datasets], dim=0)
    y = torch.cat([d.tensors[2] for d in datasets], dim=0)
    if use_mask:
        masks = torch.cat([d.tensors[3] for d in datasets], dim=0)
        return TensorDataset(x_parts, x_jets, y, masks)
    return TensorDataset(x_parts, x_jets, y)


def ddp_train(rank: int, world_size: int, config: dict) -> None:
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if config["type"] == "masked":
        dataset = load_all_labels_dataset(config["start"], config["end"], True)
        use_mask = True
        log_pt = True
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] == "new":
        dataset = load_all_labels_dataset(config["start"], config["end"], False)
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] in ["MOE_med", "MOE_large"]:
        dataset = load_all_labels_dataset(config["start"], config["end"], False)
        use_mask = False
        log_pt = False
        model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])
    else:
        dataset = load_all_labels_dataset(config["start"], config["end"], False)
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

    # Create model - same parameters for all types
    model = model_module.VQVAENormFormer(
        input_dim=3,
        latent_dim=16,
        hidden_dim=128,  # Fixed: changed from 12 to 128
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],find_unused_parameters=True)

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
    
    # Load most recent checkpoint with better error handling
    start_epoch = 0
    if rank == 0:
        ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_path = os.path.join(config["checkpoint_dir"], latest)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                # Use strict=False to handle potential model architecture changes
                missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint["model_state"], strict=False)
                
                if missing_keys:
                    print(f"⚠️  Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys in checkpoint: {unexpected_keys}")
                
                # Only load optimizer and scheduler if model loaded successfully
                if not missing_keys:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    if "scheduler_state" in checkpoint:
                        scheduler.load_state_dict(checkpoint["scheduler_state"])
                        print("✅ Loaded scheduler state")
                    print("✅ Loaded optimizer state")
                else:
                    print("⚠️  Skipping optimizer/scheduler state due to model changes")
                
                start_epoch = checkpoint["epoch"]
                print(f"🔄 Loaded checkpoint from {checkpoint_path} (epoch {start_epoch})")
                
            except Exception as e:
                print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
                print("🆕 Starting from scratch due to checkpoint error")
                start_epoch = 0
        else:
            print("🆕 No checkpoint found, starting from scratch")
    
    # Broadcast start_epoch to all processes
    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, 0)
    start_epoch = start_epoch_tensor.item()

    if rank == 0:
        print(f"🚀 Training from epoch {start_epoch + 1} to {config['num_epochs']}")
        print(f"📊 Dataset size: {len(dataset)}")
        print(f"🔢 Total batches per epoch: {len(dataloader)}")

    # Fix: Check if we need to train at all
    if start_epoch >= config["num_epochs"]:
        if rank == 0:
            print(f"⚠️  Training already completed! start_epoch ({start_epoch}) >= num_epochs ({config['num_epochs']})")
        cleanup()
        return

    for epoch in range(start_epoch, config["num_epochs"]):
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"🔄 Starting epoch {epoch + 1}/{config['num_epochs']} (LR: {current_lr:.6f})")
        
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = torch.zeros(1, device=device)
        recon_loss = torch.zeros(1, device=device)
        vq_loss = torch.zeros(1, device=device)
        aux_loss = torch.zeros(1, device=device)
        
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            
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
                out, loss_dict = model(x_norm, mask=mask) if mask is not None else model(x_norm)

                if mask is not None:
                    diff = (out - x_norm) ** 2
                    r_loss = (diff * mask.unsqueeze(-1)).sum() / mask.sum()
                else:
                    r_loss = recon_loss_fn(out, x_norm).mean()

                # Safely handle dict vs tensor loss_dict
                if isinstance(loss_dict, dict):
                    vq_loss_val = loss_dict.get("vq_loss", torch.tensor(0.0, device=device))
                    aux_loss_val = loss_dict.get("aux_loss", torch.tensor(0.0, device=device))
                    total_latent_loss = loss_dict.get("total_loss", vq_loss_val + 0.01 * aux_loss_val)
                else:
                    vq_loss_val = torch.tensor(0.0, device=device)
                    aux_loss_val = torch.tensor(0.0, device=device)
                    total_latent_loss = loss_dict  # Assume tensor loss

                loss = r_loss + total_latent_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach()
            recon_loss += r_loss.detach()
            vq_loss += vq_loss_val.detach()
            aux_loss += aux_loss_val.detach()

            # Log batch metrics
            if rank == 0 and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

        if rank == 0:
            print(f"✅ Epoch {epoch + 1} completed - Processed {batch_count} batches")

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
                f"Epoch {epoch+1}/{config['num_epochs']} - Total: {epoch_loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                f"LR: {current_lr:.6f}"
            )
            unique_codes = loss_dict["q"].unique().numel() if isinstance(loss_dict, dict) and "q" in loss_dict else 0
            print(f"🧩 Unique codes used: {unique_codes}")
            
            # Save checkpoint every 5 epochs or at the end with scheduler state
            if (epoch + 1) % 5 == 0 or epoch + 1 == config["num_epochs"]:
                checkpoint_path = os.path.join(config["checkpoint_dir"], f"vqvae_epoch_{epoch+1}.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"💾 Saved checkpoint at {checkpoint_path}")
        
        # Step the scheduler at the end of each epoch
        scheduler.step()

    cleanup()


def ddp_eval(config: dict) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config["type"] == "masked":
        use_mask = True
        log_pt = True
        dataset = load_all_labels_dataset(config["start"], config["end"], True)
        eval_dataset = load_all_labels_dataset(11, 12, True)
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] == "new":
        use_mask = False
        log_pt = False
        dataset = load_all_labels_dataset(config["start"], config["end"], False)
        eval_dataset = load_all_labels_dataset(11, 12, False)
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] in ["MOE_med", "MOE_large"]:
        use_mask = False
        log_pt = False
        dataset = load_all_labels_dataset(config["start"], config["end"], False)
        eval_dataset = load_all_labels_dataset(11, 12, False)
        model_module = __import__("models.MOE", fromlist=["VQVAENormFormer"])
    else:
        use_mask = False
        log_pt = False
        dataset = load_all_labels_dataset(config["start"], config["end"], False)
        eval_dataset = load_all_labels_dataset(11, 12, False)
        model_module = __import__("models.NormFormer", fromlist=["VQVAENormFormer"])

    mean, std = compute_global_stats(dataset, config["batch_size"], log_pt, use_mask)
    mean = mean.to(device)
    std = std.to(device)

    # Create model for evaluation - same parameters for all types
    model = model_module.VQVAENormFormer(
        input_dim=3,
        latent_dim=16,
        hidden_dim=128,  # Already correct
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(config["checkpoint_dir"], latest)
        print(f"📊 Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state"], strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys during evaluation: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys during evaluation: {unexpected_keys}")
            
            print("✅ Loaded model for evaluation")
        except Exception as e:
            print(f"❌ Failed to load checkpoint for evaluation: {e}")
            print("⚠️  Using randomly initialized model")

    model.eval()
    dataloader_eval = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)
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

    plot_tensor_jet_features(
        [all_orig_jets, all_recon_jets],
        labels=("Original", "Reconstructed"),
        filename=os.path.join(PLOT_DIR, "jet_recon_overlay_ddp_all.png"),
    )
    plot_difference(
        all_orig_jets,
        all_recon_jets,
        filename=os.path.join(PLOT_DIR, "jet_feature_difference_ddp_all.png"),
    )


def main() -> None:
    config = CONFIGS[TRAIN_TYPE].copy()
    config["type"] = TRAIN_TYPE
    mp.spawn(ddp_train, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
    ddp_eval(config)


if __name__ == "__main__":
    main()
    model.eval()
    dataloader_eval = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)
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

    plot_tensor_jet_features(
        [all_orig_jets, all_recon_jets],
        labels=("Original", "Reconstructed"),
        filename=os.path.join(PLOT_DIR, "jet_recon_overlay_ddp_all.png"),
    )
    plot_difference(
        all_orig_jets,
        all_recon_jets,
        filename=os.path.join(PLOT_DIR, "jet_feature_difference_ddp_all.png"),
    )


def main() -> None:
    config = CONFIGS[TRAIN_TYPE].copy()
    config["type"] = TRAIN_TYPE
    mp.spawn(ddp_train, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
    ddp_eval(config)


if __name__ == "__main__":
    main()
