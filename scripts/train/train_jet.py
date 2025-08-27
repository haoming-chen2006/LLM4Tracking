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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
    plot_difference,
)

LABELS = [
    "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L",
    "ZToQQ", "WToQQ", "TTBar", "TTBarLep", "ZJetsToNuNu",
]

# --- Unified model selection and configs ---
TRAIN_TYPE = "VQVAE"  # Options: "MLP", "VQVAE" (for jet training only)
WORLD_SIZE = 4

MODEL_CONFIGS = {
    "MLP": {
        "batch_size": 512,
        "num_epochs": 50,
        "learning_rate": 1e-4,  # Reduced from 2e-4 to match MOE
        "start": 40,
        "end": 45,  # Reduced range for faster testing
        "vq_kwargs": {
            "num_codes": 1024,
            "beta": 0.25,
            "affine_lr": 0.0,
            "sync_nu": 1,
            "replace_freq": 20,
            "dim": -1,
        },
        "checkpoint_dir": "checkpoints/jet_mlp_2",
        "input_dim": 3,  # Jet features: [pt, eta, phi] (no mass in dataloader)
        "hidden_dim": 256,
        "z_dim": 128,
        "model_type": "MLP",
    },
    "VQVAE": {
        "batch_size": 512,
        "num_epochs": 30,  # Reduced from 100 for faster testing
        "learning_rate": 1e-4,
        "start": 10,
        "end": 39,  # Reduced range for faster testing  
        "vq_kwargs": {
            "num_codes": 2048,
            "beta": 0.25,
            "affine_lr": 0.0,
            "sync_nu": 2,
            "replace_freq": 20,
            "dim": -1,
        },
        "checkpoint_dir": "checkpoints/jet_vqvae_2",
        "input_dim": 3,  # Jet features: [pt, eta, phi] (no mass in dataloader)
        "hidden_dim": 128,
        "z_dim": 32,
        "num_embeddings": 2048,
        "commitment_cost": 0.25,
        "model_type": "VQVAE",
    },
}

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "jet_training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)



def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12358")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)



def cleanup() -> None:
    dist.destroy_process_group()



def eval_jet(config: dict) -> None:
    """Evaluate trained jet model and create plots."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_all_labels_jet_dataset(config["start"], config["end"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    mean, std = compute_global_stats(dataset, config["batch_size"], log_pt=True)
    mean = mean.to(device)
    std = std.to(device)

    orig_jets = []
    recon_jets = []
    model_type = config.get("model_type", "MLP")

    print(f"🔍 Evaluating {model_type} jet model...")

    if model_type in ["MLP", "VQVAE"]:
        import models.vqvaeMLP_jet as vqvae
        
        # Create model based on type
        if model_type == "MLP":
            model = vqvae.VQVAEJet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                z_dim=config["z_dim"],
                num_embeddings=config["vq_kwargs"]["num_codes"],
                commitment_cost=config["vq_kwargs"]["beta"],
                mean=mean,
                std=std,
                vq_kwargs=config["vq_kwargs"],
            ).to(device)
        else:  # VQVAE
            model = vqvae.VQVAEJet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                z_dim=config["z_dim"],
                num_embeddings=config["num_embeddings"],
                commitment_cost=config["commitment_cost"],
                mean=mean,
                std=std,
                vq_kwargs=config["vq_kwargs"],
            ).to(device)
        
        # Load checkpoint
        ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("jet_epoch_") and f.endswith(".pth")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            ckpt_path = os.path.join(config["checkpoint_dir"], latest)
            print(f"📂 Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"], strict=False)
            print(f"✅ Loaded {model_type} model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print(f"⚠️ No checkpoint found in {config['checkpoint_dir']}, using random weights")
        
        model.eval()
        
        # Evaluate model with consistent preprocessing
        with torch.no_grad():
            for x_j, _ in loader:
                x_j = x_j.to(device)  # [B, 3] jet features
                
                # Apply SAME preprocessing as training using robust functions
                try:
                    x_j_norm, x_j_processed = preprocess_jet_batch(x_j, mean, std, log_pt=True, validate=True)
                    
                    out, _ = model(x_j_norm)   # Reconstruct in normalized space
                    
                    # Denormalize output using robust function
                    out_denorm = denormalize_jet_batch(out, mean, std, log_pt=True, validate=True)
                    
                    recon_jets.append(out_denorm)
                    orig_jets.append(x_j)
                    
                except Exception as e:
                    print(f"⚠️ Warning: Error processing batch during evaluation: {e}")
                    continue
    
    else:
        raise ValueError(f"Unknown model_type for jet training: {model_type}. Use 'MLP' or 'VQVAE'")

    if not orig_jets:
        print("❌ No evaluation data available")
        return

    orig_jets = torch.cat(orig_jets, dim=0)
    recon_jets = torch.cat(recon_jets, dim=0)

    print(f"📊 Evaluated {len(orig_jets)} jets")

    # Plot overlay: original vs reconstructed
    plot_tensor_jet_features(
        [orig_jets, recon_jets],
        labels=("Original Jets", f"{model_type} Reconstructed Jets"),
        filename=os.path.join(PLOT_DIR, f"jet_recon_overlay_{model_type}.png"),
    )

    # Plot difference
    plot_difference(
        orig_jets,
        recon_jets,
        filename=os.path.join(PLOT_DIR, f"jet_feature_difference_{model_type}.png"),
    )
    
    print(f"✅ Saved evaluation plots for {model_type} jet model")



def load_all_labels_jet_dataset(start: int, end: int) -> TensorDataset:
    from dataloader.dataloader import load_jetclass_label_as_dataset
    datasets = []
    for lbl in LABELS:
        try:
            ds = load_jetclass_label_as_dataset(label=lbl, start=start, end=end)
            datasets.append(ds)
        except Exception as e:
            print(f"Failed to load dataset for {lbl}: {e}")
            continue
    if not datasets:
        raise RuntimeError("No valid datasets loaded for any label")
    x_jets = torch.cat([d.tensors[1] for d in datasets], dim=0)
    y = torch.cat([d.tensors[2] for d in datasets], dim=0)
    return TensorDataset(x_jets, y)



def compute_global_stats(dataset: TensorDataset, batch_size: int, log_pt: bool = True):
    """Compute mean and std for jet features [pt, eta, phi] with robust handling."""
    print(f"🔢 Computing jet statistics with log_pt={log_pt}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    jets = []
    
    for batch_idx, batch in enumerate(loader):
        x_j, _ = batch  # x_j should be [B, 3] for jet features
        
        # Validate input data
        if torch.isnan(x_j).any() or torch.isinf(x_j).any():
            print(f"⚠️ Warning: NaN/Inf detected in jet data batch {batch_idx}")
            continue
            
        # Check for reasonable pt values
        pt_values = x_j[:, 0]
        if (pt_values <= 0).any():
            print(f"⚠️ Warning: Non-positive pt values found in batch {batch_idx}")
            # Clamp to small positive value
            x_j = x_j.clone()
            x_j[:, 0] = torch.clamp(x_j[:, 0], min=1e-6)
        
        # Apply log transform consistently
        if log_pt:
            x_j = x_j.clone()
            x_j[:, 0] = torch.log(x_j[:, 0] + 1e-6)  # Add epsilon for numerical stability
            
        jets.append(x_j)
        
        # Limit computation for very large datasets
        if batch_idx >= 100:
            print(f"⚠️ Limited stats computation to first {batch_idx + 1} batches")
            break
    
    if not jets:
        raise RuntimeError("No valid batches found for statistics computation")
    
    jets_all = torch.cat(jets, dim=0)
    mean = jets_all.mean(dim=0)
    std = jets_all.std(dim=0) + 1e-6  # Add epsilon for numerical stability
    
    print(f"📊 Jet statistics computed:")
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std: {std.tolist()}")
    print(f"  Valid samples used: {jets_all.shape[0]:,}")
    
    # Validate computed statistics
    if torch.isnan(mean).any() or torch.isnan(std).any():
        raise RuntimeError("NaN values in computed statistics")
    if (std < 1e-8).any():
        print("⚠️ Warning: Very small std values detected, adjusting...")
        std = torch.clamp(std, min=1e-6)
    
    return mean, std



def preprocess_jet_batch(x_jets, mean, std, log_pt=True, validate=True):
    """Preprocess jet batch with consistent normalization and validation."""
    if validate:
        # Validate input data
        if torch.isnan(x_jets).any() or torch.isinf(x_jets).any():
            raise ValueError("NaN/Inf detected in input jet data")
        
        # Check pt values
        pt_values = x_jets[:, 0]
        if (pt_values <= 0).any():
            print("⚠️ Warning: Non-positive pt values found, clamping to positive")
            x_jets = x_jets.clone()
            x_jets[:, 0] = torch.clamp(x_jets[:, 0], min=1e-6)
    
    # Apply preprocessing consistently
    x_processed = x_jets.clone()
    if log_pt:
        x_processed[:, 0] = torch.log(x_processed[:, 0] + 1e-6)
    
    # Apply normalization
    x_norm = (x_processed - mean) / std
    
    # Validate output
    if validate:
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            raise ValueError("NaN/Inf detected after normalization")
    
    return x_norm, x_processed

def denormalize_jet_batch(x_norm, mean, std, log_pt=True, validate=True):
    """Denormalize jet batch consistently."""
    # Denormalize
    x_processed = x_norm * std + mean
    
    # Inverse log transform
    x_output = x_processed.clone()
    if log_pt:
        x_output[:, 0] = torch.exp(x_processed[:, 0]) - 1e-6
        
        # Clamp to ensure positive pt values
        x_output[:, 0] = torch.clamp(x_output[:, 0], min=1e-6)
    
    # Validate output
    if validate:
        if torch.isnan(x_output).any() or torch.isinf(x_output).any():
            print("⚠️ Warning: NaN/Inf detected after denormalization")
            # Replace NaN/Inf with mean values
            x_output = torch.where(torch.isnan(x_output) | torch.isinf(x_output), 
                                 torch.tensor([200.0, 0.0, 0.0], device=x_output.device), x_output)
    
    return x_output


def ddp_train(rank: int, world_size: int, config: dict) -> None:
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model_type = config.get("model_type", "MLP")
    generator = seed_everything(42 + rank)

    print(f"🚀 Training {model_type} jet model on rank {rank}")

    dataset = load_all_labels_jet_dataset(config["start"], config["end"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, generator=generator, worker_init_fn=seed_worker)

    # Compute global statistics for jet features
    if rank == 0:
        mean, std = compute_global_stats(dataset, config["batch_size"], log_pt=True)
        mean = mean.to(device)
        std = std.to(device)
        print(f"📊 Jet feature stats - Mean: {mean}, Std: {std}")
    else:
        mean = torch.zeros(3, device=device)  # 3D jet features
        std = torch.ones(3, device=device)
    dist.broadcast(mean, 0)
    dist.broadcast(std, 0)

    # Create model based on type (only MLP and VQVAE for jet training)
    if model_type in ["MLP", "VQVAE"]:
        import models.vqvaeMLP_jet as vqvae
        
        if model_type == "MLP":
            model = vqvae.VQVAEJet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                z_dim=config["z_dim"],
                num_embeddings=config["vq_kwargs"]["num_codes"],
                commitment_cost=config["vq_kwargs"]["beta"],
                mean=mean,
                std=std,
                vq_kwargs=config["vq_kwargs"],
            ).to(device)
        else:  # VQVAE
            model = vqvae.VQVAEJet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                z_dim=config["z_dim"],
                num_embeddings=config["num_embeddings"],
                commitment_cost=config["commitment_cost"],
                mean=mean,
                std=std,
                vq_kwargs=config["vq_kwargs"],
            ).to(device)
        ckpt_prefix = "jet_epoch_"
    else:
        raise ValueError(f"Unknown model_type for jet training: {model_type}. Use 'MLP' or 'VQVAE'")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=config["learning_rate"] * 0.01
    )
    recon_loss_fn = nn.MSELoss()
    scaler = GradScaler()

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    start_epoch = 0
    checkpoint = None
    if rank == 0:
        # Ensure checkpoint directory exists
        if not os.path.exists(config["checkpoint_dir"]):
            print(f"⚠️ Checkpoint directory {config['checkpoint_dir']} does not exist, creating it...")
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
        
        ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith(ckpt_prefix) and f.endswith(".pth")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            path = os.path.join(config["checkpoint_dir"], latest)
            checkpoint = torch.load(path, map_location="cpu")
            start_epoch = checkpoint["epoch"]
            print(f"🔄 Resuming {model_type} from {path} (epoch {start_epoch})")
        else:
            print(f"🆕 Starting {model_type} training from scratch")
    
    obj = [checkpoint]
    dist.broadcast_object_list(obj, src=0)
    checkpoint = obj[0]
    if checkpoint:
        model.module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]

    if rank == 0:
        print(f"📋 {model_type} Training: {start_epoch + 1} -> {config['num_epochs']} epochs")
        print(f"📊 Dataset: {len(dataset)} jets, Batch size: {config['batch_size']}")

    for epoch in range(start_epoch, config["num_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        recon_total = torch.tensor(0.0, device=device)
        vq_total = torch.tensor(0.0, device=device)
        code_hist = torch.zeros(config["vq_kwargs"]["num_codes"], device=device, dtype=torch.long)

        for batch_idx, (x_jets, _) in enumerate(dataloader):
            x_jets = x_jets.to(device)  # [B, 3] jet features
            
            # Validate batch data before processing
            if torch.isnan(x_jets).any() or torch.isinf(x_jets).any():
                print(f"⚠️ Warning: NaN/Inf in batch {batch_idx}, skipping")
                continue
            
            # Check for valid pt values
            if (x_jets[:, 0] <= 0).any():
                print(f"⚠️ Warning: Non-positive pt values in batch {batch_idx}, clamping")
                x_jets = x_jets.clone()
                x_jets[:, 0] = torch.clamp(x_jets[:, 0], min=1e-6)
            
            # Apply preprocessing consistently (like MOE data module)
            x_jets_norm, x_jets_processed = preprocess_jet_batch(x_jets, mean, std, log_pt=True, validate=True)
            
            # Debug logging for first batch of first epoch
            if epoch == start_epoch and batch_idx == 0 and rank == 0:
                print(f"\n🔍 Debug info for first batch:")
                print(f"  Original jet range: [{x_jets.min():.3f}, {x_jets.max():.3f}]")
                print(f"  Processed jet range: [{x_jets_processed.min():.3f}, {x_jets_processed.max():.3f}]")
                print(f"  Normalized jet range: [{x_jets_norm.min():.3f}, {x_jets_norm.max():.3f}]")
                print(f"  Mean: {mean}")
                print(f"  Std: {std}\n")
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Direct jet-to-jet reconstruction
                out, vq_loss = model(x_jets_norm)
                
                # Validate model output
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"❌ Invalid model output in batch {batch_idx}")
                    continue
                
                # Compute loss in NORMALIZED space for stability (like MOE)
                r_loss = recon_loss_fn(out, x_jets_norm)
                
                # Handle VQ loss dictionary robustly
                if isinstance(vq_loss, dict):
                    v_loss = vq_loss.get("loss", vq_loss.get("vq_loss", torch.tensor(0.0, device=device)))
                    codes = vq_loss.get("q")
                    if codes is not None:
                        try:
                            hist = torch.bincount(codes.view(-1), minlength=config["vq_kwargs"]["num_codes"])
                            code_hist += hist.to(device)
                        except Exception as e:
                            print(f"⚠️ Warning: Error computing code histogram: {e}")
                else:
                    v_loss = vq_loss
                
                # Validate loss values
                if torch.isnan(r_loss) or torch.isinf(r_loss):
                    print(f"❌ Invalid reconstruction loss in batch {batch_idx}: {r_loss}")
                    continue
                    
                if torch.isnan(v_loss) or torch.isinf(v_loss):
                    print(f"❌ Invalid VQ loss in batch {batch_idx}: {v_loss}")
                    continue
                    
                loss = r_loss + v_loss
            
            scaler.scale(loss).backward()
            
            # Add gradient clipping for stability (like MOE)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            recon_total += r_loss.detach()
            vq_total += v_loss.detach()

            # Log batch progress with more detail
            if rank == 0 and batch_idx % 50 == 0:
                print(f"  {model_type} Epoch {epoch+1} Batch {batch_idx}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f} (Recon: {r_loss.item():.4f}, VQ: {v_loss.item():.4f})")

        scheduler.step()

        # Average losses across batches and processes
        total_loss /= len(dataloader)
        recon_total /= len(dataloader)
        vq_total /= len(dataloader)
        dist.all_reduce(total_loss)
        dist.all_reduce(recon_total)
        dist.all_reduce(vq_total)
        dist.all_reduce(code_hist)
        total_loss /= world_size
        recon_total /= world_size
        vq_total /= world_size
        unique_codes = torch.count_nonzero(code_hist).item()

        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"{model_type} Epoch {epoch+1}/{config['num_epochs']} - "
                f"Loss: {total_loss.item():.4f} | Recon: {recon_total.item():.4f} | "
                f"VQ: {vq_total.item():.4f} | Codes: {unique_codes}/{config['vq_kwargs']['num_codes']} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save checkpoint every epoch
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                os.path.join(config["checkpoint_dir"], f"{ckpt_prefix}{epoch+1}.pth"),
            )
            print(f"💾 Saved {model_type} checkpoint at epoch {epoch+1}")

    if rank == 0:
        print(f"✅ {model_type} training completed successfully!")
    cleanup()



def main() -> None:
    config = MODEL_CONFIGS[TRAIN_TYPE].copy()
    mp.spawn(ddp_train, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
    eval_jet(config)

if __name__ == "__main__":
    main()
