import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot.plot import (
    plot_tensor_jet_features,
    reconstruct_jet_features_from_particles,
    plot_difference,
)

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "checkpoint_comparison")
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_TYPE = "new"  # Change this as needed

CONFIGS = {
    "new": {
        "batch_size": 512,
        "checkpoint_dir": "checkpoints/all_checkpoints_vqvae_normformer_flash",
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
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

def load_model_and_checkpoint(config, checkpoint_path, device):
    if config["type"] == "masked":
        use_mask = True
        log_pt = True
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] == "new":
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    else:
        use_mask = False
        log_pt = False
        model_module = __import__("models.NormFormer", fromlist=["VQVAENormFormer"])

    model = model_module.VQVAENormFormer(
        input_dim=3,
        latent_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_blocks=3,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, use_mask, log_pt

def evaluate_model(model, dataloader, mean, std, use_mask, log_pt, device):
    all_orig_jets, all_recon_jets = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 200:  # Limit for faster evaluation
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

    return torch.cat(all_orig_jets, dim=0), torch.cat(all_recon_jets, dim=0)

def evaluate_model_all_labels(model, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=512):
    """Evaluate model on all labels and return original and reconstructed jet features"""
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
                        x_particles, _, _, mask = [b.to(device) for b in batch]
                    else:
                        x_particles, _, _ = [b.to(device) for b in batch]
                        mask = None
                    
                    # x_particles is [B, 3, N] - transpose to [B, N, 3]
                    x_particles = x_particles.transpose(1, 2)
                    
                    if log_pt:
                        x_particles[:, :, 0] = torch.log(x_particles[:, :, 0] + 1e-6)
                    x_norm = (x_particles - mean) / std

                    # Get model reconstruction
                    if mask is not None:
                        out, _ = model(x_norm, mask=mask)
                    else:
                        out, _ = model(x_norm)

                    # Denormalize output
                    out_denorm = out * std + mean
                    if log_pt:
                        out_denorm[:, :, 0] = torch.exp(out_denorm[:, :, 0]) - 1e-6
                        x_particles[:, :, 0] = torch.exp(x_particles[:, :, 0]) - 1e-6

                    # Reconstruct jet features from particles
                    if mask is not None:
                        # Apply mask to particles
                        orig_particles_masked = x_particles * mask.unsqueeze(-1)
                        recon_particles_masked = out_denorm * mask.unsqueeze(-1)
                        
                        orig_jet = reconstruct_jet_features_from_particles(orig_particles_masked)
                        recon_jet = reconstruct_jet_features_from_particles(recon_particles_masked)
                    else:
                        orig_jet = reconstruct_jet_features_from_particles(x_particles)
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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = CONFIGS[TRAIN_TYPE].copy()
    config["type"] = TRAIN_TYPE
    
    print(f"🔍 Comparing checkpoints for {TRAIN_TYPE} training")
    
    # Find checkpoints
    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) 
             if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
    
    if not ckpts:
        print("❌ No checkpoints found!")
        return
    
    # Sort checkpoints by epoch number
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    if len(ckpts) < 2:
        print(f"⚠️  Only {len(ckpts)} checkpoint(s) found, need at least 2 for comparison")
        return
    
    # Use the most recent 2 checkpoints
    latest_ckpt = ckpts[-1]
    earlier_ckpt = ckpts[-2]  # Second most recent
    
    latest_epoch = int(latest_ckpt.split("_")[-1].split(".")[0])
    earlier_epoch = int(earlier_ckpt.split("_")[-1].split(".")[0])
    
    print(f"📊 Comparing epoch {earlier_epoch} vs epoch {latest_epoch}")
    print(f"📁 Available checkpoints: {len(ckpts)} total")
    
    # Load evaluation dataset (all labels, parts 10-11)
    use_mask = config["type"] == "masked"
    eval_dataset = load_all_labels_dataset(10, 11, use_mask)
    dataloader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Compute normalization stats (using training data range - single file)
    if config["type"] == "masked":
        train_dataset = load_all_labels_dataset(20, 21, True)  # Load only file 20
        log_pt = True
    else:
        train_dataset = load_all_labels_dataset(10, 11, False)  # Load only file 10
        log_pt = False
    
    mean, std = compute_global_stats(train_dataset, config["batch_size"], log_pt, use_mask)
    mean, std = mean.to(device), std.to(device)
    
    # Load and evaluate earlier checkpoint on ALL LABELS
    print(f"🔄 Loading earlier checkpoint: {earlier_ckpt}")
    earlier_path = os.path.join(config["checkpoint_dir"], earlier_ckpt)
    model_earlier, _, _ = load_model_and_checkpoint(config, earlier_path, device)
    orig_jets, recon_jets_earlier = evaluate_model_all_labels(
        model_earlier, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=config["batch_size"]
    )
    
    # Load and evaluate latest checkpoint on ALL LABELS
    print(f"🔄 Loading latest checkpoint: {latest_ckpt}")
    latest_path = os.path.join(config["checkpoint_dir"], latest_ckpt)
    model_latest, _, _ = load_model_and_checkpoint(config, latest_path, device)
    _, recon_jets_latest = evaluate_model_all_labels(
        model_latest, mean, std, use_mask, log_pt, device, start=10, end=11, batch_size=config["batch_size"]
    )
    
    # Create comparison plots
    print("📈 Creating comparison plots...")
    
    # Plot original vs both reconstructions
    plot_tensor_jet_features(
        [orig_jets, recon_jets_earlier, recon_jets_latest],
        labels=("Original", f"Epoch {earlier_epoch}", f"Epoch {latest_epoch}"),
        filename=os.path.join(PLOT_DIR, f"checkpoint_comparison_{TRAIN_TYPE}_all_labels.png"),
    )
    
    # Plot differences
    plot_difference(
        orig_jets,
        recon_jets_earlier,
        filename=os.path.join(PLOT_DIR, f"difference_epoch_{earlier_epoch}_{TRAIN_TYPE}_all_labels.png"),
    )
    
    plot_difference(
        orig_jets,
        recon_jets_latest,
        filename=os.path.join(PLOT_DIR, f"difference_epoch_{latest_epoch}_{TRAIN_TYPE}_all_labels.png"),
    )
    
    # Compare the two reconstructions directly
    plot_difference(
        recon_jets_earlier,
        recon_jets_latest,
        filename=os.path.join(PLOT_DIR, f"reconstruction_evolution_{earlier_epoch}_to_{latest_epoch}_{TRAIN_TYPE}.png"),
    )
    
    print(f"✅ Plots saved to {PLOT_DIR}")
    print(f"📊 Evaluated on {len(orig_jets)} samples from ALL LABELS (parts 10-11)")

if __name__ == "__main__":
    main()
