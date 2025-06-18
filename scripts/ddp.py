import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
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
# -----------------------------------------------------------------------------
# Configuration: choose which training script to mimic
# Options: "new", "masked", "particle"
TRAIN_TYPE = "new"
WORLD_SIZE = 4

CONFIGS = {
    "new": {
        "batch_size": 512,
        "num_epochs": 1,
        "learning_rate": 2e-4,
        "start": 70,
        "end": 80,
        "vq_kwargs": {"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0,
                      "sync_nu": 2, "replace_freq": 20, "dim": -1},
        "checkpoint_dir": "checkpoints/checkpoints_vqvae_normformer_flash",
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


def ddp_train(rank: int, world_size: int, config: dict) -> None:
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if config["type"] == "masked":
        from dataloader.masked_dataloader import load_jetclass_label_as_dataset
        dataset = load_jetclass_label_as_dataset(
            label="HToBB", start=config["start"], end=config["end"])
        use_mask = True
        log_pt = True
        model_module = __import__("models.NormFormer_Flash", fromlist=["VQVAENormFormer"])
    elif config["type"] == "new":
        from dataloader.dataloader import load_jetclass_label_as_dataset
        dataset = load_jetclass_label_as_dataset(
            label="HToBB", start=config["start"], end=config["end"])
        use_mask = False
        log_pt = False
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

    for epoch in range(config["num_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = torch.zeros(1, device=device)
        recon_loss = torch.zeros(1, device=device)
        vq_loss = torch.zeros(1, device=device)

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

        epoch_loss /= len(dataloader)
        recon_loss /= len(dataloader)
        vq_loss /= len(dataloader)
        for t in (epoch_loss, recon_loss, vq_loss):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size

        if rank == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']} - Total: {epoch_loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f}")
            if epoch + 1 == config["num_epochs"]:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    os.path.join(config["checkpoint_dir"], f"vqvae_epoch_{epoch+1}.pth"),
                )

    cleanup()


def ddp_eval(config: dict) -> None:
    """Run evaluation on a single device after training."""
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
            orig_jet = reconstruct_jet_features_from_particles(x_particles)
            recon_jet = reconstruct_jet_features_from_particles(out_denorm)
            all_orig_jets.append(orig_jet)
            all_recon_jets.append(recon_jet)

    all_orig_jets = torch.cat(all_orig_jets, dim=0)
    all_recon_jets = torch.cat(all_recon_jets, dim=0)

    plot_tensor_jet_features(
        [all_orig_jets, all_recon_jets],
        labels=("Original", "Reconstructed"),
        filename=os.path.join(PLOT_DIR, "jet_recon_overlay_ddp.png"),
    )
    plot_difference(
        all_orig_jets,
        all_recon_jets,
        filename=os.path.join(PLOT_DIR, "jet_feature_difference_ddp.png"),
    )


def main() -> None:
    config = CONFIGS[TRAIN_TYPE].copy()
    config["type"] = TRAIN_TYPE
    mp.spawn(ddp_train, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
    ddp_eval(config)


if __name__ == "__main__":
    main()
