import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from plot.plot import plot_tensor_jet_features, plot_difference

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = [
    "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L",
    "ZToQQ", "WToQQ", "TTBar", "TTBarLep", "ZJetsToNuNu",
]

WORLD_SIZE = 4

CONFIG = {
    "batch_size": 512,
    "num_epochs": 20,
    "learning_rate": 2e-4,
    "start": 10,
    "end": 40,
    "vq_kwargs": {
        "num_codes": 1024,
        "beta": 0.25,
        "affine_lr": 0.0,
        "sync_nu": 1,
        "replace_freq": 20,
        "dim": -1,
    },
    "checkpoint_dir": "checkpoints/vqvae_mlp_jet",
}

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "plot", "jet_training_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12358")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup() -> None:
    dist.destroy_process_group()


def eval_jet(config: dict) -> None:
    """Evaluate trained model on all jet labels and create plots."""
    # Import inside the function so spawned workers also resolve the module.
    import models.vqvaeMLP_jet as vqvae

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_all_labels_jet_dataset(config["start"], config["end"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    mean, std = compute_global_stats(dataset, config["batch_size"])
    mean = mean.to(device)
    std = std.to(device)

    model = vqvae.VQVAEJet(
        input_dim=3,
        hidden_dim=256,
        z_dim=128,
        num_embeddings=config["vq_kwargs"]["num_codes"],
        commitment_cost=config["vq_kwargs"]["beta"],
        mean=mean,
        std=std,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)

    ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("jet_epoch_") and f.endswith(".pth")]
    if ckpts:
        latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        ckpt_path = os.path.join(config["checkpoint_dir"], latest)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    model.eval()
    orig_jets = []
    recon_jets = []
    with torch.no_grad():
        for x_j, _ in loader:
            x_j = x_j.to(device)
            out, _ = model(x_j)
            orig_jets.append(x_j)
            recon_jets.append(out)

    if not orig_jets:
        print("No evaluation data available")
        return

    orig_jets = torch.cat(orig_jets, dim=0)
    recon_jets = torch.cat(recon_jets, dim=0)

    plot_tensor_jet_features(
        [orig_jets, recon_jets],
        labels=("Original", "Reconstructed"),
        filename=os.path.join(PLOT_DIR, "jet_recon_overlay.png"),
    )

    plot_difference(
        orig_jets,
        recon_jets,
        filename=os.path.join(PLOT_DIR, "jet_feature_difference.png"),
    )


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


def compute_global_stats(dataset: TensorDataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    jets = []
    for batch in loader:
        x_j, _ = batch
        jets.append(x_j)
    jets_all = torch.cat(jets, dim=0)
    mean = jets_all.mean(dim=0)
    std = jets_all.std(dim=0) + 1e-6
    return mean, std


def ddp_train(rank: int, world_size: int, config: dict) -> None:
    # Local import required for torch.multiprocessing with the ``spawn`` method
    # to avoid ``NameError: name 'vqvae' is not defined`` on some systems.
    import models.vqvaeMLP_jet as vqvae

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    dataset = load_all_labels_jet_dataset(config["start"], config["end"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)

    if rank == 0:
        mean, std = compute_global_stats(dataset, config["batch_size"])
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean = torch.zeros(3, device=device)
        std = torch.ones(3, device=device)
    dist.broadcast(mean, 0)
    dist.broadcast(std, 0)

    model = vqvae.VQVAEJet(
        input_dim=3,
        hidden_dim=256,
        z_dim=128,
        num_embeddings=config["vq_kwargs"]["num_codes"],
        commitment_cost=config["vq_kwargs"]["beta"],
        mean=mean,
        std=std,
        vq_kwargs=config["vq_kwargs"],
    ).to(device)
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
        ckpts = [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("jet_epoch_") and f.endswith(".pth")]
        if ckpts:
            latest = max(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            path = os.path.join(config["checkpoint_dir"], latest)
            checkpoint = torch.load(path, map_location="cpu")
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from {path} (epoch {start_epoch})")
    obj = [checkpoint]
    dist.broadcast_object_list(obj, src=0)
    checkpoint = obj[0]
    if checkpoint:
        model.module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]

    for epoch in range(start_epoch, config["num_epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        recon_total = torch.tensor(0.0, device=device)
        vq_total = torch.tensor(0.0, device=device)
        code_hist = torch.zeros(config["vq_kwargs"]["num_codes"], device=device, dtype=torch.long)

        for x_jets, _ in dataloader:
            x_jets = x_jets.to(device)
            optimizer.zero_grad()
            with autocast():
                out, vq_loss = model(x_jets)
                r_loss = recon_loss_fn(out, x_jets)
                if isinstance(vq_loss, dict):
                    v_loss = vq_loss.get("loss", torch.tensor(0.0, device=device))
                    codes = vq_loss.get("q")
                    if codes is not None:
                        hist = torch.bincount(codes.view(-1), minlength=config["vq_kwargs"]["num_codes"])
                        code_hist += hist.to(device)
                else:
                    v_loss = vq_loss
                loss = r_loss + v_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            recon_total += r_loss.detach()
            vq_total += v_loss.detach()

        scheduler.step()

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
            print(
                f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {total_loss.item():.4f} | "
                f"Recon: {recon_total.item():.4f} | VQ: {vq_total.item():.4f} | "
                f"Codes: {unique_codes}/{config['vq_kwargs']['num_codes']}"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                os.path.join(config["checkpoint_dir"], f"jet_epoch_{epoch+1}.pth"),
            )

    cleanup()


def main() -> None:
    config = CONFIG.copy()
    mp.spawn(ddp_train, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
    eval_jet(config)


if __name__ == "__main__":
    main()
