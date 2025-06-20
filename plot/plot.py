import awkward as ak
import numpy as np
import torch
import vector
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend suitable for HPC
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import pandas as pd
sys.path.append("/pscratch/sd/h/haoming/Projects/hep_models")  # parent of `dataloader`
from matplotlib.lines import Line2D
from dataloader.dataloader import load_jetclass_label_as_tensor

vector.register_awkward()
plt.switch_backend("Agg")

def plot_jet_and_particle_features(label="HToBB", start=10, end=12, batch_size=512, output_prefix="HToBB"):
    dataloader = load_jetclass_label_as_tensor(label=label, start=start, end=end, batch_size=batch_size)

    all_x_particles = []
    all_x_jets = []

    for x_particles, x_jets, _ in dataloader:
        all_x_particles.append(x_particles)
        all_x_jets.append(x_jets)

    x_particles_all = torch.cat(all_x_particles, dim=0)  # (B, 4, N)
    x_jets_all = torch.cat(all_x_jets, dim=0)            # (B, 4)

    print(f"✅ Total jets: {x_jets_all.shape[0]} for {label}")

    # === Convert to numpy ===
    jet_pt = x_jets_all[:, 0].cpu().numpy()
    jet_eta = x_jets_all[:, 1].cpu().numpy()
    jet_phi = x_jets_all[:, 2].cpu().numpy()

    # Compute jet mass if not provided
    if x_jets_all.shape[1] > 3:
        jet_mass = x_jets_all[:, 3].cpu().numpy()
    else:
        p4 = vector.array({
            "pt": jet_pt,
            "eta": jet_eta,
            "phi": jet_phi,
        })
        jet_mass = p4.mass

    # Particle-level features
    part_pt = x_particles_all[:, 0, :].cpu().numpy()
    part_eta = x_particles_all[:, 1, :].cpu().numpy()
    part_phi = x_particles_all[:, 2, :].cpu().numpy()

    jets = ak.zip({"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "mass": jet_mass})
    parts = ak.zip({"pt": part_pt, "eta": part_eta, "phi": part_phi})

    # === Plot setup ===
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    def hist_plot(ax, data, bins, label):
        ax.hist(data, bins=bins, histtype="step", density=True)
        ax.set_xlabel(label)
        ax.grid(True)

    # Jet-level
    hist_plot(axs[0], ak.to_numpy(jets.pt), np.linspace(400, 1200, 100), "Jet $p_T$ [GeV]")
    hist_plot(axs[1], ak.to_numpy(jets.eta), np.linspace(-2.5, 2.5, 100), "Jet $\\eta$")
    hist_plot(axs[2], ak.to_numpy(jets.phi), np.linspace(-np.pi, np.pi, 100), "Jet $\\phi$")
    hist_plot(axs[3], ak.to_numpy(jets.mass), np.linspace(0, 300, 100), "Jet mass [GeV]")

    # Particle-level with tighter bounds
    hist_plot(axs[4], ak.to_numpy(ak.flatten(parts.pt)), np.linspace(0, 50, 100), "Part. $p_T$ [GeV]")
    hist_plot(axs[5], ak.to_numpy(ak.flatten(parts.eta)), np.linspace(-0.5, 0.5, 100), "Part. $\\eta$")
    hist_plot(axs[6], ak.to_numpy(ak.flatten(parts.phi)), np.linspace(-0.5, 0.5, 100), "Part. $\\phi$")

    axs[7].axis("off")

    fig.suptitle(f"Jet and Particle Feature Distributions ({label})", fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    save_path = f"{output_prefix}_7feature_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"📊 Saved: {save_path}")

# === Label mapping (name → JetClass folder) ===
jetclass_labels = {
    "QCD": "ZJetsToNuNu",
    "HToBB": "HToBB",
    "HToCC": "HToCC",
    "HToGG": "HToGG",
    "HToWW4Q": "HToWW4Q",
    "HToWW2Q1L": "HToWW2Q1L",
    "ZToQQ": "ZToQQ",
    "WToQQ": "WToQQ",
    "TopToBQQ": "TTBar",
    "TopToBLNu": "TTBarLep",
}

# === Run all plots ===
DEFAULT_LABELS = {
    "jet_pt": "Jet $p_T$ [GeV]",
    "jet_eta": "Jet $\\eta$",
    "jet_phi": "Jet $\\phi$",
}


def plot_tensor_jet_features(jet_tensor: torch.Tensor | list[torch.Tensor], labels=None, filename="tensor_jet_features.png"):
    """Plot 4 jet features (pt, eta, phi, mass) from one or more PyTorch tensors.

    Parameters
    ----------
    jet_tensor: Tensor or list[Tensor]
        Single tensor of shape ``[N, 4]`` or a list of such tensors.
    labels: list[str] | None
        Labels corresponding to each tensor. If ``None`` labels will be
        generated automatically.
    filename: str
        Output file name.
    """

    if isinstance(jet_tensor, torch.Tensor):
        jet_tensor = [jet_tensor]

    if labels is None:
        labels = [f"Jets {i+1}" for i in range(len(jet_tensor))]

    fig, axarr = plt.subplots(1, 4, figsize=(18, 4))
    bins_dict = {
        "pt": np.linspace(400, 1200, 100),
        "eta": np.linspace(-2.5, 2.5, 100),
        "phi": np.linspace(-np.pi, np.pi, 100),
        "mass": np.linspace(0, 300, 100),
    }

    features = ["pt", "eta", "phi", "mass"]

    for i, jets in enumerate(jet_tensor):
        values = {
            "pt": jets[:, 0].cpu().numpy(),
            "eta": jets[:, 1].cpu().numpy(),
            "phi": jets[:, 2].cpu().numpy(),
            "mass": jets[:, 3].cpu().numpy(),
        }

        for j, feat in enumerate(features):
            axarr[j].hist(values[feat], bins=bins_dict[feat], histtype="step", density=True, label=labels[i])

    for j, feat in enumerate(features):
        axarr[j].set_xlabel(DEFAULT_LABELS.get("jet_" + feat, feat))
        axarr[j].legend(frameon=False)
        axarr[j].grid(True)

    fig.suptitle("Overlay of Jet Features (Tensor)", fontsize=16)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved plot to {filename}")


def plot_all(start: int = 10, end: int = 12, batch_size: int = 512,
             filename: str = "all_jet_features.png", overlay: bool = False) -> None:
    """Plot jet feature distributions for all jet classes.

    Parameters
    ----------
    start, end, batch_size : int
        Range of files and batch size passed to the dataloader.
    filename : str
        Output plot file name.
    overlay : bool, optional
        If ``True`` (default) the distributions from different jet classes are
        overlaid.  If ``False`` all jets are concatenated and a single
        distribution is produced.
    """
    jet_tensors = []
    labels = []

    for display_name, label in jetclass_labels.items():
        dataloader = load_jetclass_label_as_tensor(
            label=label, start=start, end=end, batch_size=batch_size
        )

        jets = []
        for x_particles, _, _ in dataloader:
            # dataloader returns [B, F, N] with pt/eta/phi as features
            # Move the feature dimension to the end so shape becomes [B, N, 3]
            jets.append(x_particles.transpose(1, 2))

        if not jets:
            continue

        jets = torch.cat(jets, dim=0)  # [n_jets, N, 3]
        jet_tensors.append(jets)
        labels.append(display_name)

    if not jet_tensors:
        return
    print(
        f"✅ Total jets: {sum(j.shape[0] for j in jet_tensors)} for {start}-{end} files"
    )
    if overlay:
        jets = [reconstruct_jet_features_from_particles(j) for j in jet_tensors]
        plot_tensor_jet_features(jets, labels=labels, filename=filename)
    else:
        # Concatenate all jets and reconstruct features from all particles
        all_jets_particles = torch.cat(jet_tensors, dim=0)  # [N, n_constits, 3]
        jets = reconstruct_jet_features_from_particles(all_jets_particles)
        plot_tensor_jet_features(jets, labels=["All jets"], filename=filename)




import vector
vector.register_awkward()
def reconstruct_jet_features_from_particles(x_particles: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct jet pt, eta, phi, mass from [B, N, 3] particle outputs.
    Returns: [B, 4] tensor (pt, eta, phi, mass), same device as input
    """
    device = x_particles.device

    if x_particles.ndim != 3:
        raise ValueError("x_particles must be a 3D tensor")

    # Support both [B, N, 3] and [B, 3, N] layouts
    if x_particles.shape[-1] == 3:
        pt = x_particles[..., 0]
        eta = x_particles[..., 1]
        phi = x_particles[..., 2]
    elif x_particles.shape[1] == 3:
        pt = x_particles[:, 0, :]
        eta = x_particles[:, 1, :]
        phi = x_particles[:, 2, :]
    else:
        raise ValueError("Tensor shape must have 3 features")

    # Build particle 4-momenta assuming massless particles
    p4 = vector.arr({
        "pt": pt.detach().cpu().numpy(),
        "eta": eta.detach().cpu().numpy(),
        "phi": phi.detach().cpu().numpy(),
        # assume massless constituents
        "mass": np.zeros_like(pt.detach().cpu().numpy()),
    })

    jets = p4.sum(axis=1)

    jets_np = np.stack([jets.pt, jets.eta, jets.phi, jets.mass], axis=-1)
    jets_tensor = torch.from_numpy(jets_np).to(device).float()

    return jets_tensor


def plot_difference(orig_jets: torch.Tensor, recon_jets: torch.Tensor,
                    filename: str = "jet_feature_difference.png") -> None:
    """Plot distributions of differences between reconstructed and original jets."""
    if orig_jets.shape != recon_jets.shape:
        raise ValueError("Input tensors must have the same shape")

    diff = (recon_jets - orig_jets).cpu().numpy()
    features = ["pt", "eta", "phi", "mass"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, feat in enumerate(features):
        values = diff[:, i]
        rng = (values.min(), values.max())
        axes[i].hist(values, bins=50, histtype="step", density=True, range=rng)
        axes[i].set_xlabel(f"$\\Delta$ {DEFAULT_LABELS.get('jet_' + feat, feat)}")
        axes[i].grid(True)

    fig.suptitle("Reconstructed - Original Jet Feature Differences", fontsize=16)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved plot to {filename}")


if __name__ == "__main__":
    plot_all()

