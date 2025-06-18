
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
    jet_energy = x_jets_all[:, 3].cpu().numpy()

    # Compute jet mass
    p4 = vector.array({
        "pt": jet_pt,
        "eta": jet_eta,
        "phi": jet_phi,
        "E": jet_energy,
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
    hist_plot(axs[3], ak.to_numpy(jets.mass), np.linspace(0, 400, 100), "Jet Mass [GeV]")

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


def plot_tensor_jet_features(jet_tensor: torch.Tensor, labels=("Jets 1", "Jets 2"), filename="tensor_jet_features.png"):
    """
    Plot 4 jet features (pt, eta, phi, mass) from one or more PyTorch tensors.
    Assumes input shape is [N, 4] with features [pt, eta, phi, mass].
    """
    if isinstance(jet_tensor, torch.Tensor):
        jet_tensor = [jet_tensor, jet_tensor.clone()]

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




import vector
vector.register_awkward()
def reconstruct_jet_features_from_particles(x_particles: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct jet pt, eta, phi, mass from [B, N, 3] particle outputs.
    Returns: [B, 4] tensor (pt, eta, phi, mass), same device as input
    """
    device = x_particles.device
    pt, eta, phi = x_particles.unbind(dim=-1)

    # Build particle 4-momenta assuming massless particles
    p4 = vector.arr({
        "pt": pt.detach().cpu().numpy(),
        "eta": eta.detach().cpu().numpy(),
        "phi": phi.detach().cpu().numpy(),
        "mass": np.zeros_like(pt.detach().cpu().numpy()),  # assume massless
    })

    jets = p4.sum(axis=1)

    # Return pt, eta, phi, mass
    jets_np = np.stack([jets.pt, jets.eta, jets.phi, jets.mass], axis=-1)
    jets_tensor = torch.from_numpy(jets_np).to(device).float()

    return jets_tensor  # shape: [B, 4]


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


