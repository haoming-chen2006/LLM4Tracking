import sys
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from plot.plot import plot_tensor_jet_features, reconstruct_jet_features_from_particles
from dataloader.dataloader import load_jetclass_label_as_tensor
import vector
import matplotlib.pyplot as plt

# Register awkward vector behavior
vector.register_awkward()

# === Load a small test batch ===
dataloader = load_jetclass_label_as_tensor(label="HToBB", start=10, end=11, batch_size=512)
x_particles, orig_jet, _ = next(iter(dataloader))  # x_particles: [B, 4, 128]
x_particles = x_particles.transpose(1, 2)  # → [B, 128, 4]

# === Reconstruct jet features from particles ===
x_jets = reconstruct_jet_features_from_particles(x_particles)
print("x_jets shape:", x_jets.shape)  # Should be [B, 4]
print("Original jet shape:", orig_jet.shape)  # Should be [B, 4]
# === Plot ===
plot_tensor_jet_features([x_jets, x_jets],labels=("Test Batch","Train batch"), filename="test_jet_features.png")
