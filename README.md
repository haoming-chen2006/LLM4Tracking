# LLM4Tracking

LLM4Tracking explores foundation models for particle physics. It is an experimental code base focused on building tokenizers and generative models for jets. The project currently focuses on developing MOE-empowered VQ-VAE style models that can encode sets of jet constituents into discrete tokens, and integrating them into a language model.

---

## 1. Overview

The long--term goal is to create a unified foundation model that can understand and generate collider events. A first step toward this goal is learning a **tokenizer** for jet constituents. The repository contains utilities to read the JetClass dataset, several prototype models, and plotting utilities to visualize results. Typical use cases include:

- Training a baseline vector-quantised autoencoder (VQ-VAE).
- Experimenting with more expressive architectures such as the NormFormer based "OmniJet" approach.
- Fast prototyping with FlashAttention to scale up depth and codebook size.

The code is under active development and many components are still exploratory.

---

## 2. Dataset

All data loading utilities live in the [`dataloader/`](dataloader) folder and are built around the **JetClass** dataset. A ROOT file is read via `read_file()` which converts jet and particle features into padded NumPy arrays. Example features include jet `pt`, `eta`, `phi` and per--particle `pt`, `eta`, `phi`.

Two dataloader variants are provided:

- [`dataloader/dataloader.py`](dataloader/dataloader.py): returns tensors of shape `[B, F, N]` for particles and `[B, J]` for jets. Useful when no padding mask is required.
- [`dataloader/masked_dataloader.py`](dataloader/masked_dataloader.py): additionally yields a mask tensor `[B, N]` indicating valid particles.

Both expose helpers such as `load_jetclass_label_as_tensor()` and `load_jetclass_label_as_dataset()` so you can load either a specific label or multiple classes. There is also [`dataloader/load.py`](dataloader/load.py) which demonstrates reading file lists from `config.yaml` to assemble full training/validation splits.

---

## 3. Models

Several VQ-VAE style models are implemented under [`models/`](models):

1. **VQVAE MLP (particles)** – [`vqvaeMLP_particle.py`](models/vqvaeMLP_particle.py)
   - A simple multilayer perceptron baseline operating on per--particle features.
2. **VQVAE MLP (jets)** – [`vqvaeMLP_jet.py`](models/vqvaeMLP_jet.py)
   - Encodes global jet features and serves as a lightweight baseline.
3. **VQVAE NormFormer** – [`NormFormer.py`](models/NormFormer.py)
   - Implements the "OmniJet" idea using a stack of NormFormer blocks to process sequences of particles.
4. **VQVAE Flash** – [`NormFormer_Flash.py`](models/NormFormer_Flash.py)
   - A deeper architecture that utilises FlashAttention for faster training and supports larger codebooks.

Each model produces a reconstruction along with VQ statistics. The NormFormer variants accept optional particle masks for variable–length jets.

---

## 4. Plots and Visualisation

The [`plot/`](plot) directory hosts scripts for analysing models and datasets:

- [`plot/plot.py`](plot/plot.py) can generate histograms of jet and particle features for any subset of JetClass events. It also contains utilities to compare reconstructed jets to the originals.
- Pre–generated summaries for different JetClass labels can be found in [`plot/event_graphs/`](plot/event_graphs).
- Training curves and reconstruction overlays are stored under [`plot/training_plots/`](plot/training_plots).

Typical usage:

```bash
python plot/plot.py               # plots all jet classes
python plot/plot.py --help        # view options
```

---

This repository is a starting point for experiments in tokenising particle physics data. Contributions and ideas are welcome!


