

model = vqvae.VQVAENormFormer(
    input_dim=3,
    latent_dim=128,
    hidden_dim=256,
    num_heads=8,
    num_blocks=3,
    vq_kwargs={"num_codes": 2048, "beta": 0.25, "affine_lr": 0.0, "sync_nu": 2,
    "replace_freq": 20,},
).to(device)

model.load_state_dict("checkpoints/moe_checkpoints_vqvae_moe_large/moe_epoch_30.pth")

model.eval()
dataloader_eval = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)
all_orig_jets, all_recon_jets = [], []

with torch.no_grad():
    for i, batch in enumerate(dataloader_eval):
        if i >= 300:
            break

        x_particles, _, _ = [b.to(device) for b in batch]
        x_particles = x_particles.transpose(1, 2)
        x_norm = (x_particles - mean) / std

        out, loss_dict = model(x_norm)
        unique_codes = loss_dict["q"]
        print(unique_codes)
        out_denorm = out * std + mean

        orig_jet = reconstruct_jet_features_from_particles(x_particles)
        recon_jet = reconstruct_jet_features_from_particles(out_denorm)

        all_orig_jets.append(orig_jet)
        all_recon_jets.append(recon_jet)

all_orig_jets = torch.cat(all_orig_jets, dim=0)
all_recon_jets = torch.cat(all_recon_jets, dim=0)

plot_tensor_jet_features(
    [all_orig_jets, all_recon_jets],
    labels=("Original", f"MOE {config['type']} Reconstructed"),
    filename=os.path.join(PLOT_DIR, f"moe_{config['type']}_recon_overlay.png"),
)
plot_difference(
    all_orig_jets,
    all_recon_jets,
    filename=os.path.join(PLOT_DIR, f"moe_{config['type']}_difference.png"),
)

