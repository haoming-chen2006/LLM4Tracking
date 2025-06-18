from dataclasses import dataclass
import math
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from vqtorch.nn import VectorQuant

# Optional: Ensure FlashAttention 2 is enabled in PyTorch >= 2.0
USE_FLASH = hasattr(F, 'scaled_dot_product_attention')
if USE_FLASH:
    print("Using FlashAttention 2 for efficient attention computation.")

class FlashNormformerBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.qkv_proj = nn.Linear(input_dim, input_dim * 3)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, input_dim),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

    def forward(self, x, mask=None, return_attn_weights=False):
        B, T, C = x.shape
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.squeeze(1)
            x = x * mask.unsqueeze(-1)

        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm).chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = qkv[0].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if USE_FLASH:
            if mask is not None:
                bool_mask = (mask == 0).unsqueeze(1).expand(B, self.num_heads, T, T)
                attn_mask = bool_mask.reshape(B * self.num_heads, T, T)
            else:
                attn_mask = None
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                mask_exp = mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(mask_exp == 0, float('-inf'))
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_out = attn_probs @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_out)
        x = x + self.dropout(out)

        x = x + self.mlp(self.norm2(x))

        return x

class FlashNormformerStack(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, num_blocks=2, dropout_rate=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            FlashNormformerBlock(
                input_dim=hidden_dim,
                mlp_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x

class VQVAENormFormer(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        vq_kwargs={},
        **kwargs,
    ):
        super().__init__()

        self.vq_kwargs = vq_kwargs
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_normformer = FlashNormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        self.vqlayer = VectorQuant(feature_size=self.latent_dim, **vq_kwargs)
        self.latent_projection_out = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_normformer = FlashNormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.encoder_normformer(x, mask=mask)

        if mask is not None:
            z_embed = self.latent_projection_in(x) * mask.unsqueeze(-1)
        else:
            z_embed = self.latent_projection_in(x)

        z, vq_out = self.vqlayer(z_embed)

        x_reco = self.latent_projection_out(z)
        x_reco = self.decoder_normformer(x_reco, mask=mask)
        x_reco = self.output_projection(x_reco)

        if mask is not None:
            x_reco = x_reco * mask.unsqueeze(-1)

        return x_reco, vq_out

def plot_model(model, samples, device="cuda", n_examples_to_plot=200, masks=None, saveas=None):
    """Visualize the model.

    Parameters
    ----------
    model : nn.Module
        The model.
    samples : Tensor
        The input data.
    device : str, optional
        Device to use. The default is "cuda".
    n_examples_to_plot : int, optional
        Number of examples to plot. The default is 200.
    """

    samples = samples.to(device)
    model = model.to(device)

    # run the model on the input data
    with torch.no_grad():
        # print(f"Model device: {next(model.parameters()).device}")
        # print(f"Samples device: {samples.device}")
        r, vq_out = model(samples, masks)
        z_q = vq_out["z_q"]
        z_e = vq_out["z"]
        idx = vq_out["q"]

        if masks is not None:
            r = r[masks == 1]
            z_e = z_e[masks == 1]
            z_q = z_q[masks == 1]
            idx = idx[masks == 1]

        z_e = z_e.squeeze(1)
        z_q = z_q.squeeze(1)
        idx = idx.squeeze(1)

        # move r, z_e, z_q, idx to cpu for plotting
        r = r.detach().cpu()
        z_e = z_e.detach().cpu()
        z_q = z_q.detach().cpu()
        idx = idx.detach().cpu()

    samples = samples.detach().cpu().numpy()
    if masks is not None:
        masks = masks.detach().cpu().numpy()
        samples = samples[masks == 1]

    # create detached copy of the codebook to plot this
    fig, axarr = plt.subplots(1, 5, figsize=(15, 3))
    # axarr = axarr.flatten()

    style_tokens = dict(color="forestgreen")
    style_true = dict(color="royalblue")
    style_tokens_emb = dict(color="darkorange")
    style_true_emb = dict(color="darkorchid")

    ax = axarr[0]
    ax.scatter(
        z_e[:n_examples_to_plot, 0],
        z_e[:n_examples_to_plot, 1],
        alpha=0.4,
        marker="o",
        label="Samples",
        **style_true_emb,
    )
    ax.scatter(
        z_q[:n_examples_to_plot, 0],
        z_q[:n_examples_to_plot, 1],
        alpha=0.6,
        marker="x",
        label="Closest tokens",
        **style_tokens_emb,
    )
    ax.set_xlabel("$e_1$")
    ax.set_ylabel("$e_2$")
    ax.legend(loc="upper right")
    ax.set_title("Embeddings \n(samples and closest tokens)")

    ax = axarr[1]
    ax.scatter(
        z_e[:n_examples_to_plot, 0],
        z_e[:n_examples_to_plot, 2],
        alpha=0.2,
        s=26,
        **style_true_emb,
        label="Samples",
    )
    ax.scatter(
        z_q[:n_examples_to_plot, 0],
        z_q[:n_examples_to_plot, 2],
        alpha=0.7,
        s=26,
        **style_tokens_emb,
        marker="x",
        label="Closest tokens",
    )
    ax.set_xlabel("$e_1$")
    ax.set_ylabel("$e_3$")
    ax.set_title("Embeddings \n(samples and closest token)")
    ax.legend(loc="upper right")

    # plot the original sample and the reconstructed sample (the first sample in the batch)
    # plot original sample
    ax = axarr[2]
    ax.scatter(
        samples[:n_examples_to_plot, 0],
        samples[:n_examples_to_plot, 1],
        alpha=0.2,
        s=26,
        **style_true,
        label="Original",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Original constituents \n(first few in batch)")
    # plot reconstructed sample
    ax.scatter(
        r[:n_examples_to_plot, 0],
        r[:n_examples_to_plot, 1],
        alpha=0.7,
        s=26,
        marker="x",
        **style_tokens,
        label="Reco. token",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Data space \nTrue vs reconstructed")
    ax.legend(loc="upper right")

    # plot true vs reconstructed for deltaR and ptrel
    ax = axarr[3]
    ax.scatter(
        samples[:n_examples_to_plot, 0],
        samples[:n_examples_to_plot, 2],
        s=26,
        alpha=0.2,
        **style_true,
        label="Original",
    )
    ax.scatter(
        r[:n_examples_to_plot, 0],
        r[:n_examples_to_plot, 2],
        s=26,
        alpha=0.7,
        **style_tokens,
        marker="x",
        label="Reco. tokens",
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_3$")
    ax.legend(loc="upper right")
    ax.set_title("Data space \nTrue vs reconstructed")

    # plot the histogram of the codebook indices (i.e. a codebook_size x codebook_size
    # histogram with each entry in the histogram corresponding to one sample associated
    # with the corresponding codebook entry)
    ax = axarr[4]
    n_codes = model.vq_kwargs["num_codes"]
    bins = np.linspace(-0.5, n_codes + 0.5, n_codes + 1)
    ax.hist(idx, bins=bins)
    ax.set_title(
        "Codebook histogram\n(Each entry corresponds to one sample\nbeing associated with that"
        " codebook entry)",
        fontsize=8,
    )

    # make empty axes invisible
    def is_axes_empty(ax):
        return not (
            ax.lines
            or ax.patches
            or ax.collections
            or ax.images
            or ax.texts
            or ax.artists
            or ax.tables
        )

    for ax in axarr.flatten():
        if is_axes_empty(ax):
            ax.set_visible(False)

    fig.tight_layout()
    plt.show()
    if saveas is not None:
        fig.savefig(saveas)


def plot_loss(loss_history, lr_history, moving_average=100):
    if len(loss_history) < moving_average:
        print("Not enough steps to plot loss history")
        return
    fig, ax1 = plt.subplots(figsize=(5, 2))
    ax2 = ax1.twinx()

    # Plot loss history
    loss_history = np.array(loss_history)
    loss_history = np.convolve(loss_history, np.ones(moving_average), "valid") / moving_average
    ax1.plot(loss_history, color="blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_title(f"Loss history (moving average over {moving_average} steps)", fontsize=8)

    # Plot lr history
    ax2.plot(lr_history, color="red")
    ax2.set_ylabel("Learning Rate")

    fig.tight_layout()
    plt.show()

