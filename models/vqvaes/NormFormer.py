

from dataclasses import dataclass
import math
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from vqtorch.nn import VectorQuant

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embd
        self.n_head = config.n_head
        assert self.n_embed % self.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = self.n_embed // self.n_head

        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.block_size = config.block_size

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        else:
            print("using flash attention")
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class NormformerBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(input_dim)

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

    def forward(self, x, mask=None, return_attn_weights=False):
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.squeeze(1)
            assert mask.ndim == 2, f"Mask should be [B, T] but got {mask.shape}"
            x = x * mask.unsqueeze(-1)

        x_norm = self.norm1(x)
        if mask is not None:
            key_padding_mask = mask == 0  # True for padding
            attn_output, attn_weights = self.attn(
                x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
            )
        else:
            attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)

        attn_res = self.norm2(attn_output) + x
        output = self.mlp(attn_res) + attn_res

        if return_attn_weights:
            return output, attn_weights
        return output

class NormformerStack(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, num_blocks=2, skip_out_proj=False, dropout_rate=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            NormformerBlock(
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

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        self.vqlayer = VectorQuant(feature_size=self.latent_dim, **vq_kwargs)
        self.latent_projection_out = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_normformer = NormformerStack(
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

def plot_model(model, samples, device="cuda", n_examples_to_plot=200, masks=None, saveas=None,
               global_mean=None, global_std=None):
    """
    Visualize the model with full reconstruction and embedding analysis.

    Parameters
    ----------
    model : nn.Module
        The model.
    samples : Tensor
        The input data. Should be normalized if global_mean/std not provided.
    global_mean, global_std : Tensor or None
        If provided, will denormalize both original and reconstructed samples.
    """
    samples = samples.to(device)
    model = model.to(device)

    with torch.no_grad():
        r, vq_out = model(samples, masks)
        z_q = vq_out["z_q"]
        z_e = vq_out["z"]
        idx = vq_out["q"]

        print("Code indices stats:")
        print("Unique codes:", torch.unique(idx).numel())
        print("Code distribution:", torch.bincount(idx.flatten(), minlength=model.vq_kwargs["num_codes"]))

        if masks is not None:
            r = r[masks == 1]
            z_e = z_e[masks == 1]
            z_q = z_q[masks == 1]
            idx = idx[masks == 1]
            samples = samples[masks == 1]

        z_e = z_e.reshape(-1, z_e.shape[-1])
        z_q = z_q.reshape(-1, z_q.shape[-1])
        idx = idx.reshape(-1)

        r = r.detach().cpu()
        z_e = z_e.detach().cpu()
        z_q = z_q.detach().cpu()
        idx = idx.detach().cpu()

        samples = samples.detach().cpu()

        # Optional: De-normalize for comparison
        if global_mean is not None and global_std is not None:
            samples = samples * global_std + global_mean
            r = r * global_std + global_mean

        samples = samples.numpy()
        r = r.numpy()

    fig, axarr = plt.subplots(1, 3, figsize=(18, 3))
    style_tokens = dict(color="forestgreen")
    style_true = dict(color="royalblue")
    style_tokens_emb = dict(color="darkorange")
    style_true_emb = dict(color="darkorchid")

    # PCA projections of latent space
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    z_all = np.concatenate([z_e[:n_examples_to_plot], z_q[:n_examples_to_plot]])
    pca.fit(z_all)
    z_e_proj = pca.transform(z_e[:n_examples_to_plot])
    z_q_proj = pca.transform(z_q[:n_examples_to_plot])

    ax = axarr[0]
    ax.scatter(z_e_proj[:, 0], z_e_proj[:, 1], alpha=0.4, marker="o", label="Samples", **style_true_emb)
    ax.scatter(z_q_proj[:, 0], z_q_proj[:, 1], alpha=0.6, marker="x", label="Tokens", **style_tokens_emb)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend()
    ax.set_title("Latents: $e_1$ vs $e_2$")

    ax = axarr[1]
    ax.scatter(z_e_proj[:, 0], z_e_proj[:, 2], alpha=0.4, s=25, label="Samples", **style_true_emb)
    ax.scatter(z_q_proj[:, 0], z_q_proj[:, 2], alpha=0.6, s=25, marker="x", label="Tokens", **style_tokens_emb)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-3")
    ax.legend()
    ax.set_title("Latents: $e_1$ vs $e_3$")



    fig.tight_layout()
    if saveas:
        fig.savefig(saveas)
    else:
        plt.show()
