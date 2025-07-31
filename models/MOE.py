from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vqtorch.nn import VectorQuant

USE_FLASH = hasattr(F, 'scaled_dot_product_attention')
if USE_FLASH:
    print("Using FlashAttention 2 for efficient attention computation.")

class MOE_MLP(nn.Module):
    def __init__(self, input_dim, num_experts=4, dropout_rate=0.2, expert_capacity=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim * 4, input_dim),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_experts)
        ])
        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)
            nn.init.xavier_uniform_(expert[3].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.zeros_(expert[3].bias)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [B*T, D]

        # Router: get top-2 experts per token
        router_logits = self.router(x_flat)  # [B*T, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [B*T, E]
        expert_weights, expert_indices = torch.topk(router_probs, k=2, dim=-1)  # [B*T, 2]

        output = torch.zeros_like(x_flat, dtype=x_flat.dtype, device=x_flat.device)

        # Flatten for easier indexing
        token_indices = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1).expand(-1, 2)  # [B*T, 2]

        # For each expert, collect tokens assigned to it
        for expert_idx in range(self.num_experts):
            # Find all positions where expert_idx is in top-2
            mask = (expert_indices == expert_idx)  # [B*T, 2]
            selected = mask.nonzero(as_tuple=False)  # [[token_id, position_in_topk], ...]

            if selected.size(0) == 0:
                continue

            token_ids = selected[:, 0]
            pos_in_topk = selected[:, 1]
            expert_inputs = x_flat[token_ids]
            weights = expert_weights[token_ids, pos_in_topk].unsqueeze(-1)

            expert_output = self.experts[expert_idx](expert_inputs)
            expert_output = expert_output.to(output.dtype)

            # Accumulate weighted output (sum if used more than once)
            output.index_add_(0, token_ids, expert_output * weights)

        # Auxiliary loss: encourage uniform expert usage
        avg_probs = router_probs.mean(dim=0)  # shape [num_experts]
        aux_loss = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()

        return output.view(batch_size, seq_len, hidden_dim), {
            'router_probs': router_probs.view(batch_size, seq_len, self.num_experts),
            'expert_indices': expert_indices.view(batch_size, seq_len, 2),
            'expert_weights': expert_weights.view(batch_size, seq_len, 2),
            'aux_loss': aux_loss
        }


class FlashNormformerBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.qkv_proj = nn.Linear(input_dim, input_dim * 3)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.mlp = MOE_MLP(input_dim=input_dim)
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

    def forward(self, x, mask=None, return_attn_weights=False):
        B, T, C = x.shape
        
        # Apply input mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm).chunk(3, dim=-1)
        q = qkv[0].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if USE_FLASH:
            attn_mask = None
            if mask is not None:
                # Create attention mask from padding mask
                # Convert float mask to boolean and create attention mask
                mask_bool = mask.bool()  # Convert to boolean first
                attn_mask = mask_bool.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                attn_mask = attn_mask & attn_mask.transpose(-2, -1)  # [B, 1, T, T]
                attn_mask = ~attn_mask  # Invert the mask
                
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Standard attention with proper masking
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                mask_exp = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                attn_scores = attn_scores.masked_fill(~mask_exp, float('-inf'))
                
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_out = attn_probs @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.dropout(self.out_proj(attn_out))
        mlp_out, moe_info = self.mlp(self.norm2(x))
        self.last_aux_loss = moe_info.get("aux_loss", torch.tensor(0.0, device=x.device))
        x = x + mlp_out
        return x

class FlashNormformerStack(nn.Module):
    def __init__(self, hidden_dim, num_heads=1, num_blocks=2, dropout_rate=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            FlashNormformerBlock(
                input_dim=hidden_dim,
                mlp_dim=hidden_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        aux_losses = []
        for block in self.blocks:
            x = block(x, mask=mask)
            if hasattr(block, "last_aux_loss"):
                aux_losses.append(block.last_aux_loss)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x, sum(aux_losses)

class VQVAENormFormer(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_heads=1, num_blocks=2, vq_kwargs={}):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.encoder_normformer = FlashNormformerStack(hidden_dim, num_heads, num_blocks)
        self.latent_projection_in = nn.Linear(hidden_dim, latent_dim)
        self.vqlayer = VectorQuant(feature_size=latent_dim, **vq_kwargs)
        self.latent_projection_out = nn.Linear(latent_dim, hidden_dim)
        self.decoder_normformer = FlashNormformerStack(hidden_dim, num_heads, num_blocks)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask=None):
        # Input projection with mask
        x = self.input_projection(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        # Encoder with mask
        x, aux_loss = self.encoder_normformer(x, mask=mask)
        
        # Latent projection with mask
        z_embed = self.latent_projection_in(x)
        if mask is not None:
            z_embed = z_embed * mask.unsqueeze(-1)
            
        # VQ layer
        z, vq_out = self.vqlayer(z_embed)
        
        # Decoder with mask
        x_reco = self.latent_projection_out(z)
        x_reco, _ = self.decoder_normformer(x_reco, mask=mask)
        x_reco = self.output_projection(x_reco)
        
        # Final masking
        if mask is not None:
            x_reco = x_reco * mask.unsqueeze(-1)
            
        if isinstance(vq_out, dict) and "loss" in vq_out:
            vq_out["loss"] = vq_out["loss"] + 0.01 * aux_loss
            vq_out["aux_loss"] = aux_loss
            
        return x_reco, vq_out

    def encode(self,x,mask=None):
        x = self.input_projection(x)
        x, _ = self.encoder_normformer(x,mask=mask)
        z_embed = self.latent_projection_in(x)
        return z_embed
    def decode(self,embed,mask=None):
        z,_ = self.vqlayer(embed)
        x_recon = latent_projection_out(z)
        x_recon,_ = self.decoder_normformer(x_recon)
        x_out = self.output_projection(x_recon)
        return x_out




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
