# LionshaftR1/model/mha_latent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # Normal MHA queries
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        # Shared latent projection (shared across heads)
        self.latent_proj_kv = nn.Linear(hidden_dim, latent_dim)

        # Per-head back projection from latent space
        self.k_back_proj = nn.ModuleList([
            nn.Linear(latent_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.v_back_proj = nn.ModuleList([
            nn.Linear(latent_dim, self.head_dim) for _ in range(num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cache=None):
        B, T, _ = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # === LATENT CACHE HANDLING ===
        new_latent = self.latent_proj_kv(x)  # [B, T, latent_dim]

        if cache is not None:
            # Append new latent to cached ones along time axis
            cached_latent = torch.cat([cache['latent'], new_latent], dim=1)  # [B, T_total, latent_dim]
        else:
            cached_latent = new_latent

        # Cache output for next step
        new_cache = {'latent': cached_latent.detach()}

        # Back project keys & values
        keys = [proj(cached_latent) for proj in self.k_back_proj]  # list of [B, T_total, head_dim]
        values = [proj(cached_latent) for proj in self.v_back_proj]

        k = torch.stack(keys, dim=1)  # [B, num_heads, T_total, head_dim]
        v = torch.stack(values, dim=1)

        # Attention: [B, num_heads, T, head_dim] x [B, num_heads, head_dim, T_total]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        return self.out_proj(attn_output), new_cache