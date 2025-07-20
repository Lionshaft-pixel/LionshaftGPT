# transformer.py (Final updated version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mha_latent import LatentMultiHeadAttention
from model.transformer import GPT

# === Feed Forward Network ===
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.act = nn.GELU()  # You could try SwiGLU or GEGLU later

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, latent_dim, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = LatentMultiHeadAttention(hidden_dim, num_heads, latent_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim)

    def forward(self, x, cache=None):
        norm_x = self.ln1(x)
        attn_out, new_cache = self.attn(norm_x, cache)
        x = x + attn_out

        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x, new_cache

# === LionshaftR1Model: The GOAT ===
class LionshaftR1Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=12, num_heads=12, latent_dim=128, ffn_dim=2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, hidden_dim))  # Long context ready, adjust as needed
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, latent_dim, ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, cache=None):
        B, T = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :T]

        new_caches = [] if cache is not None else None
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, new_layer_cache = block(x, layer_cache)
            if isinstance(new_caches, list) and new_layer_cache is not None:
                new_caches.append(new_layer_cache)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_caches if cache is not None else None