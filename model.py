# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()

        # === Token + Position embeddings ===
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # === Stack of Transformer blocks ===
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        # === Final LayerNorm & Linear Head ===
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape  # Batch size, Time steps

        # === Embed tokens + positions
        tok_emb = self.token_embedding(idx)            # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)         # (1, T, n_embd)
        x = tok_emb + pos_emb                          # (B, T, n_embd)

        # === Transformer magic ✨
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits  # (B, T, vocab_size)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        # === Self-attention
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)

        # === First norm + feedforward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

        # === Second norm
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # === Multi-head self attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        x = self.ln1(x)

        # === Feedforward network
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ln2(x)

        return x