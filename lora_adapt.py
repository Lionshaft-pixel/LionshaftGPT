import torch
import torch.nn as nn
from model import GPT
from tokenizer import CharTokenizer
from train import get_data_loader
import random
import os

MODEL_PATH = "lionshaftGPT.pth"
LORE_FILE = "lionshaft_lore.txt"
LORA_SAVE_PATH = "lora_adapters.pth"

class LoRAAdapter(nn.Module):
    def __init__(self, in_features, rank=4):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, in_features, bias=False)

    def forward(self, x):
        return self.up(self.down(x))

def inject_lora(model, rank=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'attn' in name:
            adapter = LoRAAdapter(module.in_features, rank).to(module.weight.device)
            module.lora = adapter
            print(f"✨ Injected LoRA into: {name}")

            # Hook: Add LoRA output to original
            old_forward = module.forward
            def new_forward(x, orig=old_forward, lora=adapter):
                return orig(x) + lora(x)
            module.forward = new_forward
    return model

def train_lora():
    print("⚡ Starting LoRA adapter training...")

    with open(LORE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    model = GPT(tokenizer.vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    model = inject_lora(model, rank=4)
    model.train()

    data_loader = get_data_loader(text, tokenizer, block_size=128, batch_size=8)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "lora" in n or getattr(p, "is_lora", False)],
        lr=3e-4
    )

    for epoch in range(1):
        total_loss = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ LoRA training done | Total Loss: {total_loss:.4f}")

    # Save only adapter weights
    lora_weights = {
        name: module.lora.state_dict()
        for name, module in model.named_modules()
        if hasattr(module, 'lora')
    }
    torch.save(lora_weights, LORA_SAVE_PATH)
    print(f"💾 Saved LoRA adapters to {LORA_SAVE_PATH}")