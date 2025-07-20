import os
import torch
import torch.nn.functional as F
from model import GPT
from tokenizer import CharTokenizer
from train import train_on_dataset
import random

MEMORY_FILE = "memory.txt"
MODEL_PATH = "lionshaftGPT.pth"
LORE_FILE = "lionshaft_lore.txt"

positive_moods = ["excited", "curious", "neutral"]
negative_moods = ["bored", "sarcastic", "sleepy"]

def load_thoughts():
    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    positive = []
    negative = []

    for line in lines:
        if any(f"({mood}" in line for mood in positive_moods):
            positive.append(line.strip())
        elif any(f"({mood}" in line for mood in negative_moods):
            negative.append(line.strip())

    return positive, negative

def build_contrastive_dataset(tokenizer, pos, neg, block_size=128):
    data = []

    for _ in range(min(len(pos), len(neg))):
        good = random.choice(pos)
        bad = random.choice(neg)

        good_text = good.split(")", 1)[-1].strip()
        bad_text = bad.split(")", 1)[-1].strip()

        g_tokens = tokenizer.encode(good_text)
        b_tokens = tokenizer.encode(bad_text)

        if len(g_tokens) < block_size or len(b_tokens) < block_size:
            continue

        g_tensor = torch.tensor(g_tokens[:block_size])
        b_tensor = torch.tensor(b_tokens[:block_size])

        data.append((g_tensor, b_tensor))

    return data

def contrastive_loss(good_logits, bad_logits):
    good_mean = good_logits.mean()
    bad_mean = bad_logits.mean()
    return F.relu(bad_mean - good_mean + 1.0)  # margin loss

def train_contrastive():
    print("🎭 Starting contrastive training (mood-based)...")

    pos, neg = load_thoughts()
    if not pos or not neg:
        print("⚠️ Not enough good/bad thoughts to train contrastively.")
        return

    with open(LORE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    model = GPT(vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.train()

    dataset = build_contrastive_dataset(tokenizer, pos, neg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1):
        total_loss = 0
        for good_tensor, bad_tensor in dataset:
            good_tensor = good_tensor.unsqueeze(0).to(device)
            bad_tensor = bad_tensor.unsqueeze(0).to(device)

            good_logits = model(good_tensor)
            bad_logits = model(bad_tensor)

            loss = contrastive_loss(good_logits, bad_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Contrastive epoch complete | Total Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("🧠 LionshaftGPT has learned what GOOD vibes feel like ✨")