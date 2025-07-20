import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import LionshaftR1Model
from tokenizer.tokenizer import CharTokenizer
import os
import random
from datetime import datetime

# === Load Data ===
def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# === Config ===
tokenizer_path = "tokenizer.json"
text_file = "lionshaft_lore.txt"
model_path = "lionshaftGPT.pth"
epoch_path = "checkpoint.txt"
log_dir = "logs"
log_file = os.path.join(log_dir, "training_log.txt")

# === Ensure log dir exists ===
os.makedirs(log_dir, exist_ok=True)

# === Tokenizer Setup ===
if os.path.exists(tokenizer_path):
    tokenizer = CharTokenizer(load_path=tokenizer_path)
    print("🧊 Tokenizer loaded from file (vocab frozen).")
    text = load_data(text_file)
else:
    text = load_data(text_file)
    tokenizer = CharTokenizer(text)
    tokenizer.save(tokenizer_path)
    print("🆕 Tokenizer built and saved.")

encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# === Model Configs ===
vocab_size = tokenizer.vocab_size
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Setup ===
model = LionshaftR1Model(vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# === Load Checkpoint ===
start_epoch = 1
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    if os.path.exists(epoch_path):
        with open(epoch_path, "r") as f:
            start_epoch = int(f.read())
    print(f"🧠 Loaded brain from epoch {start_epoch}")
else:
    print("🆕 Starting from scratch")

print(f"🚀 Starting training from epoch {start_epoch}")

# === Training ===
total_epochs = 10000
input_seq = encoded[:-1].unsqueeze(0).to(device)
target_seq = encoded[1:].unsqueeze(0).to(device)

for epoch in range(start_epoch, start_epoch + total_epochs):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    chunks = 0

    for i in range(0, input_seq.size(1) - block_size, block_size):
        x = input_seq[:, i:i+block_size]
        y = target_seq[:, i:i+block_size]

        if x.size(1) != block_size or y.size(1) != block_size:
            continue

        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        total_loss += loss.item()
        chunks += 1

    optimizer.step()

    avg_loss = total_loss / chunks if chunks > 0 else 0
    
    if epoch % 10 == 0 or epoch == start_epoch:
        log = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{start_epoch + total_epochs - 1} | Loss: {avg_loss:.4f}"
        print(log)
        with open(log_file, "a") as lf:
            lf.write(log + "\n")

    if epoch % 50 == 0:
        torch.save(model.state_dict(), model_path)
        with open(epoch_path, "w") as f:
            f.write(str(epoch))
        print(f"💾 Checkpoint saved at epoch {epoch}")

# === Final Save ===
torch.save(model.state_dict(), model_path)
with open(epoch_path, "w") as f:
    f.write(str(start_epoch + total_epochs))
print("🎉 Training done. Brain saved to lionshaftGPT.pth")

# === Extra Train Helper (Dream Mode) ===
def train_on_dataset(model, dataset, device, epochs=1, batch_size=4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            inputs = [item[0].to(device) for item in batch]
            targets = [item[1].to(device) for item in batch]
            input_batch = torch.stack(inputs)
            target_batch = torch.stack(targets)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Dream Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")