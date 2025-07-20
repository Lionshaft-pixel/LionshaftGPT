import os
import torch
from model import GPT
from tokenizer import CharTokenizer
from train import train_on_dataset  # We’ll define this soon

# === Paths ===
DREAM_DIR = "logs/dream_logs/"
LORE_FILE = "lionshaft_lore.txt"
MODEL_PATH = "lionshaftGPT.pth"
SNAPSHOT_DIR = "data/training_snapshots/"

# === Read & Combine All Dream Logs ===
def load_dream_logs():
    logs = []
    for filename in os.listdir(DREAM_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DREAM_DIR, filename), "r", encoding="utf-8") as f:
                logs.append(f.read())
    return "\n".join(logs)

# === Prepare Training Dataset ===
def build_dataset(tokenizer, dream_text):
    tokens = tokenizer.encode(dream_text)
    inputs = []
    targets = []
    block_size = 128

    for i in range(len(tokens) - block_size):
        chunk = tokens[i:i + block_size + 1]
        input_chunk = chunk[:-1]
        target_chunk = chunk[1:]
        inputs.append(torch.tensor(input_chunk))
        targets.append(torch.tensor(target_chunk))

    return list(zip(inputs, targets))

# === Run the Retrain Process ===
def retrain():
    print("🧠 Starting retraining from dreams...")

    if not os.path.exists(MODEL_PATH):
        print("⚠️ Base model not found. Train it manually first using train.py")
        return

    # Load tokenizer
    with open(LORE_FILE, "r", encoding="utf-8") as f:
        lore_data = f.read()
    tokenizer = CharTokenizer(lore_data)
    vocab_size = tokenizer.vocab_size

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Load dream data
    dream_text = load_dream_logs()
    dataset = build_dataset(tokenizer, dream_text)

    # Train model on new data (1 epoch, small batch)
    train_on_dataset(model, dataset, device, epochs=1, batch_size=4)

    # Save new model
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ LionshaftGPT has retrained itself from dreams!")