# autopilot.py
import time
import os
from tokenizer import CharTokenizer
from model import GPT
from web_mind import browse_and_extract
import torch
import torch.nn as nn
import torch.optim as optim

# ========== Settings ==========
save_dir = "autopilot_data"
os.makedirs(save_dir, exist_ok=True)

model_path = "autopilot_model.pt"
tokenizer_path = os.path.join(save_dir, "tokenizer.json")
text_data_path = os.path.join(save_dir, "knowledge.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128
n_embd = 256
n_head = 4
n_layer = 4

# ========== Load or Init Tokenizer ==========
if os.path.exists(tokenizer_path):
    tokenizer = CharTokenizer(load_path=tokenizer_path)
else:
    tokenizer = CharTokenizer(text="hello ai", load_path=None)
    tokenizer.save(tokenizer_path)

# ========== Load or Init Model ==========
model = GPT(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

# ========== Optimizer ==========
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# ========== Train On Text ==========
def train_on_text(text, steps=5):
    data = tokenizer.encode(text)
    if len(data) < block_size + 1:
        print("⏩ Not enough data to train.")
        return

    model.train()
    for step in range(steps):
        i = torch.randint(0, len(data) - block_size, (1,))
        x = torch.tensor([data[i:i+block_size]], dtype=torch.long).to(device)
        y = torch.tensor([data[i+1:i+1+block_size]], dtype=torch.long).to(device)

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Step {step+1}/{steps}] Loss: {loss.item():.4f}")

# ========== Self-Learning Loop ==========
while True:
    print("\n🤖 Autopilot waking up...")

    try:
        # Step 1: Browse & Extract
        print("🌐 Browsing...")
        content = browse_and_extract("https://en.wikipedia.org/wiki/Artificial_intelligence")
        with open(text_data_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")

        # Step 2: Train on it
        print("📚 Training on new data...")
        train_on_text(content, steps=10)

        # Step 3: Save progress
        tokenizer.save(tokenizer_path)
        torch.save(model.state_dict(), model_path)

    except Exception as e:
        print("🚨 Error during autopilot:", str(e))

    # Step 4: Nap time 😴
    print("😴 Sleeping for 10 minutes...\n")
    time.sleep(600)