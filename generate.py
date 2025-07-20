import torch
from model.transformer import LionshaftR1Model
from tokenizer.tokenizer import CharTokenizer

# === Load Tokenizer ===
import os
tokenizer = CharTokenizer(load_path=os.path.join("tokenizer", "tokenizer.json"))
vocab_size = tokenizer.vocab_size

# === Model config (must match train.py) ===
block_size = 128
hidden_dim = 768
num_layers = 12
num_heads = 12
latent_dim = 128
ffn_dim = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load model ===
model = LionshaftR1Model(
    vocab_size=vocab_size,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    latent_dim=latent_dim,
    ffn_dim=ffn_dim
).to(device)

model.load_state_dict(torch.load("lionshaftGPT.pth", map_location=device), strict=False)
model.eval()

# === Generation Function with KV Cache ===
@torch.no_grad()
def generate_text(prompt, max_tokens=100, temperature=1.0, top_k=50):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    cache = None
    generated = input_ids

    for _ in range(max_tokens):
        logits, cache = model(generated[:, -1:], cache=cache)
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, -1].unsqueeze(1)] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0].tolist())

# === Terminal Chat UI ===
print("🔥 Model loaded and ready for chat!")
if __name__ == "__main__":
    while True:
        prompt = input("\n💬 Prompt (or 'exit'): ")
        if prompt.strip().lower() == 'exit':
            break
        output = generate_text(prompt)
        print("\n🧠 LionshaftR1 says:\n" + output)