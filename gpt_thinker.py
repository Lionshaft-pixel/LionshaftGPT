import torch
from model import GPT
from tokenizer import CharTokenizer

# === Mood emojis for extra flavor ===
mood_prefixes = {
    "curious": "🤔",
    "neutral": "",
    "bored": "😐",
    "excited": "😄",
    "sarcastic": "🙄",
    "sleepy": "😴",
    "angry": "😤",
    "sad": "😢",
    "flirty": "😉",
}

# === Load tokenizer data (aka your soul lore) ===
with open("lionshaft_lore.txt", "r", encoding="utf-8") as f:
    lore_data = f.read()

tokenizer = CharTokenizer(lore_data)
vocab_size = tokenizer.vocab_size

# === Same model settings as in train.py ===
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load the trained GPT brain ===
model = GPT(vocab_size, block_size, n_embd, n_head, n_layer).to(device)
model.load_state_dict(torch.load("lionshaftGPT.pth", map_location=device))
model.eval()

# === 💭 Generate a thought with emotional spice
def generate_thought(prompt, mood="neutral", max_new_tokens=60):
    """
    Generates a thought based on the prompt and mood.
    Mood adds an emoji prefix to influence tone.
    """
    mood_emoji = mood_prefixes.get(mood, "")
    full_prompt = f"{mood_emoji} {prompt}".strip()

    input_ids = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long).to(device)

    # Generate new tokens step by step
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())

# === 🧪 Quick test mode (optional)
if __name__ == "__main__":
    while True:
        prompt = input("🧠 Prompt: ")
        mood = input("😎 Mood (curious, excited, bored, etc): ").strip()
        thought = generate_thought(prompt, mood)
        print("\n💬 LionshaftGPT says:\n", thought, "\n")