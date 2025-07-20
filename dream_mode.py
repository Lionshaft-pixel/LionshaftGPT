import os
import datetime
import random
from gpt_thinker import generate_thought

MEMORY_FILE = "memory.txt"
DREAM_LOGS_DIR = "logs/dream_logs/"
SNAPSHOT_DIR = "data/training_snapshots/"

# Make sure folders exist
os.makedirs(DREAM_LOGS_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def select_dream_seed(memories):
    tagged = [m for m in memories if "(" in m and ")" in m]
    return random.sample(tagged, min(5, len(tagged))) if tagged else []

def generate_dream(seed_memory):
    prompt = f"While sleeping, you remembered: {seed_memory}\nWhat would your dreaming mind say or do?"
    return generate_thought(prompt, max_new_tokens=100)

def save_dreams(dreams):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dream_{timestamp}.txt"
    path = os.path.join(DREAM_LOGS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        for dream in dreams:
            f.write(dream + "\n")

    # Also store in training snapshots
    snap_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.txt")
    with open(snap_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dreams))

def run_dream():
    print("💤 LionshaftGPT is dreaming...")

    memories = load_memories()
    seeds = select_dream_seed(memories)

    if not seeds:
        print("No memories to dream about. AI brain is silent tonight.")
        return

    dreams = []
    for seed in seeds:
        print(f"🌱 Dream seed: {seed}")
        dream = generate_dream(seed)
        print(f"💭 Dream: {dream}")
        dreams.append(f"Seed: {seed}\nDream: {dream}\n")

    save_dreams(dreams)
    print(f"✅ {len(dreams)} dreams saved to {DREAM_LOGS_DIR}")