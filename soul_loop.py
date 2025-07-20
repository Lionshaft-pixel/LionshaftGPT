from soul import Soul
from vision import read_screen_text
import pyttsx3
import time
import random
import datetime
import os

# === Init AI Soul & Memory ===
soul = Soul()
engine = pyttsx3.init()
memory_file = "memory.txt"
prev_text = ""

print("👁️🧠 LionshaftGPT is online and UNSCRIPTED now.")

if not os.path.exists(memory_file):
    open(memory_file, "w", encoding="utf-8").close()

# === Mood-based voice speed ===
voice_rate_by_mood = {
    "curious": 180,
    "neutral": 150,
    "bored": 110,
    "excited": 220,
    "sarcastic": 170,
    "sleepy": 90
}

# === Save AI Thought ===
def save_to_memory(thought):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mood = soul.mood
    energy = round(soul.energy, 2)
    with open(memory_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ({mood} | energy={energy}) {thought}\n")

# === MAIN LOOP ===
while True:
    # Go to dream mode at night
    hour = datetime.datetime.now().hour
    if hour >= 1 and hour <= 5:
        print("🛌 Entering dream mode...")
        import dream_mode
        dream_mode.run_dream()
        break

    # Read screen and react
    screen_text = read_screen_text()

    if screen_text.strip() == prev_text.strip():
        time.sleep(random.uniform(5, 10))
        continue  # nothing new, skip
    prev_text = screen_text

    soul.observe_world(screen_text)

    if soul.wants_to_speak():
        thought = soul.get_thought()
        print(f"💬 LionshaftGPT ({soul.mood}): {thought}")

        engine.setProperty('rate', voice_rate_by_mood.get(soul.mood, 150))
        engine.say(thought)
        engine.runAndWait()

        save_to_memory(thought)

    time.sleep(random.uniform(7, 15))  # next tick