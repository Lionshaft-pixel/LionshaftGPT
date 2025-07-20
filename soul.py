from gpt_thinker import generate_thought
import random
import time

class Soul:
    def __init__(self):
        self.energy = 0.4  # 0 = sleepy, 1 = hyper
        self.mood = "neutral"
        self.last_seen_text = ""
        self.emotion_drift_timer = time.time()
        self.moods = ["curious", "neutral", "bored", "excited", "sarcastic", "sleepy"]

    def observe_world(self, visible_text):
        visible_text = visible_text.strip()
        self.last_seen_text = visible_text

        if len(visible_text) > 20 and random.random() < 0.4:
            self.energy += random.uniform(0.05, 0.15)
            self._adjust_mood("stimulated")
        else:
            self.energy -= random.uniform(0.01, 0.05)
            self._adjust_mood("bored")

        self.energy = max(0, min(self.energy, 1))
        self._drift_emotion()

    def _adjust_mood(self, event):
        if event == "stimulated":
            if self.mood in ["bored", "sleepy"]:
                self.mood = "curious"
            elif self.mood == "neutral":
                self.mood = "excited"
        elif event == "bored":
            if self.energy < 0.3:
                self.mood = "sleepy"
            elif self.energy < 0.5:
                self.mood = "bored"

    def _drift_emotion(self):
        if time.time() - self.emotion_drift_timer > 30:  # every 30 seconds
            self.mood = random.choice(self.moods)
            self.emotion_drift_timer = time.time()

    def wants_to_speak(self):
        threshold = 0.4 + random.uniform(-0.1, 0.2)
        return self.energy > threshold and random.random() < self.energy

    def get_thought(self):
        if not self.last_seen_text:
            return generate_thought("The screen is blank. Say what you're thinking.", style=self.mood)

        prompt = f"You saw this on the screen: '{self.last_seen_text}'. What do you think?"
        return generate_thought(prompt, mood=self.mood, max_new_tokens=60)