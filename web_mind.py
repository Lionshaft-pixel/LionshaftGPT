from bs4 import BeautifulSoup
from transformers import pipeline
from urllib.parse import urlparse
from soul import Soul
import requests
import time
import random
import re
import hashlib
import os

# === CONFIG ===
memory_file = "memory.txt"
visited_urls_file = "visited_urls.txt"
user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
summary_model = "sshleifer/distilbart-cnn-12-6"
max_summary_tokens = 130

# === Load Summarizer ===
try:
    summarizer = pipeline("summarization", model=summary_model)
except Exception as e:
    print(f"🛑 Failed to load summarizer: {e}")
    summarizer = None

# === URL Visit Tracking ===
def has_visited(url):
    if not os.path.exists(visited_urls_file):
        return False
    with open(visited_urls_file, "r", encoding="utf-8") as f:
        return hashlib.md5(url.encode()).hexdigest() in f.read()

def mark_visited(url):
    with open(visited_urls_file, "a", encoding="utf-8") as f:
        f.write(hashlib.md5(url.encode()).hexdigest() + "\n")

# === Web Utils ===
def fetch_page_text(url):
    try:
        res = requests.get(url, headers=user_agent, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer"]):
            tag.decompose()
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching {url}: {e}"

def clean_web_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(cookies?|privacy|ads?|login).*', '', text, flags=re.IGNORECASE)
    return text.strip()

def save_to_memory(thought):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(memory_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {thought}\n")

# === Core Function ===
def learn_from_url(url, soul: Soul):
    if has_visited(url):
        print(f"🔁 Already learned from {url}")
        return

    print(f"🌐 Reading from: {url}")
    raw_text = fetch_page_text(url)
    if "Error fetching" in raw_text:
        print(raw_text)
        return

    clean_text = clean_web_text(raw_text)
    if not clean_text:
        print("❌ Page was empty or filtered too much.")
        return

    try:
        chunks = [clean_text[i:i+1024] for i in range(0, len(clean_text), 1024)]
        summaries = []
        for chunk in chunks[:3]:  # Read max 3 chunks per page
            result = summarizer(chunk, max_length=max_summary_tokens, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
        summary = " ".join(summaries)
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        summary = clean_text[:300] + "..."

    print(f"📚 Learned: {summary}")
    soul.observe_world(summary)
    save_to_memory(f"Web Knowledge ({urlparse(url).netloc}): {summary}")
    mark_visited(url)

# === Infinite Loop ===
if __name__ == "__main__":
    soul = Soul()

    seed_urls = [
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://www.ibm.com/topics/contrastive-learning",
        "https://www.kaggle.com/learn/intro-to-deep-learning"
    ]

    while True:
        for url in seed_urls:
            learn_from_url(url, soul)
            time.sleep(random.uniform(10, 20))  # Chill so we don't get IP banned

        print("🧠 Loop complete. Sleeping for a bit...\n")
        time.sleep(random.uniform(60, 120))  # Nap time before looping again