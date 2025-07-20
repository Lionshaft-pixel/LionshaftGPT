import wikipedia
import datetime
import os

def learn_from_wikipedia(topic, log_dir="knowledge_logs"):
    try:
        print(f"🔎 Searching Wikipedia for '{topic}'...")
        summary = wikipedia.summary(topic, sentences=10)

        # Make sure the logs folder exists
        os.makedirs(log_dir, exist_ok=True)

        # Create a timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{log_dir}/{topic.replace(' ', '_')}_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== TOPIC: {topic} ===\n\n")
            f.write(summary)

        print(f"✅ Learned about '{topic}' and saved to {filename}")
        return summary

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Topic too broad. Options:\n{e.options[:5]}")
    except wikipedia.exceptions.PageError:
        print(f"❌ No page found for '{topic}'")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    while True:
        topic = input("\n📚 What should Agent 001 learn from Wikipedia? (or type 'exit')\n> ")
        if topic.lower() == "exit":
            break
        learn_from_wikipedia(topic)