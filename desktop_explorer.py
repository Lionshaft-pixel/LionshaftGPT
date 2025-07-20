import os
import time
import pyautogui
import psutil
import subprocess

# === Basic app map (customize this) ===
APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "vscode": "code",
    "chrome": "chrome.exe",
    "paint": "mspaint.exe",
}

def open_app(app_name):
    if app_name not in APP_MAP:
        print(f"❌ App '{app_name}' not mapped.")
        return False
    try:
        print(f"🚀 Launching {app_name}...")
        subprocess.Popen(APP_MAP[app_name])
        time.sleep(2)
        return True
    except Exception as e:
        print(f"💥 Failed to open {app_name}: {e}")
        return False

def is_running(process_name):
    for p in psutil.process_iter(['name']):
        if process_name.lower() in p.info['name'].lower():
            return True
    return False

def type_text(text):
    print(f"⌨️ Typing: {text[:30]}...")
    pyautogui.typewrite(text, interval=0.05)

def close_app(app_name):
    os.system(f"taskkill /f /im {APP_MAP[app_name]}")

# === Simple autonomous routine ===
def explore_with_goal(topic):
    success = open_app("notepad")
    if success:
        from agent import learn_from_wikipedia  # Import here to avoid circular imports
        content = learn_from_wikipedia(topic)
        if content:
            time.sleep(1)
            type_text(content)
        else:
            type_text(f"Couldn't find info on {topic}")
        time.sleep(2)

if __name__ == "__main__":
    while True:
        topic = input("\n🎯 Topic for desktop learning? (or 'exit')\n> ")
        if topic.lower() == "exit":
            break
        explore_with_goal(topic)