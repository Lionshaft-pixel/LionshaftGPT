import pytesseract
import pyautogui
from PIL import Image
import time

# === 🛠️ Set Tesseract path (adjust if yours is different) ===
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# === 🧠 Vision Core ===
def read_screen_text(region=None, grayscale=True):
    """
    Takes a screenshot (full or region) and uses OCR to extract text.
    region: tuple (x, y, width, height)
    grayscale: improve OCR by converting to black & white
    """
    screenshot = pyautogui.screenshot(region=region)
    if grayscale:
        screenshot = screenshot.convert('L')  # Convert to grayscale (improves OCR accuracy)
    text = pytesseract.image_to_string(screenshot)
    return text.strip()

# === 🧪 Live Testing ===
if __name__ == "__main__":
    print("👁️ LionshaftGPT Vision Online. Starting scan loop...\n")
    try:
        while True:
            print("📸 Scanning screen...")
            text = read_screen_text()
            if text:
                print("🧠 Text Detected:\n", text[:500])
            else:
                print("⚠️ No readable text.")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 Vision terminated by user.")