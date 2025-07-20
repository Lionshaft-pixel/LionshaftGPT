import json

class CharTokenizer:
    def __init__(self, text=None, load_path=None):
        if load_path:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(i): ch for i, ch in data["itos"].items()}
            print("🧊 Tokenizer loaded from file.")
        elif text is not None:
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            print("🧪 Tokenizer built from text.")
        else:
            raise ValueError("🛑 Tokenizer needs either `text` or `load_path`. Both are missing!")

        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, indices: list) -> str:
        return ''.join([self.itos[i] for i in indices if i in self.itos])

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": {str(i): ch for i, ch in self.itos.items()}
            }, f, indent=2)
        print(f"💾 Tokenizer saved to {path}")