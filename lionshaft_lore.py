def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()