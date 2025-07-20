# generate_cli.py
import torch, sys, time
from model.transformer import LionshaftR1Model
from training.tokenizer import CharTokenizer
from rich import print
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text

# (load tokenizer & model exactly as before...)

def generate_stream(prompt, max_tokens=100, temperature=1.0, top_k=50):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    cache = None
    text = prompt
    with Live(Text(text, style="bold cyan"), refresh_per_second=10) as live:
        for _ in range(max_tokens):
            logits, cache = model(input_ids[:, -1:], cache=cache)
            logits = logits[:, -1, :] / temperature
            # top-k filter...
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            token = tokenizer.decode(nxt[0].item())
            text += token
            input_ids = torch.cat((input_ids, nxt), dim=1)
            live.update(Text(text, style="bold cyan"))
            time.sleep(0.02)
    return text

if __name__=="__main__":
    while True:
        prompt = Prompt.ask("[bold green]Prompt[/]")
        if prompt.lower()=="exit": sys.exit()
        out = generate_stream(prompt)
        print(f"[bold magenta]🦁 LionshaftR1:[/] {out}")