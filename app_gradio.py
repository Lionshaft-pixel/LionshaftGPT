# app_gradio.py
import torch
import gradio as gr
from model.transformer import LionshaftR1Model
from training.tokenizer import CharTokenizer

# (load tokenizer & model as before...)

def chat_fn(prompt, history, temperature, top_k, max_tokens):
    # append prompt to history, generate new text:
    out = generate_text(prompt, max_tokens, temperature, top_k)
    history = history or []
    history.append((prompt, out))
    return history, history

with gr.Blocks(css="""
    body { background: #1f1f2e; color: #e0e0e0; }
    .gradio-container { max-width: 800px; margin: auto; }
    .message { border-radius: 1rem; padding: 0.75rem; margin: 0.5rem 0; }
    .user { background: #4d4dff; color: white; text-align: right; }
    .bot { background: #2e2e3e; color: #a0a0ff; text-align: left; }
""") as demo:
    gr.Markdown("## 🦁 LionshaftR1 Premium Chat")
    chat = gr.Chatbot(elem_id="chatbot", label="Chat")
    with gr.Row():
        txt = gr.Textbox(placeholder="Type your prompt...", show_label=False)
        btn = gr.Button("Send")
    with gr.Row():
        temp = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
        k = gr.Slider(1, 100, value=50, step=1, label="Top‑k")
        mt = gr.Slider(10, 512, value=100, label="Max Tokens")
    btn.click(chat_fn, [txt, chat, temp, k, mt], [chat, chat])
    txt.submit(btn.click, [txt, chat, temp, k, mt], [chat, chat])

demo.launch(server_name="0.0.0.0", server_port=7860)