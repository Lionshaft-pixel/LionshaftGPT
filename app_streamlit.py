# app_streamlit.py
import streamlit as st
from model.transformer import LionshaftR1Model
from training.tokenizer import CharTokenizer

# (load tokenizer & model...)

st.set_page_config(page_title="🦁 LionshaftR1", layout="centered", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#f5c518;'>LionshaftR1 Premium Chat</h1>", unsafe_allow_html=True)

temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)
top_k = st.sidebar.slider("Top‑k", 1, 100, 50)
max_tokens = st.sidebar.slider("Max Tokens", 10, 512, 100)

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("Your prompt:", "")
if st.button("Send") and prompt:
    response = generate_text(prompt, max_tokens, temperature, top_k)
    st.session_state.history.append(("You", prompt))
    st.session_state.history.append(("LionshaftR1", response))

for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div style='text-align:right;'><b>{speaker}:</b> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; color:#a0a0ff;'><b>{speaker}:</b> {text}</div>", unsafe_allow_html=True)