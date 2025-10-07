# LionshaftGPT

**Author:** Lionshaft 15 year old coder  
**Objective:** To experiment with building a GPT model on a low end PC and learn foundational AI concepts.

---

## Abstract

LionshaftGPT is my personal project exploring how a models like ChatGPT can be created from scratch without high end GPUs but with extremely less epoch or you can also say parameters.  

The model currently produces mostly random outputs due to extremely low training epochs and limited computational resources. Despite this, it still can run simple inference loops in the terminal, serving as a learning platform for experimenting with language models and AI concepts.

---

## Features

### 1. Minimal Transformer Architecture
A lightweight transformer model designed for low memory and processing power.  

- It runs on CPUs or low end GPUs but it's not ideal for training.
- I added small parameter count for quick experiments.

### 2. Overcoming Catastrophic Forgetting with Dynamic Context Compression
A simple system to keep track of “knowledge” across training steps, Persistent state storage and Helps slightly reduce random outputs.

### 3. Generative Text Loop
Generates text sequences using repeated sampling from the model outputs. It produces terminal based conversations but it's not "perfect" because outputs are mostly random due to limited training but it definetely provides foundation for future development!

---

## Technical Requirements

- Python 3.10 or higher  
- Optional: GPU support (any GPU will help, but not required)  
- Minimal system resources (low-end PC friendly)  

---

## Installation
```
git clone https://github.com/Lionshaft-pixel/LionshaftGPT.git
cd LionshaftGPT
