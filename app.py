# 🤖 Step 8: Long-Term Memory + Language-Aware Attention (Flask Upgrade)

from flask import Flask, request, jsonify, send_from_directory
import torch
from chatbot_model import Chatbot
import json
import torch.nn.functional as F
from collections import deque
from langdetect import detect
import os

# ✅ Load Dataset and Vocab
with open("data/dataset.json", encoding="utf-8") as f:
    raw_data = json.load(f)

words = [word for item in raw_data for sentence in [item["input"], item["response"]] for word in sentence.lower().split()]
vocab = {word: i+2 for i, word in enumerate(set(words))}  # 0=PAD, 1=UNK
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
ivocab = {i: w for w, i in vocab.items()}

def encode(sentence):
    return [vocab.get(word.lower(), vocab["<UNK>"]) for word in sentence.split()]

def decode(tensor):
    words = []
    for token in tensor:
        idx = token.item()
        if idx == vocab["<PAD>"]:
            continue
        word = ivocab.get(idx, "<UNK>")
        if word not in ["<PAD>", "<UNK>"]:
            words.append(word)
    return " ".join(words)

# ✅ Load Model
model = Chatbot(vocab_size=len(vocab), embed_dim=64, hidden_dim=128)
model.load_state_dict(torch.load("weights/model_weights.pt"))
model.eval()

# ✨ Create Flask App
app = Flask(__name__)
chat_memory = deque(maxlen=6)  # memory window
LOG_FILE = "chat_memory.log"

@app.route("/", methods=["GET"])
def home():
    return send_from_directory("static", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Say something!"})

    # 🌐 Detect language
    try:
        lang = detect(user_input)
    except:
        lang = "unknown"

    # ⌛ Combine memory context + input
    memory_context = " ".join(chat_memory)
    full_input = memory_context + " " + user_input if memory_context else user_input

    # 🤮 Encode input
    encoded = encode(full_input)[-20:]
    padded = encoded + [0]*(20 - len(encoded))
    input_tensor = torch.tensor([padded])

    with torch.no_grad():
        output = model(input_tensor)[0]  # shape: (seq_len, vocab_size)
        temperature = 0.9
        probs = F.softmax(output / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)

    response = decode(sampled)

    # 🚀 Save short memory and log long memory
    chat_memory.append(user_input)
    chat_memory.append(response)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[{lang.upper()}] User: {user_input}\n")
        log.write(f"[{lang.upper()}] Bot: {response}\n")

    return jsonify({"response": response, "lang": lang})

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "w", encoding="utf-8").close()
    app.run(debug=True)
