# 🔧 Step 11: Auto-Retrain When Enough Feedback Is Collected
# Add this to app.py or a background script

import json
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chatbot_model import Chatbot
import os

APPROVED_FILE = "../data/approved_pairs.json"
REJECTED_FILE = "../data/rejected_pairs.json"
MODEL_WEIGHTS_PATH = "weights/model_weights.pt"
TRAIN_LIMIT = 20
# 🔧 Step 11: Auto-Retrain When Enough Feedback Is Collected
# Add this to app.py or a background script


# ✅ Training Logic
print("Checking for approved data...")
if os.path.exists(APPROVED_FILE):
    with open(APPROVED_FILE, encoding="utf-8") as f:
        approved_data = json.load(f)

    if len(approved_data) >= TRAIN_LIMIT:
        print(f"Retraining model on {len(approved_data)} approved pairs...")

        # Step 1: Rebuild vocab
        all_text = [item["input"] for item in approved_data] + [item["response"] for item in approved_data]
        words = [w for s in all_text for w in s.lower().split()]
        vocab = {w: i+2 for i, w in enumerate(set(words))}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1

        def encode(sentence):
            return [vocab.get(w.lower(), vocab["<UNK>"]) for w in sentence.split()]

        # Step 2: Prepare tensors
        pairs = [(encode(p["input"]), encode(p["response"])) for p in approved_data]
        max_input = max(len(i) for i, _ in pairs)
        max_output = max(len(o) for _, o in pairs)

        def pad(x, l): return x + [0]*(l - len(x))

        X = torch.tensor([pad(i, max_input) for i, _ in pairs])
        y = torch.tensor([pad(o, max_output) for _, o in pairs])

        class Dataset(torch.utils.data.Dataset):
            def __getitem__(self, i): return X[i], y[i]
            def __len__(self): return len(X)

        loader = torch.utils.data.DataLoader(Dataset(), batch_size=2, shuffle=True)

        # Step 3: Train model
        model = Chatbot(vocab_size=len(vocab), embed_dim=64, hidden_dim=128)
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            total_loss = 0
            for xi, yi in loader:
                opt.zero_grad()
                out = model(xi)
                if out.size(1) != yi.size(1):
                    pad_len = abs(out.size(1) - yi.size(1))
                    pad_tensor = torch.zeros((out.size(0), pad_len, out.size(2)))
                    out = out[:, :yi.size(1), :] if out.size(1) > yi.size(1) else torch.cat([out, pad_tensor], 1)
                loss = loss_fn(out.reshape(-1, out.size(-1)), yi.reshape(-1))
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")

        # Save trained model
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        print(f"Model saved to {MODEL_WEIGHTS_PATH}")

        # Clear approved list
        os.remove(APPROVED_FILE)
        print("Approved data cleared after training.")

    else:
        print(f"Waiting for more data: {len(approved_data)}/{TRAIN_LIMIT}")
else:
    print("No approved data found.")

# ❌ Track rejected feedback
if os.path.exists(REJECTED_FILE):
    with open(REJECTED_FILE, encoding="utf-8") as r:
        rejected = json.load(r)
    print(f"Note: {len(rejected)} rejected examples for analysis.")
else:
    print("No rejected feedback recorded yet.")
