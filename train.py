# 🤮 Step 4: Tokenize and Train the AI
# This script takes the dataset and teaches your LSTM model how to respond.

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from chatbot_model import Chatbot
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


# ✅ A. Load Dataset
with open("data/dataset.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# ✅ B. Build Vocabulary
all_text = [item["input"] for item in raw_data] + [item["response"] for item in raw_data]
words = [word for sentence in all_text for word in sentence.lower().split()]
vocab = {word: i+2 for i, word in enumerate(set(words))}  # 0=PAD, 1=UNK
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# ✅ C. Encode Sentences
def encode(sentence):
    return [vocab.get(word.lower(), vocab["<UNK>"]) for word in sentence.split()]

pairs = [(encode(item["input"]), encode(item["response"])) for item in raw_data]

# ✅ D. Pad Sequences to Same Length
def pad(seq, max_len):
    return seq + [0]*(max_len - len(seq))

input_max = max(len(i) for i, _ in pairs)
output_max = max(len(o) for _, o in pairs)

X = torch.tensor([pad(i, input_max) for i, _ in pairs])
y = torch.tensor([pad(o, output_max) for _, o in pairs])

# ✅ E. Create Dataset and Loader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

data = ChatDataset(X, y)
loader = DataLoader(data, batch_size=2, shuffle=True)

# ✅ F. Initialize Model
model = Chatbot(vocab_size=len(vocab), embed_dim=64, hidden_dim=128)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ G. Train
for epoch in range(10):
    total_loss = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # Flatten outputs and targets
        outputs = model(inputs)

        # Ensure output matches target length
        if outputs.size(1) > targets.size(1):
            outputs = outputs[:, :targets.size(1), :]
        elif outputs.size(1) < targets.size(1):
            pad_len = targets.size(1) - outputs.size(1)
            pad_tensor = torch.zeros((outputs.size(0), pad_len, outputs.size(2)), device=outputs.device)
            outputs = torch.cat([outputs, pad_tensor], dim=1)

        # Flatten for loss
        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ✅ H. Save Model
torch.save(model.state_dict(), "weights/model_weights.pt")
print("🔖 Model trained and saved to weights/model_weights.pt")


        