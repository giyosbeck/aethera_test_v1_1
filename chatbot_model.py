# 🧠 Step 2: Build the Baby Brain (Simple AI Model)

# This file will define the AI's brain using PyTorch
# It's a simple LSTM-based model that can learn small language patterns

import torch
import torch.nn as nn

class Chatbot(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Chatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Example of how to create the model (we'll do this later in training script):
# model = Chatbot(vocab_size=5000, embed_dim=64, hidden_dim=128)

# Save this as chatbot_model.py
