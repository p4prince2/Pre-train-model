#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import FocusLSTM

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load preprocessed sequences and labels
X = np.load("X_sequences.npy")   # Shape: [N, 30, 4]
y = np.load("y_labels.npy")      # Shape: [N]

# Create DataLoader
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = FocusLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 12
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        out = model(batch_x)
        loss = criterion(out, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved as 'model.pth'")

