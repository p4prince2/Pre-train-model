#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# model.py
import torch
import torch.nn as nn

class FocusLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2):
        super(FocusLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # Last layer's hidden state
        return out

