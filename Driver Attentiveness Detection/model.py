#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocusLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2):
        super(FocusLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Attention parameters
        self.attn_fc = nn.Linear(hidden_dim, 1)  # for scoring each time step

        # Final classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

    def attention(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        attn_scores = self.attn_fc(lstm_output)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_output, dim=1)  # [batch, hidden_dim]
        return context, attn_weights

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        context, attn_weights = self.attention(lstm_out)
        out = self.fc(context)  # [batch, num_classes]
        return out
