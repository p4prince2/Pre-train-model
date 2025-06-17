#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# evaluate_model.py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import FocusLSTMWithAttention as FocusLSTM

# Load test data
X = np.load("X_sequences.npy")        # shape: (num_samples, seq_len, feature_dim)
y_true = np.load("y_labels.npy")       # shape: (num_samples,)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_true_tensor = torch.tensor(y_true, dtype=torch.long)

# Load model
model = FocusLSTM()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_tensor)
    y_pred = torch.argmax(outputs, dim=1).numpy()
    probs = torch.softmax(outputs, dim=1)[:, 1].numpy()  # For AUC-ROC

# Metrics
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(" Confusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print("\n Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("AUC-ROC  :", roc_auc_score(y_true, probs))

