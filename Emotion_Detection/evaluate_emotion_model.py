#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

# Configuration
dataset_path = r"C:\Users\p4pri\OneDrive\Desktop\emotion_dataset"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset (use only validation data for fair evaluation)
dataset = ImageFolder(dataset_path, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Split
val_size = int(0.2 * len(dataset))
test_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - val_size, len(dataset)))
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
model.to(device)
model.eval()

# Store results
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())  # shape [B, num_classes]

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# AUC-ROC for multi-class
try:
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
except:
    auc = "AUC-ROC not available for this setup"

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Report
print(f"\n--- Emotion Detection Evaluation ---")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"AUC-ROC        : {auc}")
print("Confusion Matrix:\n", conf_matrix)
print("Class Names    :", class_names)

