# #!/usr/bin/env python
# # coding: utf-8
#
# # In[ ]:
#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
#
# # Config
# dataset_path = r"C:\Users\p4pri\OneDrive\Desktop\emotion_dataset"
# epochs = 10
# batch_size = 64
# lr = 0.0001
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Transforms
# transform = transforms.Compose([
#     transforms.Resize((48, 48)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # Dataset
# dataset = ImageFolder(dataset_path, transform=transform)
# class_names = dataset.classes
# num_classes = len(class_names)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
# # Model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.to(device)
#
# # Loss & Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
#
# # Training Loop
# for epoch in range(epochs):
#     model.train()
#     total, correct, loss_epoch = 0, 0, 0
#     for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         loss_epoch += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print(f"[{epoch+1}] Loss: {loss_epoch:.4f}, Accuracy: {correct/total:.4f}")
#
# # Save model
# torch.save(model.state_dict(), 'emotion_model.pth')
# print("Model saved as 'emotion_model.pth'")
#

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
#
# # Config
# dataset_path = r"C:\Users\p4pri\OneDrive\Desktop\emotion_dataset"
# epochs = 10
# batch_size = 64
# lr = 0.0001
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Transforms
# transform = transforms.Compose([
#     transforms.Resize((48, 48)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # Dataset
# dataset = ImageFolder(dataset_path, transform=transform)
# class_names = dataset.classes
# num_classes = len(class_names)
#
# # Split dataset
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
# # Model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model = model.to(device)
#
# # Loss & optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
#
# best_val_acc = 0.0
#
# for epoch in range(epochs):
#     model.train()
#     train_loss, train_correct, train_total = 0, 0, 0
#     loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
#     for inputs, labels in loop:
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         _, preds = torch.max(outputs, 1)
#         train_correct += (preds == labels).sum().item()
#         train_total += labels.size(0)
#
#         loop.set_postfix(loss=loss.item(), acc=train_correct/train_total)
#
#     train_acc = train_correct / train_total
#     train_loss_avg = train_loss / len(train_loader)
#
#     # Validation
#     model.eval()
#     val_loss, val_correct, val_total = 0, 0, 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             val_loss += loss.item()
#             _, preds = torch.max(outputs, 1)
#             val_correct += (preds == labels).sum().item()
#             val_total += labels.size(0)
#
#     val_acc = val_correct / val_total
#     val_loss_avg = val_loss / len(val_loader)
#
#     print(f"Epoch {epoch+1}/{epochs} => "
#           f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} | "
#           f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")
#
#     # Save best model
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), 'best_emotion_model.pth')
#         print(f"Best model saved with val acc: {best_val_acc:.4f}")
#
# print("Training complete.")


#train_emotion_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

# Config
dataset_path = r"C:\Users\p4pri\OneDrive\Desktop\emotion_dataset"
epochs = 20
batch_size = 64
lr = 0.0001
patience = 5  # for early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
dataset = ImageFolder(dataset_path, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=train_correct/train_total)

    train_acc = train_correct / train_total
    train_loss_avg = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} => Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_emotion_model.pth')
        print(f"Best model saved with val acc: {best_val_acc:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

print("Training complete.")
