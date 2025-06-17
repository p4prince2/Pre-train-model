#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#create_dataset.py
import os
import numpy as np
import cv2
from feature_extractor import extract_features_from_frame

dataset_dir = r"C:\Users\p4pri\OneDrive\Desktop\project\DataSet1"
labels_map = {"focused": 1, "distracted": 0}

seq_len = 30
all_sequences = []
all_labels = []

for label_name, label_val in labels_map.items():
    folder_path = os.path.join(dataset_dir, label_name)
    images = sorted([
        os.path.join(folder_path, img)
        for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png', '.jpeg'))
    ])

    features = []
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        feat = extract_features_from_frame(frame)
        if feat is not None:
            features.append(feat)
            if len(features) >= seq_len:
                sequence = np.array(features[-seq_len:])
                all_sequences.append(sequence)
                all_labels.append(label_val)

X = np.array(all_sequences)
y = np.array(all_labels)

np.save("X_sequences.npy", X)
np.save("y_labels.npy", y)

