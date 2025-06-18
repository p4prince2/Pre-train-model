#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# prepare_sequences.py
import os
import cv2
import numpy as np
from feature_extractor import extract_features_from_frame

def extract_sequences_from_video(video_path, label, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    features = []
    sequences = []
    labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feat = extract_features_from_frame(frame)
        if feat is not None:
            features.append(feat)
            if len(features) >= seq_len:
                sequences.append(np.array(features[-seq_len:]))
                labels.append(label)

    cap.release()
    return sequences, labels

