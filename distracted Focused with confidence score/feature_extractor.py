#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# feature_extractor.py
import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh

def extract_features_from_frame(frame):
    h, w = frame.shape[:2]
    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            def to_px(lm): return np.array([lm.x * w, lm.y * h])
            nose = to_px(landmarks[1])
            left_eye = to_px(landmarks[33])
            right_eye = to_px(landmarks[263])
            left_iris = to_px(landmarks[468])
            right_iris = to_px(landmarks[473])

            eye_center = (left_eye + right_eye) / 2
            iris_center = (left_iris + right_iris) / 2

            gaze = iris_center - eye_center
            head = nose - eye_center

            # Normalize
            gaze /= np.linalg.norm(gaze) + 1e-6
            head /= np.linalg.norm(head) + 1e-6

            return np.concatenate([gaze, head])  # 4D vector
        else:
            return None

