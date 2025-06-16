#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import torch
import mediapipe as mp
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import threading

# ----- Config -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
buffer_size = 60  # Store last 60 seconds of emotions

# ----- Load model -----
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_labels))
model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
model.eval().to(device)

# ----- Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- Face Detection -----
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ----- Emotion Buffer -----
emotion_history = deque(maxlen=buffer_size)
timestamp_history = deque(maxlen=buffer_size)

# ----- Plotting Thread -----
def plot_emotion_timeline():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    while True:
        if len(emotion_history) > 0:
            ax.clear()
            times = list(timestamp_history)
            labels = list(emotion_history)
            ax.set_title("Driver Emotion Over Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Emotion")
            ax.set_ylim(-1, len(emotion_labels))
            ax.set_yticks(range(len(emotion_labels)))
            ax.set_yticklabels(emotion_labels)
            ax.plot(times, [emotion_labels.index(e) for e in labels], marker='o', linestyle='-')
        plt.pause(1)

# Start plotting thread
plot_thread = threading.Thread(target=plot_emotion_timeline, daemon=True)
plot_thread.start()

# ----- Webcam Stream -----
#cap = cv2.VideoCapture(r"C:\Users\p4pri\OneDrive\Desktop\run2_2018-05-24-14-08-46.ids_1.mp4")
cap = cv2.VideoCapture(0)


start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                label = emotion_labels[pred]

            # Save emotion + time
            current_time = round(time.time() - start_time, 2)
            emotion_history.append(label)
            timestamp_history.append(current_time)

            # Display
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

    cv2.imshow("Driver Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

