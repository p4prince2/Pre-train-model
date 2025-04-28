#!/usr/bin/env python
# coding: utf-8

# # Multi-class object detector using bounding box

# In[30]:


import cv2
import numpy as np


# Load YOLO model with correct file paths
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")  # Ensure these files are correctly placed in your directory
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class labels from coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread('your_image_data.jpg')
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Perform forward pass to get predictions
net.setInput(blob)
outs = net.forward(output_layers)

# Process the predictions (bounding boxes, class IDs, etc.)
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Filter predictions with confidence > 50%
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Display the detected objects
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])  # Get human-readable label
    confidence = str(round(confidences[i], 2))

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#Show image with detected objects
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






# # Key Changes:
# 
# **Face Detection with Haar Cascade:** Added code to detect faces within the image, which can be used to estimate the head pose.
# 
# **Bounding Boxes for Faces:** Faces are drawn with a yellow bounding box, and attention tracking could be added based on head pose estimation.

# In[28]:


import cv2
import numpy as np


# Load YOLO model with correct file paths
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class labels from coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load HaarCascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
img = cv2.imread('your_image_data.jpg')
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Perform forward pass to get predictions
net.setInput(blob)
outs = net.forward(output_layers)

# Process the predictions (bounding boxes, class IDs, etc.)
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Filter predictions with confidence > 50%
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Face detection for attention tracking
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # Draw face bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Optionally, estimate the head pose here using facial landmarks (e.g., with OpenCV's solvePnP)

# Display the detected objects
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])  # Get human-readable label
    confidence = str(round(confidences[i], 2))

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#Show image with detected objects
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# In[ ]:




