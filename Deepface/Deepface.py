#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
deepface.py

A demonstration of DeepFace functionalities for face verification, facial analysis,
face recognition, and model export using DeepFace's pre-trained models.

Dependencies:
    - deepface

Author: [Your Name or Team]
"""

# Install the DeepFace library (only needed in Colab or first-time use)
get_ipython().system('pip install deepface')

from deepface import DeepFace
import pprint

# ---------------------------------------------------------------------
# Face Verification
# ---------------------------------------------------------------------
"""
Verifies whether two given images belong to the same person.

Inputs:
    img1.jpg: Path to the first image.
    img2.jpg: Path to the second image.

Output:
    Dictionary containing 'verified' status, distance metric, model details, etc.
"""
result = DeepFace.verify("img1.jpg", "img2.jpg")
print(result)

# ---------------------------------------------------------------------
# Facial Attribute Analysis
# ---------------------------------------------------------------------
"""
Analyzes facial attributes from a given image.

Inputs:
    img_path: Path to the image file.
    actions : List of facial attributes to analyze, e.g., ['age', 'gender', 'emotion', 'race'].

Output:
    List of dictionaries (one per detected face) with predicted attributes and confidence scores.
"""
result = DeepFace.analyze(img_path="img1.jpg", actions=['age', 'gender', 'emotion', 'race'])
data = result[0]  # Access analysis of the first detected face

# Display summary of facial analysis
print("\n Facial Analysis Summary")
print("----------------------------")
print(f"Estimated Age       : {data['age']}")
print(f"Face Confidence     : {round(data['face_confidence'] * 100, 2)}%")
print(f"Dominant Gender     : {data['dominant_gender'].capitalize()} ({round(data['gender'][data['dominant_gender']], 2)}%)")
print(f"Dominant Emotion    : {data['dominant_emotion'].capitalize()} ({round(data['emotion'][data['dominant_emotion']], 2)}%)")
print(f"Dominant Race       : {data['dominant_race'].capitalize()} ({round(data['race'][data['dominant_race']], 2)}%)")

# Display facial region details
print("\n Facial Region")
print("----------------------------")
region = data['region']
print(f"Bounding Box        : x={region['x']}, y={region['y']}, w={region['w']}, h={region['h']}")
print(f"Left Eye            : {region['left_eye']}")
print(f"Right Eye           : {region['right_eye']}")

# Display detailed prediction probabilities
print("\n Detailed Prediction Probabilities")
print("----------------------------")

print("\nGender:")
for gender, prob in data['gender'].items():
    print(f"  {gender.capitalize():<10}: {round(prob, 2)}%")

print("\nEmotion:")
for emotion, prob in data['emotion'].items():
    print(f"  {emotion.capitalize():<10}: {round(prob, 2)}%")

print("\nRace:")
for race, prob in data['race'].items():
    print(f"  {race.capitalize():<18}: {round(prob, 2)}%")

# ---------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------
"""
Performs facial recognition by comparing an unknown image to a database of known faces.

Inputs:
    img_path: Path to the unknown image (e.g., "khan.png").
    db_path : Path to the image database directory.

Output:
    Prints the identity if a match is found.
"""
img_path = "khan.png"
results = DeepFace.find(img_path=img_path, db_path="/content/sample_data/image")

if len(results[0]) > 0:
    print("Match Found!")
    identity = results[0].iloc[0]['identity']
    name = identity.split("/")[-1].split(".")[0]
    print(f"Person identified: {name}")
else:
    print("No match found.")

# ---------------------------------------------------------------------
# Model Export (.h5 format)
# ---------------------------------------------------------------------
"""
Builds a DeepFace model (VGG-Face) and saves it to a .h5 file for further use.

Output:
    'vgg_face_model.h5' file containing the serialized Keras model.
"""
client = DeepFace.build_model("VGG-Face")
keras_model = client.model
keras_model.save("vgg_face_model.h5")

