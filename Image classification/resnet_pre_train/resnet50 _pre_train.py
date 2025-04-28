#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import json
# Load pre-trained ResNet50 model
emotion_model = ResNet50(weights='imagenet')



# In[9]:


# Prepare an image for prediction
#img_path = 'your_input_image_.jpg'  # path to your image
img_path = '2.jpg'  # path to your image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get predictions
predictions = emotion_model.predict(img_array)


# Example: the model's prediction output array
output_array = np.array(predictions)

# Get the index of the maximum value in the output array
predicted_class = np.argmax(output_array)

#print(f"Predicted class index: {predicted_class}")



# Load the ImageNet class labels from a JSON file
with open('imagenet_class_index.json', 'r') as f:
    class_idx = json.load(f)

# Example: Predicted class index from your model
predicted_class_index = predicted_class

# Get the corresponding class label
predicted_class_label = class_idx[str(predicted_class_index)][1]

print("Predicted Class Label:", predicted_class_label)



# In[ ]:




