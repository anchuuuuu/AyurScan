import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("medicinal_leaf_80_classes_perfect.h5")

with open("class_indices.json", "r") as f:
    class_names = json.load(f)

# Find an image to test
dataset_path = "Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset"
for class_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(folder_path):
        images = [img for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')]
        if images:
            img_path = os.path.join(folder_path, images[0])
            
            image = Image.open(img_path).convert('RGB')
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)
            predicted_label = class_names.get(str(predicted_index), "Unknown Plant")
            
            print(f"Testing image from folder: {class_folder}")
            print(f"Predicted: {predicted_label} (Index: {predicted_index}) with Confidence: {confidence:.2f}%")
            break
