import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
print("Loading model...")
model = tf.keras.models.load_model("medicinal_leaf_final_optimized.h5")

# Load classes
with open("class_indices.json", "r") as f:
    class_names = json.load(f)

dataset_path = os.path.join(os.path.dirname(__file__), "Indian Medicinal Leaves Image Datasets", "Medicinal Leaf dataset")

for class_name in ['Aloevera', 'Neem', 'Tulsi']:
    class_dir = os.path.join(dataset_path, class_name)
    images = os.listdir(class_dir)
    # pick the first image that is a jpg
    img_name = next(img for img in images if img.endswith(".jpg"))
    img_path = os.path.join(class_dir, img_name)
    
    # Preprocess exactly like app.py
    image = Image.open(img_path).convert('RGB')
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)
    predicted_label = class_names.get(str(predicted_index), "Unknown")
    
    print(f"True Class: {class_name} | Predicted: {predicted_label} | Confidence: {confidence:.2f}%")
