import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import random
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("medicinal_leaf_80_classes_perfect.h5")

with open("class_indices.json", "r") as f:
    class_names = json.load(f)

dataset_path = "Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset"

for target in ["Aloevera", "Neem", "Tulsi", "Mint"]:
    folder_path = os.path.join(dataset_path, target)
    if os.path.exists(folder_path):
        images = [img for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')]
        if images:
            # test a few images
            for img_file in random.sample(images, min(3, len(images))):
                img_path = os.path.join(folder_path, img_file)
                image = Image.open(img_path).convert('RGB')
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                predictions = model.predict(img_array, verbose=0)
                predicted_index = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]) * 100)
                predicted_label = class_names.get(str(predicted_index), "Unknown")
                print(f"True: {target} | Pred: {predicted_label} | Conf: {confidence:.2f}%")
