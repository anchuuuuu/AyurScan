import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'medicinal_leaf_80_classes_perfect.h5'
JSON_PATH = 'class_indices.json'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the AI Brain
print("⏳ Loading AI Model...")
model = load_model(MODEL_PATH)
print("✅ Model Loaded.")

# Load the Plant Names
with open(JSON_PATH, 'r') as f:
    class_indices = json.load(f)

# Optional: Add Ayurvedic database
ayurvedic_info = {
    "Neem": "Used for skin disorders, blood purification, and antimicrobial properties.",
    "Tulsi": "Helps with respiratory issues, stress relief, and boosting immunity.",
    "Basil": "Rich in antioxidants; used for digestive health and anti-inflammatory benefits.",
    # Add more info here or fetch from a database/CSV
}

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    
    preds = model.predict(x)
    pred_idx = str(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]) * 100)
    
    plant_name = class_indices.get(pred_idx, "Unknown Species")
    return plant_name, round(confidence, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Get AI prediction
    plant_name, confidence = model_predict(file_path)
    
    # Get Ayurvedic details
    details = ayurvedic_info.get(plant_name, "Traditional medicinal plant used in Ayurvedic treatments for various ailments.")

    return jsonify({
        'plant': plant_name,
        'confidence': confidence,
        'details': details
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)