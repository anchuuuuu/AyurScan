import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D, InputLayer
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageEnhance
import io
import requests
import base64
import concurrent.futures
from dotenv import load_dotenv
import time
from flask import Response, stream_with_context

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_API_KEY_2 = os.getenv("NVIDIA_API_KEY_2")

app = Flask(__name__)

# --- CONNECTION POOLING ---
session = requests.Session()
# Configure session with retries for robustness
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
retry_strategy = Retry(total=2, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# --- CONFIGURATION ---
MODEL_PATH = 'medicinal_leaf_80_classes_perfect.h5'
MODEL_PATH_2 = 'medicinal_leaf_final_optimized.h5'
JSON_PATH = 'class_indices.json'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- KERAS 3 COMPATIBILITY FIX ---
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

class FixedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        # Strip Keras 3 unrecognized arguments
        for arg in ['batch_shape', 'optional']:
            if arg in kwargs:
                kwargs.pop(arg)
        super().__init__(**kwargs)

CUSTOM_OBJECTS = {
    'DepthwiseConv2D': FixedDepthwiseConv2D,
    'InputLayer': FixedInputLayer
}

# Specialist Model Mapping (Aloevera, Neem, Tulsi)
SPECIALIST_CLASSES = {0: "Aloevera", 1: "Neem", 2: "Tulsi"}
# Map specialist names back to 80-class indices for easier comparison
# Note: These must match the labels in class_names.json exactly
SPECIALIST_TO_GENERAL_MAP = {
    "Aloevera": "0", 
    "Neem": "49",
    "Tulsi": "74"
}

# Load the AI Brain (Ensemble Mode)
print("⏳ Loading Primary AI Model...")
try:
    model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
except Exception as e:
    print(f"❌ Primary Model Load Failed: {e}")
    # Fallback to model 2 or raise
    raise e

print("⏳ Loading Secondary AI Model for Ensemble...")
try:
    model2 = load_model(MODEL_PATH_2, custom_objects=CUSTOM_OBJECTS)
    HAS_ENSEMBLE = True
    print("✅ Ensemble Active (Dual Models).")
except Exception as e:
    HAS_ENSEMBLE = False
    print(f"⚠️ Secondary model not found or incompatible: {e}")
    print("⚠️ Running in Single Model mode.")

# Load the Plant Names
with open(JSON_PATH, 'r') as f:
    class_indices = json.load(f)

# Optional: Add Ayurvedic database
ayurvedic_info = {
    "Neem": "🌿 Ayurvedic Species Profile\nNeem\n\nBotanical Name: Azadirachta indica\nAyurvedic Name: Nimba\n\n📋 1. Medicinal Benefits\nPrimary Action: Blood purifier & Antimicrobial\nDosha Impact: Pitta & Kapha reducing\nKey Properties: Antiseptic, antibacterial, antifungal\n\n💊 2. Common Treatments\n- Skin issues: Eczema, acne, psoriasis\n- Immunity: Fever reduction and detoxification\n\n🍵 3. Method of Use\nInternal Use: Leaf tea or extract\nExternal Application: Paste or Oil\nPrecautions: Avoid during pregnancy.",
    "Tulsi": "🌿 Ayurvedic Species Profile\nTulsi (Holy Basil)\n\nBotanical Name: Ocimum sanctum\nAyurvedic Name: Tulasi\n\n📋 1. Medicinal Benefits\nPrimary Action: Adaptogen & Respiratory support\nDosha Impact: Kapha & Vata reducing\nKey Properties: Anti-stress, antibacterial, antioxidant\n\n💊 2. Common Treatments\n- Cough/Cold: Eases congestion and sore throat\n- Stress: Reduces cortisol and boosts mood\n\n🍵 3. Method of Use\nInternal Use: Fresh leaf tea\nExternal Application: Juice for skin infections\nPrecautions: May lower blood sugar.",
}

def get_ayurvedic_details_api(plant_name):
    """Fetch detailed benefits and treatments from NVIDIA Llama API."""
    if not NVIDIA_API_KEY:
        return ayurvedic_info.get(plant_name, "Traditional medicinal plant used in Ayurvedic treatments.")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Generate a medicinal profile for '{plant_name}' using this EXACT template:

🌿 Ayurvedic Species Profile
[Common Name]

Botanical Name: [Scientific Name]
Ayurvedic Name: [Sanskrit Name]

📋 1. Medicinal Benefits
Primary Action: [Main effect]
Dosha Impact: [Vata/Pitta/Kapha]
Key Properties: [Therapeutic qualities]

💊 2. Common Treatments
- [Condition A]: [How it helps]
- [Condition B]: [How it helps]

🍵 3. Method of Use
Internal Use: [How to consume]
External Application: [How to apply]
Precautions: [Any warnings]

INSTRUCTIONS:
1. Stay under 150 words total.
2. Use professional, expert tone.
3. Keep the emojis and section numbers as shown."""
    
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 500
    }
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"⚠️ Details API Error: {e}")
    
    # Fallback to structured local dict or a generic template
    local_data = ayurvedic_info.get(plant_name)
    if local_data:
        return local_data
        
    return f"""🌿 Ayurvedic Species Profile
{plant_name}

Botanical Name: [Consulting...]
Ayurvedic Name: [Consulting...]

📋 1. Medicinal Benefits
Primary Action: Traditional therapeutic use
Dosha Impact: Varies by preparation
Key Properties: Natural medicinal qualities

💊 2. Common Treatments
- Traditional uses: Specific to '{plant_name}'
- Wellness support: General Ayurvedic benefits

🍵 3. Method of Use
Internal Use: Consult practitioner
External Application: Leaf paste or oil
Precautions: Always perform a patch test."""

def vision_predict_api(img_path, api_key, prompt_style=1):
    """Call NVIDIA Vision API with resized image for speed."""
    if not api_key:
        return None
    
    # Resize image to 512x512 for faster upload/processing
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((512, 512)) # Maintain aspect ratio, max 512px
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"⚠️ Image Processing Error: {e}")
        return None

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if prompt_style == 1:
        prompt = "Identify this medicinal plant leaf. Provide ONLY the common name (e.g., 'Neem')."
    else:
        prompt = "Carefully examine this medicinal leaf and provide its common plant name. Be precise. Name only."

    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 50
    }
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip().replace(".", "").replace('"', '').replace("'", "")
    except Exception as e:
        print(f"⚠️ Vision API Error: {e}")
    return None

import random

def model_predict(img_path, epoch_num):
    # Load and Enhance
    img = Image.open(img_path).convert("RGB")
    
    # --- ACCURACY BOOST 1: Contrast & Sharpness Enhancement ---
    # Enhance contrast to make leaf veins more visible
    img = ImageEnhance.Contrast(img).enhance(1.2)
    # Enhance sharpness for better edge detection
    img = ImageEnhance.Sharpness(img).enhance(1.6) 
    
    # SMART RESIZE: Detect if model needs 224 (old) or 300 (new EfficientNet)
    try:
        target_size = model.input_shape[1] 
    except:
        target_size = 224 
    
    # --- ACCURACY BOOST 2: Multiscale TTA (Random Zoom) ---
    if epoch_num > 1:
        # Random zoom between 100% and 120%
        zoom_factor = random.uniform(1.0, 1.2)
        w, h = img.size
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        img = img.crop((left, top, left + new_w, top + new_h))

    img = img.resize((target_size, target_size))
    x = image.img_to_array(img)
    
    # --- ACCURACY BOOST 3: Geometry TTA ---
    if epoch_num > 1:
        # Random rotation/flip TTA
        if random.random() > 0.5: x = np.fliplr(x)
        if random.random() > 0.5: x = np.flipud(x)
        
        angle = random.uniform(-25, 25)
        temp_img = Image.fromarray(x.astype('uint8')).rotate(angle)
        x = np.array(temp_img)
        
    # --- ACCURACY BOOST 4: Lighting TTA (Always apply a little) ---
    brightness = random.uniform(0.8, 1.2) if epoch_num > 1 else random.uniform(0.95, 1.05)
    x = x * brightness
    x = np.clip(x, 0, 255)

    x = np.expand_dims(x, axis=0)
    
    # --- MODEL 1 PREDICTION (80-CLASS) ---
    x1 = np.copy(x)
    if target_size == 224:
        x1 = (x1 / 127.5) - 1.0 # Legacy scaling
    else:
        x1 = preprocess_input(x1) # EfficientNet scaling
    
    preds1 = model.predict(x1, verbose=0)
    
    # --- MODEL 2 PREDICTION (SPECIALIST 3-CLASS) ---
    if HAS_ENSEMBLE:
        # Specialist model was trained with 224x224 and 1/255 rescaling
        x2 = np.copy(x)
        if x2.shape[1] != 224:
            # Resize if necessary (though usually target_size handles this)
            temp_img = Image.fromarray(x2[0].astype('uint8')).resize((224, 224))
            x2 = np.expand_dims(np.array(temp_img), axis=0)
            
        x2 = x2 / 255.0 # Specialist 1/255 scaling
        preds2 = model2.predict(x2, verbose=0)
        
        # We return both for the smarter ensemble in predict()
        return preds1, preds2
    
    return preds1, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
    
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
    
        # --- 5 EPOCH LOCAL ANALYSIS PROCESS ---
        all_preds1 = []
        all_preds2 = []
        print(f"📡 Starting 5-Epoch Local Botanical Scan (Ensemble Mode)...")
        for i in range(1, 6):
            p1, p2 = model_predict(file_path, i)
            all_preds1.append(p1)
            if p2 is not None: all_preds2.append(p2)
        
        # Calculate Final Local Results
        avg_preds1 = np.mean(all_preds1, axis=0)
        pred_idx = str(np.argmax(avg_preds1[0]))
        confidence = float(np.max(avg_preds1[0]) * 100)
        plant_name = class_indices.get(pred_idx, "Unknown Species")
        
        # Specialist Cross-Check
        if HAS_ENSEMBLE and all_preds2:
            avg_preds2 = np.mean(all_preds2, axis=0)
            spec_idx = np.argmax(avg_preds2[0])
            spec_conf = float(np.max(avg_preds2[0]) * 100)
            spec_name = SPECIALIST_CLASSES.get(spec_idx)
            
            print(f"🛡️ Specialist Model Check: {spec_name} ({spec_conf:.2f}%)")
            
            # If specialist is very sure and general model somewhat agrees
            if spec_conf > 85:
                # Get the 80-class mapping for this specialist name
                gen_idx = SPECIALIST_TO_GENERAL_MAP.get(spec_name)
                # Check if this name is in top 5 of general model
                top_5_indices = np.argsort(avg_preds1[0])[-5:]
                if int(gen_idx) in top_5_indices:
                    # HEAVY BOOST: specialist overruled/boosted general prediction
                    plant_name = spec_name
                    confidence = (confidence + spec_conf) / 2
                    print(f"🔥 SPECIALIST BOOST TRIGGERED: {plant_name}")

        confidence = round(confidence, 2)
        print(f"✅ Ensemble Analysis Complete: {plant_name} ({confidence}%)")
    
        # --- POST-SCAN API VERIFICATION ---
        print(f"🔍 Starting Cloud Verification...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(vision_predict_api, file_path, NVIDIA_API_KEY, 1)
            future2 = executor.submit(vision_predict_api, file_path, NVIDIA_API_KEY_2, 2)
            api1_name = future1.result(timeout=60)
            api2_name = future2.result(timeout=60)
        
        # --- REFINED CONSENSUS LOGIC ---
        final_plant = plant_name
        status = "Local Prediction (Low Confidence)"
        
        if confidence >= 80:
            status = "High-Confidence Local Scan"
        elif confidence >= 60:
            status = "Standard Local Scan"

        if api1_name or api2_name:
            l_low = plant_name.lower()
            a1_low = (api1_name.lower() if api1_name else "")
            a2_low = (api2_name.lower() if api2_name else "")
            
            api_agreement = (a1_low and a2_low and (a1_low in a2_low or a2_low in a1_low))
            local_api1_match = (a1_low and (a1_low in l_low or l_low in a1_low))
            local_api2_match = (a2_low and (a2_low in l_low or l_low in a2_low))
            
            # Priority 1: If Cloud APIs agree but local doesn't, trust Cloud (Handles Overfitting)
            if api_agreement and not local_api1_match:
                final_plant = api1_name
                status = "Cloud Expert Consensus (Overruled Local)"
            
            # Priority 2: If local matches any Cloud API, confirm it
            elif local_api1_match or local_api2_match:
                status = "Botanical Match Confirmed (Local + Cloud)"
                # Use the Cloud name for better formatting if they match
                if local_api1_match: final_plant = api1_name
                elif local_api2_match: final_plant = api2_name
                
            # Priority 3: If local is weak (<60%) and we have ANY cloud result, trust cloud
            elif confidence < 60 and (api1_name or api2_name):
                final_plant = api1_name if api1_name else api2_name
                status = "Cloud Expert Identification"
    
        # Fetch Ayurvedic Details
        details = get_ayurvedic_details_api(final_plant)
    
        return jsonify({
            'plant': final_plant,
            'confidence': confidence,
            'status': status,
            'details': details
        })
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return jsonify({
            'error': str(e),
            'status': 'Error during analysis'
        }), 500

@app.route('/consultation')
def consultation():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are an expert Ayurvedic Consultant. When users describe symptoms, provide a clear, step-by-step response. 1. Identify 1-2 medicinal plants. 2. Explain their specific benefits. 3. Provide a simple 'How-to-Use' step. Keep responses concise (under 100 words), professional, and easy to follow. Use bullet points for steps."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 250,
        "stream": True
    }
    
    def generate():
        try:
            response = session.post(url, headers=headers, json=payload, timeout=30, stream=True)
            if response.status_code != 200:
                yield json.dumps({'error': f'API Error: {response.status_code}'})
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data_json = json.loads(data_str)
                            content = data_json['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                yield content
                        except:
                            continue
        except Exception as e:
            print(f"Chat Stream Error: {e}")
            yield "I apologize, but I am having trouble connecting to my Ayurvedic knowledge base right now."

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)