import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")

def identify_leaf(image_path, model="meta/llama-3.2-11b-vision-instruct"):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f'What medicinal plant is in this image? Provide only the common name (e.g., "Neem"). <img src="data:image/jpeg;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 50
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    test_img = "Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset/Common rue(naagdalli)/88.jpg"
    print("Llama 3.2 11B Vision:", identify_leaf(test_img, "meta/llama-3.2-11b-vision-instruct"))
    print("Neva 22B:", identify_leaf(test_img, "nvidia/neva-22b"))
