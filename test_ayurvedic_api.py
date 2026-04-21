import os
from dotenv import load_dotenv
import requests

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

def test_api(plant_name):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"As an Ayurvedic expert, provide the medicinal benefits and traditional treatments for the plant: {plant_name}. Keep it under 50 words."
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 150
    }
    
    print(f"Testing for: {plant_name}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Success!")
            print(response.json()['choices'][0]['message']['content'].strip())
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    test_api("Neem")
