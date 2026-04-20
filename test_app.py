import requests
import json

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('Indian Medicinal Leaves Image Datasets/Medicinal Leaf dataset/Neem/1000.jpg', 'rb')}
try:
    response = requests.post(url, files=files)
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
