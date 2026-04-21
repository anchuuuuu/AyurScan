import requests

url = "http://127.0.0.1:5001/chat"
payload = {"message": "I have a cough"}
try:
    response = requests.post(url, json=payload, timeout=20)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
