import requests

url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}
payload = {
    "model": "meta/llama-3.2-11b-vision-instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 10
}
res = requests.post(url, headers=headers, json=payload)
print(res.status_code)
print(res.text)
