"""import requests

url = "https://7o4235z4ugzjh1-8000.proxy.runpod.net/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing simply."}
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
"""

from openai import OpenAI

# Create the client, but override the base_url
client = OpenAI(
    base_url="https://60b3ldvwurvt72-8000.proxy.runpod.net/v1",
    api_key="not-needed"  # or your actual key if you secured the pod
)

response = client.chat.completions.create(
    model="Agaba-Embedded4/Deepfund_Medical_Assistant_Merged",
    messages=[
       {"role": "system", "content": "You are a medical Doctor with expert knowledge respond accurately and concise"},
        {"role": "user", "content": "what are the symptoms of malaria?"}
    ],
    max_tokens = 400
)

print(response.choices[0].message.content)
