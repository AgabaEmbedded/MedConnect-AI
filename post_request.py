import requests
import json

# The server URL and endpoint path

API_URL = "http://127.0.0.1:8000/conversation"
#API_URL = "https://medconnect-ai-4fnj.onrender.com/conversation"


# The data payload (matching the UserMessage model)
language = input("Enter the Language of choice[english, hausa, igbo, yoruba]: ")
while True:
    # Making the POST request
    user_entry = input("\n👤 YOU: ")

    payload =  {
    "message": user_entry,
    "language": language
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # The response body (matching the AgentResponse model)
        response_data = response.json()
        
        #print("\n--- Extracted Greeting ---")
        print(f"\n🤖 ASSISTANT: {response_data.get('message')+response_data.get('doctorid')}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")