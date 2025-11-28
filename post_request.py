import requests
import json

# The server URL and endpoint path
API_URL = "http://127.0.0.1:8000/conversation"
#API_URL = "https://medconnect-ai-4fnj.onrender.com/conversation"

# The data payload (matching the UserMessage model)

while True:
    # Making the POST request
    try:
        user_entry = input("\n👤 YOU: ")
        payload =  {
  "message": user_entry,
  "isdoctorlist": False,
  "doctor_list": [{} ]
}
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        #print("--- Request Sent Successfully ---")
        #print(f"Status Code: {response.status_code}")
        
        # The response body (matching the AgentResponse model)
        response_data = response.json()
        
        #print("--- Server Response Body ---")
        #print(json.dumps(response_data, indent=4))
        
        #print("\n--- Extracted Greeting ---")
        print(f"\n🤖 ASSISTANT: {response_data.get('message')}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")