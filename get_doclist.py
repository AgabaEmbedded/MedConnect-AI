import requests
import json

# The server URL and endpoint path

#API_URL = "http://127.0.0.1:8000/conversation"
#API_URL = "https://medconnect-ai-4fnj.onrender.com/conversation"
API_URL = "https://medconnect-api-xrmi.onrender.com/api/agents"


 
try:
    response = requests.get(API_URL)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    
    # The response body (matching the AgentResponse model)
    response_data = response.json()
    
    #print("\n--- Extracted Greeting ---")
    print(f"\n🤖 ASSISTANT: {response_data}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")