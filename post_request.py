import requests
import json

# The server URL and endpoint path
API_URL = "http://127.0.0.1:8000/conversation"
#API_URL = "https://medconnect-ai-4fnj.onrender.com/conversation"
mock_doctors = [
        {
            "id": "DOC001",
            "name": "Dr. Sarah Johnson",
            "specialty": "General Practitioner",
            "rating": 4.8,
            "years_experience": 12,
            "consultation_fee": 75,
            "location": "Lagos, Nigeria",
            "languages": ["English", "Yoruba"],
            "available_slots": ["Today 2PM", "Today 5PM", "Tomorrow 9AM"],
            "response_time_avg": "15 minutes",
            "experience_level": "senior"
        },
        {
            "id": "DOC002",
            "name": "Dr. Michael Okonkwo",
            "specialty": "Internal Medicine",
            "rating": 4.9,
            "years_experience": 15,
            "consultation_fee": 100,
            "location": "Abuja, Nigeria",
            "languages": ["English", "Igbo"],
            "available_slots": ["Today 3PM", "Tomorrow 10AM"],
            "response_time_avg": "10 minutes",
            "experience_level": "senior"
        },
        {
            "id": "DOC003",
            "name": "Dr. Amina Bello",
            "specialty": "Pediatrics",
            "rating": 4.7,
            "years_experience": 8,
            "consultation_fee": 80,
            "location": "Kano, Nigeria",
            "languages": ["English", "Hausa"],
            "available_slots": ["Tomorrow 11AM", "Tomorrow 2PM"],
            "response_time_avg": "20 minutes",
            "experience_level": "mid-level"
        },
        {
            "id": "DOC004",
            "name": "Dr. James Adebayo",
            "specialty": "Cardiology",
            "rating": 4.9,
            "years_experience": 20,
            "consultation_fee": 150,
            "location": "Lagos, Nigeria",
            "languages": ["English"],
            "available_slots": ["Today 4PM", "Tomorrow 9AM"],
            "response_time_avg": "5 minutes",
            "experience_level": "senior"
        },
        {
            "id": "DOC005",
            "name": "Dr. Fatima Mohammed",
            "specialty": "General Practitioner",
            "rating": 4.6,
            "years_experience": 5,
            "consultation_fee": 50,
            "location": "Kano, Nigeria",
            "languages": ["English", "Hausa", "Arabic"],
            "available_slots": ["Today 1PM", "Today 3PM", "Tomorrow 10AM"],
            "response_time_avg": "25 minutes",
            "experience_level": "junior"
        }
    ]
# The data payload (matching the UserMessage model)
language = input("Enter the Language of choice[english, hausa, igbo, yoruba]: ")
response_data = {"doctorlist_request" : False}
while True:
    # Making the POST request
    user_entry = input("\n👤 YOU: ")
    if response_data["doctorlist_request"]:
                  
        payload =  {
        "message": user_entry,
        "isdoctorlist": True,
        "doctor_list": mock_doctors,
        "language": language
        }
        print(f"\n{'*'*60}")
        print("doctor list sent")
        print(f"{'*'*60}")
    
    else:
         payload =  {
            "message": user_entry,
            "isdoctorlist": False,
            "doctor_list": [{} ],
            "language": language
            }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # The response body (matching the AgentResponse model)
        response_data = response.json()
        
        #print("\n--- Extracted Greeting ---")
        print(f"\n🤖 ASSISTANT: {response_data.get('message')}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")