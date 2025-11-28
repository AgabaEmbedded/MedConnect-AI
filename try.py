from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Dict, List

# --- 1. FastAPI Application Initialization ---
app = FastAPI(
    title="Medical Assistant Agent API",
    description="A POST endpoint that processes a user message, pass it to the medical assistant agent and return the agent response."
)


class UserMessage(BaseModel):
    """Defines the expected structure for the incoming POST request body."""
    message: str # The message from the user


# Model for the outgoing response
class AgentResponse(BaseModel):
    """Defines the structure for the response sent back to the user."""
    message: str


# --- 4. The POST Endpoint ---

@app.post(
    "/conversation", 
    response_model=AgentResponse,
    summary="Receives a user message, pass to the agent and return agent response."
)
def handle_agent_interaction(user_data: UserMessage):
    """
    This POST request:
    1. Receives the user's and message.
    2. pass it to the agent
    3. Generates and returns a agent response.
    """
    
    message = user_data.message
    return AgentResponse(
        message=message
    )