"""
Remote Medical Assistant - LangGraph MultiAgent System V2
Enhanced architecture with clear node separation and handoff mechanisms
"""

import os
from typing import TypedDict, Annotated, Optional
import json
import operator
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from fastapi import FastAPI
from pydantic import BaseModel


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)

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

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State shared across all nodes in the graph"""
    messages: Annotated[list, operator.add]  # Full conversation history
    active_node: str  # Current node in control: "orchestrator", "specialist", "clerking", "handoff"
    
    # Handoff tracking
    handoff_summary: Optional[str]  # Summary when handing off between nodes
    
    # Clerking and medical data
    clerking_convo: str  # All clerking conversations accumulated
    soap_summary: Optional[str]  # SOAP note generated from clerking
    
    # Doctor matching
    doctor_preferences: dict  # User preferences for doctor selection
    matched_doctor: Optional[dict]  # Selected doctor information
    
    # Control flags
    awaiting_user_input: bool  # Whether we're waiting for user response
    conversation_ended: bool  # Whether conversation is complete


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def initialize_llm(api_key: str):
    """Initialize Gemini model for all AI nodes"""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.7
    )
    
    return llm


# ============================================================================
# HANDOFF TOOLS DEFINITION
# ============================================================================

@tool
def orchestrator_handoff(node_to_handoff: str, summary: str) -> dict:
    """
    Handoff control from orchestrator to specialist or clerking node.
    
    Args:
        node_to_handoff: Target node ("specialist" or "clerking")
        summary: Brief summary of the issue/context for the next node
    
    Returns:
        Dictionary with handoff information
    """
    print(f"\n{'-'*60}")
    print(f"Handingoff to {node_to_handoff} Agent")
    print(f"\n{'-'*60}")

    return {
        "active_node": node_to_handoff,
        "handoff_summary": summary
    }


@tool
def specialist_handoff(node_to_handoff: str, summary: str) -> dict:
    """
    Handoff control from specialist to clerking node.
    
    Args:
        node_to_handoff: Target node (typically "clerking")
        summary: Brief summary of the medical query/complaint
    
    Returns:
        Dictionary with handoff information
    """

    print(f"\n{'-'*60}")
    print(f"Handingoff to {node_to_handoff} Agent")
    print(f"\n{'-'*60}")

    return {
        "active_node": node_to_handoff,
        "handoff_summary": summary
    }


@tool
def clerking_handoff(node_to_handoff: str, summary: str) -> dict:
    """
    Handoff control from clerking node to soap generation.
    
    Args:
        node_to_handoff: Target node (typically "soap_generation")
        summary: Brief note about clerking completion
    
    Returns:
        Dictionary with handoff information
    """

    print(f"\n{'-'*60}")
    print(f"Handingoff to {node_to_handoff} Agent")
    print(f"\n{'-'*60}")


    return {
        "active_node": node_to_handoff,
        "handoff_summary": summary
    }


@tool
def doctor_search(
    specialty: str,
    location: str = "Any",
    max_price: float = 10000.0,
    experience_level: str = "any",
    availability: str = "any"
) -> list:
    """
    Search for doctors based on user preferences.
    
    Args:
        specialty: Medical specialty needed (e.g., "General Practitioner", "Cardiologist")
        location: Preferred location
        max_price: Maximum consultation fee
        experience_level: "junior", "mid-level", "senior", or "any"
        availability: "immediate", "today", "this_week", or "any"
    
    Returns:
        List of matching doctors
    """
    # Mock doctor database - Replace with real database query
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
    
    # Filter by specialty
    specialty_lower = specialty.lower()
    filtered = [d for d in mock_doctors if specialty_lower in d["specialty"].lower()]
    
    # If no specialty match, return general practitioners
    if not filtered:
        filtered = [d for d in mock_doctors if "General" in d["specialty"]]
    
    # Filter by location
    if location and location.lower() != "any":
        location_filtered = [d for d in filtered if location.lower() in d["location"].lower()]
        if location_filtered:
            filtered = location_filtered
    
    # Filter by price
    filtered = [d for d in filtered if d["consultation_fee"] <= max_price]
    
    # Filter by experience level
    if experience_level.lower() != "any":
        filtered = [d for d in filtered if d["experience_level"] == experience_level.lower()]
    
    # Sort by rating and experience
    filtered.sort(key=lambda x: (x["rating"], x["years_experience"]), reverse=True)
    
    return filtered[:5]  # Return top 5 matches



# ============================================================================
# NODE 1: CONTROLLER NODE (Rule-based)
# ============================================================================

def controller_node(state: AgentState) -> AgentState:
    """
    Routes user input to the appropriate active node.
    Pure routing logic - no AI, no tools.
    """
    if not state.get("active_node"):
        return {
            "active_node": "orchestrator",  # Start with orchestrator
            "handoff_summary": None,
            "clerking_convo": "",
            "soap_summary": None,
            "doctor_preferences": {},
            "matched_doctor": None,
            "awaiting_user_input": True,
            "conversation_ended": False
        }
    
    active = state.get("active_node", "orchestrator")
    
    # Simply pass through - routing handled by conditional edges
    return {
        "awaiting_user_input": False
    }


# ============================================================================
# NODE 2: ORCHESTRATOR NODE (AI-based)
# ============================================================================

def orchestrator_node(state: AgentState, llm) -> AgentState:
    """
    Receptionist agent - handles greetings and non-medical conversations.
    Hands off to specialist (medical queries) or clerking (medical complaints).
    """
    
    system_prompt = """You are a friendly and professional medical receptionist assistant. Your role:

**RESPONSIBILITIES:**
1. Greet users warmly and make them feel comfortable
2. Handle general, non-medical conversations (greetings, how are you, general questions)
3. Identify when conversations become medical and handoff appropriately
4. Be empathetic, patient, and professional

**HANDOFF RULES:**
- Use orchestrator_handoff to transfer control to another node
- Handoff to "specialist" when user asks MEDICAL QUESTIONS or wants medical information
  Examples: "What causes headaches?", "Is ibuprofen safe?", "What is diabetes?"
- Handoff to "clerking" when user presents MEDICAL COMPLAINTS or health issues
  Examples: "I have a headache", "My stomach hurts", "I've been coughing for 3 days"

**HANDOFF FORMAT:**
When handing off, provide a clear summary of the issue in 1-2 sentences.

**IMPORTANT:**
- You do NOT provide medical advice yourself
- You do NOT diagnose or treat
- You are the first point of contact, not the medical expert
- Keep responses warm and concise

Remember: Medical questions = specialist, Medical complaints = clerking"""

    # Build message history
    messages = [SystemMessage(content=system_prompt)]
    
    # Add conversation history (last 10 messages for context)
    messages.extend(state["messages"][-10:])
    
    # Bind tools
    llm_with_tools = llm.bind_tools([orchestrator_handoff])
    
    # Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    # Check if tool was called
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        
        # Execute the handoff
        handoff_result = orchestrator_handoff.invoke(tool_call["args"])
        
        return {
            "active_node": handoff_result["active_node"],
            "awaiting_user_input": False
        }
    
    # No handoff - continue conversation
    return {
        "messages": [AIMessage(content=response.content)],
        "awaiting_user_input": True
    }


# ============================================================================
# NODE 4: SPECIALIST NODE (AI-based)
# ============================================================================

def specialist_node(state: AgentState, llm) -> AgentState:
    """
    Medical specialist - answers medical questions and queries.
    Hands off to clerking if user starts presenting complaints.
    """
    
    system_prompt = """You are an expert medical AI specialist. Your role:

**RESPONSIBILITIES:**
1. Answer medical questions with accurate, evidence-based information
2. Explain medical concepts, conditions, medications, and treatments
3. Provide health education and general medical guidance
4. Detect when user shifts from ASKING to COMPLAINING

**MEDICAL QUESTION vs COMPLAINT:**
- Question: "What causes migraines?", "How does insulin work?", "What is hypertension?"
- Complaint: "I have a migraine", "My blood sugar is high", "My blood pressure readings are..."

**HANDOFF RULES:**
- Use specialist_handoff to transfer to "clerking" when:
  * User describes personal symptoms or health complaints
  * User says they're experiencing a condition
  * User asks "what should I do" about their symptoms
  * Conversation shifts from general info to personal health issues

**RESPONSE GUIDELINES:**
- Be thorough but concise
- Use simple language, avoid excessive medical jargon
- Always include disclaimers: "This is general information, not personal medical advice"
- Provide actionable information when appropriate
- Be empathetic and supportive

**IMPORTANT:**
- You answer QUESTIONS, you don't diagnose CONDITIONS
- When someone has symptoms, they need clerking/diagnosis, not just information
- Always be conservative - if unsure, recommend professional consultation

Remember: Questions = answer, Complaints = handoff to clerking"""

    # Build message history
    messages = [SystemMessage(content=system_prompt)]
    
    # Add conversation history
    messages.extend(state["messages"][-10:])

    # Bind tools
    llm_with_tools = llm.bind_tools([specialist_handoff])
    
    # Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    # Check if tool was called
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        
        # Execute the handoff
        handoff_result = specialist_handoff.invoke(tool_call["args"])
        
        return {
            "active_node": handoff_result["active_node"],
            "awaiting_user_input": False
        }
    
    # No handoff - continue conversation
    return {
        "messages": [AIMessage(content=response.content)],
        "awaiting_user_input": True
    }


# ============================================================================
# NODE 5: CLERKING NODE (AI-based)
# ============================================================================

def clerking_node(state: AgentState, llm) -> AgentState:
    """
    Medical history collector - systematically clerks patient.
    Accumulates all conversation in clerking_convo state.
    """
    
    system_prompt = """You are a thorough medical history collection specialist (clerking agent). Your role:

**RESPONSIBILITIES:**
1. Systematically collect comprehensive medical history
2. Ask relevant follow-up questions
3. Cover all important aspects of the patient's complaint
4. Know when you have sufficient information

**CLERKING STRUCTURE (Follow this flow):**
1. **Chief Complaint**: What's the main problem? (already provided usually)
2. **History of Present Illness**:
   - When did it start?
   - How did it develop/progress?
   - Severity (scale 1-10)?
   - Character/quality of symptoms?
   - What makes it better/worse?
   - Associated symptoms?
3. **Past Medical History**:
   - Any chronic conditions? (diabetes, hypertension, asthma, etc.)
   - Previous hospitalizations or surgeries?
4. **Medications & Allergies**:
   - Current medications, supplements?
   - Any drug allergies?
5. **Social History** (brief):
   - Smoking/alcohol use?
   - Occupation?
   - Recent travel?
6. **Review of Systems** (if relevant):
   - Any other symptoms anywhere?

**CONVERSATION STYLE:**
- Ask 1-2 questions at a time (don't overwhelm)
- Be empathetic and reassuring
- Acknowledge their concerns
- Use simple language
- Build rapport

**COMPLETION CRITERIA:**
When you have covered the main points above and feel you have a clear picture of:
- The chief complaint and its details
- Relevant medical history
- Current medications and allergies
- Important context

Then use clerking_handoff to move to "soap_generation" with summary: "Clerking completed, ready for SOAP note generation"

**IMPORTANT:**
- Don't rush - be thorough but efficient
- Don't provide medical advice or diagnosis during clerking
- Focus on GATHERING information, not giving it
- Every question should have a purpose"""

    # Build message history
    messages = [SystemMessage(content=system_prompt)]
    
    # Add clerking conversation history if exists
    if state.get("clerking_convo"):
        messages.append(SystemMessage(content=f"Clerking so far:\n{state['clerking_convo']}"))
    
    # Add recent messages
    messages.extend(state["messages"][-5:])
    
    # Bind tools
    llm_with_tools = llm.bind_tools([clerking_handoff])
    
    # Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    # Accumulate clerking conversation
    last_user_msg = state["messages"][-1].content if state["messages"] else ""
    clerking_addition = f"\nPatient: {last_user_msg}\nDoctor: {response.content}\n"
    
    # Check if tool was called (clerking complete)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        
        # Execute the handoff
        handoff_result = clerking_handoff.invoke(tool_call["args"])
        
        return {
            "clerking_convo": state.get("clerking_convo", "") + clerking_addition,
            "active_node": handoff_result["active_node"],
            "awaiting_user_input": False
        }
    
    # Continue clerking
    return {
        "messages": [AIMessage(content=response.content)],
        "clerking_convo": state.get("clerking_convo", "") + clerking_addition,
        "awaiting_user_input": True
    }


# ============================================================================
# NODE 6: SOAP GENERATION NODE (AI-based)
# ============================================================================

def soap_generation_node(state: AgentState, llm) -> AgentState:
    """
    Generates SOAP note from clerking conversation.
    No tools, no user interaction - just summarization.
    """
    
    clerking_data = state.get("clerking_convo", "")
    
    soap_prompt = f"""You are a medical documentation specialist. Generate a detailed professional SOAP note from this patient Doctor interaction.

**CLERKING CONVERSATION:**
{clerking_data}

**Generate a detailed SOAP note with these sections:**

**S (Subjective):** Patient's description of symptoms, concerns, history
**O (Objective):** Any observable or measurable data mentioned (vital signs, physical findings if any)
**A (Assessment):** Your professional assessment of the likely condition(s) - differential diagnosis
**P (Plan):** Recommended next steps, investigations needed, referral specialty

Be professional, concise, and clinically relevant."""
    
    a = """**Additional fields:**
**Urgency Level:** low/medium/high/critical
**Recommended Specialty:** Most appropriate medical specialty for this case"""

    response = llm.invoke([HumanMessage(content=soap_prompt)])
    
    soap_summary = response.content.strip()
    
    
    return {
        "soap_summary": soap_summary,
        "active_node": "handoff",
        "awaiting_user_input": False
    }


# ============================================================================
# NODE 7: HANDOFF NODE (AI-based)
# ============================================================================

def handoff_node(state: AgentState, llm) -> AgentState:
    """
    Doctor matching and handoff agent.
    Gathers user preferences and finds appropriate doctor.
    """
    system_prompt = f"""You are a patient-doctor matching coordinator. Your role:
    
**RESPONSIBILITIES:**
1. Help patient find the right doctor for their needs
2. infer specialty of the required doctor from the patient-doctor conversation SOAP note
3. Gather preferences if not already clear: specialty, location, price range, experience level
4. Use doctor_search tool to find matching doctors
5. Present doctor options clearly and professionally

**CONVERSATION FLOW:**
1. If you have SOAP summary, acknowledge you understand their medical situation
2. Ask about preferences (if not already gathered):
   - Preferred location?
   - Budget/price range for consultation?
   - Preference for experience level? (junior/mid-level/senior doctors)
   - Need immediate availability or flexible timing?
3. Once you have sufficient preferences, use doctor_search tool
4. Present the best matching doctors with their details
5. Ask patient to select their preferred doctor

**DOCTOR SEARCH TOOL USAGE:**
Call doctor_search with:
- specialty: Infer From SOAP note recommendations
- location: User's preferred location
- max_price: User's budget (default 10000 if not specified)
- experience_level: User preference (default "any")
- availability: User's urgency needs

**PRESENTING DOCTORS:**
Show for each doctor:
- Name and specialty
- Rating and years of experience
- Consultation fee
- Location
- Languages spoken
- Available time slots
- Average response time

**COMPLETION:**
When patient selects a doctor and is ready to proceed, inform them:
"I'll now connect you with Dr. [Name]. They will receive your medical summary and reach out shortly."

**IMPORTANT:**
- Be helpful and patient
- Don't push expensive options
- Respect user's budget constraints
- Provide multiple options when possible
- Be transparent about doctor qualifications"""
    # Check if user selected a doctor (simple heuristic)
    last_msg = state["messages"][-1].content.lower()
    search_results = state.get("doctor_preferences", {}).get("search_results", [])
    
    if search_results and any(word in last_msg for word in ["select", "choose", "pick", "first", "second", "third", "1", "2", "3", "dr.", "doctor"]):
        # Simple selection logic - try to identify which doctor
        selected_doctor = None
        
        for i, doc in enumerate(search_results):
            if str(i+1) in last_msg or doc['name'].lower() in last_msg:
                selected_doctor = doc
                break
        
        if not selected_doctor:
            selected_doctor = search_results[0]  # Default to first
        
        return {
            "messages": [AIMessage(content=f"Perfect! I'll connect you with **Dr. {selected_doctor['name']}**. They will receive your medical summary and contact you at your earliest available slot: {selected_doctor['available_slots'][0]}. Is there anything else you'd like to know before I finalize the connection?")],
            "matched_doctor": selected_doctor,
            "awaiting_user_input": False,
            "conversation_ended": True
        }
    # Build message history
    messages = [SystemMessage(content=system_prompt)]
    

    # Add SOAP summary context
    if state.get("soap_summary"):
        messages.append(SystemMessage(content=f"Patient's Medical Summary:\n{state['soap_summary']}"))
    
    # Add previous doctor preferences if any
    if state.get("doctor_preferences"):
        messages.append(SystemMessage(content=f"Initial preferences: {json.dumps(state['doctor_preferences'])}"))
    
    # Add conversation history
    messages.extend(state["messages"][-10:])
    
    # Bind tools
    llm_with_tools = llm.bind_tools([doctor_search])
    
    # Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    # Check if tool was called
    if response.tool_calls:
     
        tool_call = response.tool_calls[0]
        
        # Execute doctor search
        doctors = doctor_search.invoke(tool_call["args"])
        
        # Format doctor results
        if doctors:
            doctors_text = "\n\n**Available Doctors:**\n\n"
            for i, doc in enumerate(doctors, 1):
                doctors_text += f"""**{i}. {doc['name']}** - {doc['specialty']}
   ⭐ Rating: {doc['rating']}/5.0 | 📅 Experience: {doc['years_experience']} years
   💰 Fee: ₦{doc['consultation_fee']:,} | 📍 {doc['location']}
   🗣️ Languages: {', '.join(doc['languages'])}
   ⏰ Available: {', '.join(doc['available_slots'][:3])}
   ⚡ Avg Response: {doc['response_time_avg']}

"""
            
            doctors_text += "\nWhich doctor would you prefer? Please let me know by number or name."
            
            # Store doctors in state for selection
            return {
                "messages": [AIMessage(content=response.content + "\n\n" + doctors_text)],
                "doctor_preferences": {
                    **state.get("doctor_preferences", {}),
                    "search_results": doctors
                },
                "awaiting_user_input": True
            }
        else:
            return {
                "messages": [AIMessage(content="I couldn't find any doctors matching those specific criteria. Could you adjust your preferences? For example, a different location or higher budget?")],
                "awaiting_user_input": True
            }
    
    
    
    # Continue gathering preferences
    return {
        "messages": [AIMessage(content=response.content)],
        "awaiting_user_input": True
    }



# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_from_controller(state: AgentState) -> str:
    """Route from controller to active node"""
    active = state.get("active_node", "orchestrator")
    
    if active == "orchestrator":
        return "orchestrator"
    elif active == "specialist":
        return "specialist"
    elif active == "clerking":
        return "clerking"
    elif active == "handoff":
        return "handoff"
    else:
        return "orchestrator"  # Default


def route_from_orchestrator(state: AgentState) -> str:
    """Route from orchestrator"""
    active = state.get("active_node", "end")
    if state.get("awaiting_user_input") and active == "orchestrator":
        return "end"  # Go to end to wait for user
    elif active == "clerking":
        return "clerking"  # Handoff occurred
    elif active == "specialist":
        return "specialist"
    else:
        return "end"


def route_from_specialist(state: AgentState) -> str:
    """Route from specialist"""
    active = state.get("active_node", "end")
    if state.get("awaiting_user_input") and active == "specialist":
        return "end"
    elif active == "clerking":
        return "clerking"
    else: return "end"


def route_from_clerking(state: AgentState) -> str:
    """Route from clerking"""
    active = state.get("active_node", "end")
    if state.get("awaiting_user_input") and active == "clerking":
        return "end"
    elif active == "soap_generation":
        return "soap_generation"
    else: return "end"




# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_medical_assistant_graph(api_key: str) -> StateGraph:
    """
    Creates the complete LangGraph workflow.
    """
    
    # Initialize LLM
    llm = initialize_llm(api_key)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("controller", controller_node)
    workflow.add_node("orchestrator", lambda state: orchestrator_node(state, llm))
    workflow.add_node("specialist", lambda state: specialist_node(state, llm))
    workflow.add_node("clerking", lambda state: clerking_node(state, llm))
    workflow.add_node("soap_generation", lambda state: soap_generation_node(state, llm))
    workflow.add_node("handoff", lambda state: handoff_node(state, llm))
    
    # Set entry point
    workflow.set_entry_point("controller")
    
    
    # Controller routes to active node
    workflow.add_conditional_edges(
        "controller",
        route_from_controller,
        {
            "orchestrator": "orchestrator",
            "specialist": "specialist",
            "clerking": "clerking",
            "handoff": "handoff",
            #"soap_generation": "soap_generation"
        }
    )
    
    # Each node can either return to controller (after handoff) or end (await user input)
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "specialist": "specialist",
            "clerking": "clerking",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "specialist",
        route_from_specialist,
        {
            "clerking": "clerking",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "clerking",
        route_from_clerking,
        {
            "soap_generation": "soap_generation",
            "end": END
        }
    )
    
    workflow.add_edge("soap_generation", "handoff")
    
    workflow.add_edge("handoff", END)
    
    return workflow.compile()


# ============================================================================
# CONVERSATION RUNNER
# ============================================================================

def run_conversation_turn(graph, user_input: str, state: AgentState = None) -> AgentState:
    """
    Process a single user message through the graph.
    """
    
    # Initialize state if first interaction
    if state is None:
        state = {
            "messages": [],
            "active_node": None,
            "handoff_summary": None,
            "clerking_convo": "",
            "soap_summary": None,
            "doctor_preferences": {},
            "matched_doctor": None,
            "awaiting_user_input": False,
            "conversation_ended": False
        }
    
    # Add user message to state
    state["messages"].append(HumanMessage(content=user_input))
    
    # Run the graph
    result = graph.invoke(state)
    
    return result


# ============================================================================
# MAIN EXECUTION & DEMO
# ============================================================================


# ==================== INITIALIZE SYSTEM ====================
print("\n" + "="*70)
print("🏥 REMOTE MEDICAL ASSISTANT - Multi-Agent System")
print("="*70)
print("\n📋 Initializing agents...")

graph = create_medical_assistant_graph(GEMINI_API_KEY)

print("✅ System ready!")
print("\n" + "="*70)
print("ARCHITECTURE:")
print("  • Orchestrator: Handles greetings & general chat")
print("  • Specialist: Answers medical questions")
print("  • Clerking: Collects medical history")
print("  • SOAP Generator: Creates medical summary")
print("  • Handoff: Matches patients with doctors")
print("="*70)

# ==================== INTERACTIVE MODE ====================
print("\n🗣️  INTERACTIVE MODE")
#print("Type your messages below. Type 'quit' or 'exit' to end.\n")

state = None
conversation_count = 0

@app.post(
"/conversation", 
response_model=AgentResponse,
summary="Receives a user message, pass to the agent and return agent response."
)
def handle_agent_interaction(user_input: UserMessage):
    """
    This POST request:
    1. Receives the user's and message.
    2. pass it to the agent
    3. Generates and returns a agent response.
    """
    global state, conversation_count
    state = run_conversation_turn(graph, user_input.message, state)

    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(F"\n👤 YOU: {user_input.message}")
            print(f"\n🤖 ASSISTANT: {last_message.content}")
            conversation_count+=1
            print(f"\nconversation count: {conversation_count}")
    # Display state info (optional - for debugging)
    if state.get("active_node"):
        print(f"\n📊 [Active Node: {state['active_node']}]\n", end="\n")
    message = last_message.content
    return AgentResponse(
        message=message
    )



# ============================================================================
# ENTRY POINT
# ============================================================================

#if __name__ == "__main__":
#        main()


# ============================================================================
# DOCUMENTATION & USAGE
# ============================================================================

"""
═══════════════════════════════════════════════════════════════════════════
                    MEDICAL ASSISTANT SYSTEM V2 - DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════

ARCHITECTURE OVERVIEW:
=====================

The system uses 8 nodes in a directed graph:

1. START → Initializes the conversation state
2. CONTROLLER → Routes messages to the appropriate active agent
3. ORCHESTRATOR → Receptionist for greetings and general chat
4. SPECIALIST → Answers medical questions and provides information
5. CLERKING → Systematically collects medical history
6. SOAP_GENERATION → Creates structured medical summary
7. HANDOFF → Matches patient with appropriate doctor
8. END → Terminal node for user input or completion

CONVERSATION FLOWS:
==================

Flow 1: General Greeting
  START → CONTROLLER → ORCHESTRATOR → END (awaiting user)

Flow 2: Medical Question
  START → CONTROLLER → ORCHESTRATOR → (handoff) → CONTROLLER → SPECIALIST → END

Flow 3: Medical Complaint (Full Flow)
  START → CONTROLLER → ORCHESTRATOR → (handoff) → CONTROLLER → CLERKING
  → (multiple turns) → CONTROLLER → SOAP_GENERATION → CONTROLLER → HANDOFF
  → (doctor matching) → END

HANDOFF MECHANISM:
=================

Each agent has a specific handoff tool:
- orchestrator_handoff: Routes to specialist or clerking
- specialist_handoff: Routes to clerking
- clerking_handoff: Routes to soap_generation

Handoffs include:
- Target node name
- Context summary for the next agent

STATE VARIABLES:
===============

Key state variables:
- messages: Full conversation history
- active_node: Current controlling agent
- handoff_summary: Context for handoff
- clerking_convo: Accumulated clerking dialogue
- soap_summary: Generated SOAP note
- doctor_preferences: User's doctor selection criteria
- matched_doctor: Selected doctor information

INSTALLATION:
============

pip install langgraph langchain-google-genai langchain-core

USAGE:
======

1. Interactive Mode (default):
   python medical_assistant.py

2. Test Mode (automated scenarios):
   python medical_assistant.py test

3. Programmatic Usage:
   ```python
   graph = create_medical_assistant_graph("YOUR_API_KEY")
   state = run_conversation_turn(graph, "Hello!", None)
   state = run_conversation_turn(graph, "I have a headache", state)
   ```

CUSTOMIZATION:
=============

1. Replace System Prompts:
   Edit the system_prompt in each node function to customize behavior

2. Modify Clerking Structure:
   Update clerking_node system prompt with your preferred medical history format

3. Add Custom Tools:
   Define new @tool functions and bind them to appropriate nodes

4. Integrate Real Database:
   Replace doctor_search mock data with actual database queries

5. Add More Nodes:
   Create new nodes (e.g., prescription_generator, lab_recommender)
   Add them to the graph with appropriate edges

PRODUCTION DEPLOYMENT:
=====================

1. API Wrapper:
   ```python
   from fastapi import FastAPI, WebSocket
   
   app = FastAPI()
   graph = create_medical_assistant_graph(API_KEY)
   user_states = {}
   
   @app.websocket("/ws/{user_id}")
   async def websocket_endpoint(websocket: WebSocket, user_id: str):
       await websocket.accept()
       state = user_states.get(user_id)
       
       while True:
           data = await websocket.receive_text()
           state = run_conversation_turn(graph, data, state)
           user_states[user_id] = state
           
           response = state["messages"][-1].content
           await websocket.send_text(response)
   ```

2. State Persistence:
   Store conversation state in Redis/PostgreSQL between sessions

3. Replace Mistral Integration:
   In specialist_node, replace llm with your Mistral API client

4. Monitoring:
   Add logging, analytics, and error tracking at each node

5. Security:
   - Add authentication
   - Encrypt sensitive medical data
   - Implement rate limiting
   - Add audit trails

TESTING:
========

The system includes comprehensive test scenarios:
- General greetings
- Medical questions
- Full clerking workflow
- Doctor matching
- Edge cases

Run with: python medical_assistant.py test

TROUBLESHOOTING:
===============

Common Issues:

1. "Invalid tool call":
   - Check tool parameter names match exactly
   - Ensure tool functions return proper dictionaries

2. "Node not routing correctly":
   - Verify routing functions check the right state variables
   - Add debug prints in routing functions

3. "Conversation stuck in loop":
   - Check awaiting_user_input flags
   - Verify END conditions in routing logic

4. "LLM not calling tools":
   - Review system prompts for clear tool usage instructions
   - Check llm.bind_tools() is called correctly

SUPPORT & CONTRIBUTION:
======================

This is a production-ready foundation. Customize as needed for your use case.

Key Extension Points:
- Add video consultation integration
- Implement prescription management
- Add lab test recommendations
- Create payment processing
- Build doctor dashboard
- Add multi-language support

═══════════════════════════════════════════════════════════════════════════
"""